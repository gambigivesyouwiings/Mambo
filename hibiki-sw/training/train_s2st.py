"""Stage 3: Speech-to-Speech Translation training.

Trains the full Hibiki model (Temporal + Depth) on paired en<->sw audio
using the multi-stream architecture with inner monologue text prediction.

Loads Stage 2 checkpoint (temporal + single-stream depth) and duplicates
the depth transformer weights for two-stream (source + target) processing.

Usage (DDP, 2 GPUs):
    torchrun --nproc_per_node=2 training/train_s2st.py \
        --config configs/model_100m.yaml \
        --manifest /path/to/s2st_manifest.tsv \
        --stage2_ckpt /path/to/stage2/checkpoint_final.pt \
        --output_dir /kaggle/working/stage3
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.hibiki_model import HibikiModel
from data.dataset import S2STDataset, collate_s2st
from training.utils import (
    setup_ddp, cleanup_ddp, get_device,
    create_optimizer, create_cosine_scheduler,
    save_checkpoint, load_checkpoint, set_seed,
    TrainingLogger, enable_gradient_checkpointing,
    augment_audio_tokens, save_to_hf_hub,
)


def duplicate_depth_weights(stage2_ckpt: dict, model: HibikiModel) -> None:
    """Initialize the 2-stream depth transformer from a 1-stream Stage 2 checkpoint.

    Stage 2 trains a single-stream depth transformer. For Stage 3, the depth
    transformer handles two streams (target + source). We copy the Stage 2
    weights into both the target and source output heads, and duplicate the
    token embedding tables for the source stream.
    """
    model_state = model.state_dict()
    ckpt_state = stage2_ckpt

    # Map Stage 2 keys to the full model
    loaded = 0
    for key, val in ckpt_state.items():
        # Strip DDP/wrapper prefixes
        clean_key = key.replace("module.", "")

        # Stage 2 wraps as AudioPretrainModel with .temporal and .depth
        if clean_key.startswith("temporal."):
            target_key = clean_key
        elif clean_key.startswith("depth."):
            target_key = clean_key
        else:
            continue

        if target_key in model_state and model_state[target_key].shape == val.shape:
            model_state[target_key] = val
            loaded += 1

    model.load_state_dict(model_state, strict=False)
    print(f"  Loaded {loaded} parameter tensors from Stage 2 checkpoint")

    # Duplicate source output heads from target output heads
    # (input_output_norms/heads initialized from output_norms/heads)
    with torch.no_grad():
        for q in range(model.depth.num_codebooks):
            model.depth.input_output_norms[q].weight.copy_(
                model.depth.output_norms[q].weight
            )
            model.depth.input_output_heads[q].weight.copy_(
                model.depth.output_heads[q].weight
            )
    print("  Duplicated depth output heads for source stream")


def train_step(
    model: nn.Module,
    batch: dict,
    scaler: GradScaler,
    optimizer: torch.optim.Optimizer,
    grad_accum: int,
    step_in_accum: int,
    device: torch.device,
    noise_augmentation: bool = False,
    predict_source: bool = True,
) -> dict:
    """Single training step with gradient accumulation and mixed precision."""
    source_audio = batch["source_audio"].to(device)
    target_audio = batch["target_audio"].to(device)
    text_tokens = batch["text"].to(device)
    voice_category = batch["voice_category"].to(device)

    # Noise augmentation on source audio
    if noise_augmentation:
        source_audio = augment_audio_tokens(source_audio, noise_prob=0.05)

    with autocast(dtype=torch.float16):
        outputs = model(
            target_audio_tokens=target_audio,
            source_audio_tokens=source_audio,
            text_tokens=text_tokens,
            voice_category=voice_category,
            predict_source=predict_source,
        )

        # Use the model's compute_loss (handles DDP wrapping)
        inner_model = model.module if hasattr(model, "module") else model
        losses = inner_model.compute_loss(
            outputs,
            target_audio_tokens=target_audio,
            source_audio_tokens=source_audio if predict_source else None,
            text_tokens=text_tokens,
        )

        loss = losses["total_loss"] / grad_accum

    scaler.scale(loss).backward()

    metrics = {k: v.item() for k, v in losses.items()}

    # Optimizer step at the end of accumulation
    if (step_in_accum + 1) % grad_accum == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Stage 3: S2ST Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to TSV manifest for S2ST data")
    parser.add_argument("--stage2_ckpt", type=str, required=True,
                        help="Path to Stage 2 checkpoint")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from Stage 3 checkpoint")
    parser.add_argument("--hf_repo", type=str, default=None,
                        help="HuggingFace repo for checkpoint backup")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace API token")
    args = parser.parse_args()

    # DDP setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    setup_ddp(local_rank, world_size)
    device = get_device(local_rank)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]["stage3_s2st"]
    common_cfg = config["training"]["common"]
    set_seed(common_cfg["seed"] + local_rank)

    # Create full Hibiki model
    model = HibikiModel.from_config(config)

    if args.resume:
        # Resume from Stage 3 checkpoint
        if local_rank == 0:
            print(f"Resuming Stage 3 from {args.resume}")
    else:
        # Initialize from Stage 2 checkpoint
        if local_rank == 0:
            print(f"Loading Stage 2 checkpoint from {args.stage2_ckpt}")
        ckpt = torch.load(args.stage2_ckpt, map_location="cpu")
        ckpt_state = ckpt.get("model", ckpt.get("model_state_dict", {}))
        duplicate_depth_weights(ckpt_state, model)

    enable_gradient_checkpointing(model)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if local_rank == 0:
        print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Dataset
    dataset = S2STDataset(
        manifest_path=args.manifest,
        max_frames=config["model"]["temporal"]["max_seq_len"],
        acoustic_delay=config["model"]["tokens"]["acoustic_delay"],
        noise_augmentation=train_cfg.get("noise_augmentation", False),
        noise_prob=0.05,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["local_batch_size"],
        sampler=sampler,
        collate_fn=collate_s2st,
        num_workers=common_cfg["num_workers"],
        pin_memory=common_cfg["pin_memory"],
        drop_last=True,
    )

    # Optimizer & scheduler
    optimizer = create_optimizer(
        model, lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        betas=tuple(train_cfg["betas"]),
    )
    scheduler = create_cosine_scheduler(
        optimizer,
        warmup_steps=train_cfg["warmup_steps"],
        max_steps=train_cfg["max_steps"],
    )
    scaler = GradScaler()

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        if local_rank == 0:
            print(f"Resumed from step {start_step}")

    # Logger
    logger = TrainingLogger(
        log_dir=os.path.join(args.output_dir, "logs"),
        rank=local_rank,
        log_interval=common_cfg.get("log_every_steps", 100),
    )

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    model.train()
    step = start_step
    grad_accum = train_cfg["gradient_accumulation"]
    predict_source = train_cfg.get("predict_source_stream", True)

    if local_rank == 0:
        print(f"Starting Stage 3 training: {train_cfg['max_steps']} steps")
        print(f"  Effective batch size: {train_cfg['local_batch_size'] * grad_accum * world_size}")
        print(f"  Predict source stream: {predict_source}")
        print(f"  Noise augmentation: {train_cfg.get('noise_augmentation', False)}")

    while step < train_cfg["max_steps"]:
        sampler.set_epoch(step // len(loader) if len(loader) > 0 else step)
        for batch in loader:
            logger.start_step()

            metrics = train_step(
                model, batch, scaler, optimizer,
                grad_accum=grad_accum,
                step_in_accum=step % grad_accum,
                device=device,
                noise_augmentation=train_cfg.get("noise_augmentation", False),
                predict_source=predict_source,
            )

            # Only count a "step" after full gradient accumulation
            if (step + 1) % grad_accum == 0:
                scheduler.step()

            step += 1

            if local_rank == 0:
                # Compute token count for throughput
                B = batch["source_audio"].shape[0]
                T = batch["source_audio"].shape[2]
                Q = batch["source_audio"].shape[1]
                tokens = B * T * (Q * 2 + 1)  # source audio + target audio + text

                logger.log_step(
                    step, metrics,
                    lr=scheduler.get_last_lr()[0],
                    tokens_processed=tokens,
                )

            # Save checkpoint
            if step % common_cfg["save_every_steps"] == 0 and local_rank == 0:
                ckpt_path = os.path.join(args.output_dir, f"checkpoint_{step}.pt")
                save_checkpoint(model, optimizer, scheduler, scaler, step, ckpt_path)
                print(f"Saved checkpoint at step {step}")

                # Optional: backup to HuggingFace Hub
                if args.hf_repo and args.hf_token:
                    try:
                        save_to_hf_hub(
                            model, args.hf_repo, args.hf_token,
                            commit_message=f"Stage 3 checkpoint step {step}",
                        )
                        print(f"  Uploaded to {args.hf_repo}")
                    except Exception as e:
                        print(f"  HF upload failed: {e}")

            if step >= train_cfg["max_steps"]:
                break

    # Final save
    if local_rank == 0:
        final_path = os.path.join(args.output_dir, "checkpoint_final.pt")
        save_checkpoint(model, optimizer, scheduler, scaler, step, final_path)
        print(f"Stage 3 complete. Final checkpoint at step {step}")

        if args.hf_repo and args.hf_token:
            try:
                save_to_hf_hub(
                    model, args.hf_repo, args.hf_token,
                    commit_message="Stage 3 final checkpoint",
                )
            except Exception as e:
                print(f"HF upload failed: {e}")

    logger.close()
    cleanup_ddp()


if __name__ == "__main__":
    main()
