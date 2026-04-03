"""Stage 4: Fine-tuning on curated aligned data with classifier-free guidance.

Fine-tunes the Stage 3 model on high-quality aligned speech translation pairs.
Enables classifier-free guidance (CFG) training by randomly dropping the voice
conditioning embedding with probability cfg_drop_prob.

Usage (DDP, 2 GPUs):
    torchrun --nproc_per_node=2 training/train_finetune.py \
        --config configs/model_100m.yaml \
        --manifest /path/to/finetune_manifest.tsv \
        --stage3_ckpt /path/to/stage3/checkpoint_final.pt \
        --output_dir /kaggle/working/stage4
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
    save_to_hf_hub,
)


def train_step_cfg(
    model: nn.Module,
    batch: dict,
    scaler: GradScaler,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    cfg_drop_prob: float = 0.2,
) -> dict:
    """Training step with classifier-free guidance dropout.

    With probability cfg_drop_prob, the voice_category is set to None,
    training the model to generate without voice conditioning. At inference
    time, CFG uses the difference between conditioned and unconditioned
    logits to improve quality.
    """
    source_audio = batch["source_audio"].to(device)
    target_audio = batch["target_audio"].to(device)
    text_tokens = batch["text"].to(device)
    voice_category = batch["voice_category"].to(device)

    # CFG dropout: randomly drop voice conditioning
    B = source_audio.shape[0]
    drop_mask = torch.rand(B, device=device) < cfg_drop_prob
    if drop_mask.any():
        # Set dropped samples' voice category to None by using a special
        # "unconditional" forward pass. We do two passes: conditioned and
        # unconditional, then combine the losses.
        voice_category_cfg = voice_category.clone()
        # We handle this by setting dropped categories to a neutral value
        # and masking the embedding addition in the model
        # For simplicity, set dropped to category 2 (neutral) and zero out
        # the voice embedding contribution for dropped samples
        pass

    with autocast(dtype=torch.float16):
        # Conditioned forward pass (all samples)
        outputs = model(
            target_audio_tokens=target_audio,
            source_audio_tokens=source_audio,
            text_tokens=text_tokens,
            voice_category=voice_category if not drop_mask.all() else None,
            predict_source=False,  # Stage 4: no source reconstruction
        )

        inner_model = model.module if hasattr(model, "module") else model
        losses = inner_model.compute_loss(
            outputs,
            target_audio_tokens=target_audio,
            source_audio_tokens=None,
            text_tokens=text_tokens,
        )
        loss = losses["total_loss"]

        # If we have dropped samples, also compute unconditional loss
        if drop_mask.any() and not drop_mask.all():
            uncond_outputs = model(
                target_audio_tokens=target_audio[drop_mask],
                source_audio_tokens=source_audio[drop_mask],
                text_tokens=text_tokens[drop_mask],
                voice_category=None,
                predict_source=False,
            )
            uncond_losses = inner_model.compute_loss(
                uncond_outputs,
                target_audio_tokens=target_audio[drop_mask],
                source_audio_tokens=None,
                text_tokens=text_tokens[drop_mask],
            )
            # Weight the unconditional loss proportionally
            uncond_weight = drop_mask.float().mean()
            cond_weight = 1.0 - uncond_weight
            loss = cond_weight * loss + uncond_weight * uncond_losses["total_loss"]

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()

    return {k: v.item() for k, v in losses.items()}


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Fine-tuning with CFG")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to TSV manifest for fine-tuning data")
    parser.add_argument("--stage3_ckpt", type=str, required=True,
                        help="Path to Stage 3 checkpoint")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--hf_repo", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    setup_ddp(local_rank, world_size)
    device = get_device(local_rank)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]["stage4_finetune"]
    common_cfg = config["training"]["common"]
    set_seed(common_cfg["seed"] + local_rank)

    # Create model and load Stage 3 weights
    model = HibikiModel.from_config(config)

    if args.resume:
        if local_rank == 0:
            print(f"Resuming Stage 4 from {args.resume}")
    else:
        if local_rank == 0:
            print(f"Loading Stage 3 checkpoint from {args.stage3_ckpt}")
        ckpt = torch.load(args.stage3_ckpt, map_location="cpu")
        state_dict = ckpt.get("model", ckpt.get("model_state_dict", {}))
        # Strip DDP prefix
        clean_state = {}
        for k, v in state_dict.items():
            clean_state[k.replace("module.", "")] = v
        model.load_state_dict(clean_state, strict=False)

    enable_gradient_checkpointing(model)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if local_rank == 0:
        print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Dataset (curated aligned data — typically smaller, higher quality)
    dataset = S2STDataset(
        manifest_path=args.manifest,
        max_frames=config["model"]["temporal"]["max_seq_len"],
        acoustic_delay=config["model"]["tokens"]["acoustic_delay"],
        noise_augmentation=False,  # No noise in fine-tuning
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

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        if local_rank == 0:
            print(f"Resumed from step {start_step}")

    logger = TrainingLogger(
        log_dir=os.path.join(args.output_dir, "logs"),
        rank=local_rank,
        log_interval=common_cfg.get("log_every_steps", 100),
    )

    # CFG configuration
    cfg_drop_prob = config["model"]["voice_transfer"]["cfg_drop_prob"]
    cfg_training = train_cfg.get("cfg_training", True)

    os.makedirs(args.output_dir, exist_ok=True)
    model.train()
    step = start_step

    if local_rank == 0:
        print(f"Starting Stage 4 fine-tuning: {train_cfg['max_steps']} steps")
        print(f"  CFG training: {cfg_training}, drop prob: {cfg_drop_prob}")
        print(f"  Batch size (local): {train_cfg['local_batch_size']}")
        print(f"  Learning rate: {train_cfg['learning_rate']}")

    while step < train_cfg["max_steps"]:
        sampler.set_epoch(step // len(loader) if len(loader) > 0 else step)
        for batch in loader:
            logger.start_step()

            metrics = train_step_cfg(
                model, batch, scaler, optimizer, scheduler,
                device=device,
                cfg_drop_prob=cfg_drop_prob if cfg_training else 0.0,
            )

            step += 1

            if local_rank == 0:
                B = batch["source_audio"].shape[0]
                T = batch["source_audio"].shape[2]
                Q = batch["source_audio"].shape[1]
                tokens = B * T * (Q * 2 + 1)

                logger.log_step(
                    step, metrics,
                    lr=scheduler.get_last_lr()[0],
                    tokens_processed=tokens,
                )

            if step % common_cfg["save_every_steps"] == 0 and local_rank == 0:
                ckpt_path = os.path.join(args.output_dir, f"checkpoint_{step}.pt")
                save_checkpoint(model, optimizer, scheduler, scaler, step, ckpt_path)

                if args.hf_repo and args.hf_token:
                    try:
                        save_to_hf_hub(
                            model, args.hf_repo, args.hf_token,
                            commit_message=f"Stage 4 checkpoint step {step}",
                        )
                    except Exception as e:
                        print(f"  HF upload failed: {e}")

            if step >= train_cfg["max_steps"]:
                break

    # Final save
    if local_rank == 0:
        final_path = os.path.join(args.output_dir, "checkpoint_final.pt")
        save_checkpoint(model, optimizer, scheduler, scaler, step, final_path)
        print(f"Stage 4 complete. Final checkpoint at step {step}")

        if args.hf_repo and args.hf_token:
            try:
                save_to_hf_hub(
                    model, args.hf_repo, args.hf_token,
                    commit_message="Stage 4 final checkpoint",
                )
            except Exception as e:
                print(f"HF upload failed: {e}")

    logger.close()
    cleanup_ddp()


if __name__ == "__main__":
    main()
