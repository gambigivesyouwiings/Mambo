"""Stage 2: Audio Pretraining — train on monolingual en+sw audio.

Continues from the text-pretrained Temporal Transformer and adds
the Depth Transformer. Trains on single-stream audio (no translation)
using next-token prediction over all Q codebook levels.

After this stage, Depth Transformer weights are duplicated for
multistream modeling in Stage 3.

Usage (DDP, 2 GPUs):
    torchrun --nproc_per_node=2 training/train_audio.py \
        --config configs/model_100m.yaml \
        --data_dir /path/to/audio_tokens \
        --stage1_ckpt /path/to/stage1/checkpoint_final.pt \
        --output_dir /kaggle/working/stage2
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

from model.temporal_transformer import TemporalTransformer
from model.depth_transformer import DepthTransformer
from data.dataset import AudioPretrainDataset
from training.utils import (
    setup_ddp, cleanup_ddp, get_device,
    create_optimizer, create_cosine_scheduler,
    save_checkpoint, load_checkpoint, set_seed,
    TrainingLogger, enable_gradient_checkpointing,
)


class AudioPretrainModel(nn.Module):
    """Wrapper combining Temporal + Depth for single-stream audio pretraining."""

    def __init__(self, temporal: TemporalTransformer, depth: DepthTransformer, num_codebooks: int = 8):
        super().__init__()
        self.temporal = temporal
        self.depth = depth
        self.num_codebooks = num_codebooks
        self.pad_id = 0
        self.bos_id = 1

    def forward(self, audio_tokens: torch.Tensor):
        """
        Args:
            audio_tokens: (B, Q, T) pre-encoded Mimi tokens

        Returns:
            audio_logits: (B, Q, T, codebook_size) logits
            loss: scalar cross-entropy loss
        """
        B, Q, T = audio_tokens.shape

        # Shift right for teacher forcing
        input_tokens = torch.full_like(audio_tokens, self.bos_id)
        input_tokens[:, :, 1:] = audio_tokens[:, :, :-1]

        # Embed: only target stream (single stream pretraining)
        h = self.temporal.embed_tokens(
            text_tokens=None,
            target_audio_tokens=input_tokens,
            source_audio_tokens=None,
        )

        # Temporal transformer
        z, _, _ = self.temporal(h)

        # Depth transformer: predict audio tokens
        audio_logits, _ = self.depth(
            z, target_tokens=audio_tokens, source_tokens=None,
        )

        # Loss over all codebooks
        loss = F.cross_entropy(
            audio_logits.reshape(-1, audio_logits.shape[-1]),
            audio_tokens.reshape(-1),
            ignore_index=self.pad_id,
        )

        return audio_logits, loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    setup_ddp(local_rank, world_size)
    device = get_device(local_rank)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]["stage2_audio"]
    set_seed(config["training"]["common"]["seed"] + local_rank)

    # Create temporal transformer and load Stage 1 weights
    model_cfg = config["model"]
    temporal = TemporalTransformer(
        d_model=model_cfg["temporal"]["d_model"],
        ffn_dim=model_cfg["temporal"]["ffn_dim"],
        num_layers=model_cfg["temporal"]["num_layers"],
        num_heads=model_cfg["temporal"]["num_heads"],
        head_dim=model_cfg["temporal"]["head_dim"],
        max_seq_len=model_cfg["temporal"]["max_seq_len"],
        dropout=model_cfg["temporal"]["dropout"],
        text_vocab_size=model_cfg["tokens"]["text_vocab_size"],
        audio_codebook_size=model_cfg["tokens"]["audio_codebook_size"],
        num_codebooks=model_cfg["codec"]["num_codebooks"],
    )

    # Load Stage 1 checkpoint (temporal transformer only)
    if local_rank == 0:
        print(f"Loading Stage 1 checkpoint from {args.stage1_ckpt}")
    ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
    # Extract temporal transformer state dict (strip DDP prefix if present)
    state_dict = {}
    for k, v in ckpt["model_state_dict"].items():
        k = k.replace("module.", "")
        if not k.startswith("depth."):
            state_dict[k] = v
    temporal.load_state_dict(state_dict, strict=False)

    # Create depth transformer (randomly initialized)
    depth = DepthTransformer(
        d_model=model_cfg["depth"]["d_model"],
        ffn_dim=model_cfg["depth"]["ffn_dim"],
        num_layers_per_codebook=model_cfg["depth"]["num_layers_per_codebook"],
        num_codebooks=model_cfg["codec"]["num_codebooks"],
        num_streams=1,  # single stream for pretraining
        weight_sharing_start=model_cfg["depth"]["weight_sharing_start"],
        audio_codebook_size=model_cfg["tokens"]["audio_codebook_size"],
        temporal_d_model=model_cfg["temporal"]["d_model"],
        dropout=model_cfg["depth"]["dropout"],
    )

    model = AudioPretrainModel(temporal, depth, model_cfg["codec"]["num_codebooks"])
    enable_gradient_checkpointing(model)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    if local_rank == 0:
        print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    # Dataset
    dataset = AudioPretrainDataset(
        data_dir=args.data_dir,
        num_codebooks=model_cfg["codec"]["num_codebooks"],
        max_len=model_cfg["temporal"]["max_seq_len"],
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["local_batch_size"],
        sampler=sampler,
        num_workers=config["training"]["common"]["num_workers"],
        pin_memory=config["training"]["common"]["pin_memory"],
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
        enabled=(local_rank == 0),
    )

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    model.train()
    step = start_step
    grad_accum = train_cfg["gradient_accumulation"]
    accum_loss = 0.0

    while step < train_cfg["max_steps"]:
        sampler.set_epoch(step)
        for batch in loader:
            audio_tokens = batch["audio_tokens"].to(device)

            with autocast(dtype=torch.float16):
                _, loss = model(audio_tokens)
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            accum_loss += loss.item()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                if local_rank == 0:
                    logger.log_step(step, {
                        "loss": accum_loss,
                        "lr": scheduler.get_last_lr()[0],
                    })
                accum_loss = 0.0

            step += 1

            if step % config["training"]["common"]["save_every_steps"] == 0 and local_rank == 0:
                save_checkpoint(
                    model, optimizer, scheduler, scaler, step,
                    os.path.join(args.output_dir, f"checkpoint_{step}.pt"),
                )

            if step >= train_cfg["max_steps"]:
                break

    # Final save
    if local_rank == 0:
        save_checkpoint(
            model, optimizer, scheduler, scaler, step,
            os.path.join(args.output_dir, "checkpoint_final.pt"),
        )
        print(f"Stage 2 complete. Final checkpoint at step {step}")
        print("NOTE: Before Stage 3, duplicate Depth Transformer weights for multistream.")

    cleanup_ddp()


if __name__ == "__main__":
    main()
