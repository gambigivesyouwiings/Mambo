"""Stage 1: Text Adaptation — continue pretraining a small LM on en+sw text.

Initializes the Temporal Transformer from a pretrained multilingual LM
and adapts it on English + Swahili text data using next-token prediction.

Usage (DDP, 2 GPUs):
    torchrun --nproc_per_node=2 training/train_text.py \
        --config configs/model_100m.yaml \
        --data_dir /path/to/text_tokens \
        --output_dir /kaggle/working/stage1
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.temporal_transformer import TemporalTransformer
from data.dataset import TextDataset
from training.utils import (
    setup_ddp, cleanup_ddp, get_device,
    create_optimizer, create_cosine_scheduler,
    save_checkpoint, load_checkpoint, set_seed,
    TrainingLogger, enable_gradient_checkpointing,
)


def train_step(model, batch, scaler, optimizer, scheduler, device):
    """Single training step with mixed precision."""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    with autocast(dtype=torch.float16):
        # Embed text tokens only (no audio in Stage 1)
        h = model.module.text_embed(input_ids)

        # Forward through temporal transformer
        z, text_logits, _ = model.module(h) if hasattr(model, 'module') else model(h)

        # Next-token prediction loss
        loss = nn.functional.cross_entropy(
            text_logits.reshape(-1, text_logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=0,  # pad token
        )

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()

    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrained_lm", type=str, default=None,
                        help="HuggingFace model ID for weight initialization")
    args = parser.parse_args()

    # DDP setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    setup_ddp(local_rank, world_size)
    device = get_device(local_rank)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]["stage1_text"]
    set_seed(config["training"]["common"]["seed"] + local_rank)

    # Create model
    model_cfg = config["model"]
    model = TemporalTransformer(
        d_model=model_cfg["temporal"]["d_model"],
        ffn_dim=model_cfg["temporal"]["ffn_dim"],
        num_layers=model_cfg["temporal"]["num_layers"],
        num_heads=model_cfg["temporal"]["num_heads"],
        head_dim=model_cfg["temporal"]["head_dim"],
        max_seq_len=train_cfg["seq_length"],
        dropout=model_cfg["temporal"]["dropout"],
        text_vocab_size=model_cfg["tokens"]["text_vocab_size"],
        audio_codebook_size=model_cfg["tokens"]["audio_codebook_size"],
        num_codebooks=model_cfg["codec"]["num_codebooks"],
    )

    if args.pretrained_lm:
        # Load and adapt weights from pretrained LM
        # This is model-specific; users should implement weight mapping
        if local_rank == 0:
            print(f"Loading pretrained weights from {args.pretrained_lm}")
        # Placeholder: actual weight loading depends on the specific pretrained model
        # from transformers import AutoModel
        # pretrained = AutoModel.from_pretrained(args.pretrained_lm)
        # ... map weights ...

    enable_gradient_checkpointing(model)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    if local_rank == 0:
        print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Dataset
    dataset = TextDataset(
        data_dir=args.data_dir,
        seq_length=train_cfg["seq_length"],
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
        enabled=(local_rank == 0),
    )

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    model.train()
    step = start_step
    grad_accum = train_cfg["gradient_accumulation"]

    while step < train_cfg["max_steps"]:
        sampler.set_epoch(step)
        for batch in loader:
            loss = train_step(model, batch, scaler, optimizer, scheduler, device)

            step += 1
            if local_rank == 0:
                logger.log_step(step, {
                    "loss": loss,
                    "lr": scheduler.get_last_lr()[0],
                })

            if step % config["training"]["common"]["save_every_steps"] == 0:
                if local_rank == 0:
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
        print(f"Stage 1 complete. Final checkpoint at step {step}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
