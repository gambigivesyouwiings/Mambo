"""Training utilities for Hibiki-Sw on Kaggle (2x T4 GPUs).

Provides DDP setup, mixed precision, optimizer/scheduler creation,
checkpointing, logging, gradient checkpointing, noise augmentation,
and reproducibility helpers.
"""

import os
import math
import time
import random
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


# ---------------------------------------------------------------------------
# DDP Setup
# ---------------------------------------------------------------------------

def setup_ddp(rank: int, world_size: int) -> None:
    """Initialize the distributed process group with NCCL backend.

    Args:
        rank: Process rank (0 or 1 for 2-GPU Kaggle).
        world_size: Total number of processes (2 for Kaggle 2x T4).
    """
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device(rank: int) -> torch.device:
    """Return the CUDA device for the given rank.

    Args:
        rank: Process rank.

    Returns:
        torch.device for the corresponding GPU.
    """
    return torch.device(f"cuda:{rank}")


# ---------------------------------------------------------------------------
# Mixed Precision
# ---------------------------------------------------------------------------

def create_scaler() -> GradScaler:
    """Create a GradScaler for FP16 mixed-precision training.

    Usage in training loop::

        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(...)
            loss = compute_loss(outputs, ...)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    Returns:
        GradScaler instance.
    """
    return GradScaler()


# ---------------------------------------------------------------------------
# Optimizer & Scheduler
# ---------------------------------------------------------------------------

def create_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
) -> AdamW:
    """Create an AdamW optimizer with proper parameter groups.

    Bias and normalization parameters are excluded from weight decay.

    Args:
        model: The model to optimize.
        lr: Peak learning rate.
        weight_decay: Weight decay coefficient.
        betas: AdamW beta coefficients.

    Returns:
        Configured AdamW optimizer.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No weight decay on biases, LayerNorm/RMSNorm weights, and embedding weights
        if param.ndim == 1 or "bias" in name or "norm" in name or "embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(param_groups, lr=lr, betas=betas, fused=True)


def create_cosine_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create a cosine annealing scheduler with linear warmup.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of linear warmup steps.
        max_steps: Total training steps.
        min_lr_ratio: Minimum LR as a fraction of peak LR.

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(step: int) -> float:
        # Linear warmup
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # Cosine decay
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: GradScaler,
    step: int,
    path: str,
) -> None:
    """Save a full training checkpoint.

    When using DDP, extracts the underlying module state dict.

    Args:
        model: The model (possibly DDP-wrapped).
        optimizer: The optimizer.
        scheduler: The LR scheduler.
        scaler: The GradScaler.
        step: Current training step.
        path: File path to save the checkpoint.
    """
    # Unwrap DDP if necessary
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    checkpoint = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[AdamW] = None,
    scheduler: Optional[LambdaLR] = None,
    scaler: Optional[GradScaler] = None,
) -> int:
    """Load a training checkpoint and restore state.

    Args:
        path: Path to the checkpoint file.
        model: The model to load weights into (possibly DDP-wrapped).
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        scaler: Optional GradScaler to restore.

    Returns:
        The training step at which the checkpoint was saved.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Load into the underlying module if DDP-wrapped
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    return checkpoint.get("step", 0)


def save_to_hf_hub(
    model: nn.Module,
    repo_id: str,
    token: str,
    commit_message: str = "Upload Hibiki-Sw checkpoint",
) -> None:
    """Upload model weights to HuggingFace Hub.

    Useful for transferring checkpoints between Kaggle accounts or
    persisting models beyond the 9-hour session limit.

    Args:
        model: The model (possibly DDP-wrapped).
        repo_id: HuggingFace repo ID (e.g. "username/hibiki-sw-checkpoint").
        token: HuggingFace API token.
        commit_message: Commit message for the upload.
    """
    from huggingface_hub import HfApi

    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    tmp_path = "/tmp/hibiki_sw_checkpoint.pt"
    torch.save({"model": model_state}, tmp_path)

    api = HfApi(token=token)
    api.create_repo(repo_id, exist_ok=True, private=True)
    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo="checkpoint.pt",
        repo_id=repo_id,
        commit_message=commit_message,
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TrainingLogger:
    """Logger that writes to both console and TensorBoard.

    Tracks loss, learning rate, throughput (tokens/sec), and GPU memory.

    Args:
        log_dir: Directory for TensorBoard logs.
        rank: Process rank. Only rank 0 writes to TensorBoard.
        log_interval: Steps between console log messages.
    """

    def __init__(
        self,
        log_dir: str = "runs/hibiki-sw",
        rank: int = 0,
        log_interval: int = 10,
    ):
        self.rank = rank
        self.log_interval = log_interval
        self.writer = None
        self._step_start_time = None
        self._tokens_since_last_log = 0

        # Console logger
        self.logger = logging.getLogger("hibiki-sw")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(asctime)s][Rank %(name)s] %(message)s", datefmt="%H:%M:%S")
            )
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        self.logger.name = str(rank)

        # TensorBoard (rank 0 only)
        if rank == 0:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                self.logger.warning("TensorBoard not available, skipping TB logging.")

    def start_step(self) -> None:
        """Mark the beginning of a training step for throughput measurement."""
        self._step_start_time = time.time()

    def log_step(
        self,
        step: int,
        losses: Dict[str, float],
        lr: float,
        tokens_processed: int,
    ) -> None:
        """Log metrics for a training step.

        Args:
            step: Current global step.
            losses: Dict of loss name -> value.
            lr: Current learning rate.
            tokens_processed: Number of tokens in this step's batch.
        """
        self._tokens_since_last_log += tokens_processed
        elapsed = time.time() - self._step_start_time if self._step_start_time else 1.0

        # TensorBoard logging (every step, rank 0 only)
        if self.writer is not None:
            for name, val in losses.items():
                self.writer.add_scalar(f"loss/{name}", val, step)
            self.writer.add_scalar("lr", lr, step)
            self.writer.add_scalar("throughput/tokens_per_sec", tokens_processed / elapsed, step)

            mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            self.writer.add_scalar("gpu/memory_allocated_gb", mem_allocated, step)
            self.writer.add_scalar("gpu/memory_reserved_gb", mem_reserved, step)

        # Console logging (every log_interval steps)
        if step % self.log_interval == 0:
            throughput = self._tokens_since_last_log / max(elapsed, 1e-6)
            mem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            loss_str = " | ".join(f"{k}: {v:.4f}" for k, v in losses.items())
            self.logger.info(
                f"step {step} | {loss_str} | lr: {lr:.2e} | "
                f"{throughput:.0f} tok/s | mem: {mem_gb:.1f}GB"
            )
            self._tokens_since_last_log = 0

    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()


# ---------------------------------------------------------------------------
# Gradient Checkpointing
# ---------------------------------------------------------------------------

def enable_gradient_checkpointing(model: nn.Module) -> None:
    """Enable gradient checkpointing on temporal transformer layers.

    Reduces peak GPU memory at the cost of ~30% slower backward pass.
    Essential for fitting the model on T4 16GB GPUs.

    Args:
        model: The HibikiModel (possibly DDP-wrapped).
    """
    target = model.module if hasattr(model, "module") else model

    # Enable on temporal transformer layers
    if hasattr(target, "temporal") and hasattr(target.temporal, "layers"):
        for layer in target.temporal.layers:
            layer._orig_forward = layer.forward

            def make_ckpt_forward(mod):
                orig = mod._orig_forward

                def ckpt_forward(*args, **kwargs):
                    if mod.training:
                        return torch.utils.checkpoint.checkpoint(
                            orig, *args, use_reentrant=False, **kwargs
                        )
                    return orig(*args, **kwargs)

                return ckpt_forward

            layer.forward = make_ckpt_forward(layer)


# ---------------------------------------------------------------------------
# Noise Augmentation
# ---------------------------------------------------------------------------

def augment_audio_tokens(
    tokens: torch.Tensor,
    noise_prob: float = 0.1,
    codebook_size: int = 2048,
    num_special_tokens: int = 4,
) -> torch.Tensor:
    """Randomly corrupt audio tokens for training robustness.

    Replaces a fraction of tokens with random valid token IDs, simulating
    codec errors and improving model robustness to noisy inputs.

    Args:
        tokens: (B, Q, T) audio token tensor.
        noise_prob: Probability of corrupting each token.
        codebook_size: Size of the audio codebook (excluding special tokens).
        num_special_tokens: Number of special tokens (pad, bos, eos, epad).

    Returns:
        Augmented token tensor with the same shape.
    """
    if noise_prob <= 0.0:
        return tokens

    augmented = tokens.clone()
    mask = torch.rand_like(tokens, dtype=torch.float32) < noise_prob

    # Only corrupt non-special tokens (id >= num_special_tokens)
    is_regular = tokens >= num_special_tokens
    mask = mask & is_regular

    # Replace with random valid token IDs
    random_tokens = torch.randint(
        num_special_tokens,
        num_special_tokens + codebook_size,
        tokens.shape,
        device=tokens.device,
        dtype=tokens.dtype,
    )
    augmented[mask] = random_tokens[mask]

    return augmented


# ---------------------------------------------------------------------------
# Seed & Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: The random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic operations (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
