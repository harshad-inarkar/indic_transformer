from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from multilingual_transformer.configs.config import AppConfig
from multilingual_transformer.models.transformer import MultilingualTransformer, make_src_mask, make_tgt_mask
from multilingual_transformer.utils.helpers import free_memory
import shutil

class Trainer:
    def __init__(
        self,
        model: MultilingualTransformer,
        config: AppConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        src_tok: Any,
        tgt_tok: Any,
    ) -> None:
        assert torch.cuda.is_available(), "Execution restricted to GPU (CUDA)."
        self.device = torch.device("cuda")
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.selected: dict[str, Any] = {}  # info about the checkpoint currently on disk


        self.best_info: dict[str, Any] = {}
        self.last_info: dict[str, Any] = {}
        
        # raw_model is always the plain module (used for saving/loading weights).
        self.raw_model = model.to(self.device)
        self.model: nn.Module = self.raw_model
        if config.training.torch_compile:
            self.model = torch.compile(self.model, dynamic=True)
        # Enable Multi-GPU Training
        if torch.cuda.device_count() > 1:
            print(f"🚀 Utilizing {torch.cuda.device_count()} GPUs for training!")
            self.model = nn.DataParallel(self.model)

        self.pad_src = src_tok.token_to_id("<pad>")
        self.pad_tgt = tgt_tok.token_to_id("<pad>")
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_tgt, label_smoothing=config.training.label_smoothing
        )
        # Validation loss is plain NLL (no smoothing) so it is comparable across settings.
        self.eval_criterion = nn.CrossEntropyLoss(ignore_index=self.pad_tgt)

        # No weight decay on biases / LayerNorm (1-D params).
        decay = [p for p in self.raw_model.parameters() if p.dim() > 1]
        no_decay = [p for p in self.raw_model.parameters() if p.dim() <= 1]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": config.training.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=config.training.lr,
            fused=True,
        )
        total_steps = len(train_loader) * config.training.epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=config.training.lr, total_steps=total_steps, pct_start=0.1
        )
        self.scaler = torch.amp.GradScaler(self.device.type)

    def _loss(self, src: torch.Tensor, tgt: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        src = src.to(self.device, non_blocking=True)
        tgt = tgt.to(self.device, non_blocking=True)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        src_mask = make_src_mask(src, self.pad_src)
        tgt_mask = make_tgt_mask(tgt_in, self.pad_tgt)
        with torch.amp.autocast(self.device.type, dtype=torch.float16):
            logits = self.model(src, tgt_in, src_mask, tgt_mask)
            return criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

    def train_epoch(self, epoch_idx: int) -> float:
        self.model.train()
        total_loss = torch.zeros((), device=self.device)  # accumulate on GPU: no per-step host sync
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx}/{self.config.training.epochs}", leave=False)

        for step, (src, tgt) in enumerate(pbar, 1):
            loss = self._loss(src, tgt, self.criterion)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.detach()
            if step % 25 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"})

        return total_loss.item() / len(self.train_loader)

    @torch.no_grad()
    def evaluate_loss(self) -> float:
        self.model.eval()
        total = torch.zeros((), device=self.device)
        for src, tgt in self.val_loader:
            total += self._loss(src, tgt, self.eval_criterion).detach()
        return total.item() / max(len(self.val_loader), 1)

    def fit(self) -> tuple[float, float, float]:
        torch.cuda.reset_peak_memory_stats()
        best_loss = float("inf")

        start_time = time.time()
        for epoch in range(1, self.config.training.epochs + 1):
            ep_start = time.time()
            tr_loss = self.train_epoch(epoch)
            va_loss = self.evaluate_loss()
            elapsed = int(time.time() - ep_start)
            m, s = divmod(elapsed, 60)
            print(f"Epoch {epoch:2d}/{self.config.training.epochs} | Time: {m}m {s:02d}s | Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f}")

            best_path = self.config.checkpoint_path("best")
            last_path = self.config.checkpoint_path("last")
            payload = {
                "state_dict": self.raw_model.state_dict(),
                "loss": va_loss, "train_loss": tr_loss, "epoch": epoch,
            }
            info = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss}

            torch.save(payload, last_path)  # always overwritten, so at the end it is the final epoch
            self.last_info = {**info, "path": last_path}
            if va_loss < best_loss:
                best_loss = va_loss
                shutil.copyfile(last_path, best_path)  # copy instead of serialising twice
                self.best_info = {**info, "path": best_path}


            free_memory()

        total_time = time.time() - start_time
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        peak_res = torch.cuda.max_memory_reserved() / 1e9
        return total_time, peak_mem, peak_res

    def restore(self) -> None:
        """save_best=True: reload the best-validation weights. Otherwise the in-memory weights are already the latest."""
        if not self.config.training.save_best:
            return
        path = self.config.checkpoint_path()
        if path.exists():
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            self.raw_model.load_state_dict(ckpt["state_dict"])
            del ckpt

    def release(self) -> None:
        """Drop optimizer / scheduler / loaders so decoding has the GPU memory to itself."""
        del self.optimizer, self.scheduler, self.scaler, self.train_loader, self.val_loader, self.model
        free_memory()
