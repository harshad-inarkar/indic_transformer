from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import AppConfig
from models.transformer import MultilingualTransformer, make_src_mask, make_tgt_mask


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
        self.model = model.to(self.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok

        self.pad_src = src_tok.token_to_id("<pad>")
        self.pad_tgt = tgt_tok.token_to_id("<pad>")
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_tgt, label_smoothing=config.training.label_smoothing
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
            fused=True,
        )
        total_steps = len(train_loader) * config.training.epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=config.training.lr, total_steps=total_steps, pct_start=0.1
        )
        self.scaler = torch.amp.GradScaler(self.device.type)

    def train_epoch(self, epoch_idx: int) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx}/{self.config.training.epochs}", leave=False)

        for src, tgt in pbar:
            src = src.to(self.device, non_blocking=True)
            tgt = tgt.to(self.device, non_blocking=True)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

            src_mask = make_src_mask(src, self.pad_src)
            tgt_mask = make_tgt_mask(tgt_in, self.pad_tgt)

            with torch.amp.autocast(self.device.type, dtype=torch.float16):
                logits = self.model(src, tgt_in, src_mask, tgt_mask)
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            l_val = float(loss.item())
            total_loss += l_val
            pbar.set_postfix({"loss": f"{l_val:.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"})

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate_loss(self) -> float:
        self.model.eval()
        total = 0.0
        for src, tgt in self.val_loader:
            src = src.to(self.device, non_blocking=True)
            tgt = tgt.to(self.device, non_blocking=True)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            src_mask = make_src_mask(src, self.pad_src)
            tgt_mask = make_tgt_mask(tgt_in, self.pad_tgt)
            with torch.amp.autocast(self.device.type, dtype=torch.float16):
                logits = self.model(src, tgt_in, src_mask, tgt_mask)
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            total += float(loss.item())
        return total / max(len(self.val_loader), 1)

    def fit(self) -> tuple[float, float, float]:
        torch.cuda.reset_peak_memory_stats()
        best_loss = float("inf")
        tag = self.config.language.lang_pair.lower().replace("-", "_")
        best_path = self.config.data.checkpoint_dir / f"transformer_{tag}_best.pt"

        start_time = time.time()
        for epoch in range(1, self.config.training.epochs + 1):
            ep_start = time.time()
            tr_loss = self.train_epoch(epoch)
            va_loss = self.evaluate_loss()
            elapsed = int(time.time() - ep_start)
            m, s = divmod(elapsed, 60)
            print(f"Epoch {epoch:2d}/{self.config.training.epochs} | Time: {m}m {s:02d}s | Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f}")

            if va_loss < best_loss:
                best_loss = va_loss
                self.config.data.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"state_dict": self.model.state_dict(), "loss": best_loss, "epoch": epoch},
                    best_path,
                )

        total_time = time.time() - start_time
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        peak_res = torch.cuda.max_memory_reserved() / 1e9
        return total_time, peak_mem, peak_res