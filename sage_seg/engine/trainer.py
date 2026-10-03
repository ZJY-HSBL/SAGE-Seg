from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
import torch
from torch.nn import functional as F
from tqdm import tqdm

from sage_seg.augment.weak_strong import weak_photometric, strong_augment
from sage_seg.core.scd import semantic_consistency_mask
from sage_seg.core.srf import sparse_region_filter
from sage_seg.core.gating import normalized_entropy_confidence, sample_bernoulli_gate
from sage_seg.core.mixing import semantic_mix
from sage_seg.losses.segmentation import supervised_ce, per_sample_ce
from sage_seg.models.ema import update_ema
from sage_seg.utils.image import imagenet_normalize
from sage_seg.utils.checkpoint import save_checkpoint
from .evaluator import evaluate


class Trainer:
    def __init__(self, cfg, model, teacher, optimizer, labeled_loader, unlabeled_loader, val_loader, device):
        self.cfg = cfg
        self.model = model
        self.teacher = teacher
        self.optimizer = optimizer
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.val_loader = val_loader
        self.device = device
        self.ssl = cfg["ssl"]
        self.train_cfg = cfg["train"]
        self.ignore_index = int(cfg["data"].get("ignore_index", 255))
        self.num_classes = int(cfg["data"]["num_classes"])
        self.output_dir = Path(cfg["project"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        amp_enabled = bool(self.train_cfg.get("amp", True) and device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        self.amp_enabled = amp_enabled

    def _adjust_lr(self, global_iter, total_iters):
        power = float(self.train_cfg.get("poly_power", 0.9))
        factor = (1.0 - min(global_iter, total_iters) / max(total_iters, 1)) ** power
        for group in self.optimizer.param_groups:
            group["lr"] = group["initial_lr"] * factor

    def train_epoch(self, epoch: int):
        self.model.train()
        self.teacher.eval()
        steps = max(len(self.labeled_loader), len(self.unlabeled_loader))
        if steps == 0:
            raise RuntimeError("Empty labeled or unlabeled DataLoader")
        l_iter = iter(self.labeled_loader)
        u_iter = iter(self.unlabeled_loader)
        total_iters = int(self.train_cfg["epochs"]) * steps
        base_iter = epoch * steps
        sums = {"loss": 0.0, "sup": 0.0, "unsup": 0.0, "gate": 0.0}
        bar = tqdm(range(steps), desc=f"train {epoch+1}/{self.train_cfg['epochs']}")
        for step in bar:
            self._adjust_lr(base_iter + step, total_iters)
            try:
                lb = next(l_iter)
            except StopIteration:
                l_iter = iter(self.labeled_loader)
                lb = next(l_iter)
            try:
                ub = next(u_iter)
            except StopIteration:
                u_iter = iter(self.unlabeled_loader)
                ub = next(u_iter)
            x_l = lb["image"].to(self.device, non_blocking=True)
            y_l = lb["mask"].to(self.device, non_blocking=True)
            x_u = ub["image"].to(self.device, non_blocking=True)

            weak_strength = float(self.ssl.get("weak_strength", 0.08))
            x_l_w = weak_photometric(x_l, weak_strength)
            x_u_t = weak_photometric(x_u, weak_strength)
            x_u_s = weak_photometric(x_u, weak_strength)

            amp_ctx = torch.cuda.amp.autocast(enabled=self.amp_enabled) if self.device.type == "cuda" else nullcontext()
            with torch.no_grad():
                with amp_ctx:
                    teacher_logits = self.teacher(imagenet_normalize(x_u_t))
                    teacher_prob = F.softmax(teacher_logits.float(), dim=1)
                    student_weak_logits = self.model(imagenet_normalize(x_u_s))
                    student_prob = F.softmax(student_weak_logits.float(), dim=1)
                e_mask, _ = semantic_consistency_mask(
                    student_prob,
                    teacher_prob,
                    threshold=float(self.ssl.get("consistency_threshold", 0.7)),
                )
                m_mask = sparse_region_filter(e_mask, int(self.ssl.get("area_threshold", 16)))
                confidence = normalized_entropy_confidence(teacher_prob)
                gate = sample_bernoulli_gate(confidence)
                pseudo = teacher_prob.argmax(dim=1)
                x_mix, y_mix = semantic_mix(x_u_t, x_l_w, pseudo, y_l, m_mask)
                x_strong = strong_augment(x_u)

            self.optimizer.zero_grad(set_to_none=True)
            amp_ctx = torch.cuda.amp.autocast(enabled=self.amp_enabled) if self.device.type == "cuda" else nullcontext()
            with amp_ctx:
                sup_logits = self.model(imagenet_normalize(x_l_w))
                mix_logits = self.model(imagenet_normalize(x_mix))
                strong_logits = self.model(imagenet_normalize(x_strong))
                loss_sup = supervised_ce(sup_logits, y_l, self.ignore_index)
                loss_mix = per_sample_ce(mix_logits, y_mix, self.ignore_index)
                loss_strong = per_sample_ce(strong_logits, pseudo, self.ignore_index)
                z = gate.to(dtype=loss_mix.dtype)
                loss_unsup = (z * loss_strong + (1.0 - z) * loss_mix).mean()
                loss = loss_sup + float(self.ssl.get("unsupervised_weight", 1.0)) * loss_unsup

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            update_ema(self.model, self.teacher, float(self.ssl.get("ema_decay", 0.99)))

            sums["loss"] += float(loss.detach())
            sums["sup"] += float(loss_sup.detach())
            sums["unsup"] += float(loss_unsup.detach())
            sums["gate"] += float(gate.float().mean())
            if (step + 1) % int(self.train_cfg.get("log_interval", 20)) == 0 or step == steps - 1:
                n = step + 1
                bar.set_postfix(loss=f"{sums['loss']/n:.4f}", sup=f"{sums['sup']/n:.4f}",
                                unsup=f"{sums['unsup']/n:.4f}", strong=f"{sums['gate']/n:.2f}")
        return {k: v / steps for k, v in sums.items()}

    def fit(self, start_epoch=0, best_miou=0.0):
        epochs = int(self.train_cfg["epochs"])
        save_every = int(self.train_cfg.get("save_every", 10))
        for epoch in range(start_epoch, epochs):
            train_stats = self.train_epoch(epoch)
            miou, class_iou = evaluate(self.teacher, self.val_loader, self.device, self.num_classes,
                                       self.ignore_index, desc=f"val {epoch+1}")
            print(f"epoch={epoch+1} train_loss={train_stats['loss']:.4f} mIoU={miou:.4f}")
            if miou >= best_miou:
                best_miou = miou
                save_checkpoint(self.output_dir / "best.pt", self.model, self.teacher, self.optimizer,
                                epoch + 1, best_miou, self.cfg)
            save_checkpoint(self.output_dir / "latest.pt", self.model, self.teacher, self.optimizer,
                            epoch + 1, best_miou, self.cfg)
            if (epoch + 1) % save_every == 0:
                save_checkpoint(self.output_dir / f"epoch_{epoch+1:03d}.pt", self.model, self.teacher,
                                self.optimizer, epoch + 1, best_miou, self.cfg)
        return best_miou
