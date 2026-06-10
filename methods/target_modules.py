from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def infer_image_stats(dataset_name: str) -> tuple[tuple[float, float, float], tuple[float, float, float], int]:
    if dataset_name == "cifar10":
        return (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 32
    if dataset_name == "cifar100":
        return (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), 32
    raise ValueError(f"TARGET currently supports cifar10/cifar100 only, got: {dataset_name}")


class TensorImagePool:
    def __init__(self) -> None:
        self._items: List[torch.Tensor] = []

    def add(self, images_01: torch.Tensor) -> None:
        if images_01.ndim != 4:
            raise ValueError("Expected images with shape [N, C, H, W].")
        self._items.append(images_01.detach().cpu().clamp(0.0, 1.0))

    def as_tensor(self) -> torch.Tensor:
        if len(self._items) == 0:
            return torch.empty((0, 3, 32, 32), dtype=torch.float32)
        return torch.cat(self._items, dim=0)

    def size(self) -> int:
        return int(sum(item.size(0) for item in self._items))


class DeepInversionHook:
    def __init__(self, module: nn.BatchNorm2d, momentum_rate: float = 0.0) -> None:
        self.module = module
        self.momentum_rate = float(momentum_rate)
        self.mmt: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.tmp_val: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.r_feature = torch.tensor(0.0)
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, inputs, outputs) -> None:
        del outputs
        feature = inputs[0]
        nch = feature.shape[1]
        mean = feature.mean([0, 2, 3])
        var = feature.permute(1, 0, 2, 3).contiguous().view(nch, -1).var(1, unbiased=False)
        if self.mmt is None:
            r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(module.running_mean.data - mean, 2)
        else:
            mean_mmt, var_mmt = self.mmt
            r_feature = torch.norm(
                module.running_var.data - (1 - self.momentum_rate) * var - self.momentum_rate * var_mmt,
                2,
            )
            r_feature = r_feature + torch.norm(
                module.running_mean.data - (1 - self.momentum_rate) * mean - self.momentum_rate * mean_mmt,
                2,
            )
        self.r_feature = r_feature
        self.tmp_val = (mean, var)

    def update_mmt(self) -> None:
        if self.tmp_val is None:
            return
        mean, var = self.tmp_val
        if self.mmt is None:
            self.mmt = (mean.detach(), var.detach())
        else:
            mean_mmt, var_mmt = self.mmt
            self.mmt = (
                self.momentum_rate * mean_mmt + (1 - self.momentum_rate) * mean.detach(),
                self.momentum_rate * var_mmt + (1 - self.momentum_rate) * var.detach(),
            )

    def remove(self) -> None:
        self.hook.remove()


class TargetGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        base_channels: int = 64,
        image_size: int = 32,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        self.params = (latent_dim, base_channels, image_size, out_channels)
        self.init_size = image_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, base_channels * 2 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(base_channels * 2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(base_channels * 2, base_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, out_channels, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        return self.conv_blocks(out)

    def clone_to(self, device: torch.device) -> "TargetGenerator":
        clone = TargetGenerator(*self.params)
        clone.load_state_dict(copy.deepcopy(self.state_dict()))
        return clone.to(device)


class ReplayTensorDataset(Dataset):
    def __init__(self, images_01: torch.Tensor) -> None:
        self.images_01 = images_01.float().contiguous()

    def __len__(self) -> int:
        return int(self.images_01.size(0))

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.images_01[idx]


@dataclass
class TargetReplayBuffer:
    images_01: torch.Tensor
    class_ids: List[int]

    def num_samples(self) -> int:
        return int(self.images_01.size(0))

    def build_loader(self, batch_size: int, num_workers: int, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            ReplayTensorDataset(self.images_01),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )


def normalize_batch(x01: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    return (x01 - mean_t) / std_t


def random_crop_batch(x01: torch.Tensor, padding: int = 4) -> torch.Tensor:
    batch, _, height, width = x01.shape
    x_pad = F.pad(x01, (padding, padding, padding, padding), mode="constant", value=0.0)
    i = torch.randint(0, 2 * padding + 1, (batch,), device=x01.device)
    j = torch.randint(0, 2 * padding + 1, (batch,), device=x01.device)
    ar_h = torch.arange(height, device=x01.device)
    ar_w = torch.arange(width, device=x01.device)
    b = torch.arange(batch, device=x01.device)[:, None, None]
    hh = i[:, None, None] + ar_h[None, :, None]
    ww = j[:, None, None] + ar_w[None, None, :]
    return x_pad[b, :, hh, ww].permute(0, 3, 1, 2).contiguous()


def random_flip_batch(x01: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    mask = torch.rand(x01.size(0), device=x01.device) < p
    if not mask.any():
        return x01
    x01 = x01.clone()
    x01[mask] = torch.flip(x01[mask], dims=[3])
    return x01


def augment_and_normalize(
    x01: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
    do_augment: bool = True,
) -> torch.Tensor:
    x = x01
    if do_augment:
        x = random_crop_batch(x, padding=4)
        x = random_flip_batch(x, p=0.5)
    return normalize_batch(x, mean=mean, std=std)


def temperature_kldiv(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    reduction: str = "batchmean",
) -> torch.Tensor:
    q = F.log_softmax(student_logits / temperature, dim=1)
    p = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(q, p, reduction=reduction) * (temperature ** 2)


class KLDivLoss(nn.Module):
    def __init__(self, temperature: float = 1.0, reduction: str = "batchmean") -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.reduction = reduction

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        return temperature_kldiv(student_logits, teacher_logits, self.temperature, self.reduction)


class CyclingDataIter:
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self._iter = iter(loader)

    def next(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter)


def init_model_weights(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, mean=1.0, std=0.02)
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias)


class TargetSynthesizer:
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        dataset_name: str,
        class_ids: Sequence[int],
        device: torch.device,
        syn_rounds: int,
        g_steps: int,
        kd_steps: int,
        warmup_rounds: int,
        synthesis_batch_size: int,
        sample_batch_size: int,
        latent_dim: int,
        generator_lr: float,
        noise_lr: float,
        student_lr: float,
        bn_weight: float,
        ce_weight: float,
        div_weight: float,
        kd_temperature: float,
        use_fomaml: bool,
        divergence_mask_mode: str,
        bn_momentum: float = 0.0,
        max_images: Optional[int] = None,
    ) -> None:
        self.teacher = copy.deepcopy(teacher_model).to(device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.student = copy.deepcopy(student_model).to(device)
        init_model_weights(self.student)
        self.student.train()

        self.mean, self.std, image_size = infer_image_stats(dataset_name)
        self.image_size = int(image_size)
        self.class_ids = list(map(int, class_ids))
        self.device = device
        self.syn_rounds = int(syn_rounds)
        self.g_steps = int(g_steps)
        self.kd_steps = int(kd_steps)
        self.warmup_rounds = int(warmup_rounds)
        self.synthesis_batch_size = int(synthesis_batch_size)
        self.sample_batch_size = int(sample_batch_size)
        self.latent_dim = int(latent_dim)
        self.generator_lr = float(generator_lr)
        self.noise_lr = float(noise_lr)
        self.student_lr = float(student_lr)
        self.bn_weight = float(bn_weight)
        self.ce_weight = float(ce_weight)
        self.div_weight = float(div_weight)
        self.kd_temperature = float(kd_temperature)
        self.use_fomaml = bool(use_fomaml)
        self.divergence_mask_mode = str(divergence_mask_mode)
        self.max_images = None if max_images is None else int(max_images)

        self.generator = TargetGenerator(
            latent_dim=self.latent_dim,
            base_channels=64,
            image_size=self.image_size,
            out_channels=3,
        ).to(device)
        self.meta_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.generator_lr * max(self.g_steps, 1),
            betas=(0.5, 0.999),
        )
        self.student_optimizer = torch.optim.SGD(
            self.student.parameters(),
            lr=self.student_lr,
            momentum=0.9,
            weight_decay=1e-4,
        )
        self.student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.student_optimizer,
            T_max=max(self.syn_rounds, 1),
            eta_min=2e-4,
        )
        self.student_criterion = KLDivLoss(temperature=self.kd_temperature)
        self.pool = TensorImagePool()
        self.hooks = [
            DeepInversionHook(module, momentum_rate=bn_momentum)
            for module in self.teacher.modules()
            if isinstance(module, nn.BatchNorm2d)
        ]
        self._episode = 0

    def _sample_targets(self) -> torch.Tensor:
        sampled_indices = torch.randint(
            low=0,
            high=len(self.class_ids),
            size=(self.synthesis_batch_size,),
            device=self.device,
        )
        class_ids = torch.tensor(self.class_ids, device=self.device, dtype=torch.long)
        return class_ids[sampled_indices]

    def _divergence_mask(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        student_pred = student_logits.argmax(dim=1)
        teacher_pred = teacher_logits.argmax(dim=1)
        if self.divergence_mask_mode == "paper":
            return (student_pred != teacher_pred).float()
        if self.divergence_mask_mode == "official":
            return (student_pred == teacher_pred).float()
        raise ValueError(f"Unknown divergence_mask_mode: {self.divergence_mask_mode}")

    def _synthesize_batch(self) -> None:
        self._episode += 1
        self.teacher.eval()
        self.student.eval()
        best_cost = float("inf")
        best_inputs = None

        z = torch.randn(
            (self.synthesis_batch_size, self.latent_dim),
            device=self.device,
            requires_grad=True,
        )
        targets = self._sample_targets()

        fast_generator = self.generator.clone_to(self.device)
        optimizer = torch.optim.Adam(
            [
                {"params": fast_generator.parameters(), "lr": self.generator_lr},
                {"params": [z], "lr": self.noise_lr},
            ],
            betas=(0.5, 0.999),
        )

        if self.use_fomaml:
            self.meta_optimizer.zero_grad()

        for _ in range(self.g_steps):
            inputs_01 = fast_generator(z)
            inputs = augment_and_normalize(inputs_01, mean=self.mean, std=self.std, do_augment=True)
            teacher_logits = self.teacher(inputs)
            loss_bn = sum(h.r_feature.to(self.device) for h in self.hooks)
            loss_ce = F.cross_entropy(teacher_logits, targets)

            if self.div_weight > 0.0 and self._episode > self.warmup_rounds:
                student_logits = self.student(inputs)
                mask = self._divergence_mask(student_logits, teacher_logits)
                div_per_sample = temperature_kldiv(
                    student_logits,
                    teacher_logits,
                    temperature=1.0,
                    reduction="none",
                ).sum(dim=1)
                loss_div = -(div_per_sample * mask).mean()
            else:
                loss_div = loss_ce.new_zeros(())

            total_loss = (
                self.ce_weight * loss_ce
                + self.div_weight * loss_div
                + self.bn_weight * loss_bn
            )

            if total_loss.item() < best_cost or best_inputs is None:
                best_cost = float(total_loss.item())
                best_inputs = inputs_01.detach().cpu()

            optimizer.zero_grad()
            total_loss.backward()

            if self.use_fomaml:
                for meta_param, fast_param in zip(self.generator.parameters(), fast_generator.parameters()):
                    if fast_param.grad is None:
                        continue
                    if meta_param.grad is None:
                        meta_param.grad = torch.zeros_like(meta_param)
                    meta_param.grad.add_(fast_param.grad.detach())

            optimizer.step()

        if self.use_fomaml:
            self.meta_optimizer.step()
        else:
            self.meta_optimizer.zero_grad()
            for meta_param, fast_param in zip(self.generator.parameters(), fast_generator.parameters()):
                if meta_param.grad is None:
                    meta_param.grad = torch.zeros_like(meta_param)
                meta_param.grad.add_(meta_param.data - fast_param.data, alpha=1.0)
            self.meta_optimizer.step()

        for hook in self.hooks:
            hook.update_mmt()

        if best_inputs is not None:
            self.pool.add(best_inputs)

    def _kd_train_student(self) -> None:
        images_01 = self.pool.as_tensor()
        if images_01.numel() == 0:
            return
        if self.max_images is not None and images_01.size(0) > self.max_images:
            images_01 = images_01[: self.max_images]

        loader = DataLoader(
            ReplayTensorDataset(images_01),
            batch_size=self.sample_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        iterator = CyclingDataIter(loader)

        self.student.train()
        self.teacher.eval()
        for _ in range(self.kd_steps):
            images_01_batch = iterator.next().to(self.device, non_blocking=True)
            images = augment_and_normalize(images_01_batch, mean=self.mean, std=self.std, do_augment=True)
            with torch.no_grad():
                teacher_logits = self.teacher(images)
            student_logits = self.student(images)
            loss = self.student_criterion(student_logits, teacher_logits)
            self.student_optimizer.zero_grad()
            loss.backward()
            self.student_optimizer.step()

        self.student_scheduler.step()

    def run(self) -> TargetReplayBuffer:
        for round_idx in range(self.syn_rounds):
            self._synthesize_batch()
            if round_idx >= self.warmup_rounds:
                self._kd_train_student()

        images_01 = self.pool.as_tensor()
        if self.max_images is not None and images_01.size(0) > self.max_images:
            images_01 = images_01[: self.max_images]

        for hook in self.hooks:
            hook.remove()

        return TargetReplayBuffer(images_01=images_01, class_ids=self.class_ids)