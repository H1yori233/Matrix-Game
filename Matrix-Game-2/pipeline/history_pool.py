from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from utils.camera_memory import (
    build_plucker_signature_from_extrinsic,
    latent_idx_to_frame_idx,
    rank_candidate_extrinsics_by_fov,
)


@dataclass
class HistoryPoolEntry:
    latent_idx: int
    frame_idx: int
    latent_cpu: torch.Tensor
    cond_concat_cpu: torch.Tensor
    keyboard_chunk_cpu: torch.Tensor
    mouse_chunk_cpu: Optional[torch.Tensor]
    extrinsic_cpu: torch.Tensor
    plucker_signature_cpu: torch.Tensor


def extract_action_chunk(action_tensor: torch.Tensor, latent_idx: int) -> torch.Tensor:
    if latent_idx == 0:
        return action_tensor[:, :1].detach().contiguous()
    start = 1 + 4 * (latent_idx - 1)
    end = start + 4
    return action_tensor[:, start:end].detach().contiguous()


def extract_cond_concat_slice(cond_concat: torch.Tensor, latent_idx: int) -> torch.Tensor:
    return cond_concat[:, :, latent_idx : latent_idx + 1].detach().contiguous()


class HistoryPool:
    def __init__(self):
        self.entries: List[HistoryPoolEntry] = []

    def reset(self):
        self.entries = []

    def add_generated_block(
        self,
        latents: torch.Tensor,
        conditional_dict: dict,
        extrinsics_all: torch.Tensor,
        latent_start_idx: int,
    ):
        mouse_cond = conditional_dict.get("mouse_cond")
        keyboard_cond = conditional_dict["keyboard_cond"]
        cond_concat = conditional_dict["cond_concat"]

        for offset in range(latents.shape[2]):
            latent_idx = latent_start_idx + offset
            frame_idx = latent_idx_to_frame_idx(latent_idx)
            if frame_idx >= extrinsics_all.shape[0]:
                continue

            extrinsic = extrinsics_all[frame_idx].detach().float().cpu().contiguous()
            self.entries.append(
                HistoryPoolEntry(
                    latent_idx=latent_idx,
                    frame_idx=frame_idx,
                    latent_cpu=latents[:, :, offset : offset + 1]
                    .detach()
                    .to(device="cpu", dtype=torch.bfloat16)
                    .contiguous(),
                    cond_concat_cpu=extract_cond_concat_slice(cond_concat, latent_idx)
                    .to(device="cpu", dtype=torch.bfloat16),
                    keyboard_chunk_cpu=extract_action_chunk(keyboard_cond, latent_idx)
                    .to(device="cpu", dtype=torch.float32),
                    mouse_chunk_cpu=None
                    if mouse_cond is None
                    else extract_action_chunk(mouse_cond, latent_idx)
                    .to(device="cpu", dtype=torch.float32),
                    extrinsic_cpu=extrinsic,
                    plucker_signature_cpu=build_plucker_signature_from_extrinsic(extrinsic)
                    .to(device="cpu", dtype=torch.float32),
                )
            )

    def get_entry(self, latent_idx: int) -> Optional[HistoryPoolEntry]:
        for entry in self.entries:
            if entry.latent_idx == latent_idx:
                return entry
        return None

    def get_recent_entries(
        self,
        current_start_idx: int,
        recent_size: int,
    ) -> List[HistoryPoolEntry]:
        start_idx = max(0, current_start_idx - recent_size)
        recent_entries = []
        for latent_idx in range(start_idx, current_start_idx):
            entry = self.get_entry(latent_idx)
            if entry is not None:
                recent_entries.append(entry)
        return recent_entries

    def select_history_entries(
        self,
        current_start_idx: int,
        current_extrinsic: torch.Tensor,
        excluded_latent_indices: Sequence[int],
        max_history: int,
        device: torch.device,
    ) -> List[HistoryPoolEntry]:
        if max_history <= 0:
            return []

        candidates = [
            entry
            for entry in self.entries
            if entry.latent_idx < current_start_idx
            and entry.latent_idx not in excluded_latent_indices
        ]
        if not candidates:
            return []

        candidate_extrinsics = torch.stack(
            [entry.extrinsic_cpu for entry in candidates],
            dim=0,
        ).to(device=device, dtype=torch.float32)
        selected_indices, _ = rank_candidate_extrinsics_by_fov(
            current_extrinsic=current_extrinsic.to(device=device, dtype=torch.float32),
            candidate_extrinsics=candidate_extrinsics,
            topk=max_history,
        )
        selected_entries = [candidates[idx] for idx in selected_indices.tolist()]
        selected_entries.sort(key=lambda entry: entry.latent_idx)
        return selected_entries


def build_context_latents(
    context_entries: Sequence[HistoryPoolEntry],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if not context_entries:
        return None
    return torch.cat(
        [
            entry.latent_cpu.to(device=device, dtype=dtype)
            for entry in context_entries
        ],
        dim=2,
    )


def build_compact_conditions(
    context_entries: Sequence[HistoryPoolEntry],
    current_latent_indices: Sequence[int],
    conditional_dict: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    cond_concat_slices = [
        entry.cond_concat_cpu.to(device=device, dtype=dtype)
        for entry in context_entries
    ]
    keyboard_chunks = [
        entry.keyboard_chunk_cpu.to(device=device, dtype=dtype)
        for entry in context_entries
    ]
    mouse_chunks = []
    if conditional_dict.get("mouse_cond") is not None:
        mouse_chunks = [
            entry.mouse_chunk_cpu.to(device=device, dtype=dtype)
            for entry in context_entries
            if entry.mouse_chunk_cpu is not None
        ]

    for latent_idx in current_latent_indices:
        cond_concat_slices.append(
            extract_cond_concat_slice(
                conditional_dict["cond_concat"],
                latent_idx,
            ).to(device=device, dtype=dtype)
        )
        keyboard_chunks.append(
            extract_action_chunk(
                conditional_dict["keyboard_cond"],
                latent_idx,
            ).to(device=device, dtype=dtype)
        )
        if conditional_dict.get("mouse_cond") is not None:
            mouse_chunks.append(
                extract_action_chunk(
                    conditional_dict["mouse_cond"],
                    latent_idx,
                ).to(device=device, dtype=dtype)
            )

    compact_conditions = {
        "cond_concat": torch.cat(cond_concat_slices, dim=2),
        "visual_context": conditional_dict["visual_context"].to(
            device=device,
            dtype=dtype,
        ),
        "keyboard_cond": torch.cat(keyboard_chunks, dim=1),
    }
    if conditional_dict.get("mouse_cond") is not None:
        compact_conditions["mouse_cond"] = torch.cat(mouse_chunks, dim=1)
    return compact_conditions
