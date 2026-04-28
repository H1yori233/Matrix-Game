from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from utils.cam_utils import select_memory_idx_fov


@dataclass
class MemoryPoolEntry:
    frame_idx: int
    latent_idx: int
    latent_cpu: torch.Tensor
    signature_cpu: torch.Tensor
    extrinsic_cpu: torch.Tensor


@dataclass
class MemorySelection:
    x_memory: Optional[torch.Tensor]
    memory_latent_idx: Optional[List[int]]
    memory_extrinsics: Optional[torch.Tensor]
    memory_frame_indices: Optional[List[int]]
    query_frame_indices: Optional[List[int]]


class MemoryPool:
    def __init__(
        self,
        similarity_threshold: float,
        coarse_topk: int,
        temporal_stride: int = 4,
    ):
        self.similarity_threshold = similarity_threshold
        self.coarse_topk = coarse_topk
        self.temporal_stride = temporal_stride
        self.entries: List[MemoryPoolEntry] = []

    def reset(self):
        self.entries = []

    def build_query_frame_indices(self, current_end_frame_idx: int) -> List[int]:
        return [
            current_end_frame_idx - (1 + 8 * idx)
            for idx in range(5)
            if current_end_frame_idx - (1 + 8 * idx) >= 0
        ]

    def add_clip(
        self,
        latents: torch.Tensor,
        plucker: torch.Tensor,
        latent_start_idx: int,
        extrinsics_all: torch.Tensor,
    ):
        if latents is None or plucker is None or latents.shape[2] == 0:
            return

        assert latents.shape[2] == plucker.shape[2], (
            latents.shape,
            plucker.shape,
            latent_start_idx,
        )

        for offset in range(latents.shape[2]):
            latent_idx = latent_start_idx + offset
            frame_idx = self._latent_idx_to_frame_idx(latent_idx)
            if frame_idx >= extrinsics_all.shape[0]:
                continue

            signature_cpu = self._signature_from_plucker_slice(
                plucker[:, :, offset]
            )
            if self.entries:
                existing_signatures = torch.stack(
                    [entry.signature_cpu for entry in self.entries], dim=0
                )
                max_similarity = torch.matmul(
                    existing_signatures, signature_cpu.to(existing_signatures.dtype)
                ).max().item()
                if max_similarity >= self.similarity_threshold:
                    continue

            latent_cpu = (
                latents[:, :, offset : offset + 1]
                .detach()
                .to(device="cpu", dtype=torch.bfloat16)
                .contiguous()
            )
            extrinsic_cpu = (
                extrinsics_all[frame_idx]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
            )
            self.entries.append(
                MemoryPoolEntry(
                    frame_idx=frame_idx,
                    latent_idx=latent_idx,
                    latent_cpu=latent_cpu,
                    signature_cpu=signature_cpu,
                    extrinsic_cpu=extrinsic_cpu,
                )
            )

    def retrieve(
        self,
        query_frame_indices: Sequence[int],
        plucker: torch.Tensor,
        current_latent_start_idx: int,
        current_start_frame_idx: int,
        extrinsics_all: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> MemorySelection:
        candidate_entries = [
            entry for entry in self.entries if entry.frame_idx < current_start_frame_idx
        ]
        if not candidate_entries:
            return MemorySelection(None, None, None, None, None)

        candidate_signatures = torch.stack(
            [entry.signature_cpu for entry in candidate_entries], dim=0
        ).to(device=device, dtype=torch.float32)
        selected_entries: List[MemoryPoolEntry] = []
        selected_queries: List[int] = []

        for query_frame_idx in query_frame_indices:
            query_signature = self._query_signature_from_plucker(
                query_frame_idx,
                plucker,
                current_latent_start_idx,
                device,
            )
            if query_signature is None:
                continue

            topk = min(self.coarse_topk, candidate_signatures.shape[0])
            similarities = torch.matmul(candidate_signatures, query_signature)
            shortlist_idx = torch.topk(similarities, k=topk).indices.tolist()
            shortlist_entries = [candidate_entries[idx] for idx in shortlist_idx]
            shortlist_frame_indices = [entry.frame_idx for entry in shortlist_entries]
            if len(shortlist_frame_indices) == 0:
                continue

            selected_frame = select_memory_idx_fov(
                extrinsics_all,
                current_start_frame_idx,
                [query_frame_idx],
                use_gpu=True,
                candidate_indices=shortlist_frame_indices,
            )[0]
            frame_to_entry = {
                entry.frame_idx: entry for entry in shortlist_entries
            }
            selected_entry = frame_to_entry.get(selected_frame)
            if selected_entry is None:
                continue

            selected_entries.append(selected_entry)
            selected_queries.append(query_frame_idx)

        if not selected_entries:
            return MemorySelection(None, None, None, None, None)

        x_memory = torch.cat(
            [
                entry.latent_cpu.to(device=device, dtype=dtype)
                for entry in selected_entries
            ],
            dim=2,
        )
        memory_extrinsics = torch.stack(
            [entry.extrinsic_cpu for entry in selected_entries], dim=0
        ).to(device=device, dtype=torch.float32)
        return MemorySelection(
            x_memory=x_memory,
            memory_latent_idx=[entry.latent_idx for entry in selected_entries],
            memory_extrinsics=memory_extrinsics,
            memory_frame_indices=[entry.frame_idx for entry in selected_entries],
            query_frame_indices=selected_queries,
        )

    def _query_signature_from_plucker(
        self,
        query_frame_idx: int,
        plucker: torch.Tensor,
        current_latent_start_idx: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        query_latent_idx = self._frame_to_latent_idx(query_frame_idx)
        local_idx = query_latent_idx - current_latent_start_idx
        if local_idx < 0 or local_idx >= plucker.shape[2]:
            return None
        return self._signature_from_plucker_slice(
            plucker[:, :, local_idx]
        ).to(device=device, dtype=torch.float32)

    def _frame_to_latent_idx(self, frame_idx: int) -> int:
        return (frame_idx - 1) // self.temporal_stride + 1

    def _latent_idx_to_frame_idx(self, latent_idx: int) -> int:
        return latent_idx * self.temporal_stride

    def _signature_from_plucker_slice(self, plucker_slice: torch.Tensor) -> torch.Tensor:
        if plucker_slice.dim() == 4:
            plucker_slice = plucker_slice.squeeze(0)
        signature = plucker_slice.float().mean(dim=(-1, -2))
        signature = F.normalize(signature, dim=0, eps=1e-6)
        return signature.to(device="cpu", dtype=torch.float32).contiguous()
