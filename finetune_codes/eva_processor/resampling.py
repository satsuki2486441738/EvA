# coding=utf-8
"""Time alignment: resample CED processor outputs to Whisper-style audio-token steps."""

import torch


def resample_proc_to_whisper_timeaware(
    x_t: torch.Tensor,      # [T_c, H], ced_processor output at 160 ms steps.
    feat_len: int,          # Target number of Whisper tokens in the segment.
    T_mel: int,             # Actual mel-frame count, roughly feat_len * 8.
    L_t: int = 1012,        # CED block length in mel frames.
    t_st: int = 16,         # CED token stride in mel frames.
    t_sz: int = 16,         # CED token window size in mel frames.
    step_mel: int = 8,      # Whisper mel frames per token (=80 ms).
    center_mel: int = 4,    # Whisper token center offset.
) -> torch.Tensor:          # -> [feat_len, H]
    T_c, H = x_t.shape
    device = x_t.device
    if T_c <= 1:
        return x_t.expand(feat_len, H)

    Tpb = (L_t - t_sz) // t_st + 1  # = 63
    idx = torch.arange(T_c, device=device)
    s = idx // Tpb
    j = idx %  Tpb

    start = s * L_t + j * t_st
    end   = start + t_sz - 1
    t_c   = start + (t_sz // 2)

    cov = (torch.clamp(end, max=T_mel-1) - start + 1).clamp(min=0) / t_sz   # [T_c] ∈ [0,1]

    k = torch.arange(feat_len, device=device)
    t_w = k * step_mel + center_mel

    right = torch.searchsorted(t_c, t_w).clamp(1, T_c-1)
    left  = right - 1
    t_l, t_r = t_c[left], t_c[right]
    alpha = ((t_w - t_l) / (t_r - t_l + 1e-8)).unsqueeze(-1)

    x_l, x_r = x_t[left], x_t[right]
    c_l, c_r = cov[left].unsqueeze(-1), cov[right].unsqueeze(-1)

    num = (1 - alpha) * (x_l * c_l) + alpha * (x_r * c_r)
    den = (1 - alpha) *  c_l          + alpha *  c_r
    x_w = num / (den + 1e-8)
    return x_w
