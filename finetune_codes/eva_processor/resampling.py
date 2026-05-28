# coding=utf-8
"""时间对齐：将 CED processor 的输出重采样到 whisper-style audio token 步长。"""

import torch


def resample_proc_to_whisper_timeaware(
    x_t: torch.Tensor,      # [T_c, H]  （ced_processor 输出，160ms 时间步）
    feat_len: int,          # 目标 whisper token 个数（该段内）
    T_mel: int,             # 该段真实 mel 帧数（≈ feat_len * 8）
    L_t: int = 1012,        # CED block 长度（mel 帧）
    t_st: int = 16,         # CED token 步长（mel 帧）
    t_sz: int = 16,         # CED token 窗宽（mel 帧）
    step_mel: int = 8,      # whisper 每 token 的 mel 帧数 (=80ms)
    center_mel: int = 4,    # whisper token 中心偏移
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
