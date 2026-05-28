# kimia_infer/utils/sampler.py
import torch


class KimiASampler:
    def __init__(
        self,
        audio_top_k: int,
        audio_temperature: float,
        audio_repetition_penalty: float,
        audio_repetition_window_size: int,
        text_top_k: int,
        text_temperature: float,
        text_repetition_penalty: float,
        text_repetition_window_size: int,
    ):
        self.audio_top_k = int(audio_top_k)
        self.audio_temperature = float(audio_temperature)
        self.text_top_k = int(text_top_k)
        self.text_temperature = float(text_temperature)

        self.audio_repetition_penalty = float(audio_repetition_penalty)
        self.audio_repetition_window_size = int(audio_repetition_window_size)
        self.text_repetition_penalty = float(text_repetition_penalty)
        self.text_repetition_window_size = int(text_repetition_window_size)

    @staticmethod
    def _last_step_logits(logits: torch.Tensor) -> torch.Tensor:
        # [B, T, V] -> [B, V]
        return logits[:, -1] if logits.dim() == 3 else logits

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor, recent_tokens: torch.Tensor, penalty: float, window: int
    ) -> torch.Tensor:
        """
        假设 batch=1。对最近 window 个 token 做 penalty（<0 乘以 p，>=0 除以 p）
        """
        if penalty <= 1.0 or recent_tokens is None or len(recent_tokens) == 0:
            return logits
        logits_ = logits.squeeze(0)  # [V]
        recent_window = recent_tokens[-window:].long()
        scores = torch.gather(logits_, dim=0, index=recent_window)
        scores = torch.where(scores < 0, scores * penalty, scores / penalty)
        logits_.scatter_(dim=0, index=recent_window, src=scores)
        return logits_.unsqueeze(0)

    @staticmethod
    def _topk_sample_from_logprobs(logprobs: torch.Tensor, k: int) -> torch.Tensor:
        # logprobs: [B, V]
        probs = torch.exp(logprobs)  # 保持 float32 计算
        if k > 0:
            top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
            sampled = torch.multinomial(top_k_probs, num_samples=1).squeeze(1)
            return top_k_indices.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        else:
            return torch.multinomial(probs, num_samples=1).squeeze(1)

    def sample_audio_logits(self, logits: torch.Tensor, recent_tokens=None) -> torch.Tensor:
        """
        从音频 logits 采样（支持 top-k / 温度 / 重复惩罚）。期望 batch=1。
        logits: [B, V] 或 [B, T, V]
        """
        logits = self._last_step_logits(logits)
        logits = self._apply_repetition_penalty(
            logits, recent_tokens, self.audio_repetition_penalty, self.audio_repetition_window_size
        )
        # 统一在 float32 上做 softmax 更稳
        logprobs = torch.log_softmax(logits.float(), dim=-1)

        if self.audio_temperature > 1e-6:
            logprobs = logprobs / self.audio_temperature
            next_token = self._topk_sample_from_logprobs(logprobs, self.audio_top_k)
        else:
            next_token = torch.argmax(logprobs, dim=-1)
        return next_token

    def sample_text_logits(self, logits: torch.Tensor, recent_tokens=None) -> torch.Tensor:
        """
        从文本 logits 采样（支持 top-k / 温度 / 重复惩罚）。期望 batch=1。
        logits: [B, V] 或 [B, T, V]
        """
        logits = self._last_step_logits(logits)
        logits = self._apply_repetition_penalty(
            logits, recent_tokens, self.text_repetition_penalty, self.text_repetition_window_size
        )
        logprobs = torch.log_softmax(logits.float(), dim=-1)

        if self.text_temperature > 1e-6:
            logprobs = logprobs / self.text_temperature
            next_token = self._topk_sample_from_logprobs(logprobs, self.text_top_k)
        else:
            next_token = torch.argmax(logprobs, dim=-1)
        return next_token
