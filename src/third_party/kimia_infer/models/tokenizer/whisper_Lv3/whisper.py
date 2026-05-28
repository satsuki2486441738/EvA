# models/tokenizer/whisper_Lv3/whisper.py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from subprocess import CalledProcessError, run, Popen, PIPE
import os
from functools import lru_cache
from typing import Optional, Union
from .modeling_whisper import WhisperModel

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 120
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk


def load_bytesio_audio(content, sr: int = SAMPLE_RATE):
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        "pipe:",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "pipe:",
    ]
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, bufsize=-1)
    out, _ = p.communicate(input=content)
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = ["ffmpeg", "-nostdin", "-threads", "0", "-i", file, "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr), "-"]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = 128) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    with np.load(
        os.path.join(os.path.dirname(__file__), "mel_filters.npz")  # todo
        # os.path.join("assets", "mel_filters.npz")
    ) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 128,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


class WhisperEncoder(nn.Module):
    def __init__(self, model_path, mel_batch_size=40):
        super().__init__()
        self.speech_encoder = WhisperModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).encoder
        self.speech_encoder.eval()
        self.mel_batch_size = mel_batch_size

    @torch.no_grad()
    def forward(self, audio_waveform: torch.Tensor, kimia_whisper_clip_silence=False):
        device = self.speech_encoder.device

        # 统一成 [B, T]
        if audio_waveform.dim() == 1:
            audio_waveform = audio_waveform.unsqueeze(0)
        assert audio_waveform.dim() == 2, f"expect [B,T], got {tuple(audio_waveform.shape)}"
        B, T = audio_waveform.shape

        per_sample_feats = []
        max_L = 0

        for b in range(B):
            x = audio_waveform[b]  # [T]
            # 30s 分块
            segs = []
            t0 = 0
            while t0 < x.shape[0]:
                seg = x[t0 : t0 + N_SAMPLES]        # 右界安全截断
                t0 += N_SAMPLES                      # 每次前进 30s
                L = int(seg.shape[0])
                token_len = (L - 1) // (HOP_LENGTH * 8) + 1

                seg = pad_or_trim(seg.flatten())     # pad 到 30s
                mel = log_mel_spectrogram(seg, device=device)
                mel_input = mel.unsqueeze(0).to(device, dtype=self.speech_encoder.dtype)

                if kimia_whisper_clip_silence:
                    input_seq_lens = torch.LongTensor([token_len * 4]).to(device)
                    h = self.speech_encoder(mel_input, return_dict=True, input_seq_lens=input_seq_lens).last_hidden_state
                else:
                    h = self.speech_encoder(mel_input, return_dict=True).last_hidden_state

                h = h[:, : token_len * 4, :]  # 裁掉右侧 padding
                segs.append(h)

            if len(segs) == 0:
                feats_b = torch.empty(1, 0, self.speech_encoder.config.d_model, device=device, dtype=self.speech_encoder.dtype)
            else:
                feats_b = torch.cat(segs, dim=1)     # [1, Lb, D]
            per_sample_feats.append(feats_b)
            max_L = max(max_L, feats_b.size(1))

        # 右侧 0 填充到同长，拼成 [B, Lmax, D]
        if max_L == 0:
            return torch.empty(B, 0, self.speech_encoder.config.d_model, device=device, dtype=self.speech_encoder.dtype)

        out = []
        for h in per_sample_feats:
            if h.size(1) < max_L:
                pad = (0, 0, 0, max_L - h.size(1))  # (last dim, seq dim)
                h = F.pad(h, pad)
            out.append(h)
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def tokenize_waveform(self, audio, kimia_whisper_clip_silence=False):
        audio_embedding = self.forward(audio, kimia_whisper_clip_silence)
        return audio_embedding.cpu()
