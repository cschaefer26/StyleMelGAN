import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, Union
from librosa.filters import mel as librosa_mel_fn
import torch.nn.functional as F
import torch

from stylemelgan.losses import stft
from stylemelgan.utils import read_config


def load_wav(path: Union[str, Path], sample_rate: int) -> np.array:
    wav, _ = librosa.load(path, sr=sample_rate)
    return wav


def save_wav(wav: np.array,
             path: Union[str, Path],
             sample_rate: int) -> None:
    wav = wav.astype(np.float32)
    sf.write(str(path), wav, samplerate=sample_rate)


# from https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py
class Audio2Mel(torch.nn.Module):

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        sample_rate: int,
        n_mels: int,
        fmin: float,
        fmax: float
    ):
        super().__init__()
        mel_basis = librosa_mel_fn(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.n_mels = n_mels

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False,
        )
        mel_output = torch.matmul(self.mel_basis.to(fft.device), fft)
        log_mel_spec = torch.log(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Audio2Mel':
        return Audio2Mel(**config['audio'])


if __name__ == '__main__':
    cfg = read_config('configs/melgan_config.yaml')
    audio2mel = Audio2Mel.from_config(cfg)

    wav, _ = librosa.load('/Users/cschaefe/datasets/ASVoice4_incl_english/r_00001_snippets/206110025_001.wav', sr=22050)
    mel = audio2mel(torch.from_numpy(wav).unsqueeze(0).unsqueeze(1))
    torch.save(mel, '/Users/cschaefe/workspace/ForwardTacotron/model_outputs/0002.mel')


