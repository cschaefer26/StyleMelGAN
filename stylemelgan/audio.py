import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, Union


class Audio:

    def __init__(self,
                 n_mels: int,
                 sample_rate: int,
                 hop_length: int,
                 win_length: int,
                 n_fft: int,
                 fmin: float,
                 fmax: float) -> None:

        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Audio':
        return Audio(**config['audio'])

    def load_wav(self, path: Union[str, Path]) -> np.array:
        wav, _ = librosa.load(path, sr=self.sample_rate)
        return wav

    def save_wav(self, wav: np.array, path: Union[str, Path]) -> None:
        wav = wav.astype(np.float32)
        sf.write(str(path), wav, samplerate=self.sample_rate)

    def wav_to_mel(self, y: np.array, normalize=True) -> np.array:
        spec = librosa.stft(
            y=y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length)
        spec = np.abs(spec)
        mel = librosa.feature.melspectrogram(
            S=spec,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax)
        if normalize:
            mel = self.normalize(mel)
        return mel

    def normalize(self, mel: np.array) -> np.array:
        mel = np.clip(mel, a_min=1.e-5, a_max=None)
        return np.log(mel)

    def denormalize(self, mel: np.array) -> np.array:
        return np.exp(mel)