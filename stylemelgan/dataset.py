import random
from pathlib import Path
from typing import Dict, Union
from librosa.filters import mel as librosa_mel_fn
import librosa
import torch
import numpy as np
from librosa.util import normalize

from torch.utils.data import Dataset, DataLoader


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec




class MelDataset(Dataset):

    def __init__(self,
                 data_path: Path,
                 hop_len: int,
                 segment_len: Union[int, None],
                 padding_val: float = -11.5129) -> None:
        mel_names = list(data_path.glob('**/*.mel'))
        self.data_path = data_path
        self.hop_len = hop_len
        self.segment_len = segment_len
        self.padding_val = padding_val
        self.file_ids = [n.stem for n in mel_names]
        if segment_len is not None:
            self.mel_segment_len = segment_len // hop_len + 2

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, item_id: int) -> Dict[str, torch.Tensor]:
        file_id = self.file_ids[item_id]
        mel_path = self.data_path / f'{file_id}.mel'
        mel = torch.load(mel_path)
        mel_pad_len = 2 * self.mel_segment_len - mel.size(-1)
        if mel_pad_len > 0:
            mel_pad = torch.full((mel.size(0), mel_pad_len), fill_value=self.padding_val)
            mel = torch.cat([mel, mel_pad], dim=-1)
        return {'mel': mel}


class AudioDataset(Dataset):

    def __init__(self,
                 data_path: Path,
                 hop_len: int,
                 segment_len: Union[int, None],
                 sample_rate: int,
                 padding_val: float = -11.5129) -> None:
        mel_names = list(data_path.glob('**/*.mel'))
        self.data_path = data_path
        self.hop_len = hop_len
        self.segment_len = segment_len
        self.padding_val = padding_val
        self.sample_rate = sample_rate
        self.file_ids = [n.stem for n in mel_names]
        if segment_len is not None:
            self.mel_segment_len = segment_len // hop_len + 2

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, item_id: int) -> Dict[str, torch.Tensor]:
        file_id = self.file_ids[item_id]
        mel_path = self.data_path / f'{file_id}.mel'
        wav_path = self.data_path / f'{file_id}.wav'
        pitch_path = self.data_path / f'{file_id}.pitch'
        wav, _ = librosa.load(wav_path, sr=self.sample_rate)
        wav = torch.tensor(wav).float()
        mel = torch.load(mel_path).squeeze(0)
        pitch = torch.load(pitch_path).float()

        if self.segment_len is not None:
            mel_pad_len = 2 * self.mel_segment_len - mel.size(-1)
            if mel_pad_len > 0:
                mel_pad = torch.full((mel.size(0), mel_pad_len), fill_value=self.padding_val)
                mel = torch.cat([mel, mel_pad], dim=-1)
                pitch_pad = torch.full((mel_pad_len, ), fill_value=0)
                pitch = torch.cat([pitch, pitch_pad], dim=0)

            wav_pad_len = mel.size(-1) * self.hop_len - wav.size(0)
            if wav_pad_len > 0:
                wav_pad = torch.zeros((wav_pad_len, ))
                wav = torch.cat([wav, wav_pad], dim=0)
            max_mel_start = mel.size(-1) - self.mel_segment_len
            mel_start = random.randint(0, max_mel_start)
            mel_end = mel_start + self.mel_segment_len
            mel = mel[:, mel_start:mel_end]
            wav_start = mel_start * self.hop_len
            wav_end = wav_start + self.segment_len
            wav = wav[wav_start:wav_end]
            wav = wav + (1 / 32768) * torch.randn_like(wav)
            pitch = pitch[mel_start:mel_end]
        wav = wav.unsqueeze(0)
        pitch = pitch.unsqueeze(0)
        #print(pitch)
        return {'mel': mel, 'wav': wav, 'pitch': pitch}



def new_mel_dataloader(data_path: Path,
                       segment_len: int,
                       hop_len: int,
                       batch_size: int,
                       num_workers: int = 0) -> DataLoader:

    dataset = MelDataset(data_path=data_path, segment_len=segment_len, hop_len=hop_len)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True)
    return dataloader

def new_dataloader(data_path: Path,
                   segment_len: int,
                   hop_len: int,
                   batch_size: int,
                   sample_rate: int,
                   num_workers: int = 0) -> DataLoader:

    dataset = AudioDataset(data_path=data_path, segment_len=segment_len, hop_len=hop_len, sample_rate=sample_rate)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True)
    return dataloader


if __name__ == '__main__':
    data_path = Path('/Users/cschaefe/datasets/bild_melgan_small')
    dataloader = new_dataloader(data_path=data_path, segment_len=16000, hop_len=256, batch_size=2, sample_rate=22050)
    for item in dataloader:
        print(item['mel'].size())
        print(item['wav'].size())
