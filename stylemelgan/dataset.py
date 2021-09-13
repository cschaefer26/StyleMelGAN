# copied from https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/dataset.py
from typing import Union

import torch
import torch.utils.data
import torch.nn.functional as F

from librosa.core import load
from librosa.util import normalize

from pathlib import Path
import numpy as np
import random

from torch.utils.data import DataLoader


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


class AudioDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self,
                 data_path: Path,
                 sample_rate: int,
                 segment_len: Union[int, None],
                 augment: bool = True):
        self.audio_files = list(data_path.glob('**/*.wav'))
        self.sample_rate = sample_rate
        self.segment_length = segment_len
        self.augment = augment

    def __getitem__(self, index: int) -> torch.Tensor:
        filename = self.audio_files[index]
        audio = self._load_wav_to_torch(filename)
        if self.segment_length is None:
            return audio.unsqueeze(0).data
        else:
            if audio.size(0) >= self.segment_length:
                max_audio_start = audio.size(0) - self.segment_length
                audio_start = random.randint(0, max_audio_start)
                audio = audio[audio_start:audio_start + self.segment_length]
            else:
                audio = F.pad(
                    audio, (0, self.segment_length - audio.size(0)), "constant"
                ).data
        return audio.unsqueeze(0)

    def __len__(self) -> int:
        return len(self.audio_files)

    def _load_wav_to_torch(self, full_path: Path) -> torch.Tensor:
        data, sampling_rate = load(full_path, sr=self.sample_rate)
        data = 0.95 * normalize(data)
        if self.augment:
            amplitude = np.random.uniform(low=0.3, high=1.0)
            data = data * amplitude
        return torch.from_numpy(data).float()


def new_dataloader(data_path: Path,
                   sample_rate: int,
                   segment_len: int,
                   batch_size: int,
                   augment: bool,
                   num_workers: int = 0) -> DataLoader:

    dataset = AudioDataset(data_path=data_path, sample_rate=sample_rate,
                           segment_len=segment_len, augment=augment)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True)
    return dataloader


if __name__ == '__main__':
    data_path = Path('/Users/cschaefe/datasets/asvoice2_splitted_train')
    dataloader = new_dataloader(data_path=data_path, sample_rate=22050, segment_len=16000, batch_size=2, augment=True)
    for item in dataloader:
        print(item.size())
