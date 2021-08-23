import os
import glob
from pathlib import Path
from typing import Dict
import librosa
import torch
import random
from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):

    def __init__(self,
                 data_path: Path,
                 segment_len: int,
                 hop_len: int,
                 padding_val: float = -11.5129) -> None:
        mel_names = list(data_path.glob('**/*.mel'))
        self.data_path = data_path
        self.hop_len = hop_len
        self.segment_len = segment_len
        self.padding_val = padding_val
        self.file_ids = [n.stem for n in mel_names]
        self.mel_segment_len = segment_len // hop_len + 2

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, item_id: int) -> Dict[str, torch.Tensor]:
        file_id = self.file_ids[item_id]
        mel_path = self.data_path / f'{file_id}.mel'
        wav_path = self.data_path / f'{file_id}.wav'
        wav, _ = librosa.load(wav_path)
        wav = torch.tensor(wav).float()
        mel = torch.load(mel_path).squeeze(0)
        mel_pad_len = 2 * self.mel_segment_len - mel.size(-1)
        if mel_pad_len > 0:
            mel_pad = torch.full((mel.size(0), mel_pad_len), fill_value=self.padding_val)
            mel = torch.cat([mel, mel_pad], dim=-1)
        wav_pad_len = mel.size(-1) * self.hop_len - wav.size(0)
        if wav_pad_len > 0:
            wav_pad = torch.zeros((wav_pad_len, ))
            wav = torch.cat([wav, wav_pad], dim=0)
        max_mel_start = mel.size(-1) - self.mel_segment_len
        mel_start = random.randint(0, max_mel_start)
        mel_end = mel_start+self.mel_segment_len
        mel = mel[:, mel_start:mel_end]
        wav_start = mel_start * self.hop_len
        wav_end = wav_start + self.mel_segment_len * self.hop_len
        wav = wav[wav_start:wav_end]
        wav = wav + (1 / 32768) * torch.randn_like(wav)
        return {'mel': mel, 'wav': wav}


def new_dataloader(data_path: Path,
                   segment_len: int,
                   hop_len: int,
                   batch_size: int,
                   num_workers: int = 0) -> DataLoader:

    dataset = AudioDataset(data_path=data_path, segment_len=segment_len, hop_len=hop_len)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True)
    return dataloader


if __name__ == '__main__':
    data_path = Path('/Users/cschaefe/datasets/asvoice2_splitted_train')
    dataloader = new_dataloader(data_path=data_path, segment_len=16000, hop_len=256, batch_size=2)
    for item in dataloader:
        print(item['mel'].size())
        print(item['wav'].size())
