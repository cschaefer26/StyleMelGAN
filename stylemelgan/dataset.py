import random
from pathlib import Path
from typing import Dict, Union

import librosa
import torch
from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):

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
        wav_path = self.data_path / f'{file_id}.wav'
        wav, _ = librosa.load(wav_path)
        wav = torch.tensor(wav).float()
        mel = torch.load(mel_path).squeeze(0)

        if self.segment_len is not None:
            max_mel_start = mel.size(1) - self.mel_segment_len
            mel_start = random.randint(0, max_mel_start)
            mel_end = mel_start + self.mel_segment_len
            mel = mel[:, mel_start:mel_end]

            audio_start = mel_start * self.hop_len
            wav = wav[audio_start:audio_start+self.segment_len]
        wav = wav.unsqueeze(0)
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
    data_path = Path('/Users/cschaefe/datasets/bild_melgan_small')
    dataloader = new_dataloader(data_path=data_path, segment_len=16000, hop_len=256, batch_size=2)
    for item in dataloader:
        print(item['mel'].size())
        print(item['wav'].size())
