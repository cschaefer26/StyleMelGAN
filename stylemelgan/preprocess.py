import argparse
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from pathlib import Path

import torch
import tqdm

from stylemelgan.audio import Audio
from stylemelgan.utils import read_config


def valid_n_workers(num):
    n = int(num)
    if n < 1:
        raise argparse.ArgumentTypeError('%r must be an integer greater than 0' % num)
    return n


class Preprocessor:

    def __init__(self, audio: Audio) -> None:
        self.audio = audio

    def __call__(self, file: Path) -> None:
        wav = self.audio.load_wav(file)
        mel = self.audio.wav_to_mel(wav)
        mel = torch.from_numpy(mel).unsqueeze(0).float()
        save_path = str(file).replace('.wav', '.mel')
        torch.save(mel, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for mel preprocessing.')
    parser.add_argument('--num_workers', '-w', metavar='N', type=valid_n_workers, default=cpu_count() - 1,
                        help='The number of worker threads to use for preprocessing.')
    parser.add_argument('--config', metavar='FILE', default='config.yaml',
                        help='The config containing all hyperparams.')
    args = parser.parse_args()

    config = read_config(args.config)
    audio = Audio.from_config(config)
    preprocessor = Preprocessor(audio)
    train_data_path = Path(config['paths']['train_dir'])
    val_data_path = Path(config['paths']['val_dir'])
    train_files = list(train_data_path.glob('**/*.wav'))
    val_files = list(val_data_path.glob('**/*.wav'))
    all_files = train_files + val_files
    n_workers = max(1, args.num_workers)

    pool = Pool(processes=n_workers)

    for _ in tqdm.tqdm(pool.imap_unordered(preprocessor, all_files)):
        pass

    print('Preprocessing done.')

