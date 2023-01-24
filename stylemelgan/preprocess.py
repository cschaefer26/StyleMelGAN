import argparse
import librosa
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from pathlib import Path
import numpy as np
import torch
import tqdm

from stylemelgan.audio import Audio
from stylemelgan.utils import read_config

def normalize_values(phoneme_val):
    nonzeros = np.concatenate([v[np.where(v != 0.0)[0]]
                               for item_id, v in phoneme_val])
    mean, std = np.mean(nonzeros), np.std(nonzeros)
    if not std > 0:
        std = 1e10
    for item_id, v in phoneme_val:
        zero_idxs = np.where(v == 0.0)[0]
        v -= mean
        v /= std
        v[zero_idxs] = 0.0
    return mean, std


def valid_n_workers(num):
    n = int(num)
    if n < 1:
        raise argparse.ArgumentTypeError('%r must be an integer greater than 0' % num)
    return n


class Preprocessor:

    def __init__(self, audio: Audio) -> None:
        self.audio = audio

    def __call__(self, file: Path) -> tuple:
        wav = self.audio.load_wav(file)
        mel = self.audio.wav_to_mel(wav)
        mel = torch.from_numpy(mel).unsqueeze(0).float()
        save_path = str(file).replace('.wav', '.mel')
        torch.save(mel, save_path)
        try:
            pitch, _, _ = librosa.pyin(wav,
                                       fmin=50,
                                       fmax=600,
                                       sr=22050,
                                       frame_length=2048,
                                       hop_length=256)
            np.nan_to_num(pitch, copy=False, nan=0.)
        except Exception as e:
            print(e)
            pitch = np.zeros((mel.size(-1), ))
        return file, pitch.astype(float)



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
    results = []

    for res in tqdm.tqdm(pool.imap_unordered(preprocessor, all_files), total=len(all_files)):
        results.append(res)

    normalize_values(results)

    for file, pitch in results:
        print(file, pitch)
        pitch = torch.from_numpy(pitch)
        torch.save(pitch, str(file).replace('.wav', '.pitch'))

    print('Preprocessing done.')

