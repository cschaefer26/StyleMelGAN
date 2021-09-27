from pathlib import Path
import tqdm
import torch
import argparse

from stylemelgan.audio import Audio
from stylemelgan.generator.melgan import MelganGenerator
from stylemelgan.utils import remove_weight_norm_recursively

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='directory containing .mel files to vocode')
    parser.add_argument('--checkpoint', type=str, required=True, help='model checkpoint')
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    step = checkpoint['step']
    g_model = MelganGenerator(80)
    g_model.load_state_dict(checkpoint['g_model'])
    g_model.eval()
    remove_weight_norm_recursively(g_model)
    audio = Audio.from_config(checkpoint['config'])
    print(f'Loaded melgan with step {step}')

    mel_files = list(Path(args.path).glob('**/*.mel'))

    for mel_file in tqdm.tqdm(mel_files, total=len(mel_files)):
        mel = torch.load(mel_file)
        wav = g_model.inference(mel)
        wav = wav.squeeze().cpu().numpy()
        save_path = str(mel_file).replace('.mel', f'_voc_step_{step//1000}k.wav')
        audio.save_wav(wav, save_path)
