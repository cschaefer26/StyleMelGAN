from pathlib import Path
import tqdm
import torch
import soundfile as sf
from torch.cuda import is_available
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from stylemelgan.audio import Audio
from stylemelgan.dataset import new_dataloader, AudioDataset
from stylemelgan.discriminator import MultiScaleDiscriminator, MultiScaleSpecDiscriminator
from stylemelgan.generator import Generator
from matplotlib.figure import Figure
import matplotlib as mpl
from functools import partial
from stylemelgan.losses import stft, MultiResStftLoss
from stylemelgan.utils import read_config

mpl.use('agg')  # Use non-interactive backend by default
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_mel(mel: np.array) -> Figure:
    mel = np.flip(mel, axis=0)
    fig = plt.figure(figsize=(12, 6), dpi=150)
    plt.imshow(mel, interpolation='nearest', aspect='auto')
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='stylemelgan/configs/melgan_config.yaml', help='points to config.yaml')
    args = parser.parse_args()

    config = read_config(args.config)
    model_name = config['model_name']

    audio = Audio.from_config(config)
    train_data_path = Path(config['paths']['train_dir'])
    val_data_path = Path(config['paths']['val_dir'])

    device = torch.device('cuda') if is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True

    step = 0

    g_model = Generator(audio.n_mels).to(device)
    d_model = MultiScaleDiscriminator().to(device)
    d_spec_model = MultiScaleSpecDiscriminator().to(device)
    train_cfg = config['training']
    g_optim = torch.optim.Adam(g_model.parameters(), lr=train_cfg['g_lr'], betas=(0.5, 0.9))
    d_optim = torch.optim.Adam(d_model.parameters(), lr=train_cfg['d_lr'], betas=(0.5, 0.9))
    d_spec_optim = torch.optim.Adam(d_spec_model.parameters(), lr=train_cfg['d_lr'], betas=(0.5, 0.9))

    multires_stft_loss = MultiResStftLoss().to(device)

    try:
        checkpoint = torch.load(f'checkpoints/latest_model_{model_name}.pt', map_location=device)
        g_model.load_state_dict(checkpoint['g_model'])
        g_optim.load_state_dict(checkpoint['g_optim'])
        d_model.load_state_dict(checkpoint['d_model'])
        d_optim.load_state_dict(checkpoint['d_optim'])
        d_spec_model.load_state_dict(checkpoint['d_spec_model'])
        d_spec_optim.load_state_dict(checkpoint['d_spec_optim'])
        step = checkpoint['step']
    except Exception as e:
        print(e)

    train_cfg = config['training']
    dataloader = new_dataloader(data_path=train_data_path, segment_len=train_cfg['segment_len'],
                                hop_len=audio.hop_length, batch_size=train_cfg['batch_size'],
                                num_workers=train_cfg['num_workers'], sample_rate=audio.sample_rate)
    val_dataset = AudioDataset(data_path=val_data_path, segment_len=None, hop_len=audio.hop_length,
                               sample_rate=audio.sample_rate)
    stft = partial(stft, n_fft=1024, hop_length=256, win_length=1024)

    pretraining_steps = 0

    summary_writer = SummaryWriter(log_dir='checkpoints/logs_specdisc_universal')

    best_stft = 9999

    for epoch in range(10000):
        pbar = tqdm.tqdm(enumerate(dataloader, 1), total=len(dataloader))
        for i, data in pbar:
            step += 1
            mel = data['mel'].to(device)
            wav_real = data['wav'].to(device)

            wav_fake = g_model(mel)[:, :, :train_cfg['segment_len']]

            d_loss = 0.0
            d_spec_loss = 0.0
            g_loss = 0.0
            stft_norm_loss = 0.0
            stft_spec_loss = 0.0

            d_loss_all = 0

            if step > pretraining_steps:

                # discriminator
                d_fake = d_model(wav_fake.detach())
                d_real = d_model(wav_real.detach())
                for (_, score_fake), (_, score_real) in zip(d_fake, d_real):
                    d_loss += F.relu(1.0 - score_real).mean()
                    d_loss += F.relu(1.0 + score_fake).mean()

                # spec discriminator
                d_spec_fake = d_spec_model(wav_fake.detach())
                d_spec_real = d_spec_model(wav_real.detach())
                for (_, score_fake), (_, score_real) in zip(d_spec_fake, d_spec_real):
                    d_spec_loss += F.relu(1.0 - score_real).mean()
                    d_spec_loss += F.relu(1.0 + score_fake).mean()

                d_loss_all = d_loss + d_spec_loss

                d_optim.zero_grad()
                d_spec_optim.zero_grad()
                d_loss_all.backward()
                d_optim.step()
                d_spec_optim.step()

                # generator
                d_fake = d_model(wav_fake)
                for (feat_fake, score_fake), (feat_real, _) in zip(d_fake, d_real):
                    g_loss += -score_fake.mean()
                    for feat_fake_i, feat_real_i in zip(feat_fake, feat_real):
                        g_loss += 10. * F.l1_loss(feat_fake_i, feat_real_i.detach())

                # generator
                d_spec_fake = d_spec_model(wav_fake)
                for (feat_fake, score_fake), (feat_real, _) in zip(d_spec_fake, d_spec_real):
                    g_loss += -score_fake.mean()

            factor = 1. if step < pretraining_steps else 0.
            stft_norm_loss, stft_spec_loss = multires_stft_loss(wav_fake.squeeze(1), wav_real.squeeze(1))

            g_loss_all = g_loss + factor * (stft_norm_loss + stft_spec_loss)

            g_optim.zero_grad()
            g_loss_all.backward()
            g_optim.step()

            pbar.set_description(desc=f'Epoch: {epoch} | Step {step} '
                                      f'| g_loss: {g_loss:#.4} '
                                      f'| d_loss: {d_loss:#.4} '
                                      f'| stft_norm_loss {stft_norm_loss:#.4} '
                                      f'| stft_spec_loss {stft_spec_loss:#.4} ', refresh=True)

            summary_writer.add_scalar('generator_loss', g_loss, global_step=step)
            summary_writer.add_scalar('stft_norm_loss', stft_norm_loss, global_step=step)
            summary_writer.add_scalar('stft_spec_loss', stft_spec_loss, global_step=step)
            summary_writer.add_scalar('discriminator_loss', d_loss, global_step=step)
            summary_writer.add_scalar('spec_discriminator_loss', d_spec_loss, global_step=step)

            if step % train_cfg['eval_steps'] == 0:
                g_model.eval()
                val_norm_loss = 0
                val_spec_loss = 0
                val_wavs = []

                for i, val_data in enumerate(val_dataset):
                    val_mel = val_data['mel'].to(device)
                    val_mel = val_mel.unsqueeze(0)
                    wav_fake = g_model.inference(val_mel).squeeze().cpu().numpy()
                    wav_real = val_data['wav'].detach().squeeze().cpu().numpy()
                    wav_f = torch.tensor(wav_fake).unsqueeze(0).to(device)
                    wav_r = torch.tensor(wav_real).unsqueeze(0).to(device)
                    val_wavs.append((wav_fake, wav_real))
                    size = min(wav_r.size(-1), wav_f.size(-1))
                    val_n, val_s = multires_stft_loss(wav_f[..., :size], wav_r[..., :size])
                    val_norm_loss += val_n
                    val_spec_loss += val_s

                val_norm_loss /= len(val_dataset)
                val_spec_loss /= len(val_dataset)
                summary_writer.add_scalar('val_stft_norm_loss', val_norm_loss, global_step=step)
                summary_writer.add_scalar('val_stft_spec_loss', val_spec_loss, global_step=step)
                val_wavs.sort(key=lambda x: x[1].shape[0])
                wav_fake, wav_real = val_wavs[-1]
                if val_norm_loss + val_spec_loss < best_stft:
                    best_stft = val_norm_loss + val_spec_loss
                    print(f'\nnew best stft: {best_stft}')
                    torch.save({
                        'g_model': g_model.state_dict(),
                        'g_optim': g_optim.state_dict(),
                        'd_model': d_model.state_dict(),
                        'd_optim': d_optim.state_dict(),
                        'd_spec_model': d_spec_model.state_dict(),
                        'd_spec_optim': d_spec_optim.state_dict(),
                        'config': config,
                        'step': step
                    }, f'checkpoints/best_model_{model_name}.pt')

                    torch.save({
                        'model_g': g_model.state_dict(),
                        'config': config,
                        'step': step
                    }, f'checkpoints/best_melgan_{model_name}.pt')

                    summary_writer.add_audio('best_generated', wav_fake, sample_rate=audio.sample_rate, global_step=step)

                g_model.train()
                summary_writer.add_audio('generated', wav_fake, sample_rate=audio.sample_rate, global_step=step)
                summary_writer.add_audio('target', wav_real, sample_rate=audio.sample_rate, global_step=step)
                mel_fake = audio.wav_to_mel(wav_fake)
                mel_real = audio.wav_to_mel(wav_real)
                mel_fake_plot = plot_mel(mel_fake)
                mel_real_plot = plot_mel(mel_real)
                summary_writer.add_figure('mel_generated', mel_fake_plot, global_step=step)
                summary_writer.add_figure('mel_target', mel_real_plot, global_step=step)

        # epoch end
        torch.save({
            'g_model': g_model.state_dict(),
            'g_optim': g_optim.state_dict(),
            'd_model': d_model.state_dict(),
            'd_optim': d_optim.state_dict(),
            'd_spec_model': d_spec_model.state_dict(),
            'd_spec_optim': d_spec_optim.state_dict(),
            'config': config,
            'step': step
        }, f'checkpoints/latest_model_{model_name}.pt')

        torch.save({
            'model_g': g_model.state_dict(),
            'config': config,
            'step': step
        }, f'checkpoints/latest_melgan_{model_name}.pt')
