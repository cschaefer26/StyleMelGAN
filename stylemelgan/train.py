from pathlib import Path
import tqdm
import torch
from torch.cuda import is_available
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from stylemelgan.audio import Audio2Mel
from stylemelgan.dataset import new_dataloader, AudioDataset
from stylemelgan.discriminator import MultiScaleDiscriminator
from stylemelgan.generator import MelganGenerator
from matplotlib.figure import Figure
import matplotlib as mpl
from functools import partial
from stylemelgan.losses import stft, MultiResStftLoss
from stylemelgan.utils import read_config

mpl.use('agg')  # Use non-interactive backend by default
import numpy as np
import matplotlib.pyplot as plt


def plot_mel(mel: np.array) -> Figure:
    mel = np.flip(mel, axis=0)
    fig = plt.figure(figsize=(12, 6), dpi=150)
    plt.imshow(mel, interpolation='nearest', aspect='auto')
    return fig


if __name__ == '__main__':

    config = read_config('stylemelgan/configs/melgan_config.yaml')
    train_data_path = Path(config['paths']['train_dir'])
    val_data_path = Path(config['paths']['val_dir'])

    device = torch.device('cuda') if is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True

    step = 0

    audio = Audio2Mel.from_config(config).to(device)
    g_model = MelganGenerator(audio.n_mels).to(device)
    d_model = MultiScaleDiscriminator().to(device)
    train_cfg = config['training']
    g_optim = torch.optim.Adam(g_model.parameters(), lr=train_cfg['g_lr'], betas=(0.5, 0.9))
    d_optim = torch.optim.Adam(d_model.parameters(), lr=train_cfg['d_lr'], betas=(0.5, 0.9))

    multires_stft_loss = MultiResStftLoss().to(device)

    segment_len = 8192 * 2

    try:
        checkpoint = torch.load('checkpoints/latest_model_neurips_orig_nostft.pt', map_location=device)
        g_model.load_state_dict(checkpoint['g_model'])
        g_optim.load_state_dict(checkpoint['g_optim'])
        d_model.load_state_dict(checkpoint['d_model'])
        d_optim.load_state_dict(checkpoint['d_optim'])
        step = checkpoint['step']
    except Exception as e:
        print(e)

    dataloader = new_dataloader(data_path=train_data_path, sample_rate=audio.sample_rate, segment_len=segment_len,
                                batch_size=16, augment=True, num_workers=4)
    val_dataset = AudioDataset(data_path=val_data_path, sample_rate=audio.sample_rate, segment_len=None,
                               augment=False)

    stft = partial(stft, n_fft=1024, hop_length=256, win_length=1024)

    pretraining_steps = 0

    summary_writer = SummaryWriter(log_dir='checkpoints/logs_neurips_orig_nostft')

    best_stft = 9999

    for epoch in range(10000):
        pbar = tqdm.tqdm(enumerate(dataloader, 1), total=len(dataloader))
        for i, wav_real in pbar:
            step += 1

            mel = audio(wav_real)

            wav_fake = g_model(mel)[:, :, :segment_len]

            d_loss = 0.0
            g_loss = 0.0
            stft_norm_loss = 0.0
            stft_spec_loss = 0.0

            if step > pretraining_steps:
                # discriminator
                d_fake = d_model(wav_fake.detach())
                d_real = d_model(wav_real)
                for (_, score_fake), (_, score_real) in zip(d_fake, d_real):
                    d_loss += F.relu(1.0 - score_real).mean()
                    d_loss += F.relu(1.0 + score_fake).mean()
                d_optim.zero_grad()
                d_loss.backward()
                d_optim.step()

                # generator
                d_fake = d_model(wav_fake)
                for (feat_fake, score_fake), (feat_real, _) in zip(d_fake, d_real):
                    g_loss += -score_fake.mean()
                    for feat_fake_i, feat_real_i in zip(feat_fake, feat_real):
                        g_loss += 10. * F.l1_loss(feat_fake_i, feat_real_i.detach())

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

            if step % 10000 == 0:
                g_model.eval()
                val_norm_loss = 0
                val_spec_loss = 0
                val_wavs = []

                for i, wav_real in enumerate(val_dataset):
                    val_mel = audio(wav_real.unsqueeze(0))
                    wav_fake = g_model.inference(val_mel, pad_steps=80).unsqueeze(0)
                    val_wavs.append((wav_fake, wav_real))
                    size = min(wav_real.size(-1), wav_fake.size(-1))
                    val_n, val_s = multires_stft_loss(wav_fake[..., :size], wav_real[..., :size])
                    val_norm_loss += val_n
                    val_spec_loss += val_s

                val_norm_loss /= len(val_dataset)
                val_spec_loss /= len(val_dataset)
                summary_writer.add_scalar('val_stft_norm_loss', val_norm_loss, global_step=step)
                summary_writer.add_scalar('val_stft_spec_loss', val_spec_loss, global_step=step)
                val_wavs.sort(key=lambda x: x[1].size(-1))
                wav_fake, wav_real = val_wavs[-1]
                if val_norm_loss + val_spec_loss < best_stft:
                    best_stft = val_norm_loss + val_spec_loss
                    print(f'\nnew best stft: {best_stft}')
                    torch.save({
                        'g_model': g_model.state_dict(),
                        'g_optim': g_optim.state_dict(),
                        'd_model': d_model.state_dict(),
                        'd_optim': d_optim.state_dict(),
                        'config': config,
                        'step': step
                    }, 'checkpoints/best_model_neurips_orig_nostft.pt')
                    summary_writer.add_audio('best_generated', wav_fake.squeeze().detach().cpu(),
                                             sample_rate=audio.sample_rate, global_step=step)

                g_model.train()
                summary_writer.add_audio('generated', wav_fake, sample_rate=audio.sample_rate, global_step=step)
                summary_writer.add_audio('target', wav_real, sample_rate=audio.sample_rate, global_step=step)
                mel_fake = audio(wav_fake.unsqueeze(1))
                mel_real = audio(wav_real.unsqueeze(1))
                mel_fake_plot = plot_mel(mel_fake.detach().cpu().squeeze().numpy())
                mel_real_plot = plot_mel(mel_real.detach().cpu().squeeze().numpy())
                summary_writer.add_figure('mel_generated', mel_fake_plot, global_step=step)
                summary_writer.add_figure('mel_target', mel_real_plot, global_step=step)

        # epoch end
        torch.save({
            'g_model': g_model.state_dict(),
            'g_optim': g_optim.state_dict(),
            'd_model': d_model.state_dict(),
            'd_optim': d_optim.state_dict(),
            'config': config,
            'step': step
        }, 'checkpoints/latest_model_neurips_orig_nostft.pt')