from functools import partial
from pathlib import Path
import argparse
import matplotlib as mpl
import torch
import torch.nn.functional as F
import tqdm
from matplotlib.figure import Figure
from torch.cuda import is_available
from torch.utils.tensorboard import SummaryWriter

from stylemelgan.audio import Audio
from stylemelgan.dataset import new_dataloader, AudioDataset
from stylemelgan.discriminator import MultiScaleDiscriminator
from stylemelgan.generator.melgan import Generator
from stylemelgan.losses import stft, MultiResStftLoss, TorchSTFT
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
    train_cfg = config['training']
    g_optim = torch.optim.Adam(g_model.parameters(), lr=train_cfg['g_lr'], betas=(0.5, 0.9))
    d_optim = torch.optim.Adam(d_model.parameters(), lr=train_cfg['d_lr'], betas=(0.5, 0.9))
    for g in g_optim.param_groups:
        g['lr'] = train_cfg['g_lr']
    for g in d_optim.param_groups:
        g['lr'] = train_cfg['d_lr']
    multires_stft_loss = MultiResStftLoss().to(device)

    try:
        checkpoint = torch.load(f'checkpoints/latest_model__{model_name}.pt', map_location=device)
        g_model.load_state_dict(checkpoint['model_g'])
        g_optim.load_state_dict(checkpoint['optim_g'])
        d_model.load_state_dict(checkpoint['model_d'])
        d_optim.load_state_dict(checkpoint['optim_d'])
        step = checkpoint['step']
        print(f'Loaded model with step {step}')
    except Exception as e:
        'Initializing model from scratch.'

    train_cfg = config['training']
    dataloader = new_dataloader(data_path=train_data_path, segment_len=train_cfg['segment_len'],
                                hop_len=audio.hop_length, batch_size=train_cfg['batch_size'],
                                num_workers=train_cfg['num_workers'], sample_rate=audio.sample_rate)
    val_dataset = AudioDataset(data_path=val_data_path, segment_len=None, hop_len=audio.hop_length,
                               sample_rate=audio.sample_rate)

    stft = partial(stft, n_fft=1024, hop_length=256, win_length=1024)

    torch_stft = TorchSTFT(filter_length=16, hop_length=4, win_length=16).to(device)

    pretraining_steps = train_cfg['pretraining_steps']

    summary_writer = SummaryWriter(log_dir=f'checkpoints/logs_{model_name}')

    best_stft = 9999

    for epoch in range(train_cfg['epochs']):
        pbar = tqdm.tqdm(enumerate(dataloader, 1), total=len(dataloader))
        for i, data in pbar:
            step += 1
            mel = data['mel'].to(device)
            wav_real = data['wav'].to(device)

            spec, phase, wav_fake = g_model(mel)
            wav_fake_istft = torch_stft.inverse(spec, phase)

            #print(wav_fake.size())

            d_loss = 0.0
            g_loss = 0.0
            stft_norm_loss = 0.0
            stft_spec_loss = 0.0

            if step > pretraining_steps:
                # discriminator
                d_fake = d_model(wav_fake.detach())
                d_real = d_model(wav_real)
                for (_, score_fake), (_, score_real) in zip(d_fake, d_real):
                    d_loss += torch.mean(torch.sum(torch.pow(score_real - 1.0, 2), dim=[1, 2]))
                    d_loss += torch.mean(torch.sum(torch.pow(score_fake, 2), dim=[1, 2]))
                d_optim.zero_grad()
                d_loss.backward()
                d_optim.step()

                # generator
                d_fake = d_model(wav_fake)
                for (feat_fake, score_fake), (feat_real, _) in zip(d_fake, d_real):
                    g_loss += torch.mean(torch.sum(torch.pow(score_fake - 1.0, 2), dim=[1, 2]))
                    for feat_fake_i, feat_real_i in zip(feat_fake, feat_real):
                        g_loss += 10. * F.l1_loss(feat_fake_i, feat_real_i.detach())

            factor = 10. if step < pretraining_steps else 10.

            stft_norm_loss, stft_spec_loss = multires_stft_loss(wav_fake_istft.squeeze(1), wav_real.squeeze(1))
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

            if step % train_cfg['eval_steps'] == 0:
                g_model.eval()
                val_norm_loss = 0
                val_spec_loss = 0
                val_wavs = []

                for i, val_data in enumerate(val_dataset):
                    val_mel = val_data['mel'].to(device)
                    val_mel = val_mel.unsqueeze(0)
                    s, p, waf_fake = g_model.inference(val_mel)
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
                        'model_g': g_model.state_dict(),
                        'optim_g': g_optim.state_dict(),
                        'model_d': d_model.state_dict(),
                        'optim_d': d_optim.state_dict(),
                        'config': config,
                        'step': step
                    }, f'checkpoints/best_model_{model_name}.pt')
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
            'model_g': g_model.state_dict(),
            'optim_g': g_optim.state_dict(),
            'model_d': d_model.state_dict(),
            'optim_d': d_optim.state_dict(),
            'config': config,
            'step': step
        }, f'checkpoints/latest_model__{model_name}.pt')