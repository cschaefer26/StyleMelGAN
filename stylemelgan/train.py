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
from stylemelgan.dataset import new_dataloader, AudioDataset, mel_spectrogram, new_mel_dataloader
from stylemelgan.discriminator import MultiScaleDiscriminator
from stylemelgan.generator.melgan import Generator
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='stylemelgan/configs/melgan_config.yaml', help='points to config.yaml')
    args = parser.parse_args()

    config = read_config(args.config)
    model_name = config['model_name']
    audio = Audio.from_config(config)
    train_data_path = Path(config['paths']['train_dir'])
    train_pred_data_path = Path(config['paths']['train_dir_mels'])
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
        print(e)
        raise ValueError(e)

    train_cfg = config['training']
    dataloader = new_dataloader(data_path=train_data_path, segment_len=train_cfg['segment_len'],
                                hop_len=audio.hop_length, batch_size=train_cfg['batch_size'],
                                num_workers=train_cfg['num_workers'], sample_rate=audio.sample_rate)

    mel_files = list(train_pred_data_path.glob('**/*.pt'))
    val_mel_files = mel_files[:512]
    train_mel_files = mel_files[512:]
    train_mel_dataloader = new_mel_dataloader(files=train_mel_files, segment_len=train_cfg['segment_len'],
                                        hop_len=audio.hop_length, batch_size=train_cfg['batch_size_mel'],
                                        num_workers=train_cfg['num_workers'])
    val_mel_dataloader = new_mel_dataloader(files=val_mel_files, segment_len=None,
                                        hop_len=audio.hop_length, batch_size=1,
                                        num_workers=train_cfg['num_workers'])

    val_dataset = AudioDataset(data_path=val_data_path, segment_len=None, hop_len=audio.hop_length,
                               sample_rate=audio.sample_rate)

    stft = partial(stft, n_fft=1024, hop_length=256, win_length=1024)

    pretraining_steps = train_cfg['pretraining_steps']

    summary_writer = SummaryWriter(log_dir=f'checkpoints/logs_{model_name}')

    best_stft = 9999
    best_exp = 9999

    for epoch in range(train_cfg['epochs']):
        pbar = tqdm.tqdm(enumerate(zip(dataloader, train_mel_dataloader), 1), total=len(dataloader))
        for i, (data, data_mel) in pbar:
            step += 1
            mel = data['mel'].to(device)
            wav_real = data['wav'].to(device)

            wav_fake = g_model(mel)[:, :, :train_cfg['segment_len']]

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


            mel_pred = data_mel['mel_post'].to(device)

            mel_pred_loss = 0

            for b in range(mel_pred.size(0)):
                #print(b)
                wav_pred_fake = g_model(mel_pred[b:b+1, :, :])
                mel_fake = mel_spectrogram(wav_pred_fake.squeeze(1), n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256,
                                           win_size=1024, fmin=0, fmax=8000)
                diff = (torch.exp(mel_fake) - torch.exp(mel_pred)) ** 2
                diff = diff.mean(1)
                diff[diff < 0.005] = 0
                mel_pred_loss += 100. * diff.sum()

            print(mel_pred_loss)

            factor = 1. if step < pretraining_steps else 0.

            stft_norm_loss, stft_spec_loss = multires_stft_loss(wav_fake.squeeze(1), wav_real.squeeze(1))
            g_loss_all = g_loss + mel_pred_loss + factor * (stft_norm_loss + stft_spec_loss)

            g_optim.zero_grad()
            g_loss_all.backward()
            g_optim.step()

            pbar.set_description(desc=f'Epoch: {epoch} | Step {step} '
                                      f'| g_loss: {g_loss:#.4} '
                                      f'| d_loss: {d_loss:#.4} '
                                      f'| stft_norm_loss {stft_norm_loss:#.4} '
                                      f'| mel_pred_loss {mel_pred_loss:#.4} '
                                      f'| stft_spec_loss {stft_spec_loss:#.4} ', refresh=True)

            summary_writer.add_scalar('generator_loss', g_loss, global_step=step)
            summary_writer.add_scalar('generator_mel_pred_loss', mel_pred_loss, global_step=step)
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
                    wav_fake = g_model.inference(val_mel).squeeze().cpu().numpy()
                    wav_real = val_data['wav'].detach().squeeze().cpu().numpy()
                    wav_f = torch.tensor(wav_fake).unsqueeze(0).to(device)
                    wav_r = torch.tensor(wav_real).unsqueeze(0).to(device)
                    val_wavs.append((wav_fake, wav_real, val_mel))
                    size = min(wav_r.size(-1), wav_f.size(-1))
                    val_n, val_s = multires_stft_loss(wav_f[..., :size], wav_r[..., :size])
                    val_norm_loss += val_n
                    val_spec_loss += val_s

                val_norm_loss /= len(val_dataset)
                val_spec_loss /= len(val_dataset)
                summary_writer.add_scalar('val_stft_norm_loss', val_norm_loss, global_step=step)
                summary_writer.add_scalar('val_stft_spec_loss', val_spec_loss, global_step=step)
                val_wavs.sort(key=lambda x: x[1].shape[0])
                wav_fake, wav_real, mel_val = val_wavs[-1]
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

                val_mel_loss = 0
                worst, best = (-9999, None), (9999, None)
                for i, val_mel in tqdm.tqdm(enumerate(val_mel_dataloader), total=len(val_mel_dataloader)):
                    val_mel_pred = val_mel['mel_post'].to(device)
                    with torch.no_grad():
                        wav_pred_fake = g_model(val_mel_pred)
                        mel_fake = mel_spectrogram(wav_pred_fake.squeeze(1), n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256,
                                                   win_size=1024, fmin=0, fmax=8000)
                        #mel_pred_loss = 10000. * F.mse_loss(torch.exp(mel_fake), torch.exp(mel_pred))
                        #mel_pred_loss = 1000. * torch.norm(torch.exp(mel_fake) - torch.exp(val_mel_pred), p="fro") / torch.norm(torch.exp(val_mel_pred), p="fro")
                        diff = (torch.exp(mel_fake) - torch.exp(val_mel_pred)) ** 2
                        diff = diff.mean(1)
                        diff[diff < 0.005] = 0
                        mel_pred_loss = 100. * diff.mean()

                        if mel_pred_loss > worst[0]:
                            worst = (mel_pred_loss, wav_pred_fake)
                        if mel_pred_loss < best[0]:
                            best = (mel_pred_loss, wav_pred_fake)
                        val_mel_loss += mel_pred_loss

                val_mel_loss /= len(val_mel_dataloader)
                summary_writer.add_audio('best_exp_generated', best[1].squeeze(), sample_rate=audio.sample_rate, global_step=step)
                summary_writer.add_audio('worst_exp_generated', worst[1].squeeze(), sample_rate=audio.sample_rate, global_step=step)

                summary_writer.add_scalar('generator_mel_pred_loss_val', val_mel_loss, global_step=step)
                if val_mel_loss < best_exp:
                    best_exp = val_mel_loss
                    print(f'\nnew best val exp loss: {best_stft}')
                    torch.save({
                        'model_g': g_model.state_dict(),
                        'config': config,
                        'step': step
                    }, f'checkpoints/best_model_exp_{best_exp:#.2}_{model_name}.pt')

                g_model.train()
                summary_writer.add_audio('generated', wav_fake, sample_rate=audio.sample_rate, global_step=step)
                summary_writer.add_audio('target', wav_real, sample_rate=audio.sample_rate, global_step=step)
                mel_fake = audio.wav_to_mel(wav_fake)
                mel_real = audio.wav_to_mel(wav_real)
                mel_fake_plot = plot_mel(mel_fake)
                mel_real_plot = plot_mel(mel_real)
                summary_writer.add_figure('mel_generated', mel_fake_plot, global_step=step)
                summary_writer.add_figure('mel_target', mel_real_plot, global_step=step)

                mel_input_plot = plot_mel(mel_val.cpu().squeeze().numpy())
                summary_writer.add_figure('mel_input', mel_input_plot, global_step=step)

        # epoch end
        torch.save({
            'model_g': g_model.state_dict(),
            'optim_g': g_optim.state_dict(),
            'model_d': d_model.state_dict(),
            'optim_d': d_optim.state_dict(),
            'config': config,
            'step': step
        }, f'checkpoints/latest_model__{model_name}.pt')