from pathlib import Path
import tqdm
import torch
import soundfile as sf
from torch.cuda import is_available
from torch.utils.tensorboard import SummaryWriter

from stylemelgan.dataset import new_dataloader, AudioDataset
from stylemelgan.discriminator import MultiScaleDiscriminator
from stylemelgan.generator import MelganGenerator
from matplotlib.figure import Figure
import matplotlib as mpl
mpl.use('agg')  # Use non-interactive backend by default
import numpy as np
import matplotlib.pyplot as plt


def plot_wav(wav: np.array, color='blue') -> Figure:
    fig = plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(wav, color=color)
    return fig


if __name__ == '__main__':

    device = torch.device('cuda') if is_available() else torch.device('cpu')

    step = 0

    g_model = MelganGenerator(80).to(device)
    d_model = MultiScaleDiscriminator().to(device)

    g_optim = torch.optim.Adam(g_model.parameters(), lr=1e-4, betas=(0.5, 0.9))
    d_optim = torch.optim.Adam(d_model.parameters(), lr=1e-4, betas=(0.5, 0.9))

    try:
        checkpoint = torch.load('checkpoints/latest_model.pt', map_location=device)
        g_model.load_state_dict(checkpoint['g_model'])
        g_optim.load_state_dict(checkpoint['g_optim'])
        d_model.load_state_dict(checkpoint['d_model'])
        d_optim.load_state_dict(checkpoint['d_optim'])
        step = checkpoint['step']
    except Exception as e:
        print(e)

    train_data_path = Path('/home/sysgen/chris/data/asvoice2_splitted_train')
    val_data_path = Path('/home/sysgen/chris/data/asvoice2_splitted_val')
    dataloader = new_dataloader(data_path=train_data_path, segment_len=16000, hop_len=256, batch_size=16)
    val_dataset = AudioDataset(data_path=val_data_path, segment_len=None, hop_len=256)

    summary_writer = SummaryWriter(log_dir='checkpoints/logs')

    for epoch in range(100):
        pbar = tqdm.tqdm(enumerate(dataloader, 1), total=len(dataloader))
        for data in pbar:
            step += 1
            mel_seg = data['mel'].to(device)
            wav_seg = data['wav'].to(device)

            # generator
            g_optim.zero_grad()
            wav_fake = g_model(mel_seg)[:, :, :16000]
            d_fake = d_model(wav_fake)
            d_real = d_model(wav_seg)
            g_loss = 0.0
            for (feat_fake, score_fake), (feat_real, _) in zip(d_fake, d_real):
                g_loss += torch.mean(torch.sum(torch.pow(score_fake - 1.0, 2), dim=[1, 2]))
                for feat_fake_i, feat_real_i in zip(feat_fake, feat_real):
                    g_loss += 10. * torch.mean(torch.abs(feat_fake_i - feat_real_i))
            g_loss.backward()
            g_optim.step()

            # discriminator
            wav_fake = wav_fake.detach()
            d_optim.zero_grad()
            d_fake = d_model(wav_fake)
            d_real = d_model(wav_seg)
            d_loss = 0.0
            for (_, score_fake), (_, score_real) in zip(d_fake, d_real):
                d_loss += torch.mean(torch.sum(torch.pow(score_real - 1.0, 2), dim=[1, 2]))
                d_loss += torch.mean(torch.sum(torch.pow(score_fake, 2), dim=[1, 2]))
            d_loss.backward()
            d_optim.step()

            pbar.set_description(desc=f'Epoch: {epoch} | Step {step} '
                                      f'| g_loss: {g_loss:#.4} | d_loss: {d_loss:#.4}', refresh=True)

            summary_writer.add_scalar('generator_loss', g_loss, global_step=step)
            summary_writer.add_scalar('discriminator_loss', d_loss, global_step=step)

            if step % 10000 == 1:
                g_model.eval()
                val_mel = val_dataset[0]['mel'].to(device)
                val_mel = val_mel.unsqueeze(0)
                wav_fake = g_model.inference(val_mel)
                wav_real = val_dataset[0]['wav'].detach().numpy()
                g_model.train()
                summary_writer.add_audio('generated', wav_fake, sample_rate=22050, global_step=step)
                summary_writer.add_audio('target', wav_real, sample_rate=22050, global_step=step)
                #wav_fake_plot = plot_wav(wav_fake)
                #wav_real_plot = plot_wav(wav_real)
                #summary_writer.add_figure('image_generated', wav_fake_plot, global_step=step)
                #summary_writer.add_figure('image_target', wav_real_plot, global_step=step)

        # epoch end
        torch.save({
            'g_model': g_model.state_dict(),
            'g_optim': g_optim.state_dict(),
            'd_model': d_model.state_dict(),
            'd_optim': d_optim.state_dict(),
            'step': step
        }, 'checkpoints/latest_model.pt')