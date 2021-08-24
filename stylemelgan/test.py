import torch

if __name__ == '__main__':

    x = torch.rand((22050, ))
    n_fft = 1024
    hop_length = 256
    win_length = 1024


    stft = torch.stft(x, n_fft, hop_length=hop_length, win_length=win_length, window=None, center=True, pad_mode='reflect',
               normalized=False, onesided=None, return_complex=True)
    print(stft)