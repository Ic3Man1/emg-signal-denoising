import matplotlib.pyplot as plt
import numpy as np

from metrics import snr

def plot_comparison(original, noisy, fs=2048, n_samples=300, channel_idx=0):
    t = np.arange(n_samples) / fs
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(t, original[:n_samples, channel_idx], 
             color='blue', alpha=1, label='Sygnał czysty', linewidth=1.5)
    
    plt.plot(t, noisy[:n_samples, channel_idx], 
             color='red', alpha=0.65, label='Sygnał zaszumiony', linewidth=1)
    
    plt.title(f"Porównanie sygnału EMG, kanał {channel_idx}")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda [mV]")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_three_signals(original, noisy, denoised, fs=2048, n_samples=1024, channel_idx=0):
    t = np.arange(n_samples) / fs

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(t, original[:n_samples, channel_idx], color='steelblue', linewidth=1.2)
    axes[0].set_title(f'Sygnał oryginalny, kanał {channel_idx}')
    axes[0].set_ylabel('Amplituda [mV]')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, noisy[:n_samples, channel_idx], color='crimson', linewidth=0.8, alpha=0.85)
    axes[1].set_title('Po zaszumieniu')
    axes[1].set_ylabel('Amplituda [mV]')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, denoised[:n_samples, channel_idx], color='seagreen', linewidth=1.2)
    axes[2].set_title('Po odszumieniu (wavelet db4)')
    axes[2].set_ylabel('Amplituda [mV]')
    axes[2].set_xlabel('Czas [s]')
    axes[2].grid(True, alpha=0.3)

    sb = snr(original[:, channel_idx], noisy[:, channel_idx])
    sa = snr(original[:, channel_idx], denoised[:, channel_idx])
    fig.suptitle(f'SNR przed: {sb:.2f} dB,  SNR po: {sa:.2f} dB  (poprawa: {sa-sb:+.2f} dB)',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('emg_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()