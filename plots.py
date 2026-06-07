import matplotlib.pyplot as plt
import numpy as np
import math

from metrics import snr

def plot_comparison(original, noisy, fs=2048, n_samples=300, channel_idx=0):
    """Plot original and noisy signals for one EMG channel.

    Shows a short time window of the clean and noisy signal waveforms for the
    selected channel to visually compare the effect of noise.
    """
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


def plot_three_signals(original, noisy, denoised, fs=2048, n_samples=1024, channel_idx=0, method_name='Denoised', params=''):
    """Plot original, noisy, and denoised signals for one EMG channel.

    Creates a stacked plot with three subplots and annotates the SNR before
    and after denoising so the quality improvement is easy to interpret.
    The method name is used to label the denoised plot.
    """
    t = np.arange(n_samples) / fs

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(t, original[:n_samples, channel_idx], color='steelblue', linewidth=1.2)
    axes[0].set_title(f'Sygnał oryginalny, kanał {channel_idx}')
    axes[0].set_ylabel('Amplituda [mV]')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, noisy[:n_samples, channel_idx], color='crimson', linewidth=0.8, alpha=0.85)
    axes[1].set_title('Sygnał zaszumiony')
    axes[1].set_ylabel('Amplituda [mV]')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, denoised[:n_samples, channel_idx], color='seagreen', linewidth=1.2)
    axes[2].set_title(f'Sygnał odszumiony ({method_name})')
    axes[2].set_ylabel('Amplituda [mV]')
    axes[2].set_xlabel('Czas [s]')
    axes[2].grid(True, alpha=0.3)

    sb = snr(original[:, channel_idx], noisy[:, channel_idx])
    sa = snr(original[:, channel_idx], denoised[:, channel_idx])
    if params:
        fig.suptitle(f'SNR przed: {sb:.2f} dB, SNR po: {sa:.2f} dB  (Δ{sa-sb:+.2f} dB)\nParametry: {params}',
                    fontsize=12, fontweight='bold')
    else:
        fig.suptitle(f'SNR przed: {sb:.2f} dB, SNR po: {sa:.2f} dB  (Δ{sa-sb:+.2f} dB)',
                    fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('emg_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_one_signal(signal, fs=2048, n_samples=1024, channel_idx=0, method_name='Denoised', params=''):
    """Plot a single EMG signal channel with proper labeling and grid."""
    t = np.arange(n_samples) / fs
    
    plt.figure(figsize=(12, 4))
    plt.plot(t, signal[:n_samples, channel_idx], color='navy', linewidth=1.5)
    if params:
        plt.title(f'Sygnał odszumiony ({method_name})\nParametry: {params}')
    else:
        plt.title(f'Sygnał odszumiony ({method_name})')
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda [mV]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_many(data_dict, x_data=None):
    """Plot multiple series on the same axes for EMG comparison.

    Draws each data series in `data_dict` on a common chart. Optionally uses
    provided x-axis values to align the series in time or sample index.
    """
    if not data_dict:
        print("No data provided to plot!")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for var_name, values in data_dict.items():
        if x_data is not None:
            ax.plot(x_data, values, label=var_name, linewidth=2)
        else:
            ax.plot(values, label=var_name, linewidth=2)

    ax.set_title("EMG Signals Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time / Samples", fontsize=12)
    ax.set_ylabel("Amplitude / Value", fontsize=12)

    ax.grid(True, linestyle="--", alpha=0.7)

    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    plt.show()