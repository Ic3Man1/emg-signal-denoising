import matplotlib.pyplot as plt
import numpy as np
import math

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
    axes[2].set_title('Po odszumieniu (PCA)')
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

    import math
import matplotlib.pyplot as plt


def plot_many(data_dict, x_data=None):
    if not data_dict:
        print("No data provided to plot!")
        return

    # Tworzymy jeden wykres
    fig, ax = plt.subplots(figsize=(10, 6))

    # Rysujemy każdą zmienną w pętli na tej samej osi (ax)
    for var_name, values in data_dict.items():
        if x_data is not None:
            ax.plot(x_data, values, label=var_name, linewidth=2)
        else:
            ax.plot(values, label=var_name, linewidth=2)

    # Dekoracja wykresu
    ax.set_title("EMG Signals Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time / Samples", fontsize=12)
    ax.set_ylabel("Amplitude / Value", fontsize=12)

    # Włączamy siatkę (już z poprawną metodą .grid()!)
    ax.grid(True, linestyle="--", alpha=0.7)

    # Dodajemy legendę, żeby było wiadomo, która linia to która zmienna
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    plt.show()