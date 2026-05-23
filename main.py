import wfdb
import numpy as np

from plots import plot_comparison, plot_three_signals
from noise import add_noise, add_powerline, add_crosstalk, add_motion_artifacts
from wavelet import wavelet_denoise
from metrics import snr, median_frequency
from butterwoth import butterworth_denoise
from savgol import savgol_denoise

RECORD = wfdb.rdrecord("data_set/session3_participant1_gesture10_trial1/session3_participant1_gesture10_trial1")
SIGNAL = RECORD.p_signal

EMG_INDICIES = [i for i, name in enumerate(RECORD.sig_name) if not('U' in name.upper())]
DATA = SIGNAL[:, EMG_INDICIES]

def main():
    noisy_data = add_noise(DATA, 15)

    signal_power = np.sqrt(np.mean(DATA**2))
    noisy_data = add_powerline(noisy_data, amplitude=0.05 * signal_power)
    noisy_data = add_crosstalk(noisy_data, strength=0.10)
    noisy_data = add_motion_artifacts(noisy_data, amplitude=0.05 * signal_power)

    # plot_comparison(data, noisy_data)

    denoised = np.zeros_like(noisy_data)
    for i in range(noisy_data.shape[1]):
        denoised[:, i] = wavelet_denoise(noisy_data[:, i])
        # denoised[:, i] = butterworth_denoise(noisy_data[:, i], fs=RECORD.fs, lowcut=16.0, highcut=500.0, order=1) # 14 500 1 +1.5dB | 15 500 1 +1.6dB | 
        # denoised[:, i] = savgol_denoise(noisy_data[:, i], window_length=23, polyorder=12) # 15 9 + 1.54dB | 17 11 +1.55dB |

    snr_before_all = []
    snr_after_all  = []
    mf_clean_all   = []
    mf_noisy_all   = []
    mf_denoised_all = []

    for i in range(DATA.shape[1]):
        sb = snr(DATA[:, i], noisy_data[:, i])
        sa = snr(DATA[:, i], denoised[:, i])
        mc = median_frequency(DATA[:, i])
        mn = median_frequency(noisy_data[:, i])
        md = median_frequency(denoised[:, i])

        snr_before_all.append(sb)
        snr_after_all.append(sa)
        mf_clean_all.append(mc)
        mf_noisy_all.append(mn)
        mf_denoised_all.append(md)

        print(f"Kanał {i:2d} | "
              f"SNR: {sb:5.2f} -> {sa:5.2f} dB ({sa-sb:+.2f})")

    # plot_comparison(data, denoised)
    # plot_comparison(denoised, noisy_data)
    plot_three_signals(DATA, noisy_data, denoised, channel_idx=0)

if __name__ == "__main__":
    main()