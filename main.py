import wfdb
import numpy as np

from plots import plot_comparison, plot_three_signals, plot_many
from noise import add_noise, add_powerline, add_crosstalk, add_motion_artifacts
from wavelet import wavelet_denoise
from metrics import snr, prd, correlation_coefficient, snr_improvement, spectral_distortion
from savgol import savgol_denoise as custom_savgol_denoise
from pca import pca_multichannel_denoise

RECORD = wfdb.rdrecord("data_set/session3_participant1_gesture10_trial1/session3_participant1_gesture10_trial1")
SIGNAL = RECORD.p_signal

EMG_INDICIES = [i for i, name in enumerate(RECORD.sig_name) if not('U' in name.upper())]
DATA = SIGNAL[:, EMG_INDICIES]

DENOISE_METHODS = {
    'wavelet': {
        'label': 'Wavelet',
        'func': wavelet_denoise,
        'parameter_sets': [
            {'wavelet': 'db4', 'level': 3},
            {'wavelet': 'db4', 'level': 4},
            {'wavelet': 'sym4', 'level': 3},
        ],
    },
    'savgol': {
        'label': 'Savitzky-Golay',
        'func': custom_savgol_denoise,
        'parameter_sets': [
            {'window_length': 17, 'order': 11},
            {'window_length': 15, 'order': 7},
            {'window_length': 21, 'order': 13},
        ],
    },
    'pca': {
        'label': 'PCA',
        'func': pca_multichannel_denoise,
        'parameter_sets': [
            {'variance_to_keep': 0.995},
            {'variance_to_keep': 0.99},
            {'variance_to_keep': 0.95},
        ],
    },
}

CURRENT_PARAMETERS = {
    'wavelet': {'wavelet': 'db4', 'level': 3},
    'savgol': {'window_length': 17, 'order': 11},
    'pca': {'variance_to_keep': 0.995},
}


def build_denoised_data(method_name, noisy_data, params):
    method = DENOISE_METHODS[method_name]['func']

    if method_name == 'pca':
        return method(noisy_data, **params)

    denoised = np.zeros_like(noisy_data)
    for ch in range(noisy_data.shape[1]):
        denoised[:, ch] = method(noisy_data[:, ch], **params)
    return denoised


def evaluate_denoised_data(clean_data, noisy_data, denoised_data):
    snr_before = []
    snr_after = []
    delta_snr = []
    prd_list = []
    corr_list = []
    sd_list = []

    for ch in range(clean_data.shape[1]):
        clean_ch = clean_data[:, ch]
        noisy_ch = noisy_data[:, ch]
        denoised_ch = denoised_data[:, ch]

        sb = snr(clean_ch, noisy_ch)
        sa = snr(clean_ch, denoised_ch)
        prd_val = prd(clean_ch, denoised_ch)
        corr_val = correlation_coefficient(clean_ch, denoised_ch)
        sd_val = spectral_distortion(clean_ch, denoised_ch)

        snr_before.append(sb)
        snr_after.append(sa)
        delta_snr.append(sa - sb)
        prd_list.append(prd_val)
        corr_list.append(corr_val)
        sd_list.append(sd_val)

    return {
        'snr_before': np.mean(snr_before),
        'snr_after': np.mean(snr_after),
        'delta_snr': np.mean(delta_snr),
        'prd': np.mean(prd_list),
        'corr': np.mean(corr_list),
        'sd': np.mean(sd_list),
    }


def format_parameters(params):
    return ', '.join(f'{k}={v}' for k, v in params.items())


def compare_parameters_for_method(method_name, clean_data, noisy_data, channel_idx=0):
    method_info = DENOISE_METHODS[method_name]
    method_label = method_info['label']
    print(f"\n=== {method_label} parameter comparison ===")
    print("Parameter set                          | SNR before | SNR after | ΔSNR    | PRD     | Corr    | SD     ")
    print("----------------------------------------+------------+-----------+---------+---------+---------+--------")

    best_result = None
    for params in method_info['parameter_sets']:
        denoised_data = build_denoised_data(method_name, noisy_data, params)
        metrics = evaluate_denoised_data(clean_data, noisy_data, denoised_data)
        param_text = format_parameters(params)
        print(f"{param_text:<40} | {metrics['snr_before']:10.2f} | {metrics['snr_after']:9.2f} | {metrics['delta_snr']:7.2f} | {metrics['prd']:7.2f} | {metrics['corr']:7.4f} | {metrics['sd']:6.2f}")

        if best_result is None or metrics['delta_snr'] > best_result['metrics']['delta_snr']:
            best_result = {'params': params, 'metrics': metrics, 'denoised': denoised_data}

    if best_result is not None:
        print(f"Best {method_label} variant: {format_parameters(best_result['params'])}  ΔSNR={best_result['metrics']['delta_snr']:.2f} dB")
        plot_three_signals(clean_data, noisy_data, best_result['denoised'], channel_idx=channel_idx, method_name=method_label)

    return best_result


def compare_all_methods(clean_data, noisy_data, channel_idx=0):
    current_denoised = {}
    for method_name in DENOISE_METHODS:
        best = compare_parameters_for_method(method_name, clean_data, noisy_data, channel_idx=channel_idx)
        current_denoised[method_name] = build_denoised_data(method_name, noisy_data, CURRENT_PARAMETERS[method_name])

    series_to_plot = {
        'original': clean_data[:1024, channel_idx],
        'noisy': noisy_data[:1024, channel_idx],
    }
    for method_name, denoised_data in current_denoised.items():
        label = DENOISE_METHODS[method_name]['label']
        series_to_plot[label] = denoised_data[:1024, channel_idx]

    plot_many(series_to_plot)


def main():
    noisy_data = add_noise(DATA, 15)

    signal_power = np.sqrt(np.mean(DATA**2))
    noisy_data = add_powerline(noisy_data, amplitude=0.05 * signal_power)
    noisy_data = add_crosstalk(noisy_data, strength=0.10)
    noisy_data = add_motion_artifacts(noisy_data, amplitude=0.05 * signal_power)

    compare_all_methods(DATA, noisy_data, channel_idx=0)

if __name__ == "__main__":
    main()