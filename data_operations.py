import numpy as np
from metrics import snr, prd, correlation_coefficient, spectral_distortion
from plots import plot_three_signals
from wavelet import wavelet_denoise
from savgol import savgol_denoise
from pca import pca_multichannel_denoise

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
        'func': savgol_denoise,
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



def build_denoised_data(method_name, noisy_data, params):
    """Build denoised output for a given method and parameter set.

    If the selected method is PCA, denoise the full multi-channel matrix
    in a single pass. Otherwise apply the denoising function channel-wise.
    """
    method = DENOISE_METHODS[method_name]['func']

    if method_name == 'pca':
        return method(noisy_data, **params)

    denoised = np.zeros_like(noisy_data)
    for ch in range(noisy_data.shape[1]):
        denoised[:, ch] = method(noisy_data[:, ch], **params)
    return denoised


def evaluate_denoised_data(clean_data, noisy_data, denoised_data):
    """Compute average evaluation metrics for denoised data.

    The function compares the denoised output to the clean signal on each
    channel and returns mean values for SNR, PRD, correlation, and spectral
    distortion.
    """
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


def compare_parameters_for_method(method_name, clean_data, noisy_data, channel_idx=0):
    """Compare multiple parameter sets for a given denoising method.

    Prints a comparison table for the method and returns the best-performing
    parameter variant based on mean SNR improvement.
    """
    def format_parameters(params):
        return ', '.join(f'{k}={v}' for k, v in params.items())
    
    method_info = DENOISE_METHODS[method_name]
    method_label = method_info['label']
    print(f"\n=== {method_label} parameter comparison ===")
    print("Parameter set                            | SNR before | SNR after | ΔSNR    | PRD     | Corr    | SD     ")
    print("-----------------------------------------+------------+-----------+---------+---------+---------+--------")

    best_result = None
    for params in method_info['parameter_sets']:
        denoised_data = build_denoised_data(method_name, noisy_data, params)
        metrics = evaluate_denoised_data(clean_data, noisy_data, denoised_data)
        param_text = format_parameters(params)
        print(f"{param_text:<40} | {metrics['snr_before']:10.2f} | {metrics['snr_after']:9.2f} | {metrics['delta_snr']:7.2f} | {metrics['prd']:7.2f} | {metrics['corr']:7.4f} | {metrics['sd']:6.2f}")

        if best_result is None or metrics['delta_snr'] > best_result['metrics']['delta_snr']:
            best_result = {'params': params, 'metrics': metrics, 'denoised': denoised_data}

        plot_three_signals(clean_data, noisy_data, denoised_data, channel_idx=channel_idx, method_name=method_label, params=param_text)

    if best_result is not None:
        print(f"Best {method_label} variant: {format_parameters(best_result['params'])}  ΔSNR={best_result['metrics']['delta_snr']:.2f} dB")

    return best_result


def compare_all_methods(clean_data, noisy_data, channel_idx=0):
    """Compare all denoising methods over their defined parameter sets."""
    for method_name in DENOISE_METHODS:
       compare_parameters_for_method(method_name, clean_data, noisy_data, channel_idx=channel_idx)