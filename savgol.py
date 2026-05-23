from scipy.signal import savgol_filter

def savgol_denoise(signal_1d, window_length=15, polyorder=3):
    if window_length % 2 == 0:
        window_length += 1
        
    if window_length <= polyorder:
        raise ValueError("długość okna (window_length) musi być większa niż rząd wielomianu (polyorder).")

    denoised_signal = savgol_filter(signal_1d, window_length=window_length, polyorder=polyorder)
    
    return denoised_signal