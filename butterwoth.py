import numpy as np
from scipy.signal import butter, filtfilt


def butterworth_denoise(signal_1d, fs=2048.0, lowcut=20.0, highcut=400.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    denoised_signal = filtfilt(b, a, signal_1d)
    
    return denoised_signal


def calculate_butter2_coeffs(fc, fs, btype='low'):
    """
    Oblicza współczynniki (a, b) dla cyfrowego filtru Butterwortha 2. rzędu.
    Wykorzystuje transformację dwuliniową (Bilinear Transform).
    """
    # W to tzw. wstępnie wypaczona (pre-warped) częstotliwość kątowa
    W = np.tan(np.pi * fc / fs)
    c = 1 + np.sqrt(2) * W + W**2
    
    # Współczynniki dla sprzężenia zwrotnego (a) - takie same dla obu typów
    a1 = 2 * (W**2 - 1) / c
    a2 = (1 - np.sqrt(2) * W + W**2) / c
    
    if btype == 'low':
        # Współczynniki wejściowe (b) dla filtru dolnoprzepustowego
        b0 = W**2 / c
        b1 = 2 * W**2 / c
        b2 = W**2 / c
    elif btype == 'high':
        # Współczynniki wejściowe (b) dla filtru górnoprzepustowego
        b0 = 1 / c
        b1 = -2 / c
        b2 = 1 / c
    else:
        raise ValueError("Typ filtru musi być 'low' albo 'high'")
        
    return b0, b1, b2, a1, a2


def apply_difference_equation(x, b0, b1, b2, a1, a2):
    """
    Aplikuje równanie różnicowe IIR do jednowymiarowego sygnału.
    y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
    """
    y = np.zeros_like(x)
    
    # Inicjalizacja dla pierwszych dwóch próbek (zakładamy stan zerowy przed startem)
    if len(x) > 0:
        y[0] = b0 * x[0]
    if len(x) > 1:
        y[1] = b0 * x[1] + b1 * x[0] - a1 * y[0]
        
    # Główna pętla filtrująca
    for n in range(2, len(x)):
        y[n] = b0 * x[n] + b1 * x[n-1] + b2 * x[n-2] - a1 * y[n-1] - a2 * y[n-2]
        
    return y


def diy_filtfilt(x, b0, b1, b2, a1, a2):
    """
    Filtruje sygnał w przód i w tył, aby wyzerować przesunięcie fazowe.
    """
    # 1. Filtrowanie w przód
    y_forward = apply_difference_equation(x, b0, b1, b2, a1, a2)
    
    # 2. Odwrócenie sygnału w czasie
    y_reversed = y_forward[::-1]
    
    # 3. Filtrowanie w tył
    y_backward = apply_difference_equation(y_reversed, b0, b1, b2, a1, a2)
    
    # 4. Ponowne odwrócenie, aby wrócić do oryginalnego czasu
    return y_backward[::-1]


def diy_butterworth_bandpass(signal_1d, fs, lowcut=20.0, highcut=400.0):
    """
    Odszumia sygnał za pomocą pasmowoprzepustowego filtru Butterwortha.
    Implementacja "od zera" bez użycia scipy.
    """
    # ETAP 1: Filtr górnoprzepustowy (usuwa pływanie linii izoelektrycznej)
    hp_coeffs = calculate_butter2_coeffs(lowcut, fs, btype='high')
    signal_highpassed = diy_filtfilt(signal_1d, *hp_coeffs)
    
    # ETAP 2: Filtr dolnoprzepustowy (usuwa szum wysokiej częstotliwości)
    lp_coeffs = calculate_butter2_coeffs(highcut, fs, btype='low')
    signal_bandpassed = diy_filtfilt(signal_highpassed, *lp_coeffs)
    
    return signal_bandpassed