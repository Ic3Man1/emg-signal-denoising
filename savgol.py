from scipy.signal import savgol_filter
import numpy as np
from math import factorial

def savgol_denoise(signal_1d, window_length=15, polyorder=3):
    if window_length % 2 == 0:
        window_length += 1
        
    if window_length <= polyorder:
        raise ValueError("długość okna (window_length) musi być większa niż rząd wielomianu (polyorder).")

    denoised_signal = savgol_filter(signal_1d, window_length=window_length, polyorder=polyorder)
    
    return denoised_signal

def custom_savgol_denoise(signal_1d, window_length, order):
    """
    Implementacja filtru Savitzky'ego-Golaya.
    
    Parametry:
    signal_1d : array_like - sygnał wejściowy
    window_length : int - rozmiar okna (musi być nieparzysty)
    order : int - rząd wielomianu dopasowania (musi być < window_size)
    """
    if window_length % 2 == 0:
        raise ValueError("Rozmiar okna (window_length) musi być liczbą nieparzystą.")
    if window_length <= order:
        raise ValueError("Rozmiar okna musi być większy niż rząd wielomianu (order).")
        
    half_window = window_length // 2
    
    # 1. Tworzymy lokalną oś czasu dla okna (np. dla okna 5: [-2, -1, 0, 1, 2])
    z = np.arange(-half_window, half_window + 1)
    
    # 2. Budujemy macierz Vandermonde'a
    # Kolumny to kolejne potęgi: z^0, z^1, z^2 ... z^order
    J = np.vstack([z**j for j in range(order + 1)]).T

    # 3. Obliczamy macierz pseudo-odwrotną Moore'a-Penrose'a
    # Odpowiada to rozwiązaniu układu równań (J^T * J)^(-1) * J^T
    J_pinv = np.linalg.pinv(J)
    
    # 4. Pobieramy współczynniki dla środkowego punktu okna.
    # Wiersz macierzy odpowiada rzędowi szukanej pochodnej - u nas zawsze 0, do wygladzenia.
    coeffs = J_pinv[0]
    
    # 5. Przygotowanie sygnału wejściowego (padding)
    # Rozszerzamy krawędzie przez odbicie lustrzane, by uniknąć spadków na końcach sygnału
    signal_padded = np.pad(signal_1d, half_window, mode='reflect')
    
    # 6. Splot współczynników z sygnałem
    # Odwracamy współczynniki (coeffs[::-1]), by splot działał jak korelacja krzyżowa
    signal_denoised = np.convolve(signal_padded, coeffs[::-1], mode='valid')
    
    return signal_denoised
