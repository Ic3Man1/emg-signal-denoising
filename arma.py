import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from metrics import snr

def arma_denoise(signal_1d, p=4, q=2, window_size=1000):
    """
    Odszumia sygnał za pomocą modelu ARMA(p, q) przetwarzanego w oknach.
    Wymaga biblioteki statsmodels.
    
    Parametry:
    - signal_1d: jednowymiarowa tablica z sygnałem.
    - p (ar_order): rząd części autoregresyjnej (ile pamiętamy w tył z sygnału).
    - q (ma_order): rząd części średniej ruchomej (ile pamiętamy w tył z szumu).
    - window_size: rozmiar okna w próbkach (zapobiega zawieszeniu obliczeń na długich sygnałach).
    """
    # Wyłączamy ostrzeżenia o problemach ze zbieżnością w trudnych oknach
    warnings.filterwarnings("ignore")
    
    denoised_signal = np.zeros_like(signal_1d)
    num_windows = int(np.ceil(len(signal_1d) / window_size))
    
    for i in range(num_windows):
        # Wycinamy fragment sygnału
        start_idx = i * window_size
        end_idx = min((i + 1) * window_size, len(signal_1d))
        window_data = signal_1d[start_idx:end_idx]
        
        # Zabezpieczenie przed zbyt krótkim oknem na samym końcu sygnału
        if len(window_data) <= max(p, q):
            denoised_signal[start_idx:end_idx] = window_data
            continue
            
        try:
            # Tworzymy model ARMA (technicznie ARIMA gdzie parametr różnicowania d=0)
            # trend='n' oznacza, że nie zakładamy stałego przesunięcia (EMG ma średnią w okolicach zera)
            model = ARIMA(window_data, order=(p, 0, q), trend='n')
            
            # Dopasowujemy model do danych
            fitted_model = model.fit()
            
            # Odszumiony fragment to "przewidywania" (fitted values) modelu
            denoised_signal[start_idx:end_idx] = fitted_model.fittedvalues
            
        except Exception as e:
            # Jeśli optymalizator polegnie na konkretnym, bardzo zaszumionym fragmencie,
            # po prostu zostawiamy ten ułamek sekundy bez zmian, by nie wysypać programu.
            denoised_signal[start_idx:end_idx] = window_data

    return denoised_signal

import itertools
import numpy as np
# Zakładam, że masz już tu zaimportowane funkcje: arma_denoise_fast oraz snr

def tune_arma(clean_data, noisy_data):
    """
    Przeszukuje kombinacje parametrów p i q dla modelu ARMA, 
    aby znaleźć te maksymalizujące średni SNR.
    """
    # 1. Definiujemy parametry (małe siatki, by uniknąć wielogodzinnych obliczeń)
    # p (AR order): 2, 4, 6 - wystarczy do uchwycenia kształtu potencjału EMG
    p_values = [6, 8, 10] 
    # q (MA order): 0, 1 - wyższe wartości masakrują wydajność i często nie zbiegają
    q_values = [0, 1]   

    window_size = [250, 500, 750, 1000]  # Stały rozmiar okna dla wszystkich testów, by wyniki były porównywalne 
    
    best_snr = -np.inf
    best_params = {}
    
    print("\n--- ROZPOCZYNAM STROJENIE PARAMETRÓW ARMA (Grid Search) ---")
    print("Ostrzeżenie: To może potrwać od kilku do kilkunastu minut. Cierpliwości...")
    
    for p, q, window_size in itertools.product(p_values, q_values, window_size):
        current_snr_list = []
        
        # Testujemy aktualną kombinację na kanałach
        for i in range(noisy_data.shape[1]):
            # Jeśli testowanie trwa za długo, wstaw tu warunek 'if i > 0: break' 
            # by stroić algorytm tylko na zerowym kanale!
            
            denoised_channel = arma_denoise_fast(
                noisy_data[:, i], 
                p=p, 
                q=q, 
                window_size=window_size  # Stały rozmiar okna, by testy były miarodajne
            )
            
            # Obliczenie SNR
            channel_snr = snr(clean_data[:, i], denoised_channel)
            current_snr_list.append(channel_snr)
            
        avg_snr = np.mean(current_snr_list)
        
        print(f"Test: p={p}, q={q}. window_size={window_size} | Średni SNR: {avg_snr:5.2f} dB")
        
        # Zapisanie najlepszego wyniku
        if avg_snr > best_snr:
            best_snr = avg_snr
            best_params = {'p': p, 'q': q, 'window_size': window_size}
            
    print("-" * 50)
    print(f"🏆 NAJLEPSZE PARAMETRY ARMA: {best_params}")
    print(f"🏆 NAJLEPSZY ŚREDNI SNR: {best_snr:.2f} dB")
    print("-" * 50)
    
    return best_params

import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA

def arma_denoise_fast(signal_1d, p=4, q=1, window_size=500):
    """
    Szybka wersja odszumiania ARMA(p, q) dostosowana do sygnałów biomedycznych.
    
    Parametry:
    - signal_1d: jednowymiarowa tablica numpy z zaszumionym sygnałem.
    - p (AR order): rząd autoregresji (ile próbek w tył pamięta model).
    - q (MA order): rząd średniej ruchomej (ile błędów w tył pamięta model).
    - window_size: długość okna w próbkach (zapobiega zawieszeniu na długich sygnałach).
    """
    # Wyłączamy ostrzeżenia o braku pełnej zbieżności algorytmu
    warnings.filterwarnings("ignore")
    
    denoised_signal = np.zeros_like(signal_1d)
    num_windows = int(np.ceil(len(signal_1d) / window_size))
    
    # 1. SKALOWANIE: Sygnał EMG ma bardzo małą amplitudę (rzędu 0.0001).
    # Optymalizatory giełdowe statsmodels "głupieją" przy takich wartościach i dają zera.
    # Podbijamy sygnał 10 000 razy przed matematyką.
    scale_factor = 10000.0 
    signal_scaled = signal_1d * scale_factor
    
    for i in range(num_windows):
        # Wyznaczamy granice okna
        start_idx = i * window_size
        end_idx = min((i + 1) * window_size, len(signal_scaled))
        window_data = signal_scaled[start_idx:end_idx]
        
        # Zabezpieczenie: jeśli na samym końcu sygnału zostanie ogryzek krótszy niż rząd filtru, 
        # zostawiamy go bez zmian.
        if len(window_data) <= max(p, q):
            denoised_signal[start_idx:end_idx] = window_data / scale_factor
            continue
            
        try:
            # Budowa modelu ARIMA z trendem 'n' (brak stałego przesunięcia średniej)
            model = ARIMA(window_data, order=(p, 0, q), trend='n')
            
            # 2. SZYBKIE DOPASOWANIE: method='innovations_mle' jest znacznie szybsze 
            # niż domyślne metody optymalizacyjne w statsmodels.
            fitted_model = model.fit(method='innovations_mle')
            
            # 3. OD-SKALOWANIE: Przewidziane wartości (nasz czysty sygnał) dzielimy 
            # z powrotem przez 10 000, aby wrócić do woltów.
            denoised_signal[start_idx:end_idx] = fitted_model.fittedvalues / scale_factor
            
        except Exception:
            # W razie gdyby optymalizator poległ na wyjątkowo dziwnym szumie w oknie,
            # po prostu przepisujemy zaszumiony fragment, żeby program działał dalej.
            denoised_signal[start_idx:end_idx] = window_data / scale_factor

    return denoised_signal