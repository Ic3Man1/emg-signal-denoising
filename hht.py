import numpy as np
import warnings
from PyEMD import EMD
from metrics import snr

def hht_emd_denoise(signal_1d, drop_first_n=1, drop_last_n=1):
    """
    Odszumia sygnał wykorzystując Empiryczną Dekompozycję (część algorytmu HHT).
    
    Parametry:
    - signal_1d: jednowymiarowa tablica numpy z sygnałem.
    - drop_first_n: ile początkowych IMF-ów usunąć (odpowiadają za szum wysokoczęstotliwościowy).
                    Dla EMG zazwyczaj odrzuca się 1 (lub 2 przy gigantycznym szumie).
    - drop_last_n: ile końcowych IMF-ów i rezyduum usunąć (odpowiadają za pływanie linii izoelektrycznej).
                   Wartość 1 lub 2 świetnie stabilizuje sygnał w osi Y.
    """
    # Wyłączenie logów biblioteki
    warnings.filterwarnings("ignore")
    
    # Inicjalizacja algorytmu EMD
    emd = EMD()
    
    # Dekompozycja sygnału na funkcje składowe (IMFs)
    # Wymaga formatu zmiennoprzecinkowego podwójnej precyzji dla stabilności numerycznej
    imfs = emd.emd(signal_1d.astype(np.float64))
    
    num_imfs = imfs.shape[0]
    
    # Zabezpieczenie: jeśli algorytm znalazł za mało składowych, zwracamy oryginał
    if num_imfs <= (drop_first_n + drop_last_n):
        return signal_1d
        
    # Wycinamy IMFs, które nas interesują (użyteczne pasmo mięśniowe)
    # Zostawiamy wszystko oprócz pierwszych 'drop_first_n' oraz ostatnich 'drop_last_n'
    valid_imfs = imfs[drop_first_n : num_imfs - drop_last_n]
    
    # Sygnał odszumiony to po prostu suma pozostałych składowych
    denoised_signal = np.sum(valid_imfs, axis=0)
    
    return denoised_signal

import itertools
import numpy as np

# Zakładam, że masz już zaimportowane hht_emd_denoise oraz snr

def tune_hht_emd(clean_data, noisy_data):
    """
    Przeszukuje kombinacje parametrów odcinania IMF dla algorytmu HHT (EMD),
    aby znaleźć te maksymalizujące średni SNR na kanałach.
    """
    # Definiujemy rozsądne przedziały dla sygnałów biomedycznych
    first_n_values = [0, 1, 2]  # Odrzucanie szumu wysokoczęstotliwościowego
    last_n_values = [0, 1, 2]   # Odrzucanie pływania linii izoelektrycznej (baseline wander)
    
    best_snr = -np.inf
    best_params = {}
    
    print("\n--- ROZPOCZYNAM STROJENIE PARAMETRÓW HHT (EMD) ---")
    print("Ostrzeżenie: EMD jest procesem iteracyjnym. Strojenie może zająć kilka minut...")
    
    for drop_first, drop_last in itertools.product(first_n_values, last_n_values):
        current_snr_list = []
        
        # Testujemy aktualną kombinację na wszystkich kanałach
        for i in range(noisy_data.shape[1]):
            # WSKAZÓWKA BADAWCZA: Jeśli masz 16 kanałów i czekasz za długo, 
            # odkomentuj poniższą linię, aby stroić tylko na jednym, reprezentatywnym kanale:
            # if i > 0: break 
            
            denoised_channel = hht_emd_denoise(
                noisy_data[:, i], 
                drop_first_n=drop_first, 
                drop_last_n=drop_last
            )
            
            channel_snr = snr(clean_data[:, i], denoised_channel)
            current_snr_list.append(channel_snr)
            
        avg_snr = np.mean(current_snr_list)
        
        print(f"Test: drop_first={drop_first}, drop_last={drop_last} | Średni SNR: {avg_snr:5.2f} dB")
        
        if avg_snr > best_snr:
            best_snr = avg_snr
            best_params = {'drop_first_n': drop_first, 'drop_last_n': drop_last}
            
    print("-" * 50)
    print(f"🏆 NAJLEPSZE PARAMETRY HHT: {best_params}")
    print(f"🏆 NAJLEPSZY ŚREDNI SNR: {best_snr:.2f} dB")
    print("-" * 50)
    
    return best_params

import numpy as np
import warnings
import pywt
from PyEMD import EMD

def hht_hybrid_denoise(signal_1d, drop_last_n=1, threshold_multiplier=0.2):
    """
    Zaawansowana hybryda HHT i progowania. 
    Zamiast wyrzucać wysokoczęstotliwościowy IMF1, poddaje go miękkiemu progowaniu.
    """
    warnings.filterwarnings("ignore")
    emd = EMD()
    imfs = emd.emd(signal_1d.astype(np.float64))
    
    num_imfs = imfs.shape[0]
    if num_imfs <= drop_last_n + 1:
        return signal_1d
        
    # 1. RATUJEMY IMF1: Zamiast wyrzucać, czyścimy go matematycznie
    imf1 = imfs[0]
    
    # Liczymy próg odcięcia szumu (dokładnie jak w Twojej pierwszej funkcji)
    sigma = np.median(np.abs(imf1)) / 0.6745
    threshold = threshold_multiplier * sigma * np.sqrt(2 * np.log(len(imf1)))
    
    # Soft thresholding na samym IMF1
    imf1_cleaned = pywt.threshold(imf1, threshold, mode='soft')
    
    # Podmieniamy zaszumiony IMF1 na ten wyczyszczony
    imfs[0] = imf1_cleaned
    
    # 2. ODRZUCAMY DÓŁ: Pozbywamy się pływania linii (artefaktów ruchowych)
    if drop_last_n > 0:
        valid_imfs = imfs[:-drop_last_n]
    else:
        valid_imfs = imfs
        
    # 3. SUMUJEMY: Zrekonstruowany, czysty sygnał
    denoised_signal = np.sum(valid_imfs, axis=0)
    
    return denoised_signal

import numpy as np
import warnings
from PyEMD import EEMD

def eemd_denoise(signal_1d, drop_first_n=1, drop_last_n=1, trials=20):
    """
    Odszumia sygnał wykorzystując EEMD (Ensemble EMD) - ulepszoną, 
    odporną na szum wersję czystego HHT.
    
    Parametry:
    - signal_1d: jednowymiarowa tablica numpy z sygnałem.
    - drop_first_n: ile szybkich warstw usunąć (teraz wyrzucenie 1 jest bezpieczne).
    - drop_last_n: ile wolnych warstw usunąć (pływanie linii).
    - trials: liczba prób z dodanym szumem. Im więcej, tym dokładniej, ale wolniej. 
              20-30 to dobry kompromis dla domowego komputera.
    """
    warnings.filterwarnings("ignore")
    
    # Inicjalizacja EEMD z kontrolowaną liczbą powtórzeń i siłą sztucznego szumu
    # noise_width=0.05 to standardowa wartość dla danych biomedycznych
    eemd = EEMD(trials=trials, noise_width=0.05)
    
    # Dekompozycja sygnału (to miejsce będzie chwilę liczyć!)
    imfs = eemd.eemd(signal_1d.astype(np.float64))
    
    num_imfs = imfs.shape[0]
    
    # Zabezpieczenie
    if num_imfs <= (drop_first_n + drop_last_n):
        return signal_1d
        
    # Bezpieczne wycinanie warstw szumu (góra) i pływania (dół)
    valid_imfs = imfs[drop_first_n : num_imfs - drop_last_n]
    
    # Rekonstrukcja
    denoised_signal = np.sum(valid_imfs, axis=0)
    
    return denoised_signal