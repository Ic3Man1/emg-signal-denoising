from sklearn.decomposition import PCA

def pca_multichannel_denoise(multi_channel_data, variance_to_keep=0.90):
    """
    Implementacja PCA do denoisingu wielokanałowego sygnału EMG.
    Odszumia wszystkie kanały naraz, usuwając szum nieskorelowany i przesłuchy (crosstalk).
    
    Parametry:
    - multi_channel_data: tablica numpy o kształcie (próbki, kanały).
    - variance_to_keep: ile procent "informacji" chcemy zachować (0.90 oznacza 90%).
      Reszta (10%) zostanie odrzucona jako szum.
    """
    # Inicjalizacja PCA: mówimy mu, żeby zachował zadaną wariancję (np. 90%)
    pca = PCA(n_components=variance_to_keep, svd_solver='full')
    
    # Krok 1: Transformacja sygnału do przestrzeni głównych składowych 
    # (tutaj najsłabsze składowe, czyli szum, są odcinane)
    transformed_data = pca.fit_transform(multi_channel_data)
    
    # Krok 2: Powrót do normalnego sygnału (ale już bez odrzuconego szumu)
    denoised_matrix = pca.inverse_transform(transformed_data)
    
    return denoised_matrix

import numpy as np

def diy_pca_multichannel_denoise(multi_channel_data, variance_to_keep=0.90):
    """
    Ręczna implementacja algorytmu PCA do odszumiania sygnałów wielokanałowych.
    
    Parametry:
    - multi_channel_data: tablica numpy 2D (próbki, kanały).
    - variance_to_keep: ułamek wariancji (energii sygnału) do zachowania (np. 0.90).
    """
    # 1. WYŚRODKOWANIE DANYCH (Centering)
    # Z każdego kanału odejmujemy jego średnią wartość. 
    # PCA wymaga, aby dane "kręciły się" wokół zera układu współrzędnych.
    means = np.mean(multi_channel_data, axis=0)
    X_centered = multi_channel_data - means
    
    # 2. MACIERZ KOWARIANCJI
    # Obliczamy, jak bardzo poszczególne kanały "zmieniają się razem".
    # rowvar=False oznacza, że kolumny to nasze kanały (zmienne), a wiersze to próbki.
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # 3. WEKTORY I WARTOŚCI WŁASNE (Eigendecomposition)
    # Używamy np.linalg.eigh (dla macierzy symetrycznych), co gwarantuje stabilność.
    # Wektory własne to "nowe osie", a wartości własne to "siła" (wariancja) na tych osiach.
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Z funkcji eigh() wyniki wychodzą posortowane rosnąco, więc musimy je odwrócić (malejąco),
    # aby najważniejsze składowe były na początku.
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]
    
    # 4. WYBÓR GŁÓWNYCH SKŁADOWYCH (Odcięcie szumu)
    # Obliczamy skumulowaną sumę wariancji
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Szukamy, ile składowych potrzebujemy, aby przekroczyć próg 'variance_to_keep'
    # np.argmax zwraca pierwszy indeks, gdzie warunek jest spełniony. +1 bo indeksy są od 0.
    num_components = np.argmax(cumulative_variance >= variance_to_keep) + 1
    
    # Odrzucamy słabe wektory własne (które reprezentują szum nieskorelowany / crosstalk)
    top_eigenvectors = eigenvectors[:, :num_components]
    
    # 5. TRANSFORMACJA DO PRZESTRZENI PCA
    # Rzutujemy oryginalne (wyśrodkowane) dane na nowe, wybrane osie.
    # To odpowiednik pca.transform() w scikit-learn.
    X_pca = np.dot(X_centered, top_eigenvectors)
    
    # 6. REKONSTRUKCJA (Powrót do oryginalnej przestrzeni)
    # Mnożymy z powrotem przez transponowane wektory i dodajemy średnie,
    # które odjęliśmy na samym początku.
    X_reconstructed = np.dot(X_pca, top_eigenvectors.T) + means
    
    return X_reconstructed
