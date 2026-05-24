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