from sklearn.decomposition import PCA

def pca_multichannel_denoise(multi_channel_data, variance_to_keep=0.90):
    """Denoise multichannel data using PCA reconstruction.

    The function fits PCA to the multi-channel data, keeps the principal
    components required to preserve the specified variance, and reconstructs
    the signal from those components to suppress noise.
    """
    pca = PCA(n_components=variance_to_keep, svd_solver='full')

    transformed_data = pca.fit_transform(multi_channel_data)

    denoised_matrix = pca.inverse_transform(transformed_data)

    return denoised_matrix
