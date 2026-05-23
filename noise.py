import numpy as np

RINGS = {
    'F_ring1': list(range(0, 8)),
    'F_ring2': list(range(8, 16)),
    'W_ring3': list(range(16, 22)),
    'W_ring4': list(range(22, 28)),
}

RING_NEIGHBORS = {
    'F_ring1': ['F_ring1', 'F_ring2'],
    'F_ring2': ['F_ring1', 'F_ring2', 'W_ring3'],
    'W_ring3': ['F_ring2', 'W_ring3', 'W_ring4'],
    'W_ring4': ['W_ring3', 'W_ring4'],
}

def add_noise(signal, snr_db):
    signal_power = np.mean(signal**2, axis=0, keepdims=True)
    snr_lin = 10**(snr_db / 10)
    noise_power = signal_power / snr_lin
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

def add_powerline(signal, fs=2048, freq=50, amplitude=0.05):
    t = np.arange(signal.shape[0]) / fs
    powerline = amplitude * np.sin(2 * np.pi * freq * t)
    return signal + powerline[:, np.newaxis]

def get_ring(channel_idx):
    for ring_name, indices in RINGS.items():
        if channel_idx in indices:
            return ring_name
    return None

def get_physical_neighbors(channel_idx, max_neighbors=4):
    ring = get_ring(channel_idx)
    allowed_rings = RING_NEIGHBORS[ring]
    
    candidates = []
    for ring_name in allowed_rings:
        candidates.extend(RINGS[ring_name])
    
    candidates = [c for c in candidates if c != channel_idx]
    
    candidates.sort(key=lambda c: abs(c - channel_idx))
    return candidates[:max_neighbors]

def add_crosstalk(signal, strength=0.15, max_neighbors=4):
    result = signal.copy()
    n_channels = signal.shape[1]
    
    for i in range(n_channels):
        neighbors = get_physical_neighbors(i, max_neighbors=max_neighbors)
        
        if not neighbors:
            continue

        distances = np.array([abs(i - n) for n in neighbors], dtype=float)
        weights = 1.0 / (distances + 1)
        weights /= weights.sum()
        weights *= strength
        
        for neighbor, weight in zip(neighbors, weights):
            result[:, i] += weight * signal[:, neighbor]
    
    return result

def add_motion_artifacts(signal, fs=2048, n_artifacts=3, amplitude=0.3):
    n_samples = signal.shape[0]
    t = np.arange(n_samples) / fs
    artifact = np.zeros(n_samples)
    
    for _ in range(3):
        freq = np.random.uniform(0.1, 5.0)
        phase = np.random.uniform(0, 2 * np.pi)
        artifact += np.sin(2 * np.pi * freq * t + phase)
    
    for _ in range(n_artifacts):
        position = np.random.randint(0, n_samples)
        width = np.random.randint(50, 300)
        impulse = np.zeros(n_samples)
        start = max(0, position - width//2)
        end = min(n_samples, position + width//2)
        impulse[start:end] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.0)

        from scipy.ndimage import gaussian_filter1d
        artifact += gaussian_filter1d(impulse, sigma=30)
    
    artifact = amplitude * artifact / np.max(np.abs(artifact))  # normalizacja
    return signal + artifact[:, np.newaxis]
