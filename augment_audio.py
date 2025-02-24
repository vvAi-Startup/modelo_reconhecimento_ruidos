import numpy as np
import librosa

def time_mask(audio, num_masks=1, mask_size=0.1):
    """Aplica máscara temporal no áudio"""
    audio_len = len(audio)
    mask_size = int(audio_len * mask_size)
    masked_audio = audio.copy()
    
    for _ in range(num_masks):
        start = np.random.randint(0, audio_len - mask_size)
        masked_audio[start:start + mask_size] = 0
    
    return masked_audio

def freq_mask(audio, num_masks=1, mask_size=0.1):
    """Aplica máscara de frequência no espectrograma"""
    # Converter para espectrograma
    D = librosa.stft(audio)
    S = np.abs(D)
    
    freq_bins = S.shape[0]
    mask_size = int(freq_bins * mask_size)
    
    for _ in range(num_masks):
        start = np.random.randint(0, freq_bins - mask_size)
        S[start:start + mask_size, :] = 0
    
    # Reconstruir áudio
    masked_audio = librosa.istft(S * np.exp(1.j * np.angle(D)))
    
    # Garantir mesmo tamanho do áudio original
    if len(masked_audio) > len(audio):
        masked_audio = masked_audio[:len(audio)]
    elif len(masked_audio) < len(audio):
        masked_audio = np.pad(masked_audio, (0, len(audio) - len(masked_audio)))
    
    return masked_audio

def pitch_shift(audio, sr, n_steps):
    """Altera o pitch do áudio"""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def time_stretch(audio, rate):
    """Altera a velocidade do áudio"""
    return librosa.effects.time_stretch(audio, rate=rate)

def add_noise(audio, noise_factor=0.005):
    """Adiciona ruído gaussiano ao áudio"""
    noise = np.random.normal(0, noise_factor, len(audio))
    return audio + noise

def random_gain(audio, min_factor=0.1, max_factor=2.0):
    """Aplica ganho aleatório ao áudio"""
    gain = np.random.uniform(min_factor, max_factor)
    return audio * gain 