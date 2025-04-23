import numpy as np

def perform_fft(signal, fs):
    """
    Perform Fast Fourier Transform on a signal
    
    Parameters:
    signal : input signal array
    fs : sampling frequency in Hz
    
    Returns:
    freqs : frequency array
    fft_mag : magnitude of FFT
    """
    n = len(signal)
    # Compute FFT
    fft_result = np.fft.fft(signal)
    # Compute magnitude (absolute value)
    fft_mag = np.abs(fft_result)
    # Normalize
    fft_mag = fft_mag / n
    # Multiply by 2 (except for DC and Nyquist)
    fft_mag[1:n//2] = 2 * fft_mag[1:n//2]
    
    # Compute frequency array
    freqs = np.fft.fftfreq(n, 1/fs)
    
    return freqs[:n//2], fft_mag[:n//2]  # Return only positive frequencies