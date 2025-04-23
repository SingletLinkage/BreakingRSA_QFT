import numpy as np
import matplotlib.pyplot as plt

# Define parameters
fs = 1000  # Sampling frequency (Hz)
duration = 2  # Duration of signal (seconds)

# Create time array
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Create components
amplitude_2hz = 1.0
amplitude_3hz = 0.7
wave_2hz = amplitude_2hz * np.sin(2 * np.pi * 2 * t)  # 2 Hz sine wave
wave_3hz = amplitude_3hz * np.sin(2 * np.pi * 3 * t)  # 3 Hz sine wave

# Create composite signal
composite_signal = wave_2hz + wave_3hz

# Compute FFT
N = len(t)
fft_result = np.fft.fft(composite_signal)
frequencies = np.fft.fftfreq(N, 1/fs)

# Get positive frequency components (for plotting)
positive_freq_indices = np.where(frequencies >= 0)
positive_frequencies = frequencies[positive_freq_indices]
positive_amplitudes = 2 * np.abs(fft_result)[positive_freq_indices] / N  # Normalize

# Create figure
plt.figure(figsize=(16, 10))

# Plot 1: Composite signal
plt.subplot(2, 2, 1)
plt.plot(t, composite_signal, color='#5D4DB3') 
plt.title('Composite Signal (2Hz + 3Hz waves)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot 2 & 3: Individual wave components
plt.subplot(2, 2, 2) 
plt.plot(t, wave_2hz, label=f'2 Hz Component (Amplitude = {amplitude_2hz})', color='#D683CE')  # Purple
plt.plot(t, wave_3hz, label=f'3 Hz Component (Amplitude = {amplitude_3hz})', color='#0F5D26')  # Dark green
plt.title('Individual Wave Components')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()


wrapping_freqs = np.linspace(0.5, 10, 1000)  # Hz, more resolution around our frequencies of interest
cog_values = []

# Fixed wrapping calculation using complex phasors
for wrap_freq in wrapping_freqs:
    # Calculate phase for this wrapping frequency (in radians)
    phases = 2 * np.pi * wrap_freq * t
    
    # Calculate the center of mass using complex math
    # This properly handles the circular wrapping
    x_component = np.sum(composite_signal * np.cos(phases))
    y_component = np.sum(composite_signal * np.sin(phases))
    
    # Normalize by signal energy
    magnitude = np.sqrt(x_component**2 + y_component**2) / np.sum(np.abs(composite_signal))
    cog_values.append(magnitude)

# Plot 4: Frequency spectrum
plt.subplot(2, 3, 4)
plt.plot(wrapping_freqs, cog_values, color='#E57373')
plt.title('x-coordinate of center of mass', fontsize=14)
plt.xlabel('Frequency (Hz)', fontsize=14)
plt.ylabel('Magnitude', fontsize=14)
plt.grid(True)

# Mark known frequencies
plt.axvline(x=2, color='#9E4C98', linestyle='--', alpha=0.8, label='2 Hz Component')  
plt.axvline(x=3, color='#0C3B17', linestyle='--', alpha=0.8, label='3 Hz Component')
plt.legend()

# Add visual emphasis to the peaks
peak_indices = []
for freq in [2, 3]:
    idx = np.argmin(np.abs(wrapping_freqs - freq))
    peak_indices.append(idx)
    plt.plot(wrapping_freqs[idx], cog_values[idx], 'o', color='#8CD9B3', markersize=8)


phase_rad = lambda wrap_freq: 2 * np.pi * wrap_freq * t  # Phase in radians

# Plot 5: Circular representation
wrap_freq = 2.0  
for i, wrap_freq in enumerate([2.0, 3.0]):
    plt.subplot(2, 3, i+5)
    phases = phase_rad(wrap_freq)
    x_coords = np.cos(phases) * composite_signal
    y_coords = np.sin(phases) * composite_signal
    x_cm = np.mean(x_coords)
    y_cm = np.mean(y_coords)

    plt.scatter(x_coords, y_coords, s=5, color='#8CD9B3', alpha=0.5)
    plt.plot([0, x_cm*2], [0, y_cm*2], 'r-', linewidth=2)  # Line to CM
    plt.plot(x_cm, y_cm, 'ro', markersize=8)  # Mark CMplt.title('Circular Wrapping Representation')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Circular Wrapping at {wrap_freq} Hz')
    plt.grid(True)

    theta = np.linspace(0, 2*np.pi, 100)
    radius = np.max(np.abs(composite_signal))
    circle_x = radius * np.cos(theta)
    circle_y = radius * np.sin(theta)
    plt.plot(circle_x, circle_y, 'k--', alpha=0.3)


plt.savefig('./media/images/complete_fourier.png', dpi=300, bbox_inches='tight')
plt.show()