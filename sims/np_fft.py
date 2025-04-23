import numpy as np
import matplotlib.pyplot as plt

# Define parameters
fs = 1000  # Sampling frequency (Hz)
duration = 3  # Duration of signal (seconds)

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
plt.figure(figsize=(12, 10))

# Plot 1: Composite signal
plt.subplot(3, 1, 1)
plt.plot(t, composite_signal, color='#8CD9B3')  # Light green to match your images
plt.title('Composite Signal (2Hz + 3Hz waves)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot 2: Individual components
plt.subplot(3, 1, 2)
plt.plot(t, wave_2hz, label=f'2 Hz Component (Amplitude = {amplitude_2hz})', color='#D683CE')  # Purple
plt.plot(t, wave_3hz, label=f'3 Hz Component (Amplitude = {amplitude_3hz})', color='#E5EE4F')  # Yellow
plt.title('Individual Components')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Plot 3: Frequency spectrum
plt.subplot(3, 1, 3)
plt.stem(positive_frequencies, positive_amplitudes, linefmt='r-', markerfmt='ro')
plt.title('Frequency Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, 10)  # Focus on relevant frequencies
plt.grid(True)

# Annotate peaks
idx_2hz = np.argmin(np.abs(positive_frequencies - 2))
idx_3hz = np.argmin(np.abs(positive_frequencies - 3))
plt.annotate(f'{positive_frequencies[idx_2hz]:.1f} Hz\nAmp: {positive_amplitudes[idx_2hz]:.2f}',
             xy=(2, positive_amplitudes[idx_2hz]),
             xytext=(2, positive_amplitudes[idx_2hz] + 0.2),
             ha='center')
plt.annotate(f'{positive_frequencies[idx_3hz]:.1f} Hz\nAmp: {positive_amplitudes[idx_3hz]:.2f}',
             xy=(3, positive_amplitudes[idx_3hz]),
             xytext=(3, positive_amplitudes[idx_3hz] + 0.2),
             ha='center')

plt.tight_layout()
plt.show()

# After your existing code, add this new plot for Center of Gravity vs Wrapping Frequency
# Create range of wrapping frequencies to analyze
wrapping_freqs = np.linspace(0.5, 5, 500)  # Hz, more resolution around our frequencies of interest
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

# Create a new figure for CoG vs Wrapping Frequency
plt.figure(figsize=(10, 6))
plt.plot(wrapping_freqs, cog_values, color='#E57373')  # Red line like in your images
plt.title('x-coordinate of center of mass', fontsize=14)
plt.xlabel('Frequency (Hz)', fontsize=14)
plt.ylabel('Magnitude', fontsize=14)
plt.grid(True)

# Mark known frequencies
plt.axvline(x=2, color='#D683CE', linestyle='--', alpha=0.7, label='2 Hz Component')  # Purple
plt.axvline(x=3, color='#E5EE4F', linestyle='--', alpha=0.7, label='3 Hz Component')  # Yellow
plt.legend()

# Add visual emphasis to the peaks
peak_indices = []
for freq in [2, 3]:
    idx = np.argmin(np.abs(wrapping_freqs - freq))
    peak_indices.append(idx)
    plt.plot(wrapping_freqs[idx], cog_values[idx], 'o', color='red', markersize=8)

plt.tight_layout()
plt.show()

# Add visualizations of the wrapping process
def plot_wrapping(wrap_freq, title):
    """Plot the wrapping process for a specific frequency"""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Wrapping at {wrap_freq} Hz', fontsize=16)
    
    # Plot 1: Original signal with markers
    axs[0, 0].plot(t, composite_signal, color='#8CD9B3')
    axs[0, 0].set_title('Original Signal')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].grid(True)
    
    # Calculate phase for this wrapping frequency
    phases = (wrap_freq * t) % 1.0  # Normalized phase (0-1)
    phases_rad = 2 * np.pi * phases  # Radians (0-2π)
    
    # Plot 2: Signal in wrapped phase space
    axs[0, 1].scatter(phases, composite_signal, s=5, color='#8CD9B3', alpha=0.5)
    axs[0, 1].set_title('Signal vs. Wrapped Phase')
    axs[0, 1].set_xlabel('Phase (normalized)')
    axs[0, 1].set_ylabel('Amplitude')
    axs[0, 1].grid(True)
    
    # Plot 3: Circular representation
    x_coords = np.cos(phases_rad) * composite_signal
    y_coords = np.sin(phases_rad) * composite_signal
    
    # Calculate center of mass
    x_cm = np.mean(x_coords)
    y_cm = np.mean(y_coords)
    
    # Plot the circle points
    axs[1, 0].scatter(x_coords, y_coords, s=5, color='#8CD9B3', alpha=0.5)
    axs[1, 0].plot([0, x_cm*2], [0, y_cm*2], 'r-', linewidth=2)  # Line to CM
    axs[1, 0].plot(x_cm, y_cm, 'ro', markersize=8)  # Mark CM
    
    # Add a reference circle
    theta = np.linspace(0, 2*np.pi, 100)
    radius = np.max(np.abs(composite_signal))
    circle_x = radius * np.cos(theta)
    circle_y = radius * np.sin(theta)
    axs[1, 0].plot(circle_x, circle_y, 'k--', alpha=0.3)
    
    # Equal aspect ratio for circular plot
    axs[1, 0].set_aspect('equal')
    axs[1, 0].grid(True)
    axs[1, 0].set_title('Circular Wrapping Representation')
    
    # Plot 4: Wrapping path (like in your images)
    # Sort by phase for continuous path
    sort_idx = np.argsort(phases)
    x_path = np.cos(phases_rad[sort_idx]) * composite_signal[sort_idx]
    y_path = np.sin(phases_rad[sort_idx]) * composite_signal[sort_idx]
    
    axs[1, 1].plot(x_path, y_path, color='#8CD9B3', linewidth=1)
    axs[1, 1].plot(x_cm, y_cm, 'ro', markersize=8)  # Mark CM
    axs[1, 1].set_aspect('equal')
    axs[1, 1].grid(True)
    axs[1, 1].set_title('Wrapping Path')
    
    plt.tight_layout()
    plt.show()
    
    return np.sqrt(x_cm**2 + y_cm**2)  # Return magnitude of CM

# Show wrapping for frequencies of interest
cm_1 = plot_wrapping(1.0, "1.0 Hz")
# cm_2 = plot_wrapping(2.0, "2.0 Hz (one of our component frequencies)")
# cm_3 = plot_wrapping(3.0, "3.0 Hz (one of our component frequencies)")
# cm_4 = plot_wrapping(4.0, "4.0 Hz")

print(f"Center of mass magnitudes:")
print(f"1.00 Hz: {cm_1:.4f}")
print(f"2.00 Hz: {cm_2:.4f}")
print(f"3.00 Hz: {cm_3:.4f}")
print(f"4.00 Hz: {cm_4:.4f}")