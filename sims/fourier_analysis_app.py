import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Fourier Analysis Tool", layout="wide")

# Create tabs for different functionality
tab1, tab2, tab3 = st.tabs(["Wave Composition", "Wrapping Visualization", "Fourier Analysis"])

# Function to compute FFT and get positive components
def compute_fft(signal, fs, t):
    N = len(t)
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(N, 1/fs)
    
    # Get positive frequency components
    positive_freq_indices = np.where(frequencies >= 0)
    positive_frequencies = frequencies[positive_freq_indices]
    positive_amplitudes = 2 * np.abs(fft_result)[positive_freq_indices] / N  # Normalize
    
    return positive_frequencies, positive_amplitudes

# Function to calculate wrapping metrics
def calculate_wrapping_metrics(signal, t, wrapping_freqs):
    cog_values = []
    
    for wrap_freq in wrapping_freqs:
        # Calculate phase for this wrapping frequency (in radians)
        phases = 2 * np.pi * wrap_freq * t
        
        # Calculate the center of mass using complex math
        x_component = np.sum(signal * np.cos(phases))
        y_component = np.sum(signal * np.sin(phases))
        
        # Normalize by signal energy
        magnitude = np.sqrt(x_component**2 + y_component**2) / np.sum(np.abs(signal))
        cog_values.append(magnitude)
    
    return cog_values

# Function to create wrapped visualization for a single frequency
def create_wrapping_visualization(signal, t, wrap_freq):
    # Calculate phase for this wrapping frequency
    phases = (wrap_freq * t) % 1.0  # Normalized phase (0-1)
    phases_rad = 2 * np.pi * phases  # Radians (0-2π)
    
    # Calculate coordinates
    x_coords = np.cos(phases_rad) * signal
    y_coords = np.sin(phases_rad) * signal
    
    # Calculate center of mass
    x_cm = np.mean(x_coords)
    y_cm = np.mean(y_coords)
    cm_magnitude = np.sqrt(x_cm**2 + y_cm**2)
    
    # Sort by phase for continuous path
    sort_idx = np.argsort(phases)
    x_path = np.cos(phases_rad[sort_idx]) * signal[sort_idx]
    y_path = np.sin(phases_rad[sort_idx]) * signal[sort_idx]
    
    # Create figure with subplots
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=("Original Signal", "Signal vs. Wrapped Phase", 
                                        "Circular Wrapping Representation", "Wrapping Path"),
                        specs=[[{"type": "scatter"}, {"type": "scatter"}],
                               [{"type": "scatter"}, {"type": "scatter"}]])
    
    # Plot 1: Original signal
    fig.add_trace(go.Scatter(x=t, y=signal, mode='lines', 
                             line=dict(color='#8CD9B3')), row=1, col=1)
    
    # Plot 2: Signal vs. Wrapped Phase
    fig.add_trace(go.Scatter(x=phases, y=signal, mode='markers', 
                             marker=dict(color='#8CD9B3', size=5, opacity=0.5)), row=1, col=2)
    
    # Plot 3: Circular representation
    fig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode='markers',
                             marker=dict(color='#8CD9B3', size=5, opacity=0.5)), row=2, col=1)
    
    # Add reference circle
    theta = np.linspace(0, 2*np.pi, 100)
    radius = np.max(np.abs(signal))
    circle_x = radius * np.cos(theta)
    circle_y = radius * np.sin(theta)
    fig.add_trace(go.Scatter(x=circle_x, y=circle_y, mode='lines',
                             line=dict(color='black', dash='dash', width=1), opacity=0.3), row=2, col=1)
    
    # Add CM point and line
    fig.add_trace(go.Scatter(x=[0, x_cm*2], y=[0, y_cm*2], mode='lines', 
                             line=dict(color='red', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=[x_cm], y=[y_cm], mode='markers',
                             marker=dict(color='red', size=10)), row=2, col=1)
    
    # Plot 4: Wrapping path
    fig.add_trace(go.Scatter(x=x_path, y=y_path, mode='lines',
                             line=dict(color='#8CD9B3')), row=2, col=2)
    fig.add_trace(go.Scatter(x=[x_cm], y=[y_cm], mode='markers',
                             marker=dict(color='red', size=10)), row=2, col=2)
    
    # Update layout
    fig.update_layout(height=800, width=1000, title_text=f"Wrapping at {wrap_freq} Hz")
    
    # Make circular plots square
    fig.update_xaxes(scaleanchor="y", scaleratio=1, row=2, col=1)
    fig.update_xaxes(scaleanchor="y", scaleratio=1, row=2, col=2)
    
    # Update axis labels
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    
    fig.update_xaxes(title_text="Phase (normalized)", row=1, col=2)
    fig.update_yaxes(title_text="Amplitude", row=1, col=2)
    
    fig.update_xaxes(title_text="X", row=2, col=1)
    fig.update_yaxes(title_text="Y", row=2, col=1)
    
    fig.update_xaxes(title_text="X", row=2, col=2)
    fig.update_yaxes(title_text="Y", row=2, col=2)
    
    return fig, cm_magnitude

# ---------- Wave Composition Tab ----------
with tab1:
    st.header("Composite Wave Generator")
    st.write("Create a composite signal by combining multiple sine waves with different frequencies and amplitudes.")
    
    # Sampling parameters
    st.sidebar.header("Sampling Parameters")
    fs = st.sidebar.slider("Sampling Frequency (Hz)", min_value=100, max_value=2000, value=1000, step=100)
    duration = st.sidebar.slider("Duration (seconds)", min_value=1, max_value=10, value=3, step=1)
    
    # Create time array based on sampling parameters
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Component waves configuration
    st.subheader("Wave Components")
    
    num_waves = st.number_input("Number of Component Waves", min_value=1, max_value=10, value=2, step=1)
    
    # Initialize arrays to store wave data
    frequencies = []
    amplitudes = []
    individual_waves = []
    
    # Create input fields for each wave
    cols = st.columns(2)
    for i in range(num_waves):
        with cols[0]:
            freq = st.slider(f"Frequency {i+1} (Hz)", min_value=0.1, max_value=20.0, value=float(i+2), step=0.1)
            frequencies.append(freq)
        
        with cols[1]:
            amp = st.slider(f"Amplitude {i+1}", min_value=0.1, max_value=5.0, value=1.0 if i == 0 else 0.7, step=0.1)
            amplitudes.append(amp)
        
        # Generate this component wave
        wave = amp * np.sin(2 * np.pi * freq * t)
        individual_waves.append(wave)
    
    # Create composite signal
    composite_signal = np.sum(individual_waves, axis=0)
    
    # Visualization of waves
    st.subheader("Wave Visualization")
    
    fig = plt.figure(figsize=(12, 10))
    
    # Plot 1: Composite signal
    plt.subplot(2, 1, 1)
    plt.plot(t, composite_signal, color='#8CD9B3')
    plt.title('Composite Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Plot 2: Individual components
    plt.subplot(2, 1, 2)
    for i, wave in enumerate(individual_waves):
        plt.plot(t, wave, label=f'{frequencies[i]} Hz Component (Amplitude = {amplitudes[i]})')
    
    plt.title('Individual Components')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Calculate FFT for this composite signal
    positive_freqs, positive_amps = compute_fft(composite_signal, fs, t)
    
    # Save data for other tabs to use
    st.session_state['t'] = t
    st.session_state['fs'] = fs
    st.session_state['composite_signal'] = composite_signal
    st.session_state['fft_freqs'] = positive_freqs
    st.session_state['fft_amps'] = positive_amps
    st.session_state['component_freqs'] = frequencies
    st.session_state['component_amps'] = amplitudes

# ---------- Wrapping Visualization Tab ----------
with tab2:
    st.header("Wrapping Visualization")
    st.write("Visualize how the signal wraps around different frequencies and observe the resulting center of gravity.")
    
    if 'composite_signal' not in st.session_state:
        st.warning("Please define your composite signal in the 'Wave Composition' tab first.")
    else:
        # Get data from session state
        t = st.session_state['t']
        composite_signal = st.session_state['composite_signal']
        component_freqs = st.session_state['component_freqs']
        
        # Select wrapping frequency
        wrap_freq_min = 0.5
        wrap_freq_max = max(20.0, max(component_freqs) * 2)
        wrap_freq = st.slider("Wrapping Frequency (Hz)", 
                              min_value=wrap_freq_min, 
                              max_value=wrap_freq_max, 
                              value=component_freqs[0] if component_freqs else 2.0, 
                              step=0.1)
        
        # Calculate and display wrapping
        wrap_fig, cm_magnitude = create_wrapping_visualization(composite_signal, t, wrap_freq)
        st.plotly_chart(wrap_fig, use_container_width=True)
        
        # Display CM Magnitude
        st.metric("Center of Mass Magnitude", f"{cm_magnitude:.4f}")
        
        # Center of Gravity vs Wrapping Frequency
        st.subheader("Center of Gravity vs Wrapping Frequency")
        
        # Create range of wrapping frequencies to analyze
        wrapping_freqs = np.linspace(wrap_freq_min, wrap_freq_max, 500)
        cog_values = calculate_wrapping_metrics(composite_signal, t, wrapping_freqs)
        
        # Plot CoG vs Wrapping Frequency
        cog_fig = plt.figure(figsize=(10, 6))
        plt.plot(wrapping_freqs, cog_values, color='#E57373')
        plt.title('Center of Mass Magnitude vs Wrapping Frequency', fontsize=14)
        plt.xlabel('Frequency (Hz)', fontsize=14)
        plt.ylabel('Magnitude', fontsize=14)
        plt.grid(True)
        
        # Mark known frequencies
        for freq in component_freqs:
            plt.axvline(x=freq, linestyle='--', alpha=0.7, 
                       label=f'{freq} Hz Component')
            
            # Find and mark peak
            idx = np.argmin(np.abs(wrapping_freqs - freq))
            plt.plot(wrapping_freqs[idx], cog_values[idx], 'o', color='red', markersize=8)
        
        plt.legend()
        st.pyplot(cog_fig)

# ---------- Fourier Analysis Tab ----------
with tab3:
    st.header("Fourier Analysis")
    st.write("Perform Fourier Transform analysis on the composite signal to identify its component frequencies.")
    
    if 'composite_signal' not in st.session_state:
        st.warning("Please define your composite signal in the 'Wave Composition' tab first.")
    else:
        # Get data from session state
        t = st.session_state['t']
        fs = st.session_state['fs']
        composite_signal = st.session_state['composite_signal']
        positive_freqs = st.session_state['fft_freqs']
        positive_amps = st.session_state['fft_amps']
        component_freqs = st.session_state['component_freqs']
        component_amps = st.session_state['component_amps']
        
        # Create FFT plot
        fft_fig = plt.figure(figsize=(12, 6))
        plt.stem(positive_freqs, positive_amps, linefmt='r-', markerfmt='ro')
        plt.title('Frequency Spectrum (FFT Analysis)', fontsize=16)
        plt.xlabel('Frequency (Hz)', fontsize=14)
        plt.ylabel('Amplitude', fontsize=14)
        
        # Focus on relevant frequencies
        max_freq_to_show = max(20, max(component_freqs) * 2)
        plt.xlim(0, max_freq_to_show)
        
        # Annotate known component frequencies
        for freq, amp in zip(component_freqs, component_amps):
            idx = np.argmin(np.abs(positive_freqs - freq))
            actual_freq = positive_freqs[idx]
            actual_amp = positive_amps[idx]
            
            plt.annotate(f'{actual_freq:.1f} Hz\nAmp: {actual_amp:.2f}',
                         xy=(actual_freq, actual_amp),
                         xytext=(actual_freq, actual_amp + 0.2),
                         ha='center')
        
        plt.grid(True)
        st.pyplot(fft_fig)
        
        # Comparison of methods
        st.subheader("Frequency Analysis Methods Comparison")
        
        # Create DataFrame to compare FFT peaks with Center of Gravity analysis
        comparison_data = []
        
        # Get CoG values at specific points
        wrapping_freqs = np.linspace(0.5, max(20, max(component_freqs) * 2), 500)
        cog_values = calculate_wrapping_metrics(composite_signal, t, wrapping_freqs)
        
        # Find peaks in FFT
        fft_peaks = []
        for freq in component_freqs:
            idx = np.argmin(np.abs(positive_freqs - freq))
            fft_peaks.append({
                'frequency': positive_freqs[idx],
                'amplitude': positive_amps[idx]
            })
        
        # Find peaks in CoG
        cog_peaks = []
        for freq in component_freqs:
            idx = np.argmin(np.abs(wrapping_freqs - freq))
            cog_peaks.append({
                'frequency': wrapping_freqs[idx],
                'magnitude': cog_values[idx]
            })
        
        # Display the results
        st.subheader("FFT Peak Analysis")
        for i, peak in enumerate(fft_peaks):
            st.write(f"Component {i+1}: {peak['frequency']:.2f} Hz with amplitude {peak['amplitude']:.4f}")
        
        st.subheader("Center of Gravity Analysis")
        for i, peak in enumerate(cog_peaks):
            st.write(f"Component {i+1}: {peak['frequency']:.2f} Hz with CoG magnitude {peak['magnitude']:.4f}")
        
        # Compare with original input
        st.subheader("Original Input Components")
        for i, (freq, amp) in enumerate(zip(component_freqs, component_amps)):
            st.write(f"Component {i+1}: {freq:.2f} Hz with amplitude {amp:.2f}")
        
        # Plot all methods together
        methods_fig = plt.figure(figsize=(12, 8))
        
        # Plot CoG
        plt.plot(wrapping_freqs, cog_values, color='#E57373', label='Center of Gravity')
        
        # Normalize FFT amplitudes for better visualization
        fft_norm = positive_amps / np.max(positive_amps) * np.max(cog_values)
        
        # Create interpolated FFT curve for better visualization
        from scipy.interpolate import interp1d
        
        # Limit to relevant frequency range
        mask = positive_freqs <= max_freq_to_show
        pos_freqs_limited = positive_freqs[mask]
        pos_amps_limited = fft_norm[mask]
        
        # Only interpolate if we have enough points
        if len(pos_freqs_limited) > 5:
            f_interp = interp1d(pos_freqs_limited, pos_amps_limited, kind='cubic',
                               bounds_error=False, fill_value=0)
            freqs_interp = np.linspace(0, max_freq_to_show, 1000)
            amps_interp = f_interp(freqs_interp)
            plt.plot(freqs_interp, amps_interp, color='#64B5F6', label='FFT (normalized)', alpha=0.7)
        else:
            plt.plot(pos_freqs_limited, pos_amps_limited, color='#64B5F6', label='FFT (normalized)', alpha=0.7)
        
        plt.title('Comparison of Frequency Analysis Methods', fontsize=16)
        plt.xlabel('Frequency (Hz)', fontsize=14)
        plt.ylabel('Magnitude', fontsize=14)
        plt.grid(True)
        plt.legend()
        
        # Mark known frequencies
        for freq in component_freqs:
            plt.axvline(x=freq, linestyle='--', alpha=0.5, color='gray')
        
        st.pyplot(methods_fig)