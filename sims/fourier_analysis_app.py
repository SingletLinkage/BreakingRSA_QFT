import streamlit as st
import numpy as np
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

# Function to display info tooltips
def info_tooltip(title, content):
    with st.expander(f"ℹ️ {title} Info"):
        st.markdown(content)

# ---------- Wave Composition Tab ----------
with tab1:
    st.header("Composite Wave Generator")
    st.write("Create a composite signal by combining multiple sine waves with different frequencies and amplitudes.")
    
    info_tooltip("Wave Composition", """
    This tab allows you to create complex waveforms by combining sine waves of different frequencies and amplitudes.
    
    **How to use:**
    1. Adjust the sampling parameters in the sidebar
    2. Set the number of component waves
    3. Define the frequency and amplitude for each component
    4. View the resulting composite signal and its components
    
    The composite signal is passed to other tabs for further analysis.
    """)
    
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
    
    info_tooltip("Wave Visualization", """
    This visualization shows:
    
    **Top Plot**: The composite signal resulting from the sum of all component waves.
    
    **Bottom Plot**: Each individual component wave shown separately.
    
    Observe how different frequencies and amplitudes combine to create complex waveforms.
    """)
    
    # Create plotly figure with subplots
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("Composite Signal", "Individual Components"),
                        vertical_spacing=0.15)
    
    # Plot 1: Composite signal
    fig.add_trace(
        go.Scatter(x=t, y=composite_signal, 
                  mode='lines', 
                  line=dict(color='#8CD9B3', width=2),
                  name="Composite Signal"),
        row=1, col=1
    )
    
    # Plot 2: Individual components
    for i, wave in enumerate(individual_waves):
        fig.add_trace(
            go.Scatter(x=t, y=wave, 
                      mode='lines',
                      name=f'{frequencies[i]} Hz Component (Amplitude = {amplitudes[i]})'),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(height=700, 
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                     hovermode="closest")
    
    # Update axis labels
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
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
    
    info_tooltip("Wrapping Visualization", """
    This tab demonstrates the concept of frequency wrapping, which is the basis for Fourier analysis.
    
    **What is wrapping?**
    - The signal is "wrapped" around a circle at different frequencies
    - When wrapping at a frequency present in the signal, the points will cluster on one side
    - This creates a center of mass (red dot) that moves away from the origin
    
    **The plots show:**
    1. Original signal in time domain
    2. Signal plotted against wrapped phase
    3. Circular representation of the wrapped signal
    4. The path created by connecting wrapped points in sequence
    
    The center of mass magnitude peaks when the wrapping frequency matches a component frequency in the signal.
    """)
    
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
        
        info_tooltip("Center of Gravity Analysis", """
        This plot shows how the Center of Mass (CoM) magnitude changes as we sweep through different wrapping frequencies.
        
        **Key insights:**
        - Peaks in this plot correspond to frequencies present in the original signal
        - The height of each peak relates to the amplitude of that frequency component
        - Vertical dashed lines mark the known component frequencies
        - This provides an alternative way to identify frequency components compared to FFT
        """)
        
        # Create range of wrapping frequencies to analyze
        wrapping_freqs = np.linspace(wrap_freq_min, wrap_freq_max, 500)
        cog_values = calculate_wrapping_metrics(composite_signal, t, wrapping_freqs)
        
        # Create plotly figure for CoG vs Wrapping Frequency
        cog_fig = go.Figure()
        
        # Add main line
        cog_fig.add_trace(
            go.Scatter(x=wrapping_freqs, y=cog_values, 
                      mode='lines', 
                      line=dict(color='#E57373', width=2),
                      name="Center of Mass Magnitude")
        )
        
        # Mark known frequencies with vertical lines and points
        for freq in component_freqs:
            # Add vertical line
            cog_fig.add_vline(x=freq, line_width=1.5, line_dash="dash", 
                            line_color="gray", opacity=0.7)
            
            # Find and mark peak
            idx = np.argmin(np.abs(wrapping_freqs - freq))
            cog_fig.add_trace(
                go.Scatter(x=[wrapping_freqs[idx]], y=[cog_values[idx]], 
                          mode='markers',
                          marker=dict(color='red', size=8),
                          name=f"{freq} Hz Component")
            )
        
        # Update layout
        cog_fig.update_layout(
            title='Center of Mass Magnitude vs Wrapping Frequency',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Magnitude',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="closest"
        )
        
        st.plotly_chart(cog_fig, use_container_width=True)

# ---------- Fourier Analysis Tab ----------
with tab3:
    st.header("Fourier Analysis")
    st.write("Perform Fourier Transform analysis on the composite signal to identify its component frequencies.")
    
    info_tooltip("Fourier Analysis", """
    This tab shows the results of Fast Fourier Transform (FFT) analysis on the composite signal.
    
    **What is FFT?**
    The FFT decomposes a time-domain signal into its frequency components, revealing:
    - Which frequencies are present in the signal
    - The amplitude (strength) of each frequency component
    
    **In the plots:**
    - The stem plot shows the frequency spectrum from the FFT analysis
    - The comparison plot shows both FFT and Center of Gravity methods side by side
    - Vertical lines mark the known component frequencies
    
    The FFT and Center of Gravity methods should both identify the same frequency components.
    """)
    
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
        
        # Create FFT plot using Plotly
        fft_fig = go.Figure()
        
        # Add stem plot
        for i, (freq, amp) in enumerate(zip(positive_freqs, positive_amps)):
            fft_fig.add_trace(
                go.Scatter(x=[freq, freq], y=[0, amp], 
                          mode='lines',
                          line=dict(color='red', width=1),
                          showlegend=False)
            )
            fft_fig.add_trace(
                go.Scatter(x=[freq], y=[amp], 
                          mode='markers',
                          marker=dict(color='red', size=8),
                          showlegend=(i == 0),
                          name="FFT Magnitude")
            )
        
        # Focus on relevant frequencies
        max_freq_to_show = max(20, max(component_freqs) * 2)
        fft_fig.update_xaxes(range=[0, max_freq_to_show])
        
        # Annotate known component frequencies
        for freq, amp in zip(component_freqs, component_amps):
            idx = np.argmin(np.abs(positive_freqs - freq))
            actual_freq = positive_freqs[idx]
            actual_amp = positive_amps[idx]
            
            fft_fig.add_annotation(
                x=actual_freq,
                y=actual_amp + 0.2,
                text=f"{actual_freq:.1f} Hz<br>Amp: {actual_amp:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="black",
                arrowsize=1,
                arrowwidth=1
            )
        
        # Update layout
        fft_fig.update_layout(
            title='Frequency Spectrum (FFT Analysis)',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Amplitude',
            height=500,
            hovermode="closest"
        )
        
        st.plotly_chart(fft_fig, use_container_width=True)
        
        # Comparison of methods
        st.subheader("Frequency Analysis Methods Comparison")
        
        info_tooltip("Methods Comparison", """
        This comparison shows how different frequency analysis methods identify components in the signal.
        
        **FFT Analysis:**
        - Mathematically decomposes the signal into frequency components
        - Shows precise amplitude information
        - Standard approach for spectral analysis
        
        **Center of Gravity Analysis:**
        - Based on how the signal "wraps" at different frequencies
        - Peaks correspond to frequencies present in the signal
        - Provides an intuitive geometric interpretation
        
        Both methods should identify the same frequency components in the signal, though with different scaling.
        """)
        
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
        
        # Plot all methods together using Plotly
        methods_fig = go.Figure()
        
        # Plot CoG
        methods_fig.add_trace(
            go.Scatter(x=wrapping_freqs, y=cog_values, 
                      mode='lines', 
                      line=dict(color='#E57373', width=2),
                      name='Center of Gravity')
        )
        
        # Normalize FFT amplitudes for better visualization
        fft_norm = positive_amps / np.max(positive_amps) * np.max(cog_values)
        
        # Use Plotly's native interpolation by setting line shape
        # Limit to relevant frequency range
        mask = positive_freqs <= max_freq_to_show
        pos_freqs_limited = positive_freqs[mask]
        pos_amps_limited = fft_norm[mask]
        
        # Add FFT data with interpolation
        methods_fig.add_trace(
            go.Scatter(x=pos_freqs_limited, y=pos_amps_limited, 
                      mode='lines', 
                      line=dict(color='#64B5F6', width=2, shape='spline'),
                      name='FFT (normalized)',
                      opacity=0.7)
        )
        
        # Mark known frequencies
        for freq in component_freqs:
            methods_fig.add_vline(x=freq, line_width=1.5, line_dash="dash", 
                                line_color="gray", opacity=0.5)
        
        # Update layout
        methods_fig.update_layout(
            title='Comparison of Frequency Analysis Methods',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Magnitude',
            height=600,
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(methods_fig, use_container_width=True)