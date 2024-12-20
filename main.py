import streamlit as st
import soundfile as sf
import numpy as np
from scipy import signal
import pandas as pd
import io

# Set page config with a smaller initial size
st.set_page_config(page_title="LANDR QC", layout="centered")

# Custom CSS to improve centering and sizing
st.markdown("""
    <style>
        .block-container {
            max-width: 1500px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stDataFrame {
            width: 1500px;
        }
        /* Remove fixed height constraints */
        [data-testid="stDataFrame"] div[style*="overflow: hidden scroll"] {
            height: auto !important;
        }
        [data-testid="stDataFrame"] div[style*="overflow: hidden scroll"] > div {
            height: auto !important;
        }
    </style>
""", unsafe_allow_html=True)

# Center the title
st.markdown("<h1 style='text-align: center;'>LANDR QC</h1>", unsafe_allow_html=True)


def format_duration(seconds):
    """Convert seconds to MM:SS format with milliseconds"""
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    whole_seconds = int(remaining_seconds)
    milliseconds = int((remaining_seconds - whole_seconds) * 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def calculate_true_peak(audio_data, sample_rate):
    """Calculate true peak values using 4x oversampling"""
    oversample_rate = 4

    # Design a low-pass anti-aliasing filter
    nyq = sample_rate * oversample_rate / 2
    cutoff = sample_rate / 2
    b, a = signal.butter(8, cutoff / nyq)

    # Upsample and apply filter for each channel
    channel_peaks = []

    for channel in range(audio_data.shape[1]):
        # Upsample
        upsampled = signal.resample_poly(audio_data[:, channel], oversample_rate, 1)

        # Apply filter
        filtered = signal.filtfilt(b, a, upsampled)

        # Find peak
        peak_value = np.max(np.abs(filtered))
        peak_db = 20 * np.log10(peak_value) if peak_value > 0 else -np.inf

        channel_peaks.append(peak_db)

    return channel_peaks


def calculate_lufs(audio_data, sample_rate):
    """Calculate integrated loudness (LUFS) according to ITU-R BS.1770-4"""
    # K-weighting filter coefficients
    # Pre-filter (high shelf)
    f0 = 1681.974450955533
    G = 3.999843853973347
    Q = 0.7071752369554196

    # High-pass filter
    f1 = 38.13547087602444
    Q1 = 0.5003270373238773

    # Design filters
    wb = 2 * np.pi * f0 / sample_rate
    wc = 2 * np.pi * f1 / sample_rate

    # Pre-filter coefficients
    K = np.tan(wb / 2.0)
    Vh = np.power(10.0, G / 20.0)
    Vb = np.power(Vh, 0.4996667741545416)

    a0 = 1.0 + K / Q + K * K
    b0 = (Vh + Vb * K / Q + K * K) / (1.0 + K / Q + K * K)
    b1 = 2.0 * (K * K - Vh) / (1.0 + K / Q + K * K)
    b2 = (Vh - Vb * K / Q + K * K) / (1.0 + K / Q + K * K)
    a1 = 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K)
    a2 = (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K)

    # High-pass filter coefficients
    K1 = np.tan(wc / 2.0)
    a01 = 1.0 + K1 / Q1 + K1 * K1
    b01 = 1.0 / (1.0 + K1 / Q1 + K1 * K1)
    b11 = -2.0 / (1.0 + K1 / Q1 + K1 * K1)
    b21 = 1.0 / (1.0 + K1 / Q1 + K1 * K1)
    a11 = 2.0 * (K1 * K1 - 1.0) / (1.0 + K1 / Q1 + K1 * K1)
    a21 = (1.0 - K1 / Q1 + K1 * K1) / (1.0 + K1 / Q1 + K1 * K1)

    # Apply filters and calculate power
    powers = []
    for channel in range(audio_data.shape[1]):
        # Apply K-weighting filters
        filtered = signal.lfilter([b0, b1, b2], [1.0, a1, a2], audio_data[:, channel])
        filtered = signal.lfilter([b01, b11, b21], [1.0, a11, a21], filtered)

        # Calculate power in 400ms blocks
        block_size = int(0.4 * sample_rate)
        channel_power = []

        for i in range(0, len(filtered), block_size):
            block = filtered[i:i + block_size]
            if len(block) == block_size:  # Only process full blocks
                power = np.mean(block ** 2)
                channel_power.append(power)

        powers.append(channel_power)

    # Convert to numpy array
    powers = np.array(powers)

    # Calculate gating level
    absolute_gate = -70.0  # LUFS
    relative_gate_offset = -10.0  # LU

    # First gate at -70 LUFS
    combined_power = np.sum(powers, axis=0)
    gate_indices = combined_power > 10 ** (absolute_gate / 10)
    if not np.any(gate_indices):
        return -70.0

    # Calculate relative gate
    gated_powers = combined_power[gate_indices]
    relative_gate = np.mean(gated_powers) * 10 ** (relative_gate_offset / 10)
    gate_indices = combined_power > relative_gate

    # Calculate final loudness
    gated_powers = combined_power[gate_indices]
    integrated_loudness = -0.691 + 10.0 * np.log10(np.mean(gated_powers))

    return integrated_loudness


def analyze_audio(audio_file):
    """Analyze audio file and return its properties"""
    with sf.SoundFile(audio_file) as f:
        # Get file info first
        sample_rate = f.samplerate
        channels = f.channels
        subtype = f.subtype
        frames = f.frames
        duration = frames / sample_rate

        # Then read the audio data
        audio_data = f.read()

        # Convert subtype to bit depth
        bit_depth_map = {
            'PCM_16': 16,
            'PCM_24': 24,
            'PCM_32': 32,
            'FLOAT': 32,
            'DOUBLE': 64
        }
        bit_depth = bit_depth_map.get(subtype, 'Unknown')

        # Reshape audio data for multichannel if needed
        if channels > 1:
            audio_data = audio_data.reshape(-1, channels)
        else:
            audio_data = audio_data.reshape(-1, 1)

        # Calculate true peak
        true_peaks = calculate_true_peak(audio_data, sample_rate)

        # Calculate integrated loudness
        integrated_loudness = calculate_lufs(audio_data, sample_rate)

        return {
            'Sample Rate': f'{sample_rate} Hz',
            'Bit Depth': f'{bit_depth}-bit',
            'Channels': channels,
            'Duration': format_duration(duration),
            'True Peak L': f'{true_peaks[0]:.2f} dB',
            'True Peak R': f'{true_peaks[1]:.2f} dB' if channels > 1 else 'N/A',
            'Integrated LUFS': f'{integrated_loudness:.1f} LUFS'
        }


# Center the file uploader
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_files = st.file_uploader("Upload Audio Files", type=['wav'], accept_multiple_files=True)

if uploaded_files:
    # Create a list to store results
    results = []

    # Process each file
    for uploaded_file in uploaded_files:
        try:
            # Create a temporary file-like object
            audio_bytes = io.BytesIO(uploaded_file.read())

            # Analyze the audio
            analysis = analyze_audio(audio_bytes)

            # Add filename to results
            analysis['File Name'] = uploaded_file.name
            results.append(analysis)

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")

    if results:
        # Convert results to DataFrame
        df = pd.DataFrame(results)

        # Reorder columns to match original layout
        columns = ['File Name', 'Sample Rate', 'Bit Depth', 'Channels', 'Duration', 'True Peak L', 'True Peak R',
                   'Integrated LUFS']
        df = df[columns]


        # Style the DataFrame
        def highlight_row(row):
            should_highlight = False
            try:
                # Check True Peak L and R
                if 'dB' in str(row['True Peak L']):
                    peak_val = float(row['True Peak L'].replace(' dB', ''))
                    if peak_val >= 0:
                        should_highlight = True

                if row['True Peak R'] != 'N/A' and 'dB' in str(row['True Peak R']):
                    peak_val = float(row['True Peak R'].replace(' dB', ''))
                    if peak_val >= 0:
                        should_highlight = True

                # Check Channels
                if row['Channels'] != 2:
                    should_highlight = True

                # Check Sample Rate
                if row['Sample Rate'] != '44100 Hz':
                    should_highlight = True

                # Check Bit Depth
                if row['Bit Depth'] != '24-bit':
                    should_highlight = True

            except:
                pass

            return ['background-color: #fee2e2' if should_highlight else '' for _ in row]


        # Center the table
        col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
        with col2:
            st.dataframe(
                df.style
                .apply(highlight_row, axis=1)
                .format(precision=2)
                .set_properties(**{
                    'text-align': 'center',
                    'min-height': '48px',
                    'white-space': 'nowrap'
                })
                .set_table_styles([
                    {'selector': 'th', 'props': [('text-align', 'center'), ('min-height', '48px')]},
                    {'selector': 'td', 'props': [('text-align', 'center')]}
                ]),
                hide_index=True,
                use_container_width=True
            )
