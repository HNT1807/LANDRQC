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
            max-width: 1000px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stDataFrame {
            width: 900px;
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

        return {
            'Sample Rate': f'{sample_rate} Hz',
            'Bit Depth': f'{bit_depth}-bit',
            'Channels': channels,
            'Duration': format_duration(duration),
            'True Peak L': f'{true_peaks[0]:.2f} dB',
            'True Peak R': f'{true_peaks[1]:.2f} dB' if channels > 1 else 'N/A'
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
        columns = ['File Name', 'Sample Rate', 'Bit Depth', 'Channels', 'Duration', 'True Peak L', 'True Peak R']
        df = df[columns]

        # Style the DataFrame
        def highlight_peaks(val):
            try:
                if 'dB' in str(val):
                    peak_val = float(val.replace(' dB', ''))
                    if peak_val >= 0:
                        return 'background-color: #fee2e2'  # Light red background
            except:
                pass
            return ''

        # Center the table
        col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
        with col2:
            st.dataframe(
                df.style
                .apply(lambda x: [highlight_peaks(val) for val in x], axis=1)
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
