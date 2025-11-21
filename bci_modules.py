import numpy as np
from scipy.signal import butter, lfilter, welch
from pylsl import StreamInlet, resolve_streams

class EEGStream:
    def __init__(self, stream_type='EEG'):
        print(f"Looking for an {stream_type} stream...")
        streams = resolve_streams(wait_time=1.0)
        # Filter for type manually if needed or check if resolve_streams supports arguments
        # Actually resolve_streams usually takes no args or wait time.
        # Let's check if we can filter. 
        # If resolve_streams returns all streams, we filter by type.
        target_streams = [s for s in streams if s.type() == stream_type]
        if not target_streams:
            raise Exception(f"No stream of type {stream_type} found.")
        self.inlet = StreamInlet(target_streams[0])
        print("Stream found and connected!")
        
    def get_chunk(self, max_samples=256):
        """
        Pull a chunk of data from the LSL stream.
        """
        chunk, timestamps = self.inlet.pull_chunk(max_samples=max_samples)
        return np.array(chunk), np.array(timestamps)

class SignalProcessor:
    def __init__(self, srate=256):
        self.srate = srate
        
    def bandpass_filter(self, data, lowcut=1.0, highcut=50.0, order=5):
        nyq = 0.5 * self.srate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        # Apply filter along time axis (axis=0 if shape is samples x channels)
        # If data is (n_samples, n_channels)
        return lfilter(b, a, data, axis=0)
    
    def extract_features(self, epoch):
        """
        Extracts Alpha Asymmetry features from a single epoch.
        Assumes epoch shape: (n_samples, n_channels)
        Assumes Ch0 = Left (F3), Ch1 = Right (F4)
        """
        features = []
        
        # Calculate PSD for each channel
        # epoch is (samples, channels), we want to iterate over channels
        n_channels = epoch.shape[1]
        
        alpha_powers = []
        
        for ch in range(n_channels):
            freqs, psd = welch(epoch[:, ch], fs=self.srate, nperseg=min(epoch.shape[0], 256))
            
            # Extract Alpha band (8-13 Hz)
            idx_alpha = np.logical_and(freqs >= 8, freqs <= 13)
            alpha_power = np.mean(psd[idx_alpha])
            alpha_powers.append(alpha_power)
            
        # Feature 1: Alpha Asymmetry (Right - Left) / (Right + Left)
        # If Ch0 is Left and Ch1 is Right
        if n_channels >= 2:
            left_alpha = alpha_powers[0]
            right_alpha = alpha_powers[1]
            
            # Avoid division by zero
            denom = right_alpha + left_alpha
            if denom == 0:
                asymmetry = 0
            else:
                asymmetry = (right_alpha - left_alpha) / denom
                
            features.append(asymmetry)
            features.extend(alpha_powers) # Add raw powers too
        else:
            features.extend(alpha_powers)
            
        return np.array(features)
