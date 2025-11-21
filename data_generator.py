import numpy as np
import time
from pylsl import StreamInfo, StreamOutlet

class SyntheticDataGenerator:
    def __init__(self, n_channels=4, srate=256):
        self.n_channels = n_channels
        self.srate = srate
        
    def generate_epoch(self, duration_sec=1.0, state='normal'):
        """
        Generates a chunk of synthetic EEG data.
        state: 'normal' or 'depressed'
        
        'depressed' state will have higher Alpha (8-13Hz) power in the Right Hemisphere (F4) 
        relative to Left (F3), or lower Left Alpha activity, simulating Alpha Asymmetry.
        Common theory: Left frontal hypoactivation (more alpha) -> Depression.
        Wait, actually: Higher Alpha = Less Activity.
        Depression is often associated with relatively greater Right Frontal Activity (less alpha) 
        or Lower Left Frontal Activity (more alpha).
        Let's model:
        - Normal: Balanced Alpha between Ch0 (Left) and Ch1 (Right).
        - Depressed: Higher Alpha in Ch0 (Left) compared to Ch1 (Right) -> Left Hypoactivation.
        """
        n_samples = int(self.srate * duration_sec)
        time_points = np.linspace(0, duration_sec, n_samples)
        
        data = np.zeros((self.n_channels, n_samples))
        
        # Base noise (1/f)
        noise = np.random.randn(self.n_channels, n_samples) * 5
        
        # Alpha waves (10Hz)
        alpha_freq = 10.0
        
        # Amplitudes
        if state == 'normal':
            left_amp = 10.0
            right_amp = 10.0
        else: # depressed - Left Hypoactivation (More Alpha on Left)
            left_amp = 20.0 
            right_amp = 5.0
            
        # Channel 0 (Left), Channel 1 (Right), others random
        data[0] = left_amp * np.sin(2 * np.pi * alpha_freq * time_points) + noise[0]
        data[1] = right_amp * np.sin(2 * np.pi * alpha_freq * time_points) + noise[1]
        
        # Add some other frequencies to other channels/mix
        for i in range(2, self.n_channels):
            data[i] = 5.0 * np.sin(2 * np.pi * 15 * time_points) + noise[i] # Beta
            
        return data.T # (n_samples, n_channels)

    def generate_dataset(self, n_samples_per_class=500, epoch_duration=1.0):
        """
        Generates a labeled dataset for training.
        Returns: X (n_samples, n_timepoints, n_channels), y (n_samples)
        """
        X = []
        y = []
        
        print(f"Generating {n_samples_per_class} samples for 'Normal' state...")
        for _ in range(n_samples_per_class):
            epoch = self.generate_epoch(duration_sec=epoch_duration, state='normal')
            X.append(epoch)
            y.append(0) # 0 for Normal
            
        print(f"Generating {n_samples_per_class} samples for 'Depressed' state...")
        for _ in range(n_samples_per_class):
            epoch = self.generate_epoch(duration_sec=epoch_duration, state='depressed')
            X.append(epoch)
            y.append(1) # 1 for Depressed
            
        return np.array(X), np.array(y)

class SyntheticLSLStream:
    def __init__(self, name='BioSemi', type='EEG', n_channels=4, srate=256):
        self.info = StreamInfo(name, type, n_channels, srate, 'float32', 'myuid12345')
        self.outlet = StreamOutlet(self.info)
        self.generator = SyntheticDataGenerator(n_channels, srate)
        self.running = False
        
    def start(self):
        self.running = True
        print("Synthetic LSL Stream Started...")
        while self.running:
            # Randomly switch states every few seconds to simulate real changes? 
            # For now just stream 'normal' mostly
            sample = self.generator.generate_epoch(duration_sec=1.0/256.0, state='normal') 
            # Stream sample by sample
            self.outlet.push_sample(sample[0])
            time.sleep(1.0/256.0)

if __name__ == "__main__":
    # Test generator
    gen = SyntheticDataGenerator()
    X, y = gen.generate_dataset(n_samples_per_class=10)
    print(f"Generated dataset shape: {X.shape}")
