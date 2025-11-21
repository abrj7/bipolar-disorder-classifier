import argparse
import time
import numpy as np
import threading
from bci_modules import EEGStream, SignalProcessor
from classifier import MoodClassifier
from data_generator import SyntheticDataGenerator, SyntheticLSLStream

def train_mode():
    print("=== TRAINING MODE ===")
    gen = SyntheticDataGenerator()
    
    # Generate training data
    # 1000 samples as requested
    X_raw, y = gen.generate_dataset(n_samples_per_class=500)
    
    # Process features
    processor = SignalProcessor()
    X_features = []
    
    print("Extracting features from training data...")
    for epoch in X_raw:
        # Apply filter
        filtered = processor.bandpass_filter(epoch)
        # Extract features
        feats = processor.extract_features(filtered)
        X_features.append(feats)
        
    X_features = np.array(X_features)
    
    # Train Classifier
    clf = MoodClassifier()
    clf.train(X_features, y)
    clf.save_model()
    print("Training complete. Model saved.")

def live_mode(use_synthetic=False):
    print("=== LIVE BCI MODE ===")
    
    # If using synthetic stream, start it in a separate thread
    if use_synthetic:
        print("Starting background synthetic stream...")
        synth_stream = SyntheticLSLStream()
        t = threading.Thread(target=synth_stream.start)
        t.daemon = True
        t.start()
        # Give it a moment to start
        time.sleep(2)
        
    # Connect to stream
    try:
        stream = EEGStream()
    except Exception as e:
        print(f"Error connecting to stream: {e}")
        print("Make sure an LSL stream is running (e.g., OpenViBE or the synthetic stream).")
        return

    processor = SignalProcessor()
    clf = MoodClassifier()
    
    # Try to load model
    try:
        clf.load_model()
    except:
        print("No trained model found! Please run with --train first.")
        return

    print("Starting Real-Time Processing... (Press Ctrl+C to stop)")
    buffer = []
    
    try:
        while True:
            # Get chunk
            chunk, timestamps = stream.get_chunk()
            if len(chunk) > 0:
                # chunk is (n_samples, n_channels)
                # We need to buffer until we have enough for an epoch (e.g. 1 second = 256 samples)
                if len(buffer) == 0:
                    buffer = chunk
                else:
                    buffer = np.vstack((buffer, chunk))
                
                # If we have enough data (1 second window)
                if len(buffer) >= 256:
                    epoch = buffer[:256]
                    # Slide window (overlap) - e.g., keep last 128 samples
                    buffer = buffer[128:] 
                    
                    # Process
                    filtered = processor.bandpass_filter(epoch)
                    feats = processor.extract_features(filtered)
                    
                    # Predict
                    prediction = clf.predict(feats)
                    label = "DEPRESSED" if prediction[0] == 1 else "NORMAL"
                    
                    # Simple visualization
                    print(f"State: {label} | Alpha Asymmetry: {feats[0]:.4f}")
                    
            time.sleep(0.01) # Prevent busy waiting
            
    except KeyboardInterrupt:
        print("\nStopping...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bipolar Disorder BCI Detector")
    parser.add_argument('--train', action='store_true', help="Run in training mode with synthetic data")
    parser.add_argument('--live', action='store_true', help="Run in live BCI mode")
    parser.add_argument('--synthetic', action='store_true', help="Use synthetic LSL stream for live testing")
    
    args = parser.parse_args()
    
    if args.train:
        train_mode()
    elif args.live:
        live_mode(use_synthetic=args.synthetic)
    else:
        parser.print_help()
