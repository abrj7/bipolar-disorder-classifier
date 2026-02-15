# bipolar disorder classifier

an EEG-driven Brain-Computer Interface (BCI) pipeline designed to detect neural biomarkers associated with the depressive phase of Bipolar Disorder. This project uses Python, OpenViBE/LSL for streaming, and Machine Learning to analyze brainwaves in real-time.

## features

- **Real-time EEG Streaming**: Uses Lab Streaming Layer (LSL) to acquire data from BCI hardware (or synthetic sources).
- **Signal Processing**: Implements bandpass filtering (1-50Hz) and Welch's method for Power Spectral Density (PSD) estimation.
- **Biomarker Detection**: Extracts **Alpha Asymmetry** features (Frontal Alpha Asymmetry), a well-researched biomarker for depression.
- **Machine Learning**: Uses a Random Forest Classifier to distinguish between "Normal" and "Depressive" states.
- **Synthetic Data Generator**: Includes a module to generate synthetic EEG data for training and testing without hardware.

## installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## usage

### 1. Train the Model
Generate synthetic data and train the Random Forest classifier:
```bash
python main.py --train
```
This will create a `mood_classifier.pkl` file.

### 2. Live BCI Mode
To run with a real BCI (streaming via LSL):
```bash
python main.py --live
```
Ensure your BCI hardware (e.g., OpenBCI, Muse) is streaming to LSL with the type `EEG`.

### 3. Test with Synthetic Stream
To simulate a live BCI stream for testing:
```bash
python main.py --live --synthetic
```

## technical details

- **Sampling Rate**: 256 Hz
- **Channels**: 4 (Default: F3, F4, etc.)
- **Feature Extraction**: 
    - Alpha Band: 8-13 Hz
    - Asymmetry Index: $(R - L) / (R + L)$
