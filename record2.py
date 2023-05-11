import scipy
import h5py
import numpy as np
from biosppy.signals import ecg

# set up the path to the data files
data_path = 'data'

# Define the Pan-Tompkins QRS detection algorithm
def pan_tompkins(signal, fs):
    # Define filter parameters
    f1 = 5 / fs
    f2 = 15 / fs
    b, a = scipy.signal.butter(1, [f1 * 2, f2 * 2], btype='bandpass')

    # Apply bandpass filter to the signal
    signal_filtered = scipy.signal.filtfilt(b, a, signal)

    # Differentiate the signal
    signal_diff = np.diff(signal_filtered)

    # Square the differentiated signal
    signal_sq = signal_diff ** 2

    # Apply moving window integration
    window_size = int(0.1 * fs)
    signal_int = np.convolve(signal_sq, np.ones(window_size) / window_size, mode='same')

    # Find QRS complexes
    qrs_peaks = []
    threshold = 0.5 * np.max(signal_int)
    searchback = int(0.2 * fs)
    for i in range(searchback, len(signal_int)):
        if signal_int[i] > threshold and signal_int[i] > signal_int[i-1] and signal_int[i] > signal_int[i+1]:
            qrs_peaks.append(i)
            threshold = 0.5 * np.max(signal_int[i-searchback:i])

    # Calculate R-R intervals
    r_peaks = []
    rr_intervals = []
    for qrs_peak in qrs_peaks:
        searchback = int(0.2 * fs)
        searchforward = int(0.4 * fs)
        searchrange = signal_filtered[qrs_peak-searchback:qrs_peak+searchforward]
        r_peak = np.argmax(searchrange) + qrs_peak - searchback
        r_peaks.append(r_peak)
        rr_intervals.append((r_peak - r_peaks[-2]) / fs)

    # Calculate heart rate as the inverse of the average R-R interval
    heart_rate = 60 / np.mean(rr_intervals)

    return qrs_peaks, heart_rate

# loop over all the files in the data directory
for filename in ['Part_1.mat', 'Part_2.mat', 'Part_3.mat', 'Part_4.mat']:
    try:
        # read the ECG signal from the file
        with h5py.File(f"{data_path}/{filename}", "r") as f:
            signal = np.array(f['sig']).reshape(-1)

        # process the ECG signal to detect R-peaks
        rpeaks = ecg.hamilton_segmenter(signal, f.attrs['fs'])[0]

        # Calculate R-R intervals
        rr_intervals = np.diff(rpeaks) / f.attrs['fs']

        # Calculate heart rate as the inverse of the average R-R interval
        heart_rate = 60 / np.mean(rr_intervals)

        # print the heart rate for this file
        print(f"Heart rate for {filename}: {heart_rate} bpm")
    except Exception as e:
        print(f"Error processing {filename}: {e}")
