import wfdb
import os
from biosppy.signals import ecg
import numpy as np

# set up the path to the MIT-BIH Arrhythmia Database
mitdb_path = 'D:\heart_rate\mit'

# loop over all the files in the MIT-BIH Arrhythmia Database
for filename in os.listdir(mitdb_path):
    if filename.endswith('.dat'):
        # read the ECG signal from the file
        record = wfdb.rdrecord(os.path.join(mitdb_path, filename[:-4]))
        signal = record.p_signal[:,0]

        # process the ECG signal to detect R-peaks
        rpeaks = ecg.hamilton_segmenter(signal, record.fs)[0]

        # Calculate R-R intervals
        rr_intervals = np.diff(rpeaks) / record.fs

        # Calculate heart rate as the inverse of the average R-R interval
        heart_rate = 60 / np.mean(rr_intervals)

        # print the heart rate for this file
        print(f"Heart rate for {filename}: {heart_rate} bpm")
