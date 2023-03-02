import os
import numpy as np
from datasets import Dataset

import wfdb

class EcgDataset(Dataset):
    def __init__(self, records_list):
        self.records_list = records_list
def load_data(data_dir):
    records = []
    labels = []

    for file in os.listdir(data_dir):
        if file.endswith(".hea"):
            file_path = os.path.join(data_dir, file[:-4])
            record, _ = wfdb.rdsamp(file_path)
            records.append(record[:, 0])

            with open(file_path + ".txt") as f:
                label = int(f.readlines()[0])
                labels.append(label)

    records = np.array(records)
    labels = np.array(labels)

    return records, labels


def preprocess_data(records):
    records = (records - np.mean(records)) / np.std(records)
    records = records[:, np.newaxis, :]
    return records
