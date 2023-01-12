import os
import requests
import base64
import numpy as np
import pandas as pd
import json
import torch
import wfdb


if __name__ == '__main__':
    # get real data
    # df = pd.read_csv('/home/ai_dev_02/Documents/quoctn_itrvn/data/Data-ECG-Generator/1D/data_fake/AVB3_20s/AVB3_info.csv')
    # event_ids = df['EventID']
    # waveform_dir = '/home/ai_dev_02/Documents/quoctn_itrvn/data/Data-ECG-Generator/1D/data_fake/AVB3_20s/generated_data'
    # waveforms = []
    # for i in range(8):
    #     # waveform = get_waveform_20s(waveform_dir, None, 0, event_ids[i])
    #     waveform, _ = wfdb.rdsamp(os.path.join(waveform_dir, event_ids[i]))
    #     waveforms.append(waveform)
    # waveforms = np.array(waveforms).astype(np.float32)

    # get fake data
    # waveforms = np.random.rand((32, 5000)).astype(np.float32)
    images = []
    for name in os.listdir('./mnist'):
        images.append(np.load('./mnist/' + name))
    images = np.array(images).astype(np.float32)

    url = 'http://0.0.0.0:8080/predictions/mnist_classification'
    X = []
    for row in images[:2]:
        row = row.tobytes()
        X.append(base64.b64encode(row).decode())
    obj = {
        'data': X,
    }
    response = requests.post(url, data=json.dumps(obj), headers={"Content-Type": "application/json"})
    predicts = np.frombuffer(response.content, dtype=np.float32).astype(np.int)
    print(predicts)
