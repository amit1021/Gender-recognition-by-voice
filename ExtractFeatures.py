import numpy as np
import librosa
import csv
import glob
from scipy.io import wavfile as wav
from scipy.fftpack import fft

# Extract the 10 features {mean, sd, mediam, Q25, Q75, IQR, skew, kutr, centroid}
def spectral_properties(y: np.ndarray, fs: int, gender: str, filepath: str) -> dict:
    spec = np.abs(np.fft.rfft(y, axis=0))
    freq = np.fft.rfftfreq(len(y) ,1/fs)
    spec = np.abs(spec)
    amp = spec / spec.sum()
    mean = (freq * amp).sum()
    sd = np.sqrt(np.sum(amp * ((freq - mean) ** 2)))
    amp_cumsum = np.cumsum(amp)
    median = freq[len(amp_cumsum[amp_cumsum <= 0.5]) + 1]
    mode = freq[amp.argmax()]
    Q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1]
    Q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1]
    IQR = Q75 - Q25
    z = amp - amp.mean()
    w = amp.std()
    skew = ((z ** 3).sum() / (len(spec) - 1)) / w ** 3
    kurt = ((z ** 4).sum() / (len(spec) - 1)) / w ** 4
    centroid = np.sum(spec*freq) / np.sum(spec)


    rate, data = wav.read(filepath)
    fft_out = fft(data)
    combined = fft(data).ravel()
    meanfun = float(sum(combined)/combined.size)

    result_d = {
        'meanfreq': mean,
        'sd': sd,
        'median': median,
        'Q25': Q25,
        'Q75': Q75,
        'IQR': IQR,
        'skew': skew,
        'kurt': kurt,
        # 'sp.ent': 0.895135270228839,
        # 'sfm': 0.40827905201623305,
        'mode': mode,
        'centroid':centroid,
        'meanfun': meanfun/1000,
        # 'minfun' : 0.03679882384700927,
        # 'maxfun': 0.25880081062844357,
        # 'meandom': 0.8291581697223647,
        # 'mindom': 0.05264066857299429,
        # 'maxdom': 5.0470089964508285,
        # 'dfrange':4.994368327901493,
        # 'modindx':0.17375737330307212,
        'label': gender
    }
    return result_d


def predict_folder():
    list_features = []
    # male records
    for wav_file in glob.iglob('records/male/*'):
        y, sr = librosa.load(wav_file)
        result_d = spectral_properties(y, 1 , "male", wav_file)
        list_features.append(result_d)
    # female records
    for wav_file in glob.iglob('records/female/*'):
        y, sr = librosa.load(wav_file)
        result_d = spectral_properties(y, 1 , "female", wav_file)
        list_features.append(result_d)

    with open('voiceTest.csv', 'w', newline='') as csvfile:
        myWriter = csv.writer(csvfile)
        myWriter.writerow(result_d)
        for i in list_features:
            myWriter.writerow(i.values())

def predict_one(file_name, gender):
    y, sr = librosa.load(file_name)
    result_d = spectral_properties(y, 1, gender, file_name)
    with open('voiceTest.csv', 'w', newline='') as csvfile:
        myWriter = csv.writer(csvfile)
        myWriter.writerow(result_d)
        myWriter.writerow(result_d.values())

