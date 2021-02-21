"""import matplotlib.pyplot as plt"""
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import numpy as np
from scipy.io.wavfile import read
# (fs,x) = read('records/male/Yair.wav')
rate, data = wav.read('records/female/not_gonna_get_off.wav')
# print(x)
# print(x.size)
# print(fs)
fft_out = fft(data)
print(fft_out)
combined = fft(data).ravel()
print(combined)
print(combined.size)
print(sum(combined))
meanfunfreeq = sum(combined)/combined.size
print("meanfunfreeq: ", float(meanfunfreeq / 1000))
"""a = sum(meanfunfreeq)/2
print(a)
"""
def voice(meanfun):

  if meanfun<0.14:
    return("male")
  else:
    return ("female")
print(voice(meanfunfreeq))
"""
plt.plot(data, np.abs(fft_out))
plt.show()"""
