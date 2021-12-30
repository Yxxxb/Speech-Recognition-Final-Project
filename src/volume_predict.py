# -*- coding:utf-8 -*-
"""
Author :Yxxxb & Xubing Ye
Number :1953348
Date   :2021/12/25
File   :volume_predict.py
"""
"""
运行run_DB
输入参数：test_data内语音路径 eg:'./test_data/emotion_test.wav'
返回值：平均分贝，图片路径
eg: (149.77372222869747, 'images/emotion_test_DB.jpg')
"""

import wave
import numpy as np
import matplotlib.pyplot as plt


def load_wav(wave_input_path):
    wf = wave.open(wave_input_path, 'rb')  # 读 wav 文件
    fs = wf.getframerate()
    nframes = wf.getnframes()
    str_data = wf.readframes(nframes)
    wf.close()
    wave_data = np.fromstring(str_data, dtype=np.short)
    return wave_data.astype(np.float64), fs


def SPLCal(x):
    Leng = len(x)
    pa = np.sqrt(np.sum(np.power(x, 2)) / Leng)
    p0 = 2e-5
    spl = 20 * np.log10(pa / p0)
    return spl


def run_DB(wav_file):
    x, fs = load_wav(wav_file)
    Leng = len(x)
    frameTime = 100
    frameLen = fs * frameTime // 1000
    m = np.mod(Leng, frameLen)
    if m >= frameLen / 2:
        x = np.append(x, np.zeros(int(frameLen - m)))
        Leng = len(x)
    else:
        nframe = np.floor(Leng / frameLen)
        x = x[0:int(nframe * frameLen) + 1]
        Leng = len(x)

    N = Leng // frameLen
    spl = np.array([])
    for k in range(N):
        s = x[k * frameLen: (k + 1) * frameLen]
        spl = np.append(spl, SPLCal(s))

    ans = np.array([])
    for k in range(N):
        if spl[k] > 0:
            ans = np.append(ans, spl[k])

    spl_rep = np.repeat(ans, frameLen)

    wav_list = wav_file.split('/')
    pic_name = wav_list[-1][:-4]
    print(pic_name)

    plt.plot(spl_rep)
    plt.title("Decibel(dB)")
    path = 'images/' + pic_name + '_DB.jpg'
    plt.savefig(path)
    plt.show()

    return ans.mean(), path


if __name__ == '__main__':
    print(run_DB('./test_data/emotion_test.wav'))
