# -*- coding:utf-8 -*-
"""
Author :Yxxxb & Xubing Ye
Number :1953348
Date   :2021/12/20
File   :emotion_predict.py
"""
"""
运行run_emotion
输入参数：test_data内语音路径 eg:'test_data/zlp1.wav'
返回值：情感类型 性别 声音图像路径 情感图像路径
eg: fear male images/emotion_test_speech.jpg images/emotion_test_emotion.jpg
"""

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pandas as pd
import librosa
import os


def predict(wav_file, pic_name):
    json_file = open('saved_models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("saved_models/Emotion_Model.h5")
    print("Loaded model from disk")

    data, sampling_rate = librosa.load(wav_file)
    plt.figure(figsize=(15, 5))
    librosa.display.waveplot(data, sr=sampling_rate)
    imgPath = 'images/' + pic_name + '_speech.jpg'
    plt.savefig(imgPath)

    X, sample_rate = librosa.load(wav_file, res_type='kaiser_fast', duration=2.5, sr=22050 * 2, offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    featurelive = mfccs
    livedf2 = featurelive

    livedf2 = pd.DataFrame(data=livedf2)
    livedf2 = livedf2.stack().to_frame().T
    twodim = np.expand_dims(livedf2, axis=2)
    livepreds = loaded_model.predict(twodim,
                                     batch_size=32,
                                     verbose=1)
    return livepreds, imgPath


def Radar(wav_path, pic_name, data_prob):
    class_labels = ['angry', 'calm', 'fear', 'happy', 'sad']
    angles = np.linspace(0, 2 * np.pi, len(class_labels), endpoint=False)
    fig = plt.figure()

    # polar参数
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, data_prob, 'bo-', linewidth=2)
    ax.fill(angles, data_prob, facecolor='r', alpha=0.25)
    ax.set_thetagrids(angles * 180 / np.pi, class_labels)
    ax.set_title("Emotion Recognition", va='bottom')

    # 设置雷达图的数据最大值
    ax.set_rlim(0, 1)

    ax.grid(True)
    imgPath = 'images/' + pic_name + '_emotion.jpg'
    plt.savefig(imgPath)
    plt.show()

    return imgPath


def Judge_render(livepreds):
    isMale = 0
    isFemale = 0
    livepreds = livepreds.reshape(10, )
    for i in range(5):
        isFemale = isFemale + livepreds[i]
    for j in range(5, 10):
        isMale = isMale + livepreds[j]

    class_labels = ['angry', 'calm', 'fear', 'happy', 'sad']
    male_list = livepreds[5:10].tolist()
    female_list = livepreds[0:5].tolist()

    if isMale > isFemale:
        return "male", livepreds[5:10], class_labels[male_list.index(max(male_list))]
    else:
        return "female", livepreds[0:5], class_labels[female_list.index(max(female_list))]


def set_GPU():
    # GPU setting
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    config = ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


def run_emotion(wav_path):
    wav_list = wav_path.split('/')
    pic_name = wav_list[-1][:-4]

    set_GPU()
    livepreds, speechPath = predict(wav_path, pic_name)
    render, data_prop, emotion = Judge_render(livepreds)
    emotionPath = Radar(wav_path, pic_name, data_prop)
    print(emotion, render, speechPath, emotionPath)
    return emotion, render, speechPath, emotionPath


if __name__ == '__main__':
    run_emotion('test_data/emotion_test.wav')
