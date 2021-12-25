# -*- coding:utf-8 -*-
"""
Author :Yxxxb & Xubing Ye
Number :1953348
Date   :2021/12/21
File   :voiceprint_predict.py
"""
"""
运行run_voiceprint
输入参数：test_data内语音路径(两个) eg:'test_data/audio/yxb1.wav', 'test_data/audio/yxb2.wav'
返回值：是否为同一个人(True / False) 相似度 eg:True, 0.771351
"""

import argparse
import functools
import numpy as np
import tensorflow as tf
from utils_voice_print.reader import load_audio
from utils_voice_print.utility import add_arguments, print_arguments


# 预测音频
def infer(audio_path, model, input_shape):
    data = load_audio(audio_path, mode='test', spec_len=input_shape[1])
    data = data[np.newaxis, :]
    feature = model.predict(data)
    return feature


def run_voiceprint(wav_path1, wav_path2):
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    sess = tf.compat.v1.Session(config=config)

    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('audio_path1', str, wav_path1, '预测第一个音频')
    add_arg('audio_path2', str, wav_path2, '预测第二个音频')
    add_arg('input_shape', str, '(257, 257, 1)', '数据输入的形状')
    add_arg('threshold', float, 0.7, '判断是否为同一个人的阈值')
    add_arg('model_path', str, 'saved_models/Voiceprint_Model.h5', '预测模型的路径')
    args = parser.parse_args()

    print_arguments(args)

    # 加载模型
    model = tf.keras.models.load_model(args.model_path)
    model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('batch_normalization').output)

    # 数据输入的形状
    input_shape = eval(args.input_shape)

    # 打印模型
    model.build(input_shape=input_shape)
    model.summary()

    # 要预测的两个人的音频文件
    feature1 = infer(args.audio_path1, model, input_shape)[0]
    feature2 = infer(args.audio_path2, model, input_shape)[0]
    # 对角余弦值
    dist = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    if dist > args.threshold:
        print("%s 和 %s 为同一个人，相似度为：%f" % (args.audio_path1, args.audio_path2, dist))
        return True, dist
    else:
        print("%s 和 %s 不是同一个人，相似度为：%f" % (args.audio_path1, args.audio_path2, dist))
        return False, dist


if __name__ == '__main__':
    run_voiceprint('test_data/audio/a_1.wav', 'test_data/audio/a_2.wav')
