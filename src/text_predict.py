# -*- coding:utf-8 -*-
"""
Author :Yxxxb & Xubing Ye
Number :1953348
Date   :2021/12/25
File   :text_predict.py
"""
"""
运行run_text
输入参数：test_data内语音路径 eg:'./test_data/test.wav'
返回值：预测时间（ms）, 结果, 评分
eg: (259, '近几年不但我用书给女儿压岁也劝说朋不要给女儿压岁钱改送压岁书', 99.08357018079514)
"""

import argparse
import functools
import time

from utils_data_text.audio_process import AudioInferProcess
from utils_text.predict import Predictor
from utils_text.audio_vad import crop_audio_vad
from utils_text.utility import add_arguments, print_arguments


def run_text(wav_file):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('wav_path', str, wav_file, "预测音频的路径")
    add_arg('is_long_audio', bool, False, "是否为长语音")
    add_arg('use_gpu', bool, True, "是否使用GPU预测")
    add_arg('enable_mkldnn', bool, False, "是否使用mkldnn加速")
    add_arg('to_an', bool, True, "是否转为阿拉伯数字")
    add_arg('beam_size', int, 300, "集束搜索解码相关参数，搜索的大小，范围:[5, 500]")
    add_arg('alpha', float, 1.2, "集束搜索解码相关参数，LM系数")
    add_arg('beta', float, 0.35, "集束搜索解码相关参数，WC系数")
    add_arg('cutoff_prob', float, 0.99, "集束搜索解码相关参数，剪枝的概率")
    add_arg('cutoff_top_n', int, 40, "集束搜索解码相关参数，剪枝的最大值")
    add_arg('mean_std_path', str, './saved_models/free/mean_std.npz', "数据集的均值和标准值的npy文件路径")
    add_arg('vocab_path', str, './saved_models/free/zh_vocab.txt', "数据集的词汇表文件路径")
    add_arg('model_dir', str, './saved_models/infer-thchs30/', "导出的预测模型文件夹路径")
    add_arg('lang_model_path', str, './lm/zh_giga.no_cna_cmn.prune01244.klm', "集束搜索解码相关参数，语言模型文件路径")
    add_arg('decoding_method', str, 'ctc_greedy', "结果解码方法，有集束搜索(ctc_beam_search)、贪婪策略(ctc_greedy)",
            choices=['ctc_beam_search', 'ctc_greedy'])
    args = parser.parse_args()
    print_arguments(args)

    # 获取数据生成器，处理数据和获取字典需要
    audio_process = AudioInferProcess(vocab_filepath=args.vocab_path, mean_std_filepath=args.mean_std_path)

    predictor = Predictor(model_dir=args.model_dir, audio_process=audio_process, decoding_method=args.decoding_method,
                          alpha=args.alpha, beta=args.beta, lang_model_path=args.lang_model_path,
                          beam_size=args.beam_size,
                          cutoff_prob=args.cutoff_prob, cutoff_top_n=args.cutoff_top_n, use_gpu=args.use_gpu,
                          enable_mkldnn=args.enable_mkldnn)

    if args.is_long_audio:
        return predict_long_audio(args, predictor)
    else:
        # 进入预测阶段
        return predict_audio(args, predictor)


def predict_long_audio(args, predictor):
    start = time.time()
    # 分割长音频
    audios_path = crop_audio_vad(args.wav_path)
    texts = ''
    scores = []
    # 执行识别
    for i, audio_path in enumerate(audios_path):
        score, text = predictor.predict(audio_path=audio_path, to_an=args.to_an)
        texts = texts + '，' + text
        scores.append(score)
        print("第%d个分割音频, 得分: %d, 识别结果: %s" % (i, score, text))
    print("最终结果，消耗时间：%d, 得分: %d, 识别结果: %s" % (round((time.time() - start) * 1000), sum(scores) / len(scores), texts))


def predict_audio(args, predictor):
    start = time.time()
    # 调用predictor的方法进行预测
    score, text = predictor.predict(audio_path=args.wav_path, to_an=args.to_an)
    print("消耗时间：%dms, 识别结果: %s, 得分: %d" % (round((time.time() - start) * 1000), text, score))

    return round((time.time() - start) * 1000), text, score


if __name__ == "__main__":
    run_text('./test_data/test.wav')
