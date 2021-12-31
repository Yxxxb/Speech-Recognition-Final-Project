# 项目文档

文档大概每个人的部分分成这几个模块：
1.阐述一下解决的问题与难点 
2.所利用到的数据集、数据处理方法、模型、损失函数等 可以稍微详细一点点 神经网络部分可以从简 主要说清楚用这些方法来干什么 为什么要用 作用是什么 为什么用别的不行 
3.工作流 可以贴一些重要代码 从数据处理到训练、评估流程走一遍 都用了哪些部分的代码文件（这是要求里面有的）可以贴一下你们自己部分的运行结果
4.结果分析（如果有的话）简要分析一下模型准确率等等参数 
5.优点缺点 比如模型对什么数据识别度高 什么类型的数据识别度低 语音转文字可以说一下不同数据计算训练出来的模型的优缺点等等 自己想一想总结一下 然后加一些TODO 就是我们的模型哪些功能还需要进一步优化



#### 1. 阐述一下解决的问题与难点 

本模块为声纹对比系统，即对输入的两端语音进行声纹解析，判断二者是否为同一个人的语音，并且给出两语音的相似程度。我们最初的思路是构建预测声纹的深度学习模型，但是在测试的时候我们发现模型无法预测训练集外的声纹，因此又对模型进行了简单修改，不需要预测声纹所属即可计算出语音声纹相似度。后文讲对这两个模型进行解释与分析。

此部分的主要难点在于构建预测模型以及导出特征矩阵并对其进行相似度计算。

This module is a voiceprint comparison system, that is to analyze the voiceprint of the two ends of the input voice, judge whether they are the same person's voice, and give the degree of similarity between the two voices. Our initial idea was to build a deep learning model to predict the voiceprint. However, when testing, we found that the model could not predict the voiceprint outside the training set. Therefore, we simply modified the model to calculate the voice print similarity without predicting the voice print's ownership. These two models are explained and analyzed in the following part.  

The main difficulty of voiceprint comparison system lies in the construction of the prediction model and the derivation of the feature matrix and the calculation of its similarity.  

#### 2. 所利用到的数据集、数据处理方法、模型、损失函数等 可以稍微详细一点点 神经网络部分可以从简 主要说清楚用这些方法来干什么 为什么要用 作用是什么 为什么用别的不行 

本部分简要阐述所用到的数据集、数据处理、模型选取以及损失函数等主要模型构建方法，将对模型与各个方法选取过程中的效果进行权衡分析与比较。

------

- Dataset-Zhvoice

  本模块所用到的数据集为Zhvoice中文语料库，此数据集一共有3242个人的语音数据，有1130000+条语音数据。相较于其他数据集，该数据集对任务进行了标签处理，数据量较大，人物数量能够满足模型的训练需求。

  The data set used in this module is Zhvoice Chinese corpus, which has a total of 3242 individual voice data, including 1,130,000 + voice data. Compared with other data sets, this data set labels tasks and has a large amount of data, and the number of characters can meet the training requirements of the model.

- Data Processing-STFT

  Short-Time Fourier Transform, STFT, which is defined as:

  ![[公式]](https://www.zhihu.com/equation?tex=X%28n%2C%5Comega%29%3D%5Csum_%7Bm%3D-%5Cinfty%7D%5E%5Cinfty+x%28m%29w%28n-m%29e%5E%7B-j%5Comega+m%7D)

  其中 ![[公式]](https://www.zhihu.com/equation?tex=x%28m%29) 为输入信号， ![[公式]](https://www.zhihu.com/equation?tex=w%28m%29) 是窗函数，它在时间上反转并且有n个样本的偏移量。 ![[公式]](https://www.zhihu.com/equation?tex=X%28n%2C%5Comega%29) 是时间 ![[公式]](https://www.zhihu.com/equation?tex=n) 和频率 ![[公式]](https://www.zhihu.com/equation?tex=%5Comega) 的二维函数，它将信号的时域和频域联系起来，我们可以据此对信号进行时频分析，比如 ![[公式]](https://www.zhihu.com/equation?tex=S%28n%2C%5Comega%29%3D%7CX%28n%2C%5Comega%29%7C%5E2) 就是语音信号所谓的语谱图(Spectrogram)。

  画出上节中两路扫频信号叠加后的信号的语谱图，如下图

  ![image-20211230121327968](声纹对比.assets/image-20211230121327968.png)

  可见该信号是由一个0~250Hz二次递增的扫频信号和一个250~0Hz二次递减的扫频信号的叠加。通过STFT，我们可以很容易地得出非平稳信号的时变特性。

  计算语谱 ![[公式]](https://www.zhihu.com/equation?tex=S%28n%2C%5Comega%29) 时采用不同窗长度，可以得到两种语谱图，即窄带和宽带语谱图。长时窗（至少两个基音周期)常被用于计算窄带语谱图，短窗则用于计算宽带语谱图。窄带语谱图具有较高的频率分辨率和较低的时间分辨率，良好的频率分辨率可以让语音的每个谐波分量更容易被辨别，在语谱图上显示为水平条纹。相反宽带语谱图具有较高的时间分辨率和较低的频率分辨率，低频率分辨率只能得到谱包络，良好的时间分辨率适合用于分析和检验英语语音的发音。

  如下图所示，分别为一段语音的帧长为128和512的语谱图。

  ![image-20211230121356539](声纹对比.assets/image-20211230121356539.png)

  可见，对于帧长固定的短时傅里叶变换，在全局范围内的时间分辨率和频率分辨率是固定的。

  这样我们对音频数据集进行处理，将音频转化为257*257的短时傅里叶变换（STFT）幅度谱。

- 模型构建-残差神经网络

  项目选取Resnet-50残差神经网络进行预测，详细有关网络的架构在此不进行分析，下图为网络的架构以及输入输出的shape。

  ```python
  Model: "Resnet-50"
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #   
  =================================================================
  resnet50v2_input (InputLayer [(None, 257, 257, 1)]     0         
  _________________________________________________________________
  resnet50v2 (Functional)      (None, 2048)              23558528  
  _________________________________________________________________
  batch_normalization (BatchNo (None, 2048)              8192      
  =================================================================
  Total params: 23,566,720
  Trainable params: 23,517,184
  Non-trainable params: 49,536
  ```

  每训练一轮结束之后，执行一次模型评估，计算模型的准确率，以观察模型的收敛情况。同样的，每一轮训练结束保存一次模型，分别保存了可以恢复训练的模型参数，也可以作为预训练模型参数。还保存预测模型，用于之后预测。

- 损失函数-ArcFace

  ArcFace是伦敦帝国理工学院在2018.01发表，在Sphere Face基础上改进了对特征向量归一化和加性角度间隔，提高了类间可分性同时加强类内紧度和类间差异，具有性能高，易于编程实现，复杂性低，训练效率高的优点。对特征向量和权重归一化，对θ加上角度间隔m，角度间隔比余弦间隔在对角度的影响更加直接。

  As the embedding features are distributed around each feature Centre on the hypersphere, we add an additive angular margin penalty m between xi and Wyi to simultaneously enhance the intra-class compactness and inter-class discrepancy. Since the proposed additive angular margin penalty is equal to the geodesic distance margin penalty in the normalized hypersphere, we name our method as ArcFace.

  ![image-20211230121810175](声纹对比.assets/image-20211230121810175.png)

- 重构模型-对角余弦值

  在问题简述与难点部分我们发现了如下问题：由于受训练数据集的限制，我们无法对自己的声音和更加多元的声音进行对比，而且没法给出两音频之间的相似关系。因此我们对模型进行了重构，不对输入语音进行预测，而是去对比模型过程中输出的特征值。

  ![image-20211230122411598](声纹对比.assets/image-20211230122411598.png)

  即在残差神经网络中输出的维度为(2048, 1)的特征矩阵，我们只需要对两输入语音特征值求解对角余弦值得到相似度，然后设置阈值判断两音频是否来自已同一个人。
  $$
  dist=\frac{np.dot(feature1, feature2)}{np.linalg.norm(feature1) * np.linalg.norm(feature2)}
  $$



#### 3. 工作流 可以贴一些重要代码 从数据处理到训练、评估流程走一遍 都用了哪些部分的代码文件（这是要求里面有的）可以贴一下你们自己部分的运行结果

此部分将结合代码中主要函数，简述模型的架构以及工作流，每个模型将从数据构建、数据处理到模型训练几个步骤进行分析与解释。

------

- 创建数据

  我们将利用数据集创建一个数据列表，数据列表的格式为`<语音文件路径\t语音分类标签>`，创建这个列表主要是方便之后的读取，也是方便读取使用其他的语音数据集，语音分类标签是指说话人的唯一ID，不同的语音数据集，可以通过编写对应的生成数据列表的函数，把这些数据集都写在同一个数据列表中。

  在create_data.py中进行数据标签处理，由于mp3格式的音频读取速度较慢，因此要把全部的mp3格式的音频转换为wav格式，在创建数据列表之后，可能有些数据的是错误的，所以我们要检查一下，将错误的数据删除。执行下面程序完成数据准备。形成了如下数据格式与标签。

  ```
  Speech-Recognition-Final-Project/5_895/5_895_20170614203758.wav	3238
  Speech-Recognition-Final-Project/5_895/5_895_20170614214007.wav	3238
  Speech-Recognition-Final-Project/5_941/5_941_20170613151344.wav	3239
  Speech-Recognition-Final-Project/5_941/5_941_20170614221329.wav	3239
  Speech-Recognition-Final-Project/5_941/5_941_20170616153308.wav	3239
  Speech-Recognition-Final-Project/5_968/5_968_20170614162657.wav	3240
  ```

- 数据处理

  有了上面创建的数据列表和均值标准值，就可以用于训练读取。主要是把语音数据转换短时傅里叶变换的幅度谱，并在此步骤进行数据增强，如随机翻转拼接，随机裁剪。经过处理，最终得到一个257*257的短时傅里叶变换的幅度谱。

  ```python
  # STFT
  wav, sr_ret = librosa.load(audio_path, sr=sr)
  linear = librosa.stft(extended_wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
  linear_T = linear.T
  mag, _ = librosa.magphase(linear_T)
  mag_T = mag.T
  ```

- 模型训练

  运行train.py开始训练模型，使用tensorflow的resnet50模型，数据输入层设置为[None, 1, 257, 257]，即短时傅里叶变换的幅度谱的shape。为了更好的观测模型训练效果并节省训练时间，每训练一轮结束之后，执行一次模型评估，计算模型的准确率，以观察模型的收敛情况。同样的，每一轮训练结束保存一次模型，分别保存了可以恢复训练的模型参数，也可以作为预训练模型参数。

  ```python
  def create_model(input_shape):
      # 获取模型
      model = tf.keras.Sequential()
      model.add(ResNet50V2(input_shape=input_shape, include_top=False, weights=None, pooling='max'))
      model.add(BatchNormalization())
      model.add(Dense(units=512, kernel_regularizer=tf.keras.regularizers.l2(5e-4), name='feature_output'))
      model.add(ArcNet(num_classes=args.num_classes))
      return model
  ```

  | params                | num           |
  | --------------------- | ------------- |
  | epoch_num             | 50            |
  | batch_size            | 16            |
  | input_shape           | (257, 257, 1) |
  | output_size           | (3242, 1)     |
  | ResNet50V2            | 1             |
  | feature_used_shape    | (2048, 1)     |
  | initial_learning_rate | 1e-3          |

  

- 声纹对比

  运行对比函数voiceprint_predict.py，利用残差神经网络输出的音频特征值，求解他们的对角余弦值，得到的结果作为他们相识度。

  ```python
  # 声纹对比
      feature1 = infer(args.audio_path1, model, input_shape)[0]
      feature2 = infer(args.audio_path2, model, input_shape)[0]
      # 对角余弦值
      cos = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
      if dist > 0.7:
          print("为同一个人，相似度为：%f" % (cos))
          return True, dist
      else:
          print("不是同一个人，相似度为：%f" % (cos))
          return False, dist
  ```



#### 4.结果分析（如果有的话）简要分析一下模型准确率等等参数 

我们对所构建的模型进行结果分析，计算模型的准确率、回归率、f1-score、ROC与AUC等参数，不同类别的模型由于输出形式不同所测试的评估值可能不尽相同。此外，我们会在此模块利用自己的测试音频进行测试并展示测试样例。

------

在上文中提到模型训练在每一轮训练结束之后对会被执行，并且训练结束之后会保存预测模型，我们用预测模型来预测测试集中的音频特征。由于此模块功能为模型之间的对比，因此要使用音频特征进行两两对比，阈值从0到1,步长为0.01进行控制，找到最佳的阈值并计算准确率。

```python
-----------  Configuration Arguments -----------
input_shape: (1, 257, 257)
model_path: models/voice_model.h5
------------------------------------------------
提取特征...
100%|█████████████████████████████████████████████████████| 5332/5332 [00:59<00:00, 57.28it/s]
对比特征...
100%|█████████████████████████████████████████████████████| 5332/5332 [02:01<00:00, 41.68it/s]
100%|█████████████████████████████████████████████████████| 100/100 [00:02<00:00, 31.70it/s]
当阈值为0.990000, 准确率最大，准确率为：0.999693
```

可见模型在测试集上的准确率表现非常好。由于声纹对比无法像多分类预测一样提供标准化的结果信息，因此不对回归率等其他参数进行计算。



#### 5.优点缺点 比如模型对什么数据识别度高 什么类型的数据识别度低 语音转文字可以说一下不同数据计算训练出来的模型的优缺点等等 自己想一想总结一下 然后加一些TODO 就是我们的模型哪些功能还需要进一步优化

上文对我们所构建的三部分主要的模型进行了较为详细的阐述，并对模型训练结果进行了简明扼要的分析，此部分将对项目针对三个问题所构建的三部分模型进行总结，阐述模型优缺点以及项目的一些todos。

------

项目在声纹对比模型对模型架构进行了很好的设计，利用训练得到的残差神经网络模型生成的特征矩阵而非预测结果，这样很好地解决了无法预测测试集外语音的问题，实现了最初的设计目的。本项目依然有一些声纹识别应用所共有的缺点，比如同一个人的声音具有易变性，易受身体状况、年龄、情绪等的影响；不同的麦克风和信道对识别性能有影响；环境噪音对识别有干扰；如混合说话人的情形下人的声纹特征不易提取等。但是实质上，我们在中文语音识别中使用的SpenAugment很好的解决了数据声音变形与噪音问题，因此项目将在未来针对以上问题进行模型之间的方法公用，已解决共性问题。

















