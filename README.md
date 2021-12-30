# Speech analysis and contrast System

Course Project for Speech Recognition in 2021 Fall | TJU



#### 1. Abstract

This project for Tongji university 2021 fall semester speech recognition course project, project for the theme of speech analysis system, can the content of the speech for speech for input, speech emotion, voiceprint gender, basic information, such as the volume voice voice print characters matching analysis and comparison, and presents a simple interactive interface and the result analysis.

#### 2. Repo architecture

SRC is the source code of the project interaction interface and model invocation, which directly loads the trained.H5 model. The train folder is the code for the training model. You need to import the dataset library by yourself and place it in the specified location. The data sets required by different models are explained in the project document. The above two documents are divided into three main modules: Chinese speech recognition, emotional gender analysis and voice print recognition. Doc is a brief model method record, detailed documentation is available in the final version.

#### 3. Requirement

If you just want to implement interactive prediction, you just need to install the following dependency libraries.

> python == 3.8
>
> tensorflow ~= 2.0.0
>
> librosa == 0.8.0
>
> flask ~= 1.1.2
>
> pyaudio ~= 0.2.11
>
> numpy ~= 1.19.2
>
> keras ~= 2.2.4
>
> pandas ~= 0.24.2
>
> matplotlib ~= 2.2.2
>
> scikit-learn ~= 0.21.1

#### 4. Run

After the requirement is met, you can run the main function in SRC to launch the interface. You can select a WAV audio file to test. The test process takes about 10 seconds. The audio files you upload are saved in test_Data, and the generated data and charts are saved in images, which you can preview on the front end. As the save model is loaded, you need to restart the project for a second prediction.

#### 5. Preview

![image-20211230112513952](README.assets\image-20211230112513952.png)

![image-20211230112749689](README.assets\image-20211230112749689.png)

![image-20211230112817535](README.assets\image-20211230112817535.png)

![image-20211230112837159](README.assets\image-20211230112837159.png)





