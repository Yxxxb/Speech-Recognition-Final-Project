from head import create_app
from flask import render_template, flash, redirect, url_for, request
from webform import inputform
import os


from voiceprint_predict import run_voiceprint
from emotion_predict import run_emotion

def run_voiceprint(wave_path1, wave_path2):
    print("run_voiceprint")


# def run_emotion(wave_path):
#     print("run_emotion", 'wave_path is' + str(wave_path))
#     emotion = 'fear'
#     gender = 'male'
#     path_speech = 'images/speech.jpg'
#     path_emotion = 'images/emotion.jpg'
#     return emotion, gender, path_speech, path_emotion


# fear male images/speech.jpg images/emotion.jpg


def return_img_stream(path):
    import base64
    img_stream = ''
    with open(path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    print(img_stream)
    return img_stream


app = create_app()


@app.route('/', methods=['POST', 'GET'])
def index():
    form = inputform()
    if form.validate_on_submit():
        src = "test_data"
        road = src + '/' + form.file.data.filename

        if form.file.data.filename.rsplit('.')[1] != "wav":
            flash("请上传wav文件")
            form.file.data = None
            return render_template("index.html",
                                   form=form)
        if os.path.exists(road):
            p = form.file.data.filename
            oldhead = p.rsplit('.')[0]
            back = p.rsplit('.')[1]
            newhead = '1'
            for i in range(1, 99):
                if i == 98:
                    flash("同名文件过多！！")
                    form.file.data = None
                    return render_template("index.html",
                                           form=form)
                if not os.path.exists(src + '/' + str(oldhead) + "(" + str(i) + ")" + "." + back):
                    newhead = oldhead + "(" + str(i) + ")"
                    break
            road = src + '/' + str(newhead + '.' + back)
            form.file.data.save(road)
            flash("文件上传成功")
            wave_path = 'test_data/' + str(newhead + '.' + back)
            return redirect(url_for('result', wave_path=wave_path))
        else:
            road = src + '/' + str(form.file.data.filename)
            form.file.data.save(road)
            flash("文件上传成功")
            wave_path = 'test_data/' + str(form.file.data.filename)
            return redirect(url_for('result', wave_path=wave_path))

    return render_template("index.html",
                           form=form)


@app.route("/result", methods=['GET'])
def result():
    wave_path = request.args.get("wave_path")
    emotion, gender, path_speech, path_emotion = run_emotion(wave_path)
    img_speech_stream = return_img_stream(path_speech)
    img_emotion_stream = return_img_stream(path_emotion)
    return render_template("result.html",
                           emotion=emotion,
                           gender=gender,
                           img_speech_stream=img_speech_stream,
                           img_emotion_stream=img_emotion_stream)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
