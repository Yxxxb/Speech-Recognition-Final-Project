from head import create_app
from flask import render_template, flash, redirect, url_for, request
from webform import inputform
import os
import time
import urllib.parse

wave_path1=None
wave_path2=None

from voiceprint_predict import run_voiceprint
from emotion_predict import run_emotion
from text_predict import run_text
from volume_predict import run_DB
#
# def run_text(wave_path):
#     print("run_text")
#     time = '259'
#     text = '近几年不但我用书给女儿压岁也劝说朋不要给女儿压岁钱改送压岁书'
#     score = 99.08357018079514
#     return time, text, score
#
#
# def run_voiceprint(wave_path1, wave_path2):
#     print("run_voiceprint")
#     is_one = True
#     similarity = 0.9
#     time.sleep(2)
#     return is_one, similarity
#
#
# def run_emotion(wave_path):
#     print("run_emotion", 'wave_path is ' + str(wave_path))
#     emotion = 'fear'
#     gender = 'male'
#     path_speech = 'images/yxb1(7)_speech.jpg'
#     path_emotion = 'images/yxb1(7)_emotion.jpg'
#     return emotion, gender, path_speech, path_emotion
#
#
# def run_DB(wave_path):
#     DB = 100
#     path_DB = 'images/yxb1(7)_DB.jpg'
#
#     return DB, path_DB


def return_img_stream(path):
    import base64
    img_stream = ''
    with open(path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream


def get_file_name(src, filename):
    p = filename
    old_head = p.rsplit('.')[0]
    back = p.rsplit('.')[1]
    new_head = '1'
    road = src + '/' + filename
    if os.path.exists(road):
        for i in range(1, 99):
            if i == 98:
                return 'full'
            if not os.path.exists(src + '/' + str(old_head) + "(" + str(i) + ")" + "." + back):
                new_head = old_head + "(" + str(i) + ")"
                break
    else:
        new_head = old_head
    new_name = new_head + '.' + back
    return new_name


app = create_app()


@app.route('/', methods=['POST', 'GET'])
def index():
    form = inputform()
    if form.validate_on_submit():
        src = "test_data"

        if form.file1.data.filename.rsplit('.')[1] != "wav" or form.file2.data.filename.rsplit('.')[1] != "wav":
            flash("请上传wav文件")
            form.file1.data = None
            form.file2.data = None
            return render_template("index.html",
                                   form=form)
        new_name1 = get_file_name(src, form.file1.data.filename)
        if new_name1 == 'full':
            flash("同名文件过多！！")
            form.file1.data = None
            form.file2.data = None
            return render_template("index.html",
                                   form=form)
        road1 = src + '/' + new_name1
        form.file1.data.save(road1)
        wave_path1 = 'test_data/' + new_name1

        new_name2 = get_file_name(src, form.file2.data.filename)
        if new_name2 == 'full':
            flash("同名文件过多！！")
            form.file1.data = None
            form.file2.data = None
            return render_template("index.html",
                                   form=form)
        road2 = src + '/' + new_name2
        form.file2.data.save(road2)
        wave_path2 = 'test_data/' + new_name2
        return redirect(url_for('loading',
                                wave_path1=wave_path1,
                                wave_path2=wave_path2))

    return render_template("index.html",
                           form=form)


@app.route('/loading',methods=['GET'])
def loading():
    global wave_path1,wave_path2
    wave_path1=request.args.get("wave_path1")
    wave_path2=request.args.get("wave_path2")

    some_data = "Here's some example data"
    some_data = urllib.parse.quote(some_data)
    return render_template('loading.html', my_data = some_data)

@app.route("/result", methods=['GET'])
def result():
    global wave_path1,wave_path2
    data = "No data was passed"
    if request.args.to_dict(flat=False)['data'][0]:
        pass
    emotion1, gender1, path_speech1, path_emotion1 = run_emotion(wave_path1)
    emotion2, gender2, path_speech2, path_emotion2 = run_emotion(wave_path2)
    DB1, path_DB1 = run_DB(wave_path1)
    DB2, path_DB2 = run_DB(wave_path2)
    img_speech_stream1 = return_img_stream(path_speech1)
    img_speech_stream2 = return_img_stream(path_speech2)
    img_emotion_stream1 = return_img_stream(path_emotion1)
    img_emotion_stream2 = return_img_stream(path_emotion2)
    img_DB_stream1 = return_img_stream(path_DB1)
    img_DB_stream2 = return_img_stream(path_DB2)
    time1, text1, score1 = run_text(wave_path1)
    time2, text2, score2 = run_text(wave_path2)
    is_one, similarity = run_voiceprint(wave_path1, wave_path2)
    return render_template("result.html",
                           is_one=is_one,
                           similarity=similarity,
                           emotion1=emotion1,
                           gender1=gender1,
                           DB1=DB1,
                           img_speech_stream1=img_speech_stream1,
                           img_emotion_stream1=img_emotion_stream1,
                           img_DB_stream1=img_DB_stream1,
                           emotion2=emotion2,
                           gender2=gender2,
                           DB2=DB2,
                           img_speech_stream2=img_speech_stream2,
                           img_emotion_stream2=img_emotion_stream2,
                           img_DB_stream2=img_DB_stream2,
                           time1=time1,
                           text1=text1,
                           score1=score1,
                           time2=time2,
                           text2=text2,
                           score2=score2)


if __name__ == '__main__':
    app.run(port=5000, debug=True,threaded=True)
