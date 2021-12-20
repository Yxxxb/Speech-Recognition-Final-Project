from head import create_app
from flask import render_template,flash,request
from webform import inputform
import os

app=create_app()

@app.route('/',methods=['POST','GET'])
def index():
    form = inputform()
    if form.validate_on_submit():
        src = "./test_data"
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
                    break
                if not os.path.exists(src + '/' + str(oldhead) + "(" + str(i) + ")" + "." + back):
                    newhead = oldhead + "(" + str(i) + ")"
                    print(newhead)
                    break
            road = src + '/' + str(newhead + '.' + back)
            form.file.data.save(road)
        road = src + '/' + str(form.file.data.filename)
        form.file.data.save(road)
        flash("文件上传成功")
    return render_template("index.html",
                           form=form)


if __name__=='__main__':
    app.run(debug=True)




