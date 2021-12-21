from flask_wtf import FlaskForm
from wtforms import SubmitField
from flask_wtf.file import FileField,FileRequired,FileAllowed

class inputform(FlaskForm):
    file1 = FileField(u"上传语音1", validators=[FileRequired()])
    file2 = FileField(u"上传语音2", validators=[FileRequired()])
    submit=SubmitField(u"提交")