from flask_wtf import FlaskForm
from wtforms import SubmitField
from flask_wtf.file import FileField,FileRequired,FileAllowed

class inputform(FlaskForm):
    file = FileField(u"上传文件", validators=[FileRequired()])
    submit=SubmitField(u"提交")