from flask import Flask
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_app():
    static_dir=os.path.join(BASE_DIR,'static')
    templates_dir=os.path.join(BASE_DIR,'templates')
    app=Flask(__name__,template_folder=templates_dir,static_folder=static_dir)
    app.config['SECRET_KEY'] = "111111"
    return app