from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import logging
import joblib
import json
import sys
import os

current_dir = os.path.dirname(__file__)

app = Flask(__name__, static_folder="static", template_folder="template")

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)


@app.route('/')
def hello_world():
    return "Hello, Flask!"
