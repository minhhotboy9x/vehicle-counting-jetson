from flask import Flask, Response, render_template
from flask_cors import CORS
import config

app = Flask(__name__)
CORS(app) 

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=config.PORT)