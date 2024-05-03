from flask import Flask, Response, render_template
from flask_cors import CORS
from routes.r_detect import detect_bp
import config


app = Flask(__name__)
app.register_blueprint(detect_bp)
CORS(app) 

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=config.PORT)