from flask import Flask
import sklearn.external.joblib as extjoblib
import joblib

app = Flask(__name__)

@app.route("/")
def enter():
    return 'welcome to the app'

if __name__ == '__main__':
    app.run(debug = True, port = 3000)

joblib.dump(lr, 'model.pkl')

lr = joblib.load('model.pkl')