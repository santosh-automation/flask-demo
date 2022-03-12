import numpy as np
from flask import Flask, render_template , request
import pickle

model_pk = pickle.load(open("model-icecream.pkl","rb"))
mk = 0

app = Flask(__name__)

@app.route("/", methods = ['GET','POST'])
def hello():
    if request.method == "POST":
        temp_value = request.form["temprature"]
        data = np.array([[temp_value]])
        price_pred = model_pk.predict(data)
        print(price_pred)
        mk = price_pred

    return render_template("index.html",my_price = price_pred)

@app.route("/sub",methods = ['POST'])
def submit():
    if request.method == "POST":
        return render_template("sub.html", my_price = mk)


if __name__ == "__main__":
    app.run(debug=True)