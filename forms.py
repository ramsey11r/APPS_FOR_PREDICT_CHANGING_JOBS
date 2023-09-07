from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('decision_tree_model.pkl','rb'))
model1 = pickle.load(open('svm.pkl','rb'))
model2 = pickle.load(open('random_forest_model.pkl','rb'))
model3 = pickle.load(open('naive_bayes.pkl','rb'))
model4 = pickle.load(open('linearReg.pkl','rb'))


app = Flask(__name__)



@app.route("/")
def home():
    return render_template('homepage.html')

@app.route("/forms")
def forms():
    return render_template('forms.html')

@app.route("/forms2")
def forms2():
    return render_template('forms2.html')

@app.route("/forms3")
def forms3():
    return render_template('forms3.html')

@app.route("/forms4")
def forms4():
    return render_template('forms4.html')

@app.route("/forms5")
def forms5():
    return render_template('forms5.html')


@app.route('/svm', methods=['POST'])
def svm():
    myfeatures = [float(x) for x in request.form.values()]
    features=[np.array(myfeatures)]
    pred = model1.predict(features)
    return render_template('result.html',data=pred)

@app.route('/naive_bayes', methods=['POST'])
def naive_bayes():
    myfeatures = [float(x) for x in request.form.values()]
    features=[np.array(myfeatures)]
    pred = model3.predict(features)
    return render_template('result.html',data=pred)


@app.route('/random_forest', methods=['POST'])
def random_forest():
    myfeatures = [float(x) for x in request.form.values()]
    features=[np.array(myfeatures)]
    pred = model2.predict(features)
    return render_template('result.html',data=pred)


@app.route('/linear_reg', methods=['POST'])
def linear_reg():
    myfeatures = [float(x) for x in request.form.values()]
    features=[np.array(myfeatures)]
    pred = model4.predict(features)
    return render_template('result.html',data=pred)

@app.route('/predict', methods=['POST'])
def predict():
    
    myfeatures = [float(x) for x in request.form.values()]
    features=[np.array(myfeatures)]
    pred = model.predict(features)
    return render_template('result.html',data=pred)


if __name__ == "__main__":
    app.run(debug=True)