from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

rf_model = joblib.load('models/rf.pkl')
knn_model = joblib.load('models/knn.pkl')
svm_model = joblib.load('models/svm.pkl')

scaler_data = joblib.load('models/scaler_data.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = model_name = acc = None
    if request.method == 'POST':
        lr = float(request.form['lr'])
        print(lr)
        x = float(request.form['x'])
        y = float(request.form['y'])
        z = float(request.form['z'])
        width = float(request.form['width'])
        depth = float(request.form['depth'])
        height = float(request.form['height'])
        model = request.form['model']

        if model == "rf":
            model = rf_model
            model_name = "Random Forest"
        elif model == "knn":
            model = knn_model
            model_name = "Stacking KNN"
        elif model == "svm":
            model = svm_model
            model_name = "Stacking SVM"

        feature_names = ['kneeLR', 'roiX', 'roiY', 'roiZ', 'roiHeight', 'roiWidth', 'roiDepth']
        X = pd.DataFrame([[lr, x, y, z, height, width, depth]], columns=feature_names)

        for col in X.columns:
            if col != 'kneeLR':
                mean = scaler_data[col]['mean']
                std = scaler_data[col]['std']
                X[col] = (X[col] - mean) / std if std != 0 else 0

        result = model.predict(X)
        acc = round(np.max(model.predict_proba(X)[0]), 4) * 100

        if result == 0:
            result = 'healthy'
        elif result == 1:
            result = 'partially ruptured'
        else:
            result = 'completely ruptured'

    return render_template('index.html', result=result, model_name=model_name, acc=acc)

if __name__ == '__main__':
    app.run()