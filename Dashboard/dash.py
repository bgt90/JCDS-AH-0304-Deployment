from flask import Flask, render_template, request
import joblib
import pickle
import pandas as pd

app = Flask(__name__)

#halaman home
@app.route('/')
def home():
    return render_template('home.html')

#halaman dataset
@app.route('/database', methods=['POST', 'GET'])
def dataset():
    return render_template('dataset.html')

# #halaman visualisasi
@app.route('/visualize', methods=['POST', 'GET'])
def visual():
    return render_template('plot.html')

# #halaman input prediksi
@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    return render_template('predict.html')

# #halaman hasil prediksi
@app.route('/result', methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        input = request.form

        df_predict = pd.DataFrame({
            'alcohol':[input['Alcohol']],
            'density':[input['Density']],
            'fixed acidity level':[input['fal']],
            'chlorides level':[input['cl']]
        })

        # df_predict = pd.DataFrame({
        #     'alcohol':[10],
        #     'density':[1],
        #     'fixed acidity level':['low'],
        #     'chlorides level':['low']
        # })
        # prediksi = 0.5

        prediksi = model.predict_proba(df_predict)[0][1]

        if prediksi > 0.5:
            quality = "Good"
        else:
            quality = "Bad"

        return render_template('result.html',
            data=input, pred=quality)

if __name__ == '__main__':
    # model = joblib.load('model_joblib')

    filename = 'Model Final.sav'
    model = pickle.load(open(filename,'rb'))

    app.run(debug=True)