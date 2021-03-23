from flask import Flask, render_template, request
import pickle
import pandas as pd

app= Flask(__name__)

#predict

@app.route('/', methods=['POST','GET'])
def predict():
    return render_template('predict.html')

# result

@app.route ('/result', methods=['POST','GET'])
def result():
    if request.method == 'POST':
        input= request.form

        df_predict= pd.DataFrame({
            'alcohol': [input['Alcohol']],
            'density': [input['Density']],
            'fixed_acidity_level':[input['fal']],
            'chlorides_level':[input['cl']]
        })

        prediksi= model.predict_proba(df_predict)[0][1]

        if prediksi>0.5:
            quality='Good'
        else:
            quality='Bad'

        return render_template('result.html', data=input, pred=quality, prob=round(prediksi,4))


if __name__ =='__main__':

    filename='Model Final.sav'
    model=pickle.load(open(filename,'rb'))

    app.run(debug=True)


# Tugas: Dengan menggunakan data Titanic, buatlah model dan app sederhana 
# untuk memprediksi apakah seseorang survive atau tidak menggunakan flask