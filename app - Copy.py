import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
  return render_template('index1.html')

@app.route('/predict',methods=['POST'])
def predict():
  input_features = [float(x) for x in request.form.values()]
  features_value = [np.array(input_features)]

  output = model.predict(features_value)
  if output==1:
      res_val = "The Patient Have Heart Disease,please consult the Doctor"
  else:
      res_val = "The Patient Normal"

    


  return render_template('index1.html', prediction_text='Result - {}'.format(res_val))

if __name__ == "__main__":
  app.run()
##host='0.0.0.0',debug=False, port = 4566
