from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from LungCancerPrediction.pipeline.prediction import PredictionPipeline


app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #reading the inputs given by the user
            age = int(request.form['age'])
            gender = int(request.form['gender'])
            air_pollution = int(request.form['air_pollution'])
            alcohol_use = int(request.form['alcohol_use'])
            dust_allergy = int(request.form['dust_allergy'])
            occupational_hazards = int(request.form['occupational_hazards'])
            genetic_risk = int(request.form['genetic_risk'])
            chronic_lung_disease = int(request.form['chronic_lung_disease'])
            balanced_diet = int(request.form['balanced_diet'])
            obesity = int(request.form['obesity'])
            smoking = int(request.form['smoking'])
            passive_smoker = int(request.form['passive_smoker'])
            chest_pain = int(request.form['chest_pain'])
            coughing_of_blood = int(request.form['coughing_of_blood'])
            fatigue = int(request.form['fatigue'])
            weight_loss = int(request.form['weight_loss'])
            shortness_of_breath = int(request.form['shortness_of_breath'])
            wheezing = int(request.form['wheezing'])
            swallowing_difficulty = int(request.form['swallowing_difficulty'])
            clubbing_of_finger_nails = int(request.form['clubbing_of_finger_nails'])
            frequent_cold = int(request.form['frequent_cold'])
            dry_cough = int(request.form['dry_cough'])
            snoring = int(request.form['snoring'])


         
            data = [
             age,
             gender,
             air_pollution,
             alcohol_use,
             dust_allergy,
             occupational_hazards,
             genetic_risk,
             chronic_lung_disease,
             balanced_diet,
             obesity,
             smoking,
             passive_smoker,
             chest_pain,
             coughing_of_blood,
             fatigue,
             weight_loss,
             shortness_of_breath,
             wheezing,
             swallowing_difficulty,
             clubbing_of_finger_nails,
             frequent_cold,
             dry_cough,
             snoring
             ]
            data = np.array(data).reshape(1, 23)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)

            prediction_map = {2: 'High', 1 : 'Medium', 0: 'Low'}
            return render_template('results.html', prediction = prediction_map[predict[0]])

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug = True, host='0.0.0.0', port=5000)