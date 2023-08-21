from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import logging
import joblib
import json
import sys
import os

current_dir = os.path.dirname(__file__)

app = Flask(__name__, static_folder="static", template_folder="template")

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)


def ValuePredictor(data=pd.DataFrame):
    model = 'model/model.pkl'
    model_dir = os.path.join(current_dir, model)
    loaded_model = joblib.load(open(model_dir, 'rb'))
    result = loaded_model.predict(data)
    return result[0]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/prediction', methods=['POST'])
def predict():
    if request.method == 'POST':
        name = request.form['name']
        gender = request.form['gender']
        education = request.form['education']
        self_employed = request.form['self_employed']
        marital_status = request.form['marital_status']
        dependents = request.form['dependents']
        applicant_income = request.form['applicant_income']
        coapplicant_income = request.form['coapplicant_income']
        loan_amount = request.form['loan_amount']
        loan_term = request.form['loan_term']
        credit_history = request.form['credit_history']
        property_area = request.form['property_area']

        schema_name = "model/column_set.json"

        schema_dir = os.path.join(current_dir, schema_name)
        with open(schema_dir, 'r') as f:
            cols = json.loads(f.read())
        schema_cols = cols['data_columns']

        try:
            col = ('Dependents_' + str(dependents))
            if col in schema_cols.keys():
                schema_cols[col] = 1
            else:
                pass
        except:
            pass

        try:
            col = ('Property_Area_' + str(property_area))
            if col in schema_cols.keys():
                schema_cols[col] = 1
            else:
                pass
        except:
            pass

        schema_cols['ApplicantIncome'] = applicant_income
        schema_cols['CoapplicantIncome'] = coapplicant_income
        schema_cols['LoanAmount'] = loan_amount
        schema_cols['Loan_Amount_Term'] = loan_term
        schema_cols['Gender_Male'] = gender
        schema_cols['Married_Yes'] = marital_status
        schema_cols['Education_Not Graduate'] = education
        schema_cols['Self_Employed_Yes'] = self_employed
        schema_cols['Credit_History_1.0'] = credit_history

        df = pd.DataFrame(
            data={k: [v] for k, v in schema_cols.items()},
            dtype=float
        )

        print(df.dtypes)
        result = ValuePredictor(data=df)

        if int(result) == 1:
            prediction = 'Dear Mr/Mrs/Ms {name}, your loan is approved!'.format(name=name)
        else:
            prediction = 'Dear Mr/Mrs/Ms {name}, your loan was not approved'.format(name=name)

        return render_template('prediction.html', prediction=prediction)

    else:
        return render_template('error.html')


if __name__ == '__main__':
    app.run(debug=True)
