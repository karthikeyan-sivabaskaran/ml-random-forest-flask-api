from flask import jsonify, request
from flasgger.utils import  swag_from
import os
from flask import Blueprint
import pandas as pd
import numpy as np
import pickle
import pathlib

api_routes = Blueprint('routes_api',__name__)

swagger_config_dir = str(pathlib.Path(__file__).resolve().parent.parent)

pkl_file_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent,'models','rf.pkl')
with open(pkl_file_path, 'rb') as model_file:
    model = pickle.load(model_file)

iris_category = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

@api_routes.route('/health')
def health():
    return 'ok'

@api_routes.route('/predict')
@swag_from(os.path.join(swagger_config_dir, 'swagger_configs', 'swagger_config1.yml'))
def predict_iris():
    s_length = request.args.get("sepal_length")
    s_width = request.args.get("sepal_width")
    p_length = request.args.get("petal_length")
    p_width = request.args.get("petal_width")

    prediction = model.predict(np.array([[s_length, s_width, p_length, p_width]]))

    return jsonify(Predicted_iris_Species=iris_category[prediction[0]],
                   sepal_length=s_length,
                   sepal_width=s_width,
                   petal_length=p_length,
                   petal_width=p_width)


@api_routes.route('/predict_file', methods=["POST"])
@swag_from(os.path.join(swagger_config_dir, 'swagger_configs', 'swagger_config2.yml'))
def predict_iris_file():
    input_data = pd.read_csv(request.files.get("input_file"), header=None)
    # prediction = model.predict(input_data)
    # return str(list(prediction))
    input_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    input_data['Predicted_iris_Species'] = model.predict(input_data)

    out_list = []

    for data in input_data.values:
        out_list.append(
            {"Predicted_iris_Species": iris_category[data[4]],
             "sepal_length": data[0],
             "sepal_width": data[1],
             "petal_length": data[2],
             "petal_width": data[3]
             })

    return jsonify(out_list)
