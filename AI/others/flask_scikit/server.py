# handle requests

import pickle

import numpy as np
from flask import Flask, jsonify, request


app = Flask(__name__)

with open('regr_model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/api', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict(
        [[np.array(data['exp'])]]
    )
    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
