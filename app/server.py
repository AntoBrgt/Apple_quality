from fastapi import FastAPI
import joblib
import numpy as np

model = joblib.load("random_forest_apple.pkl")

class_names = np.array(["good", "bad"])

app = FastAPI()

@app.get('/')

def reed_root():

    return {'message' : 'Apple model API'}

@app.post('/predict')

def predict(data: dict):
    """
    Predicts the class

    Args:
        data(dict): A dictionnary containing the features
    """
    features = (np.array(data['features']).reshape(1, -1))
    prediction = model.predict(features)
    class_name = class_names[prediction][0]

    return {'predicted_class': class_name}