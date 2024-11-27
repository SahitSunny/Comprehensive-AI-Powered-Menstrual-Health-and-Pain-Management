import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError


working_dir = os.path.dirname(os.path.abspath(__file__))

diabetes_model = pickle.load(
    open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(
    open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(
    open(f'{working_dir}/saved_models/updated_parkinson.pkl', 'rb'))

pco_model = pickle.load(
    open(f'{working_dir}/saved_models/updated_pcos.pkl', 'rb'))


custom_objects = {
    'mse': MeanSquaredError
}

model_path = f'{working_dir}/saved_models/updated_next_date.h5'

cycle_model = load_model(model_path, custom_objects=custom_objects)

