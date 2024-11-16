import pickle
import os

working_dir = os.path.dirname(os.path.abspath(__file__))

diabetes_model = pickle.load(
    open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(
    open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(
    open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

pco_model = pickle.load(
    open(f'{working_dir}/saved_models/pcos.pkl', 'rb'))

cycle_model = pickle.load(
    open(f'{working_dir}/saved_models/date_predictor.pkl', 'rb')
)

ususal_model = pickle.load(
    open(f'{working_dir}/saved_models/unsual_classifier.pkl', 'rb')
)
