import pickle
import os

working_dir = os.path.dirname(os.path.abspath(__file__))

parkinsons_scaler = pickle.load(
    open(f'{working_dir}/saved_models/parkinsons_scaler.pkl', 'rb'))

pcos_scaler = pickle.load(
    open(f'{working_dir}/saved_models/PCOS_scaler.pkl', 'rb'))

next_cycle_scaler = pickle.load(
    open(f'{working_dir}/saved_models/next_cycle_scaler.pkl', 'rb'))