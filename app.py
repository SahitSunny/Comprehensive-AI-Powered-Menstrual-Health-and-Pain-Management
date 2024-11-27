from datetime import datetime, timedelta
import os #provide facility to establish user and os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from models import *
from utils import *
from scalers import *
import google.generativeai as genai
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from PIL import Image
import io


genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")



with st.sidebar:
    selected = option_menu('Menstrual Cycle and Disease Prediction',
                           [
                               'Next Cycle Predictor',
                               'Diabetes Prediction',
                               'Heart Disease Prediction',
                               'Parkinsons Prediction',
                               'Polycystic Ovarian Syndrome',
                               'PCOS Using CNN',
                               'Chatbot'
                           ],
                           menu_icon='hospital-fill',
                           icons=['calendar', 'circle', 'heart',
                                  'person', 'person-circle', 'robot'],
                           default_index=0)


if selected == 'Diabetes Prediction':

    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.text_input(
            'Number of Pregnancies', placeholder='0-10')
        skin_thickness = st.text_input(
            'Skin Thickness (in mm)', placeholder='0-99')
        diabetes_pedigree_function = st.text_input(
            'Diabetes Pedigree Function', placeholder='0.0-2.5')

    with col2:
        glucose = st.text_input(
            'Glucose Level (in mg/dL)', placeholder='70-300')
        insulin = st.text_input('Insulin Level (in µU/mL)', placeholder='2-25')
        age = st.text_input('Age of the Person (in years)',
                            placeholder='0-120')

    with col3:
        blood_pressure = st.text_input(
            'Blood Pressure (in mmHg)', placeholder='80-200')
        BMI = st.text_input('BMI value', placeholder='10-60')

    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):

        user_input = [pregnancies, glucose, blood_pressure, skin_thickness, insulin,
                      BMI, diabetes_pedigree_function, age]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
            st.error(diab_diagnosis)
        else:
            diab_diagnosis = 'The person is not diabetic'
            st.success(diab_diagnosis)

if selected == 'Heart Disease Prediction':

    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    def gender_mapper(value):
        return 1 if value == 'Male' else 0

    def chestpain_mapper(value):
        if value == 'Typical Angina':
            return 0
        elif value == 'Atypical Angina':
            return 1
        elif value == 'Non-Anginal Pain':
            return 2
        else:
            return 3

    def thal_mapper(value):
        if value == 'Normal':
            return 0
        elif value == 'Fixed Defect':
            return 1
        else:
            return 2

    with col1:
        age = st.text_input('Age', placeholder='0-120')
        trestbps = st.text_input(
            'Resting Blood Pressure (in mmHg)', placeholder='90-200')
        restecg = st.text_input(
            'Resting Electrocardiographic Results', placeholder='0-2')
        oldpeak = st.text_input(
            'ST Depression Induced by Exercise (in mm)', placeholder='0.0-6.0')
        thal = thal_mapper(st.selectbox(
            'Thal: ',
            options=[
                'Normal',
                'Fixed Defect',
                'Reversible Defect'
            ],
            index=0
        ))

    with col2:
        sex = gender_mapper(st.selectbox(
            'Sex',
            options=[
                'Male',
                'Female'
            ],
            index=0
        ))
        chol = st.text_input(
            'Serum Cholesterol (in mg/dL)', placeholder='100-400')
        thalach = st.text_input(
            'Maximum Heart Rate Achieved (in bpm)', placeholder='60-220')
        slope = st.text_input(
            'Slope of the Peak Exercise ST Segment', placeholder='0-2')

    with col3:
        cp = chestpain_mapper(st.selectbox(
            'Chest Pain Types',
            options=[
                'Typical Angina',
                'Atypical Angina',
                'Non-Anginal Pain',
                'Asymptomatic'
            ],
            index=0
        ))

        fbs = yes_no_mapper(st.selectbox(
            'Fasting Blood Sugar > 120 mg/dL',
            options=[
                'Yes',
                'No'
            ],
            index=0
        ))

        exang = yes_no_mapper(st.selectbox(
            'Exercise Induced Angina',
            options=[
                'Yes',
                'No'
            ],
            index=0
        ))
        ca = st.text_input(
            'Major Vessels Colored by Fluoroscopy', placeholder='0-3')

    heart_diagnosis = ''

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs,
                      restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
            st.error(heart_diagnosis)
        else:
            heart_diagnosis = 'The person does not have any heart disease'
            st.success(heart_diagnosis)


if selected == "Parkinsons Prediction":

    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, value=40)  # Age as a numerical input
        gender = gender_mapper(st.selectbox('Gender', ['Male', 'Female']))  # Gender as a dropdown (categorical)
        tremor = yes_no_mapper(st.selectbox('Tremor', ['No', 'Yes']))  # Tremor as a dropdown (categorical)
        slowness_of_movement = yes_no_mapper(st.selectbox('Slowness of Movement', ['No', 'Yes']))  # Slowness of Movement (categorical)
        muscle_stiffness = yes_no_mapper(st.selectbox('Muscle Stiffness', ['No', 'Yes']))  # Muscle Stiffness (categorical)

# Column 2: Balance Issues, Speech Problems, Depression, Anxiety
    with col2:
        balance_issues = yes_no_mapper(st.selectbox('Balance Issues', ['No', 'Yes']))  # Balance Issues (categorical)
        speech_problems = yes_no_mapper(st.selectbox('Speech Problems', ['No', 'Yes']))  # Speech Problems (categorical)
        depression = yes_no_mapper(st.selectbox('Depression', ['No', 'Yes']))  # Depression (categorical)
        anxiety = yes_no_mapper(st.selectbox('Anxiety', ['No', 'Yes']))  # Anxiety (categorical)
        sleep_problems = yes_no_mapper(st.selectbox('Sleep Problems', ['No', 'Yes']))  # Sleep Problems (categorical)

# Column 3: Memory Problems, Fatigue, Walking Problems, Family History of Parkinson's
    with col3:
        memory_problems = yes_no_mapper(st.selectbox('Memory Problems', ['No', 'Yes']))  # Memory Problems (categorical)
        fatigue = yes_no_mapper(st.selectbox('Fatigue', ['No', 'Yes']))  # Fatigue (categorical)
        walking_problems =yes_no_mapper(st.selectbox('Walking Problems', ['No', 'Yes']))  # Walking Problems (categorical)
        family_history_of_parkinsons = yes_no_mapper(st.selectbox('Family History of Parkinson\'s', ['No', 'Yes']))  # Family History (categorical)
        handwriting_problems = yes_no_mapper(st.selectbox('Handwriting Problems', ['No', 'Yes']))  # Handwriting Problems (categorical)

# Column 4: Loss of Smell, Dizziness, Unusual Sweating, Frequent Urination
    with col4:
        loss_of_smell = yes_no_mapper(st.selectbox('Loss of Smell', ['No', 'Yes']))  # Loss of Smell (categorical)
        dizziness = yes_no_mapper(st.selectbox('Dizziness', ['No', 'Yes']))  # Dizziness (categorical)
        unusual_sweating = yes_no_mapper(st.selectbox('Unusual Sweating', ['No', 'Yes']))  # Unusual Sweating (categorical)
        frequent_urination = yes_no_mapper(st.selectbox('Frequent Urination', ['No', 'Yes']))  # Frequent Urination (categorical)
        low_blood_pressure = yes_no_mapper(st.selectbox('Low Blood Pressure', ['No', 'Yes']))  # Low Blood Pressure (categorical)

# Column 5: Weight Loss, Difficulty Swallowing
    with col5:
        weight_loss = yes_no_mapper(st.selectbox('Weight Loss', ['No', 'Yes']))  # Weight Loss (categorical)
        difficulty_swallowing = yes_no_mapper(st.selectbox('Difficulty Swallowing', ['No', 'Yes']))

    parkinsons_diagnosis = ''

    if st.button("Parkinson's Test Result"):

        user_input = [age, gender, tremor, slowness_of_movement, muscle_stiffness, balance_issues, speech_problems, depression, anxiety,sleep_problems, memory_problems, fatigue,walking_problems,family_history_of_parkinsons,handwriting_problems,loss_of_smell, dizziness,unusual_sweating, frequent_urination,low_blood_pressure,weight_loss, difficulty_swallowing]

        user_input = [float(x) for x in user_input]

        user_input = parkinsons_scaler.transform([user_input])
        
        parkinsons_prediction = parkinsons_model.predict(user_input)

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
            st.error(parkinsons_diagnosis)
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"
            st.success(parkinsons_diagnosis)


if selected == 'Polycystic Ovarian Syndrome':

    st.title('Polycystic Ovarian Syndrome Prediction')

    col1, col2, col3 = st.columns(3)

    with col1:
        bmi = st.text_input('BMI', placeholder='18.5-40.0')
        fatigue_levels = category_mapper(st.selectbox('Fatigue Levels', ['Low', 'Moderate', 'High'], index=0))
        headaches = category_mapper(st.selectbox('Headaches', ['Rarely', 'Occasionally', 'Frequently'], index=0))
        urinary_issues = yes_no_mapper(st.selectbox('Urinary Issues', ['Yes', 'No'], index=0))


    with col2:
        fertility_status = yes_no_mapper(st.selectbox('Fertility Status', ['Yes', 'No'], index=0))
        sleep_apnea = yes_no_mapper(st.selectbox('Sleep Apnea', ['Yes', 'No'], index=0))
        family_history = yes_no_mapper(st.selectbox('Family History', ['Yes', 'No'], index=0))
        palpitations = category_mapper(st.selectbox('Palpitations', ['Rarely', 'Occasionally', 'Frequently'], index=0))


    with col3:
        vision_problems = yes_no_mapper(st.selectbox('Vision Problems', ['Yes', 'No'], index=0))
        mood_disorders = category_mapper(st.selectbox('Mood Disorders', ['None', 'Mild', 'Severe'], index=0))

    outcome = ''

    if st.button('Test Results'):
        user_input = [
           bmi, fatigue_levels, headaches, urinary_issues, fertility_status, sleep_apnea, family_history, palpitations,vision_problems, mood_disorders
        ]

        user_input = [float(x) for x in user_input]

        user_input = pcos_scaler.transform([user_input])
        predict = pco_model.predict(user_input)

        if predict[0] == 1:
            outcome = 'The person has Polycystic Ovarian Syndrome'
            st.error(outcome)
        else:
            outcome = 'The person does not have Polycystic Ovarian Syndrome'
            st.success(outcome)


if selected == 'Next Cycle Predictor':
    st.title('Next Cycle Prediction')

    col1, col2, col3 = st.columns(3)

    reproductive_categories = [
        'Adolescent',
        'Reproductive',
        'Perimenopausal',
        'Menopausal'
    ]

    def reproductive_category_mapper(value):
        mapping = {
            'Adolescent': 0,
            'Reproductive': 1,
            'Perimenopausal': 2,
            'Menopausal': 3
        }
        return mapping.get(value, None)

    with col1: 
        Sleeping_Hours = st.number_input('Sleeping Hours (4-10)', min_value=4, max_value=10) 
        Habits = category_mapper(st.selectbox( 'Habits', options=['No drinking/smoking', 'Smoking', 'Alcohol'], index=0 )) 
        Stress_Level = st.number_input( 'Stress Level (1-5)', min_value=1, max_value=5) 
        Activity_Level = st.number_input('Activity Level (1-5)', min_value=1, max_value=5)
        user_date = st.date_input('Date') 
    with col2:
        Mood_Swings = st.number_input('Mood Swings (0-5)', min_value=0, max_value=5) 
        Diet_Quality = st.number_input('Diet Quality (1-5)', min_value=1, max_value=5) 
        Water_Intake = st.number_input('Water Intake (1-4 liters)', min_value=1.0, max_value=4.0) 
        Menstrual_Flow_Intensity = st.number_input('Menstrual Flow Intensity (1-3)', min_value=1, max_value=3)
    with col3:
        bmi = st.number_input('BMI (18-30)', min_value=18.0, max_value=30.0) 
        Daily_Step_Count = st.number_input('Daily Step Count (2000-15000)', min_value=2000, max_value=15000) 
        Cycle_Regularity = yes_no_mapper(st.selectbox( 'Cycle Regularity', options=['Yes', 'No'], index=0 )) 
        Hormonal_Medication = yes_no_mapper(st.selectbox( 'Hormonal Medication', options=['Yes', 'No'], index=0 ))
       


    if st.button('Predict Date'):
        user_input = [
           Sleeping_Hours,
           Habits,
           Stress_Level,
           Activity_Level,
           Mood_Swings,
           Diet_Quality,
           Water_Intake,
           Menstrual_Flow_Intensity,
           bmi,
           Daily_Step_Count,
           Cycle_Regularity,
           Hormonal_Medication
        ]

        user_input = [float(x) for x in user_input]

        user_input = next_cycle_scaler.transform([user_input])
        predict_date = cycle_model.predict(user_input)


        

        if predict_date[0][0] < 30:
            st.success("ususal")
            st.success("Next Expected Cycle on : " +
                   add_days_to_date(user_date, predict_date[0][0]))
        else:
            st.error("unsual")
            st.error("Next Expected Cycle on : " +
                   add_days_to_date(user_date, predict_date[0][0]))





if selected == 'Chatbot':
    st.title("Chatbot using Google Generative AI")

    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-002",
        generation_config=generation_config,
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    for message in st.session_state.history:
        with st.chat_message(message['role']):
            st.markdown(message['parts'][0])

    if prompt := st.chat_input("What is up?"):
        st.session_state.history.append({"role": "user", "parts": [prompt]})

        with st.chat_message("user"):
            st.markdown(prompt)

        chat_session = model.start_chat(history=st.session_state.history)

        with st.spinner("Generating response..."):
            response = chat_session.send_message(prompt)

        assistant_message = response.text
        with st.chat_message("assistant"):
            st.markdown(assistant_message)

        st.session_state.history.append(
            {"role": "model", "parts": [assistant_message]})


if selected == 'PCOS Using CNN':
    st.title("PCOS Detection using Image Proccessing")

    model = load_model("saved_models/facemodel.h5")

    def calculate_bmi(weight, height):
        try:
            return weight / (height ** 2)
        except ZeroDivisionError:
            return None

    def bmi_category(bmi):
        if bmi is None:
            return "Invalid"
        if bmi < 18.5:
            return "Underweight"
        elif 18.5 <= bmi < 24.9:
            return "Normal"
        elif 25 <= bmi < 29.9:
            return "Overweight"
        elif 30 <= bmi < 34.9:
            return "Obesity Class 1"
        elif 35 <= bmi < 39.9:
            return "Obesity Class 2"
        else:
            return "Obesity Class 3"


    def pcos_risk_analysis(bmi, acne_detected, hair_growth_detected):
        if bmi and bmi > 25 and acne_detected and hair_growth_detected:
            return "High"
        else:
            return "Low"

    def detect_acne(image):
        try:
            face = image.convert('RGB')
            face = face.resize((224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # Make prediction
            (acne, withoutAcne) = model.predict(face)[0]
            return acne > withoutAcne, max(acne, withoutAcne) * 100  # Returns True if acne is detected, and the confidence score
        except Exception as e:
            st.error(f"Acne detection error: {str(e)}")
            return False, 0.0

    def detect_hair_growth(image):
        return True



    weight = st.number_input("Weight (kg):", min_value=1.0, max_value=200.0)
    height = st.number_input("Height (m):", min_value=0.5, max_value=2.5)
    cycle_length = st.number_input("Menstrual Cycle Length (days):", min_value=1, max_value=60)
    sleep_hours = st.number_input("Average Sleep Hours:", min_value=0, max_value=24)

    acne_choice = st.selectbox("Do you have acne?", ("No", "Yes"))
    hair_growth_choice = st.selectbox("Do you have excessive hair growth?", ("No", "Yes"))

    image_option = st.radio("Choose image source:", ("Upload Image", "Use Camera"))

    if image_option == "Upload Image":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    else:
        uploaded_image = st.camera_input("Take a picture")


    if st.button("Check PCOS Risk"):
        try:
            bmi = calculate_bmi(weight, height)
            bmi_cat = bmi_category(bmi)
            
            acne_detected = False
            hair_growth_detected = hair_growth_choice == "Yes"
            
            if uploaded_image is not None:
                img = Image.open(uploaded_image)
                acne_detected, acne_confidence = detect_acne(img)
                st.write(f"Acne detected: {'Yes' if acne_detected else 'No'} with confidence {acne_confidence:.2f}%")

            pcos_risk = pcos_risk_analysis(bmi, acne_detected, hair_growth_detected)
            
            st.write(f"BMI: {bmi:.2f} ({bmi_cat})")
            st.write(f"PCOS Risk: {pcos_risk}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")