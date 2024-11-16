from datetime import datetime, timedelta
import os #provide facility to establish user and os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from models import *
from utils import *
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
        fo = st.text_input('MDVP:Fo (Hz)', placeholder='100-250 Hz')
        RAP = st.text_input('MDVP:RAP', placeholder='0.0-0.5')
        APQ3 = st.text_input('Shimmer:APQ3', placeholder='0.0-0.2')
        HNR = st.text_input('HNR', placeholder='10-30')
        D2 = st.text_input('D2', placeholder='1.0-4.0')

    with col2:
        fhi = st.text_input('MDVP:Fhi (Hz)', placeholder='150-200 Hz')
        PPQ = st.text_input('MDVP:PPQ', placeholder='0.0-0.1')
        APQ5 = st.text_input('Shimmer:APQ5', placeholder='0.0-0.1')
        RPDE = st.text_input('RPDE', placeholder='0.0-1.0')
        PPE = st.text_input('PPE', placeholder='0.0-1.0')

    with col3:
        flo = st.text_input('MDVP:Flo (Hz)', placeholder='70-100 Hz')
        DDP = st.text_input('Jitter:DDP', placeholder='0.0-0.02')
        APQ = st.text_input('MDVP:APQ', placeholder='20-30')
        DFA = st.text_input('DFA', placeholder='0.0-1.0')

    with col4:
        Jitter_percent = st.text_input(
            'MDVP:Jitter (%)', placeholder='0.0-0.1')
        Shimmer = st.text_input('MDVP:Shimmer', placeholder='0.0-0.1')
        DDA = st.text_input('Shimmer:DDA', placeholder='0.0-0.1')
        spread1 = st.text_input('spread1', placeholder='-5.0 to 0.0')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter (Abs)', placeholder='0.0-0.1')
        Shimmer_dB = st.text_input('MDVP:Shimmer (dB)', placeholder='0.0-1.0')
        NHR = st.text_input('NHR', placeholder='0.0-0.1')
        spread2 = st.text_input('spread2', placeholder='0.0-1.0')

    parkinsons_diagnosis = ''

    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
            st.error(parkinsons_diagnosis)
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"
            st.success(parkinsons_diagnosis)


if selected == 'Polycystic Ovarian Syndrome':

    st.title('Polycystic Ovarian Syndrome Prediction')

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        age = st.text_input(placeholder='1-100', label='Age (in years)')
        weight = st.text_input('Weight (in kilograms)', placeholder='45-150')
        height = st.text_input('Height (in centimeters)',
                               placeholder='150-190')
        skin_darkening = yes_no_mapper(st.selectbox(
            'Skin Darkening',
            options=['Yes', 'No'],
            index=0
        ))
        hair_loss = yes_no_mapper(st.selectbox(
            'Hair loss',
            options=['Yes', 'No'],
            index=0
        ))
       

    with col2:
        cycle = st.text_input('Cycle Regular/Irregular (R/I)',
                              placeholder='Regular or Irregular')
        cycle_length = st.text_input(
            'Cycle Length (in days)', placeholder='21-35')
        pimples = yes_no_mapper(st.selectbox(
            'Pimples',
            options=['Yes', 'No'],
            index=0
        ))
        endometrium = st.text_input(
            'Endometrium Thickness (in mm)', placeholder='5-15')
        
        avg_f_size_l = st.text_input(
            'Average Follicle Size (Left in mm)', placeholder='5-20')
       

    with col3:
        pregnant = yes_no_mapper(st.selectbox(
            'Pregnant',
            options=['Yes', 'No'],
            index=0
        ))
        abortions = st.text_input('Number of Abortions', placeholder='0-10')
        fsh = st.text_input(
            'Follicle Stimulating Hormone ', placeholder='1-12')
        lh = st.text_input('Luteinizing Hormone (mIU/mL)', placeholder='1-25')
       
        bp_systolic = st.text_input(
            'Systolic Blood Pressure (mmHg)', placeholder='90-180')

    with col4:
        tsh = st.text_input(
            'Thyroid Stimulating Hormone', placeholder='0.5-4.5')
        amh = st.text_input(
            'Anti-Mullerian Hormone (ng/mL)', placeholder='0.1-10')
        prl = st.text_input('Prolactin (PRL in ng/mL)', placeholder='5-25')
        bp_diastolic = st.text_input(
            'Diastolic Blood Pressure (mmHg)', placeholder='60-100')
        follicle_no_l = st.text_input(
            'Number of Follicles (Left)', placeholder='0-30')

    with col5:
        vit_d3 = st.text_input('Vitamin D3 (in ng/mL)', placeholder='20-100')
        prg = st.text_input('Progesterone (PRG in ng/mL)',
                            placeholder='0.1-20')
        rbs = st.text_input('Random Blood Sugar (in mg/dL)',
                            placeholder='70-200')
        weight_gain = yes_no_mapper(st.selectbox(
            'Weight Gain',
            options=['Yes', 'No'],
            index=0
        ))
        hair_growth = yes_no_mapper(st.selectbox(
            'Excess Hair Growth',
            options=['Yes', 'No'],
            index=0
        ))

    outcome = ''

    if st.button('Test Results'):
        user_input = [
            age,
            weight,
            height,
            cycle,
            cycle_length,
            pregnant,
            abortions,
            fsh,
            lh,
            tsh,
            amh,
            prl,
            vit_d3,
            prg,
            rbs,
            weight_gain,
            hair_growth,
            skin_darkening,
            hair_loss,
            pimples,
            bp_systolic,
            bp_diastolic,
            follicle_no_l,
            avg_f_size_l,
            endometrium
        ]

        user_input = [float(x) for x in user_input]

        predict = pco_model.predict([user_input])

        if predict[0] == 1:
            outcome = 'The person has Polycystic Ovarian Syndrome'
            st.error(outcome)
        else:
            outcome = 'The person does not have Polycystic Ovarian Syndrome'
            st.success(outcome)


if selected == 'Next Cycle Predictor':
    st.title('Next Cycle Prediction')

    col1, col2, col3, col4, col5 = st.columns(5)

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
        CycleNumber = st.number_input('Cycle Number (e.g., 1)', 0)
        CycleWithPeakorNot = yes_no_mapper(st.selectbox(
            'Cycle With Peak or Not',
            options=['Yes', 'No'],
            index=0
        ))
        ReproductiveCategory = reproductive_category_mapper(st.selectbox(
            'Reproductive Category',
            options=reproductive_categories,
            index=0
        ))
        user_date = st.date_input('Date')

    with col2:
        EstimatedDayofOvulation = st.number_input(
            'Day of Ovulation (1-30)', min_value=1, max_value=30)
        LengthofLuteatPhase = st.number_input(
            'Luteal Phase (1-20 days)', min_value=1, max_value=20)
        FirstDayofHigh = st.number_input(
            'First Day of High (1-30)', min_value=1, max_value=30)
        TotatNumberofHighDays = st.number_input(
            'Total Number of High Days (1-20)', min_value=1, max_value=20)

    with col3:
        TotatHighPostPeak = st.number_input(
            'Total High Post Peak (0)', min_value=0)
        TotatDaysofFertitity = st.number_input(
            'Total Days of Fertility (1-30)', min_value=1, max_value=30)
        TotatNumberofPeakDays = st.number_input(
            'Total Number of Peak Days (1-20)', min_value=1, max_value=20)
        TotatFertitityFormuta = st.number_input(
            'Total Fertility Formula (1-15)', min_value=1, max_value=15)

    with col4:
        MensesScoreDayOne = st.number_input(
            'Menses Score Day One (0-10)', min_value=0, max_value=10)
        MensesScoreDayTwo = st.number_input(
            'Menses Score Day Two (0-10)', min_value=0, max_value=10)
        MensesScoreDayThree = st.number_input(
            'Menses Score Day Three (0-10)', min_value=0, max_value=10)
        MensesScoreDayFour = st.number_input(
            'Menses Score Day Four (0-10)', min_value=0, max_value=10)
        MensesScoreDayFive = st.number_input(
            'Menses Score Day Five (0-10)', min_value=0, max_value=10)

    with col5:
        LengthofMenses = st.number_input(
            'Length of Menses (1-14 days)', min_value=1, max_value=14)
        TotatMensesScore = st.number_input(
            'Total Menses Score (0-30)', min_value=0, max_value=30)
        NumberofDaysofIntercourse = st.number_input(
            'Days of Intercourse (0-30)', min_value=0, max_value=30)

        IntercourseInFertileWindow = yes_no_mapper(st.selectbox(
            'Intercourse In Fertile Window',
            options=['Yes', 'No'],
            index=0
        ))
       


    if st.button('Predict Date'):
        user_input = [
            CycleNumber,
            CycleWithPeakorNot,
            ReproductiveCategory,
            EstimatedDayofOvulation,
            LengthofLuteatPhase,
            FirstDayofHigh,
            TotatNumberofHighDays,
            TotatHighPostPeak,
            TotatDaysofFertitity,
            TotatNumberofPeakDays,
            TotatFertitityFormuta,
            LengthofMenses,
            MensesScoreDayOne,
            MensesScoreDayTwo,
            MensesScoreDayThree,
            MensesScoreDayFour,
            MensesScoreDayFive,
            TotatMensesScore,
            NumberofDaysofIntercourse,
            IntercourseInFertileWindow
        ]

        user_input = [float(x) for x in user_input]

        predict_date = cycle_model.predict([user_input])

        predict_usual = ususal_model.predict([user_input])

        st.success("Next Expected Cycle on : " +
                   add_days_to_date(user_date, predict_date))

        if predict_usual == 0:
            st.success("ususal")
        else:
            st.error("unsual")





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