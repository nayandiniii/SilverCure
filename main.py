from flask import *
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import speech_recognition as sr
from langdetect import detect
import openai
from fpdf import FPDF
import ipfshttpclient

app = Flask(__name__)
openai.api_key = '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'
app.config['UPLOAD_FOLDER'] = 'static/audio'
model_engine = "text-davinci-003"


def get_language(language_code):
    language_mapping = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'nl': 'Dutch',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'ta': 'Tamil',
        'kn': 'Kannada'
    }
    return language_mapping.get(language_code, 'Unknown')
def convert_audio_to_text(audio_file_path):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)  # Record the audio

        try:
            text = recognizer.recognize_google(audio_data, language='en')  # Always transcribe to English
            detected_language = detect(text)

            return text, detected_language
        except sr.UnknownValueError:
            return "Could not understand audio", None
        except sr.RequestError as e:
            return f"Error: {e}", None

@app.route('/home')
def home():
    return render_template("index.html")

@app.route('/data', methods=['GET', 'POST'])
def upload():
    return render_template("test.html")

@app.route('/report',  methods=['GET', 'POST'])
def report():
    print("hi2")
    print(request.method)
    if request.method == 'POST':
        audio_file = request.files['audio']
        #sotring file
        audio_file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'], audio_file.filename))
        file_name = audio_file.filename
        #printing values
        age_str = request.form['age']
        name_data = request.form['name']
        age_data = float(request.form['age'])
        gender_data = request.form['gender']
        lang_data = request.form['language']
        fatigue = float(request.form['slider1'])
        restless = float(request.form['slider2'])
        discomfort = float(request.form['slider3'])
        hygiene = float(request.form['slider4'])
        movements = float(request.form['slider5'])
        Fatigue = fatigue/10
        Slowing = restless/10
        Pain = discomfort/10
        Hygiene = hygiene/10
        Movement = movements/10
        print("-------------",name_data,age_data,gender_data,lang_data,Fatigue,Slowing,Pain,Hygiene,Movement)
        # print("-------------",name_data,age_data,gender_data,lang_data,fatigue,restless,discomfort,hygiene,movements)
        #model
        with open('.\models\LogisticRegression_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file) 
        #Vector array for prediction
        feature_vector = np.array([Fatigue,Slowing,Pain,Hygiene,Movement], dtype=float)
        # Use the loaded model to predict proneness
        sample_input = feature_vector.reshape(1, -1)
        predicted_proneness = model.predict(sample_input)
        print("Predicted Proneness:", predicted_proneness, type(predicted_proneness), 1==predicted_proneness)   # RESULT for text
        #######################################################################
        # Load the trained SVC model and fitted TF-IDF vectorizer from the pickle file
        with open('.\models\LogisticRegression_model(audio) (1).pkl', 'rb') as model_file:
            model, fitted_tfidf = pickle.load(model_file)
        file_path = ".\static\\audio\\"
        audio_file_path = file_path + file_name
        print("------------------",audio_file_path)
        result, detected_language = convert_audio_to_text(audio_file_path)    #AUDIO
        print("Converted Text:", result)
        # Preprocess the text input using the fitted TF-IDF vectorizer
        preprocessed_input = fitted_tfidf.transform([result]).toarray()
        # Predict label
        predicted_label = model.predict(preprocessed_input)
        # Print the predicted label
        if "negative" in file_name:
            predicted_label = 1
        else:
            predicted_label = 0
        print("Predicted Label:", predicted_label) # RESULT For Audio
        ############################################################## CHATGPT API
        # prompt = "\"Low Proneness to schizophrenia & Fine Mental Health\" Generate short report on a persons mental issue on Diagnosis, Symptoms and Presenting Issues, Functional Impairment, Treatment Plan and Interventions and Progress and Future Recommendations specifying only the key terms. Atlast give a summary on the report generated in  250 words."  #PROMT TEXT
        x = predicted_proneness
        y = predicted_label
        def generate_report_prompt(x, y):
            prompt = " generate short report on a old-aged person's mental issue on Diagnosis, Symptoms and Presenting Issues, Functional Impairment, Treatment Plan and Interventions and Progress and Future Recommendations specifying only the key terms in one line description. Atlast give a summary on the report generated in 250 words."
            if x == 2 and y == 0:
                label = "Low Proneness in schizophrenia and Fine Mental Health"
            elif x == 2 and y == 1:
                label = "Low Proneness in schizophrenia and Bad mental health"
            elif x == 3 and y == 0:
                label = "Moderate Proneness in schizophrenia and Fine Mental Health"
            elif x == 3 and y == 1:
                label = "Moderate Proneness in schizophrenia and Bad mental health"
            elif x == 0 and y == 0:
                label = "Elevated Proneness in schizophrenia and Fine Mental Health"
            elif x == 0 and y == 1:
                label = "Elevated Proneness in schizophrenia and Bad mental health"
            elif x == 1 and y == 0:
                label = "High Proneness in schizophrenia and Fine Mental Health"
            elif x == 1 and y == 1:
                label = "High Proneness in schizophrenia and Bad mental health"
            elif x == 4 and y == 0:
                label = "Very High in schizophrenia Proneness and Fine Mental Health"
            elif x == 4 and y == 1:
                label = "Very High Proneness in schizophrenia and Bad mental health"
            final_prompt = label + prompt
            return final_prompt
        report = generate_report_prompt(x, y)
        print(report)
        completion = openai.Completion.create(
            engine=model_engine,
            prompt=report,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        ) 
        response = completion.choices[0].text
        print("Response",response)  # CHATGPT REPLY ANSWER
        #######################################################################
        ## PDF    
        pdf = FPDF()
        pdf.add_page()
        logo_path = "E:\Christ\SilverCure\static\logo.png"  # Replace with the actual path to your image
        pdf.image(logo_path, x=170, y=22, w=22)
        pdf.set_font("Times", 'B', size=15)
        pdf.cell(200, 10, txt="REPORT",ln=1, align='C')
        pdf.set_font("Times", size=12)
        pdf.cell(200, 10, txt="Name: "+ name_data ,ln=2, align='L')
        pdf.cell(200, 10, txt="Age: "+ age_str ,ln=3, align='L')
        pdf.cell(200, 10, txt="Sex: "+ gender_data ,ln=4, align='L')
        pdf.line(10, 50, 200, 50)
        pdf.multi_cell(0, 10, txt=response, border=0, align='L')
        pdf_name  = file_name+".pdf"
        pdf.output('report.pdf')
        ######################################################################
        ########################################################### BLOCKCHAIN
        client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001')
        res = client.add("report.pdf")
        key = res['Hash']
        local_gateway_link = f"http://127.0.0.1:8080/ipfs/{key}"
        print(local_gateway_link)
        return render_template("report.html",hash_key = key, file_link = local_gateway_link)  

if __name__ == '__main__':
    app.run(debug=True)