import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import fitz  # PyMuPDF
import docx
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('stopwords')




# Load the dataset
file_path = 'AI_resume_matchmaker_dataset.csv'
data = pd.read_csv(file_path)

# Handling missing values by dropping rows with any NaN values
data_cleaned = data.dropna()

# Combining Resumes and JD into a single feature set in a new DataFrame
combined_text = data_cleaned['Resumes'] + " " + data_cleaned['JD']
data_combined = pd.DataFrame({'text': combined_text, 'Result': data_cleaned['Result']})

# Splitting the data into features and target
X = data_combined['text']
y = data_combined['Result']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# List of models to train
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier()
}

# Vectorization
vectorizer = TfidfVectorizer()

# Dictionary to store the performance results
results = {}

# Training and evaluating each model
for model_name, model in models.items():
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('clf', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']

    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

# Creating a DataFrame for the results
results_df = pd.DataFrame(results).T

# Identifying the best model
best_model_name = results_df['Accuracy'].idxmax()
best_accuracy = results_df.loc[best_model_name, 'Accuracy']

# Train the best model on the full training set
best_pipeline = Pipeline([
    ('tfidf', vectorizer),
    ('clf', models[best_model_name])
])
best_pipeline.fit(X_train, y_train)

def read_file(file_stream, file_type):
    if file_type.lower().endswith('.pdf'):
        text = ""
        document = fitz.open(stream=file_stream.read(), filetype='pdf')
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
    elif file_type.lower().endswith('.docx'):
        document = docx.Document(file_stream)
        text = " ".join([para.text for para in document.paragraphs])
    elif file_type.lower().endswith('.txt'):
        text = file_stream.read().decode('utf-8')
    else:
        raise ValueError("Unsupported file format. Please upload a PDF, DOCX, or TXT file.")
    return text

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

st.set_page_config(page_title="AI Resume Analyzer", page_icon=":robot:", layout="wide")
st.markdown("""
    <style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

.stApp {
    background-color: #121212;
    color: #e0e0e0;
    font-family: 'Inter', sans-serif;
    padding: 10px;
}

.header {
    background: #1f1f1f;
    color: #ffffff;
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    font-size: 2.6em;
    font-weight: 600;
    margin-bottom: 30px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
}

h4 {
    color: #f0f0f0;
    font-weight: 600;
    margin-bottom: 10px;
}

.stFileUploader, .stTextInput {
    background-color: #1e1e1e !important;
    padding: 14px;
    border-radius: 10px;
    box-shadow: inset 0 0 5px rgba(255,255,255,0.05);
    color: #e0e0e0;
}

.stButton button {
    background: linear-gradient(145deg, #1db954, #1aa34a);
    color: white;
    font-weight: bold;
    font-size: 1.1em;
    padding: 12px 30px;
    border-radius: 10px;
    border: none;
    transition: 0.3s;
}

.stButton button:hover {
    background: #14833b;
    transform: scale(1.05);
}

.resizable-image img {
    width: 100%;
    height: auto;
    border-radius: 12px;
    margin-top: 20px;
    box-shadow: 0 0 10px rgba(255,255,255,0.05);
}

/* 🔥 Redesigned Results Card */
.result-card {
    background: #1f1f1f;
    padding: 30px;
    border-radius: 12px;
    margin-top: 30px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.6);
    border-left: 6px solid #1db954;
    transition: 0.3s ease-in-out;
}

.result-card:hover {
    transform: translateY(-2px);
}

.result-card h4 {
    font-size: 1.6em;
    color: #ffffff;
    margin-bottom: 10px;
}

.result-matched {
    color: #1db954;
    font-weight: bold;
    font-size: 1.2em;
}

.result-not-matched {
    color: #e53935;
    font-weight: bold;
    font-size: 1.2em;
}
</style>
""", unsafe_allow_html=True)     

   

st.markdown('<div class="header">AI Resume Match Maker</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h4>Upload Resume</h4>', unsafe_allow_html=True)
    resume_file = st.file_uploader(" ", type=["pdf", "docx", "txt"], key="resume", accept_multiple_files=False, help="Upload the resume in PDF, DOCX, or TXT format")
    st.markdown('<h4>Upload Job Description</h4>', unsafe_allow_html=True)
    jd_file = st.file_uploader(" ", type=["pdf", "docx", "txt"], key="jd", accept_multiple_files=False, help="Upload the job description in PDF, DOCX, or TXT format")
    submit_button = st.button("Submit", key="submit")

with col2:
    st.markdown('<div class="resizable-image">', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if submit_button:
    if not resume_file or not jd_file:
        st.error("Please upload both resume and job description files.")
    elif resume_file.size > 200 * 1024 * 1024 or jd_file.size > 200 * 1024 * 1024:  # Check if file exceeds 200 MB
        st.error("One or both files exceed the 200 MB size limit.")
    else:
        try:
            resume_text = read_file(resume_file, resume_file.name)
            jd_text = read_file(jd_file, jd_file.name)

            resume_text_cleaned = normalize_text(resume_text)
            jd_text_cleaned = normalize_text(jd_text)

            combined_text = resume_text_cleaned + ' ' + jd_text_cleaned

            combined_text_tfidf = best_pipeline.named_steps['tfidf'].transform([combined_text])
            match_score = best_pipeline.named_steps['clf'].predict_proba(combined_text_tfidf)[0][1] * 100

            result_text = f"Matching Score: {match_score:.2f}%"
            match_status = "Resume Matched" if match_score > 50 else "Resume Not Matched"
            match_status_class = "result-matched" if match_score > 50 else "result-not-matched"
            
            st.markdown(f'''
    <div class="result-card">
        <h4>🎯 Matching Score: <span style="color:#1db954;">{match_score:.2f}%</span></h4>
        <p class="{match_status_class}">{match_status}</p>
    </div>
''', unsafe_allow_html=True)

        except ValueError as e:
            st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")



