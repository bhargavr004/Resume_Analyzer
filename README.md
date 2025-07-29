

# 🤖 AI Resume Match Maker

A smart Streamlit-based application that matches resumes with job descriptions using machine learning. Upload a resume and a JD, and let the app tell you how well they align — with a matching score and classification result!

## 🚀 Features

- Upload resumes and job descriptions in PDF, DOCX, or TXT format
- NLP-powered text cleaning and feature extraction
- Model training with multiple classifiers
- Automatic selection of the best-performing model
- Beautiful dark UI with modern styling
- Real-time match score and classification

## 🧠 Models Used

- Naive Bayes
- Logistic Regression
- Random Forest
- SVM
- Decision Tree

The model with the highest accuracy is automatically selected for final prediction.

## 📁 Dataset

The dataset file is: `AI_resume_matchmaker_dataset.csv`  
It contains the following columns:
- `Resumes`: Text of the resumes
- `JD`: Corresponding job descriptions
- `Result`: Label indicating match (1) or no match (0)

## 📁 Project Structure

```
CodeGPT/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .venv      # Environment variables 
└── README.md          # Project documentation
```

## 🛠 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bhargavr004/Resume_Analyzer.git
   cd resume analyzer
````

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## 💻 Running the App

```bash
streamlit run app.py
```

Once the app is running, open the provided local URL in your browser. Upload a resume and job description to get your matching score.


## 📚 Requirements

All dependencies are listed in `requirements.txt`:

```
streamlit
pandas
scikit-learn
nltk
python-docx
PyMuPDF
```

> First-time users: The app will download necessary NLTK data files (`punkt`, `stopwords`).

## 📌 TODO

* Improve model performance with larger dataset
* Add file download of results
* Enable feedback collection for mismatches


Made with ❤️ using Streamlit and scikit-learn.


