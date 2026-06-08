# 🚢 Titanic Survival Prediction Web App

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit 1.28](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io/)
[![Scikit-learn 1.3](https://img.shields.io/badge/Scikit--learn-1.3-orange.svg)](https://scikit-learn.org/)

## 📌 Overview

A machine learning web application that predicts passenger survival chances from the Titanic disaster. Built with **Random Forest Classifier** achieving **84.36% accuracy** and deployed as an interactive **Streamlit** web app.


## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 84.36% |
| **Precision** | 81.94% |
| **Recall** | 79.73% |
| **F1 Score** | 80.82% |

- **Dataset:** 891 passenger records (80/20 train-test split)
- **Features:** 10 engineered attributes
- **Model:** Random Forest Classifier


## 🎯 Features

- ✅ Real-time survival prediction with probability score (0-100%)
- ✅ Interactive web interface with instant predictions
- ✅ Feature importance visualization (what influenced the prediction)
- ✅ Confidence level indicators (High/Moderate/Low)
- ✅ 10 passenger attributes including engineered features


## 🛠️ Tech Stack

| Technology | Version |
|------------|---------|
| Python | 3.11 |
| Streamlit | 1.28 |
| Scikit-learn | 1.3 |
| Pandas | 2.0 |
| NumPy | 1.24 |
| Matplotlib | 3.7 |


## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/umaimafawad711-code/titanic-survival-predictor.git
cd titanic-survival-predictor
2. Create virtual environment (optional but recommended)
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies
bash
pip install -r requirements.txt
4. Run the app
bash
streamlit run app.py
📋 How to Use
Enter passenger details using the input form

Click "PREDICT SURVIVAL" button

View results: Survival prediction, probability score, confidence level, and feature importance chart

📁 Project Structure
text
titanic-survival-predictor/
├── app.py                 # Streamlit web application
├── my_titanic_model.pkl   # Trained Random Forest model
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .gitignore             # Git ignore file
🔍 Feature Engineering
Family Size = SibSp + Parch + 1

Is Alone = 1 if Family Size = 1 else 0

Title extracted from passenger name (Mr, Mrs, Miss, Master)

📈 Model Training Details
Algorithm: Random Forest Classifier

Training/Test Split: 80/20

Cross-validation: 5-fold

Random State: 42 (reproducible results)

👩‍💻 Author
Umaima Fawad

https://img.shields.io/badge/GitHub-umaimafawad711--code-black?logo=github

⭐ Show Your Support
If you found this project helpful, please give it a ⭐ on GitHub!
