# app.py - Titanic Survival Prediction Web App
# Is code ko copy-paste karein VS Code mein

import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set page configuration (yeh pehle aana chahiye)
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# Load the trained model
@st.cache_resource  # Yeh model ko sirf ek baar load karega (performance ke liye)
def load_model():
    with open('my_titanic_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error("❌ my_titanic_model.pkl file not found! Please make sure it's in the same directory.")
    st.stop()

# App title and description
st.title("🚢 Titanic Survival Prediction Web App")
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.write("""
    ### Enter passenger details below to predict if they would have survived the Titanic disaster.
    The model uses a Random Forest classifier trained on historical Titanic data.
""")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Passenger Information")
    
    # Passenger Class
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3],
        format_func=lambda x: {1: "1st Class (Upper)", 2: "2nd Class (Middle)", 3: "3rd Class (Lower)"}[x]
    )
    
    # Sex
    sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
    
    # Age
    age = st.slider("Age", 0, 100, 30, help="Age in years")
    
    # SibSp
    sibsp = st.number_input("Siblings/Spouses Aboard", 0, 8, 0, 
                            help="Number of siblings or spouses aboard")
    
    # Parch
    parch = st.number_input("Parents/Children Aboard", 0, 6, 0,
                           help="Number of parents or children aboard")
    
    # Fare
    fare = st.number_input("Fare Amount ($)", 0, 500, 32, step=5,
                          help="Ticket fare paid")

with col2:
    st.subheader("📍 Additional Information")
    
    # Embarked
    embarked = st.selectbox(
        "Port of Embarkation",
        ["C", "Q", "S"],
        format_func=lambda x: {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}[x]
    )
    
    # Title (simplified for demo)
    title = st.selectbox(
        "Title",
        ["Mr", "Mrs", "Miss", "Master", "Other"],
        help="Title extracted from passenger's name"
    )
    
    # Feature engineering display
    st.subheader("📊 Calculated Features")
    
    # Calculate derived features
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    
    st.metric("Family Size", family_size)
    st.metric("Traveling Alone", "Yes" if is_alone == 1 else "No")

# Add a separator
st.divider()

# Feature engineering function (MUST match training!)
def preprocess_data(pclass, sex, age, sibsp, parch, fare, embarked, title):
    """Preprocess input data exactly as during training"""
    
    # Encode sex
    sex_encoded = 1 if sex == "Male" else 0
    
    # Encode embarked
    embarked_mapping = {"C": 0, "Q": 1, "S": 2}
    embarked_encoded = embarked_mapping[embarked]
    
    # Encode title
    title_mapping = {"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3, "Other": 4}
    title_encoded = title_mapping[title]
    
    # Feature engineering
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    
    # Create DataFrame with exact column names as training
    data = pd.DataFrame([[
        pclass, sex_encoded, age, sibsp, parch, fare, 
        embarked_encoded, family_size, is_alone, title_encoded
    ]], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                 'Embarked', 'FamilySize', 'IsAlone', 'Title'])
    
    return data

# Input summary
with st.expander("📝 View Input Summary"):
    st.write(f"""
    - **Class:** {pclass} ({'Upper' if pclass==1 else 'Middle' if pclass==2 else 'Lower'})
    - **Sex:** {sex}
    - **Age:** {age} years
    - **Family:** {sibsp} siblings/spouses, {parch} parents/children
    - **Fare:** ${fare}
    - **Embarked:** {embarked}
    - **Title:** {title}
    - **Family Size:** {family_size}
    - **Traveling Alone:** {'Yes' if is_alone == 1 else 'No'}
    """)

# Prediction button
col_button1, col_button2, col_button3 = st.columns([1, 2, 1])
with col_button2:
    predict_button = st.button("🔮 PREDICT SURVIVAL", type="primary", use_container_width=True)

if predict_button:
    # Preprocess input
    input_data = preprocess_data(pclass, sex, age, sibsp, parch, fare, embarked, title)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    # Display result with animation effect
    st.divider()
    
    col_result1, col_result2 = st.columns(2)
    
    with col_result1:
        if prediction == 1:
            st.success("### ✅ **PASSENGER WOULD SURVIVE!**")
            st.balloons()  # Celebration effect!
        else:
            st.error("### ❌ **PASSENGER WOULD NOT SURVIVE**")
    
    with col_result2:
        st.metric("Survival Probability", f"{probability:.1%}")
        
        # Add a progress bar for probability
        st.progress(probability)
        
        # Show confidence level
        if probability >= 0.7:
            st.info("📊 High confidence prediction")
        elif probability >= 0.4:
            st.warning("📊 Moderate confidence prediction")
        else:
            st.info("📊 Low confidence prediction")
    
    # Show feature importance (what influenced the decision)
    st.subheader("🔍 What Influenced This Prediction?")
    
    # Get feature importance from model
    feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                     'Embarked', 'FamilySize', 'IsAlone', 'Title']
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Display as bar chart
    st.bar_chart(importance_df.set_index('Feature'))
    
    # Show explanation
    top_feature = importance_df.iloc[0]['Feature']
    st.info(f"💡 **Most important factor:** {top_feature} had the biggest impact on this prediction.")
    
    # Add a tip for better predictions
    st.caption("💡 Tip: Try changing passenger class or fare to see how prediction changes!")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About This App")
    st.markdown("""
    - **Model:** Random Forest Classifier
    - **Accuracy:** ~80-85% on test data
    - **Features used:** 10 passenger attributes
    - **Training data:** Titanic passenger list
    
    ### 📊 Feature Legend
    - **Pclass:** Passenger class (1st = Upper, 2nd = Middle, 3rd = Lower)
    - **SibSp:** # of siblings/spouses aboard
    - **Parch:** # of parents/children aboard
    - **Embarked:** Port where passenger boarded
    
    ### 🚀 Made with
    - Streamlit
    - Scikit-learn
    - Pandas
    """)
    
    st.divider()
    
    # Sample test data
    st.subheader("🎯 Try These Examples")
    
    if st.button("📌 Rich Female Passenger"):
        st.session_state['pclass'] = 1
        st.session_state['sex'] = "Female"
        st.session_state['age'] = 25
        st.session_state['sibsp'] = 1
        st.session_state['parch'] = 0
        st.session_state['fare'] = 100
        st.session_state['embarked'] = "C"
        st.session_state['title'] = "Mrs"
        st.rerun()
    
    if st.button("📌 Poor Male Passenger"):
        st.session_state['pclass'] = 3
        st.session_state['sex'] = "Male"
        st.session_state['age'] = 35
        st.session_state['sibsp'] = 0
        st.session_state['parch'] = 0
        st.session_state['fare'] = 10
        st.session_state['embarked'] = "S"
        st.session_state['title'] = "Mr"
        st.rerun()

# Footer
st.divider()
st.caption("🚢 Titanic Survival Prediction App | Built with Streamlit | Model trained on Kaggle Titanic dataset")