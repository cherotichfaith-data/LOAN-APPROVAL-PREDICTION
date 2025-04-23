import streamlit as st
import pandas as pd
import pickle
import difflib

# Load the model and encoders
with open("loan_approval_model1.pkl", "rb") as f:
    obj = pickle.load(f)
    
# Check if the loaded object is a dictionary with model and encoders
if isinstance(obj, dict):
    model = obj.get('model')
    label_encoders = obj.get('label_encoders', {})
else:
    # If it's just the model directly
    model = obj
    label_encoders = {}  # Create empty dict if no encoders were saved

print(type(obj))

categorical_features = [
    'person_gender',
    'person_education',
    'person_home_ownership',
    'loan_intent',
    'previous_loan_defaults_on_file'
]

# Normalize inputs like 0 → 'No', etc.
value_normalizers = {
    'previous_loan_defaults_on_file': {
        '0': 'No', 0: 'No',
        '1': 'Yes', 1: 'Yes'
    }
}

# Helper: Fuzzy match to training classes
def match_known_value(val, known_values):
    val_clean = str(val).strip().lower()
    matches = difflib.get_close_matches(val_clean, [k.lower() for k in known_values], n=1, cutoff=0.8)
    if matches:
        idx = [k.lower() for k in known_values].index(matches[0])
        return known_values[idx]
    raise ValueError(f"Unrecognized value: '{val}'")

st.set_page_config(page_title="Skies Loan Predictor", page_icon="🏦")
st.markdown("""
    <style>
        h1 {
            font-size: clamp(1.5rem, 5vw, 3rem);
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)
st.title("🏦 Skies Loan Approval Predictor")
st.caption("Built with ❤️ by a Skies Students")
st.markdown("Fill out the form below to get started!")

# User input form
with st.form("loan_form"):
    age = st.number_input("Age", 18, 100, 30)
    income = st.number_input("Annual Income ($)", 1000, 1000000, 50000)
    exp = st.number_input("Employment Experience (years)", 0, 50, 5)
    home = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
    edu = st.selectbox("Education Level", ['High School', 'Bachelor', 'Master', 'Doctorate', 'Associate'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    amount = st.number_input("Loan Amount ($)", 500, 500000, 10000)
    intent = st.selectbox("Loan Purpose", ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
    rate = st.number_input("Interest Rate (%)", 0.0, 100.0, 10.5)
    percent_income = st.number_input("Loan % of Income", 0.0, 1.0, 0.2)
    cred_hist = st.number_input("Credit History Length (years)", 1, 50, 5)
    score = st.number_input("Credit Score", 300, 850, 700)
    defaults = st.selectbox("Previous Loan Defaults on File", ['No', 'Yes'])
    submitted = st.form_submit_button("Check Loan Status")

if submitted:
    new_data = pd.DataFrame([{
        'person_age': age,
        'person_income': income,
        'person_emp_exp': exp,
        'person_home_ownership': home,
        'person_education': edu,
        'person_gender': gender,
        'loan_amnt': amount,
        'loan_intent': intent,
        'loan_int_rate': rate,
        'loan_percent_income': percent_income,
        'cb_person_cred_hist_length': cred_hist,
        'credit_score': score,
        'previous_loan_defaults_on_file': defaults
    }])
    
    # Clean + match categorical
    for col in categorical_features:
        new_data[col] = new_data[col].astype(str).str.strip()
        if col in value_normalizers:
            new_data[col] = new_data[col].apply(lambda v: value_normalizers[col].get(v, v))
        if col in label_encoders:
            new_data[col] = new_data[col].apply(lambda v: match_known_value(v, label_encoders[col].classes_))
            new_data[col] = label_encoders[col].transform(new_data[col])
    
    # Make sure we only include features the model was trained on
    if hasattr(model, 'feature_names_in_'):
        new_data = new_data[model.feature_names_in_]
    
    # Predict
    prediction = model.predict(new_data)[0]
    
    # Display result
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: 14px;">
        © 2025 Skies Project<br>
        Created by: <strong>Faith, Jotham, Valary, Mercy, Nana</strong>
    </div>
    """,
    unsafe_allow_html=True
)
