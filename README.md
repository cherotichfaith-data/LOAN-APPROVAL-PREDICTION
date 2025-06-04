# LOAN-APPROVAL-PREDICTION

# 🏦 Skies Loan Approval Prediction App

This project is a machine learning-based **Loan Approval Predictor** built with **Streamlit**, allowing users to input financial and personal details to determine whether a loan would be approved. It was developed as part of the **Skies Project** by Faith, Jotham, Valary, Mercy, and Nana.

---

## 📌 Features

* Predicts loan approval using a trained ML model.
* Clean and interactive UI built with [Streamlit](https://streamlit.io).
* Fuzzy matching and label encoding for categorical inputs.
* Displays additional insights like loan-to-income ratio.
* Error handling for robust prediction pipeline.

---

## 🧠 Model Details

* Trained on a labeled dataset of loan applications.
* Utilizes preprocessing steps including:

  * Label encoding for categorical variables
  * Loan percentage of income feature engineering
* Model saved using `pickle` in `loan_approval_model1.pkl`.
* Includes saved label encoders for consistent deployment.

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/loan-approval-app.git
cd loan-approval-app
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit App

```bash
streamlit run app.py
```

---

## 🗃️ Files and Structure

| File/Folder                      | Description                                        |
| -------------------------------- | -------------------------------------------------- |
| `app.py`                         | Main Streamlit application.                        |
| `loan_approval_model1.pkl`       | Trained model and encoders (as dictionary).        |
| `LOAN_APPROVAL_PREDICTION.ipynb` | Jupyter notebook used for training and evaluation. |
| `requirements.txt`               | Python dependencies.                               |
| `README.md`                      | Project documentation (this file).                 |

---

## 📥 Inputs Required

Users must fill in:

* Age, Annual Income, Employment Experience
* Home Ownership, Education, Gender
* Loan Amount, Purpose, Interest Rate
* Credit History Length, Credit Score
* Previous Loan Defaults

---

## ✅ Output

* **Approved:** If the model predicts a loan will be granted.
* **Not Approved:** If the model predicts rejection.
* Shows **Loan as % of income** and handles invalid/missing inputs gracefully.

---

## 👥 Contributors

* Faith Cherotich
* Jotham
* Valary
* Mercy
* Nana

Built with ❤️ by Skies Students, 2025.

---

## 📜 License

This project is for educational and demonstration purposes.

