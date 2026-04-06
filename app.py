import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
# ------------------------------------------------
# TITLE
# ------------------------------------------------
st.title("📈 Loan Default Risk Predictor")
st.markdown("Predict the **risk score** (0–100) of a loan applicant defaulting.")
# ------------------------------------------------
# DATASET
# ------------------------------------------------
data = {
    "ApplicantID":    [f"APP{str(i).zfill(3)}" for i in range(1, 31)],
    "Age":            [28,35,42,24,55,38,30,47,26,33,
                       45,29,52,37,23,41,34,27,48,31,
                       39,25,56,36,22,44,32,50,40,29],
    "AnnualIncome_L": [4.5,8.2,12.0,3.0,18.5,9.0,5.5,15.0,3.8,7.2,
                       13.5,4.2,20.0,8.8,2.8,11.0,7.8,4.0,16.5,6.0,
                       9.5,3.2,22.0,8.5,2.5,12.5,7.0,17.5,10.0,5.0],
    "LoanAmount_L":   [2.0,4.0,6.0,1.5,8.0,4.5,2.5,7.0,2.0,3.5,
                       6.5,2.2,9.0,4.2,1.8,5.5,3.8,2.0,7.5,3.0,
                       4.8,1.6,10.0,4.0,1.2,6.0,3.5,8.0,5.0,2.5],
    "CreditScore":    [720,680,760,620,800,700,650,780,610,690,
                       750,630,820,710,600,770,680,640,790,660,
                       700,615,830,695,595,775,670,800,720,645],
    "ExistingLoans":  [1,2,0,2,0,1,3,0,2,1,
                       0,2,0,1,3,0,1,2,0,2,
                       1,3,0,1,2,0,2,0,1,2],
    "EmploymentYears":[3,8,15,1,25,10,4,18,2,6,
                       14,2,22,9,1,12,7,3,20,5,
                       8,1,28,9,0,13,6,19,11,3],
    "DefaultRiskScore":[35,28,15,65,10,30,72,12,70,32,
                        18,68,8,25,80,14,35,62,10,50,
                        27,75,5,28,85,12,40,9,22,58]
}
df = pd.DataFrame(data)
st.subheader("🏦 Loan Applicant Dataset")
st.dataframe(df)
# ------------------------------------------------
# PREPROCESSING
# ------------------------------------------------
st.subheader("🔧 Data Preprocessing")
st.write("**Missing Values:**")
st.write(df.isnull().sum())
st.write("**Duplicate Rows:**", df.duplicated().sum())
df = df.drop_duplicates()
df = df.drop(["ApplicantID"], axis=1)
st.write("**Cleaned Data:**")
st.dataframe(df)
# ------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------
st.subheader("⚙️ Feature Engineering")
# Debt-to-Income Ratio
df["DebtToIncome"] = df["LoanAmount_L"] / df["AnnualIncome_L"]
# Loan Burden Index = Loan * ExistingLoans
df["LoanBurden"] = df["LoanAmount_L"] * (df["ExistingLoans"] + 1)
# Credit Health (normalized)
df["CreditHealth"] = (df["CreditScore"] - 300) / 550  # score range 300-850
st.write("**New Features:** `DebtToIncome`, `LoanBurden`, `CreditHealth`")
st.dataframe(df[["DebtToIncome","LoanBurden","CreditHealth","DefaultRiskScore"]].head(10))
# ------------------------------------------------
# VISUALIZATION
# ------------------------------------------------
st.subheader("📊 Visualizations")
col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots()
    ax1.scatter(df["CreditScore"], df["DefaultRiskScore"], color="#e74c3c", alpha=0.8)
    ax1.set_xlabel("Credit Score")
    ax1.set_ylabel("Default Risk Score")
    ax1.set_title("Credit Score vs Risk")
    st.pyplot(fig1)
with col2:
    fig2, ax2 = plt.subplots()
    ax2.scatter(df["AnnualIncome_L"], df["DefaultRiskScore"], color="#2ecc71", alpha=0.8)
    ax2.set_xlabel("Annual Income (Lakhs)")
    ax2.set_ylabel("Default Risk Score")
    ax2.set_title("Income vs Risk")
    st.pyplot(fig2)
fig3, ax3 = plt.subplots()
ax3.scatter(df["DebtToIncome"], df["DefaultRiskScore"], color="#e67e22", alpha=0.8)
ax3.set_xlabel("Debt-to-Income Ratio")
ax3.set_ylabel("Default Risk Score")
ax3.set_title("Debt Ratio vs Risk")
st.pyplot(fig3)
loan_groups = df.groupby("ExistingLoans")["DefaultRiskScore"].mean()
fig4, ax4 = plt.subplots()
ax4.bar(loan_groups.index.astype(str), loan_groups.values,
        color=["#2ecc71","#f39c12","#e74c3c","#8e44ad"])
ax4.set_xlabel("Existing Loans Count")
ax4.set_ylabel("Avg Default Risk Score")
ax4.set_title("Existing Loans vs Avg Risk")
st.pyplot(fig4)
st.subheader("🔗 Correlation Matrix")
st.dataframe(df.corr(numeric_only=True))
# ------------------------------------------------
# OUTLIER DETECTION
# ------------------------------------------------
st.subheader("📦 Outlier Detection")
fig5, ax5 = plt.subplots()
ax5.boxplot([df["DefaultRiskScore"], df["CreditScore"]/10],
            labels=["Risk Score", "CreditScore (÷10)"])
ax5.set_title("Outlier Check")
st.pyplot(fig5)
# ------------------------------------------------
# MODEL TRAINING
# ------------------------------------------------
st.subheader("🤖 Model Training")
X = df.drop("DefaultRiskScore", axis=1)
y = df["DefaultRiskScore"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=99)
st.write(f"Training: **{len(X_train)}** rows | Testing: **{len(X_test)}** rows")
model = LinearRegression()
model.fit(X_train, y_train)
st.success("✅ Model Trained Successfully!")
# ------------------------------------------------
# EVALUATION
# ------------------------------------------------
st.subheader("📈 Model Evaluation")
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
col1, col2, col3 = st.columns(3)
col1.metric("MAE",  f"{mae:.2f}")
col2.metric("MSE",  f"{mse:.2f}")
col3.metric("RMSE", f"{rmse:.2f}")
fig6, ax6 = plt.subplots()
ax6.barh(range(len(y_test)),
         [abs(a - p) for a, p in zip(y_test.values, y_pred)],
         color="#3498db")
ax6.set_xlabel("Absolute Error")
ax6.set_title("Prediction Error per Test Sample")
st.pyplot(fig6)
# ------------------------------------------------
# USER INPUT & PREDICTION
# ------------------------------------------------
st.subheader("🎯 Predict Loan Default Risk")
col1, col2 = st.columns(2)
with col1:
    age            = st.slider("Applicant Age",             18, 65, 30)
    income         = st.slider("Annual Income (Lakhs)",     1.0, 30.0, 6.0)
    loan_amt       = st.slider("Loan Amount (Lakhs)",       0.5, 15.0, 3.0)
with col2:
    credit_score   = st.slider("Credit Score",              300, 850, 680)
    existing_loans = st.slider("Existing Loans",            0,   5,   1)
    emp_years      = st.slider("Employment Years",          0,   35,  5)
if st.button("📊 Predict Risk Score"):
    dti           = loan_amt / income
    loan_burden   = loan_amt * (existing_loans + 1)
    credit_health = (credit_score - 300) / 550
    input_df = pd.DataFrame([[age, income, loan_amt, credit_score,
                               existing_loans, emp_years,
                               dti, loan_burden, credit_health]],
                            columns=X.columns)
    pred = model.predict(input_df)[0]
    pred = np.clip(pred, 0, 100)
    st.success(f"⚠️ Predicted Default Risk Score: **{pred:.1f} / 100**")
    if pred < 30:
        st.success("✅ Low Risk — Loan likely to be approved")
    elif pred < 60:
        st.warning("🟡 Moderate Risk — Further review recommended")
    else:
        st.error("🔴 High Risk — Loan likely to be rejected")