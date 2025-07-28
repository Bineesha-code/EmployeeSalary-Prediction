import streamlit as st
import pandas as pd
import joblib
import requests

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="💵 Employee Salary Predictor ", layout="wide")
st.title(" Employee Salary Prediction")
st.markdown("Explore Salary Potential Across Borders")

# ---------------------- LOAD MODELS ----------------------
best_gb_model = joblib.load("model_files/best_gb_model.pkl")
le_edu = joblib.load("model_files/le_edu.pkl")
le_emp = joblib.load("model_files/le_emp.pkl")
le_dev = joblib.load("model_files/le_dev.pkl")
le_country = joblib.load("model_files/le_country.pkl")

# ---------------------- CURRENCY API ----------------------
@st.cache_data(show_spinner=False)
def get_exchange_rates():
    url = "https://v6.exchangerate-api.com/v6/7515210001ea1bb24d28e240/latest/USD"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get("conversion_rates", {})
    except Exception:
        return {}

rates = get_exchange_rates()
usd_to_inr = rates.get("INR", 83)  # fallback to 83 if API fails

# ---------------------- LAYOUT ----------------------
col_left, col_right = st.columns([2, 1])

# ---------------------- USER INPUT (LEFT) ----------------------
with col_left:
    st.subheader("📋 Enter Employee Details")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            education = st.selectbox("🎓 Education Level", le_edu.classes_)
            employment = st.selectbox("💼 Employment Type", le_emp.classes_)
            dev_type = st.selectbox("🖥️ Developer Role", le_dev.classes_)
            experience = st.number_input("📈 Years of Experience", min_value=0.0, step=0.5)
        with col2:
            country = st.selectbox("🌐 Country", le_country.classes_)
            orgsize = st.selectbox("🏢 Organization Size", ['Enterprise', 'Large', 'Medium', 'Micro', 'Small'])
            target_currency = st.selectbox("💱 Convert predicted salary to:", sorted(rates.keys()), index=sorted(rates.keys()).index("INR"))

        submitted = st.form_submit_button("🚀 Predict Salary")

    if submitted:
        try:
            # Encode inputs
            edu_enc = le_edu.transform([education])[0]
            emp_enc = le_emp.transform([employment])[0]
            dev_enc = le_dev.transform([dev_type])[0]
            country_enc = le_country.transform([country])[0]

            # Feature engineering
            edu_exp = edu_enc * experience
            country_exp = country_enc * experience

            # One-hot encode OrgSize
            orgsize_features = {
                'OrgSize_Enterprise': 1 if orgsize == 'Enterprise' else 0,
                'OrgSize_Large': 1 if orgsize == 'Large' else 0,
                'OrgSize_Medium': 1 if orgsize == 'Medium' else 0,
                'OrgSize_Micro': 1 if orgsize == 'Micro' else 0,
                'OrgSize_Small': 1 if orgsize == 'Small' else 0
            }

            # Construct dataframe
            new_data = pd.DataFrame([{
                'Education': edu_enc,
                'Employment': emp_enc,
                'DevType': dev_enc,
                'Experience': experience,
                'Country': country_enc,
                'Edu_Exp': edu_exp,
                'Country_Exp': country_exp,
                **orgsize_features
            }])

            # Predict salary in USD
            predicted_usd = best_gb_model.predict(new_data)[0]

            # Convert USD to INR and to other currencies
            predicted_inr = predicted_usd * usd_to_inr
            rate = rates.get(target_currency, 1.0)
            predicted_converted = predicted_usd * rate
            predicted_lakhs = predicted_inr / 1e5

            # Display results
            st.success(f"💰 Predicted Annual Salary: {target_currency} {predicted_converted:,.0f}")
            st.info(f"Equivalent in ₹: ₹{predicted_inr:,.0f} (~ ₹{predicted_lakhs:.2f} Lakhs)")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ---------------------- OUTPUT (RIGHT) ----------------------
with col_right:
    if not submitted:
        st.image("home.png", use_container_width=True)
    else:
        st.subheader("📋 Prediction Summary")

        summary_data = {
            "Education": education,
            "Employment Type": employment,
            "Developer Role": dev_type,
            "Experience (Years)": experience,
            "Country": country,
            "Organization Size": orgsize,
            "Salary (USD)": f"${predicted_usd:,.0f}",
            "Salary (INR)": f"₹{predicted_inr:,.0f}",
            f"Salary ({target_currency})": f"{target_currency} {predicted_converted:,.0f}"
        }

        for title, value in summary_data.items():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**{title}**")
            with col2:
                st.markdown(f"{value}")

        # CSV export
        summary_df = pd.DataFrame([summary_data])
        csv = summary_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Summary as CSV",
            data=csv,
            file_name='salary_prediction_summary.csv',
            mime='text/csv'
        )
