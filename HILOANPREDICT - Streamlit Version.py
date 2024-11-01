import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import plotly.graph_objects as go
import plotly.express as px

# Load the trained CatBoost model
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    return model

model = load_model()

# Define categorical columns
categorical_columns = ['person_age', 'person_income', 'person_home_ownership', 'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_default_on_file', 'cb_person_cred_hist_length']

st.title("Loan Approval Prediction App 💼💰")
st.markdown("""
This app predicts the probability of loan approval based on various factors.
Fill in the form below to get your loan approval prediction!
""")

# Create input form
st.header("Applicant Information")
col1, col2 = st.columns(2)
with col1:
    person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
    person_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
    person_home_ownership = st.selectbox(
        "Home Ownership",
        options=["RENT", "MORTGAGE", "OWN", "OTHER"]
    )
    person_emp_length = st.number_input("Employment Length (years)", min_value=0.0, max_value=50.0, value=5.0)
    loan_intent = st.selectbox(
        "Loan Intent",
        options=["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOME_IMPROVEMENT", "DEBT_CONSOLIDATION"]
    )
with col2:
    loan_grade = st.selectbox("Loan Grade", options=["A", "B", "C", "D", "E", "F", "G"])
    loan_amnt = st.number_input("Loan Amount ($)", min_value=1000, max_value=1000000, value=10000)
    loan_int_rate = st.slider("Loan Interest Rate (%)", min_value=1.0, max_value=30.0, value=10.0)
    loan_percent_income = st.slider("Loan Percent Income", min_value=0.0, max_value=1.0, value=0.1)
    cb_person_default_on_file = st.selectbox("Has the person defaulted before?", options=["Y", "N"])
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=5)

# Prediction button
if st.button("Predict Loan Approval"):
    # Prepare user input for prediction
    user_data = pd.DataFrame({
        'person_age': [str(person_age)],
        'person_income': [str(person_income)],
        'person_home_ownership': [person_home_ownership],
        'person_emp_length': [str(person_emp_length)],
        'loan_intent': [loan_intent],
        'loan_grade': [loan_grade],
        'loan_amnt': [str(loan_amnt)],
        'loan_int_rate': [str(loan_int_rate)],
        'loan_percent_income': [str(loan_percent_income)],
        'cb_person_default_on_file': [cb_person_default_on_file],
        'cb_person_cred_hist_length': [str(cb_person_cred_hist_length)]
    })

    # Convert all columns to strings (categorical)
    for col in categorical_columns:
        user_data[col] = user_data[col].astype(str)

    # Create CatBoost Pool with all columns as categorical features
    pool = Pool(user_data, cat_features=categorical_columns)

    try:
        # Make prediction
        approval_probability = model.predict_proba(pool)[0][1]

        # Create a gauge chart for the approval probability
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=approval_probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Approval Probability", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': 'red'},
                    {'range': [50, 75], 'color': 'yellow'},
                    {'range': [75, 100], 'color': 'green'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50}}))

        st.plotly_chart(fig)

        # Display result
        st.subheader("Loan Approval Prediction")
        if approval_probability > 0.5:
            st.success(f"Congratulations! Your loan is likely to be approved with a {approval_probability:.2%} probability.")
        else:
            st.error(f"We're sorry, but your loan is likely to be denied. The approval probability is {approval_probability:.2%}.")

        # Display important factors
        st.subheader("Important Factors")
        feature_importance = model.get_feature_importance(type='ShapValues', data=pool)
        feature_importance_df = pd.DataFrame({
            'Feature': user_data.columns,
            'Importance': np.abs(feature_importance[0]).mean(axis=0)
        }).sort_values('Importance', ascending=False)

        # 1. Horizontal Bar Chart
        fig1 = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h',
                      title="Feature Importance - Bar Chart",
                      labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'},
                      color='Importance', color_continuous_scale='Viridis')
        fig1.update_layout(coloraxis_colorbar=dict(title="Importance"))
        st.plotly_chart(fig1)

        # 2. Treemap
        fig2 = px.treemap(feature_importance_df, path=['Feature'], values='Importance',
                          title="Feature Importance - Treemap",
                          color='Importance', color_continuous_scale='Turbo')
        fig2.update_layout(coloraxis_colorbar=dict(title="Importance"))
        st.plotly_chart(fig2)

        # 3. Radar Chart
        top_5_features = feature_importance_df.head()
        fig3 = go.Figure()
        fig3.add_trace(go.Scatterpolar(
            r=top_5_features['Importance'],
            theta=top_5_features['Feature'],
            fill='toself',
            line=dict(color='rgba(255, 0, 0, 0.8)'),
            fillcolor='rgba(255, 0, 0, 0.3)'
        ))
        fig3.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, top_5_features['Importance'].max()])),
            showlegend=False,
            title="Top 5 Feature Importance - Radar Chart"
        )
        st.plotly_chart(fig3)

        # 4. Bubble Chart
        fig4 = px.scatter(feature_importance_df, x="Feature", y="Importance", 
                          size="Importance", color="Feature", 
                          hover_name="Feature", size_max=60, 
                          title="Feature Importance - Bubble Chart",
                          color_discrete_sequence=px.colors.qualitative.Bold)
        fig4.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig4)

        # 5. Animated Bar Chart
        max_importance = feature_importance_df['Importance'].max()
        feature_importance_df['Importance'] = feature_importance_df['Importance'] / max_importance * 2 - 1

        fig5 = px.bar(feature_importance_df, x="Importance", y="Feature", orientation='h',
                      title="Feature Importance - Animated Bar Chart",
                      animation_frame="Feature", animation_group="Feature",
                      range_x=[-1, 1],
                      color="Feature", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig5.update_layout(showlegend=False, xaxis_title="Importance Score", yaxis_title="Feature")
        st.plotly_chart(fig5)

    except Exception as e:
        st.error(f"An error occurred while making the prediction: {str(e)}")
        st.error("Please check if the model file 'catboost_model.cbm' is present in the correct location.")

st.markdown("""
### How to interpret the results:
- A probability above 50% suggests a higher likelihood of loan approval.
- The gauge chart visually represents the approval probability.
- The feature importance charts show which factors had the most impact on the prediction.

Please note that this is a predictive model and the actual loan approval decision may involve additional factors and human judgment.
""")
