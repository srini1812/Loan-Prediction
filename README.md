# 🏦 HILOANPREDICT: Intelligent Loan Approval Prediction System

![Loan Prediction](https://img.shields.io/badge/AI-Loan%20Prediction-blue)
![Python](https://img.shields.io/badge/Python-3.7%2B-brightgreen)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Ensemble-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

## 🌟 Overview

LoanSage AI is a sophisticated machine learning system that predicts loan approval probabilities using advanced ensemble methods. Our system combines the power of multiple state-of-the-art algorithms to provide accurate and reliable loan approval predictions.

## 🎯 Key Features

- **Multi-Model Ensemble Architecture**
  - CatBoost
  - XGBoost
  - LightGBM
  - Gradient Boosting
  - Hill Climbing Optimization

- **Interactive Web Interface**
  - Real-time predictions
  - Dynamic visualizations
  - User-friendly input forms

- **Comprehensive Analysis**
  - Feature importance visualization
  - Prediction confidence scores
  - Multiple performance metrics

## 🛠️ Technical Stack

- **Machine Learning**: 
  - `catboost`
  - `xgboost`
  - `lightgbm`
  - `scikit-learn`
  - `hillclimbers`

- **Data Processing**:
  - `pandas`
  - `numpy`
  - `scipy`

- **Visualization**:
  - `plotly`
  - `seaborn`
  - `matplotlib`

- **Web Interface**:
  - `streamlit`

## 📊 Model Features

The system considers various factors for prediction:

- Personal Information:
  - Age
  - Income
  - Employment length
  - Home ownership status

- Loan Details:
  - Amount
  - Interest rate
  - Purpose
  - Grade
  - Percent of income

- Credit Information:
  - Default history
  - Credit history length

## 🚀 Getting Started

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/loansage-ai.git
cd loansage-ai
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Application**
```bash
streamlit run app6.py
```

## 📈 Model Performance

Our ensemble approach achieves:
- High accuracy in loan approval prediction
- Robust performance across different customer segments
- Real-time prediction capabilities
- Cross-validated results for reliability

## 💻 Usage Example

```python
from catboost import CatBoostClassifier

# Load the trained model
model = CatBoostClassifier()
model.load_model("catboost_model.cbm")

# Make predictions
prediction = model.predict_proba(user_data)
approval_probability = prediction[0][1]
```

## 🎨 Visualization Examples

The application provides various interactive visualizations:
- Gauge charts for approval probability
- Feature importance treemaps
- Interactive radar charts
- Dynamic bar charts
- Bubble charts for feature analysis

## 🔍 Model Training Process

1. **Data Preprocessing**
   - Handle missing values
   - Encode categorical variables
   - Feature scaling and normalization

2. **Model Training**
   - Cross-validation with 5 folds
   - Hyperparameter optimization using Optuna
   - Ensemble model creation

3. **Model Evaluation**
   - ROC-AUC scoring
   - Cross-validation metrics
   - Feature importance analysis

## 📝 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ✨ Future Enhancements

- [ ] Add API endpoints for model serving
- [ ] Implement batch prediction capabilities
- [ ] Add more visualization options
- [ ] Enhance model interpretability
- [ ] Add support for more languages

## 📧 Contact

For questions and feedback, please reach out to:
- Email: 1812srini@gmail.com

---
*Made with ❤️ by [Srinivas K M]*
