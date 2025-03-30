# Diabetes Prediction Model for Women  

## Overview  
This project is predicts diabetes outcomes for women using a machine learning pipeline. The dataset contains health records of **15,000 women aged 20-80** from the **Taipei Municipal Medical Center (2018–2022)**. Features include pregnancies, glucose levels, BMI, and more, with the target variable indicating diabetes diagnosis (**1: diagnosed, 0: not diagnosed**). 
The project is a part of the 'Machine Learning with Python Labs' course, DSTI, 2025. 

## Key Features  
- **Data Analysis**: Explored dataset characteristics, handled duplicates, and assessed outliers.  
- **Feature Engineering**: Identified significant predictors (e.g., SerumInsulin, Glucose) and retained all features due to their statistical relevance.  
- **Model Selection**: **XGBoost** outperformed other models (e.g., GLM, Decision Trees) with **95.6% accuracy**.  
- **Deployment**: Integrated into a **Streamlit web app** for real-time diabetes risk assessment.  

## Usage  
### Resources  
- **Code Instructions**: Refer to the `instructions.md` file for setup and execution details.  
- **Core Functions**: Major utilities are implemented in `util.py` for modularity.  
- **Notebook**: Follow the analysis and model training steps in `Diabetes_Prediction_Insights.ipynb`.  
- **Web App**:  
  - Run locally: `streamlit run app.py`  
  - Access deployed app: [Streamlit App Link](https://mahamadoukeita-dsti-project-diabeties-prediction-in--app-crsm5r.streamlit.app/)  

## Requirements  
- **Python 3.12+**  
- **Libraries**:  
  ```plaintext
  pandas, numpy, scikit-learn, xgboost, optuna, streamlit
  ```
  Install via:  
  ```bash
  pip install -r requirements.txt
  ```

## GitHub  
Explore the full project: [GitHub Repository](https://github.com/AndreiRRR/DSTI-Project-Diabeties-Prediction-In-Women/tree/main)  

## License  
MIT License. See `LICENSE` for details.  
