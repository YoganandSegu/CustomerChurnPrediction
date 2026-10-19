# Customer Churn Prediction

This Streamlit app estimates whether a telecommunications customer is likely to
leave the company using a trained machine learning model.

## Requirements

- Python 3.13 (recommended; use a Python version supported by the dependencies)
- The packages listed in [`requirements.txt`](requirements.txt)
- The trained model bundle, [`customer_churn_bundle.joblib`](customer_churn_bundle.joblib)

The model bundle was serialized with scikit-learn 1.6.1. Keep
`scikit-learn==1.6.1` in `requirements.txt` when loading this artifact; loading
scikit-learn models with a different version can fail or produce unreliable
results.

## Setup and run

From the project directory, create and activate a virtual environment, then
install the dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the app:

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally <http://localhost:8501>) in
your browser.

The model bundle must remain in the project directory beside `app.py`.

## Using the app

Enter the customer's account and service details, then select **Predict
Customer Churn**. The app displays the estimated churn probability and
classifies the customer using the threshold stored in the model bundle.
