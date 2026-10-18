# Student Performance Analysis & Prediction System

## What is included
- `data/students.csv`: small illustrative sample dataset (20 rows)
- `analysis.py`: data checks, descriptive statistics, and a scatter plot
- `train_model.py`: trains and evaluates a Linear Regression model
- `app.py`: Streamlit dashboard with dataset overview, charts, and mark estimation
- `requirements.txt`: required Python packages
- `Project_Report.docx`: editable college-project report

## Important limitation
The included dataset is synthetic and very small. This project is for learning and demonstration only. Predictions are not validated for real students and should not be used for consequential decisions.

## Run locally
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (optional but recommended).
4. Install packages: `pip install -r requirements.txt`
5. Start dashboard: `streamlit run app.py`

## iPad workflow
You can edit files and use notebooks on iPad, but Streamlit apps are not normally hosted directly inside a standard Google Colab notebook. For an online dashboard, put this project in a GitHub repository and deploy it through Streamlit Community Cloud. Follow the current instructions on the official Streamlit website. Do not upload private student information.
