# AI Flood Predictor

AI Flood Predictor is a machine learning-based flood risk prediction dashboard.

## Project Description

This project predicts flood risk using rainfall, river level, humidity, temperature, elevation, distance from river, and past flood history.

The system uses a trained Random Forest machine learning model and provides flood risk levels:

- Low
- Medium
- High
- Critical

## Features

- Machine learning flood risk prediction
- Open-Meteo live weather API
- Automatic rainfall, humidity, and temperature loading
- Flood risk map view
- Prediction history saving
- Analytics dashboard
- Telegram alert for High and Critical risk
- Downloadable PDF flood risk report

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Open-Meteo API
- Folium
- Streamlit-Folium
- ReportLab
- Telegram Bot API

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## Important Files

```text
app.py
requirements.txt
data/flood_data.csv
models/flood_model.pkl
```

## Deployment

This project can be deployed using Streamlit Community Cloud.

Deployment steps:

1. Upload project to GitHub.
2. Open Streamlit Community Cloud.
3. Click New app.
4. Select GitHub repository.
5. Select branch: main.
6. Main file path: app.py.
7. Add secrets in Advanced settings.
8. Click Deploy.

## Streamlit Secrets

Do not upload `.streamlit/secrets.toml` to GitHub.

For Telegram alerts, add these secrets in Streamlit Cloud Advanced Settings:

```toml
TELEGRAM_BOT_TOKEN = "your_real_bot_token"
TELEGRAM_CHAT_ID = "your_real_chat_id"
```

## Safety Note

This is an AI decision-support prototype, not an official government flood warning system.

Always follow official alerts from disaster management authorities.

## Project Report

Full project report is available in:

```text
PROJECT_REPORT.md
```

