# AI Flood Predictor

## 1. Title

AI Flood Predictor: Machine Learning-Based Flood Risk Prediction and Disaster Decision Support System

---

## 2. Abstract

AI Flood Predictor is a machine learning-based disaster decision support system that predicts flood risk using environmental and historical data. The system uses rainfall, river level, humidity, temperature, elevation, distance from river, and past flood count to predict flood risk as Low, Medium, High, or Critical.

The project includes live weather data integration using Open-Meteo API, flood risk map visualization, prediction history saving, Supabase database storage, Telegram and email alerts, PDF report generation, Gemini AI explanation, admin dashboard, river station data, and machine learning model comparison.

This system is designed as an AI decision-support prototype and is not an official government flood warning system.

---

## 3. Introduction

Floods are one of the most dangerous natural disasters. They can damage houses, roads, farms, and public infrastructure. Early flood risk prediction can help people and disaster management teams prepare before the situation becomes dangerous.

This project uses artificial intelligence and machine learning to predict flood risk from weather and environmental data. The system provides a simple dashboard where users can search a location, get live weather data, select river station data, and predict flood risk.

---

## 4. Problem Statement

Many places face floods due to heavy rainfall, rising river levels, low elevation, and poor drainage. People may not understand flood risk early because data is spread across different sources.

The main problem is to create a simple AI-based system that can analyze flood-related data and provide a clear flood risk level.

---

## 5. Objectives

The main objectives of this project are:

- To collect rainfall, weather, river level, and historical flood-related data.
- To train a machine learning model for flood risk prediction.
- To classify flood risk as Low, Medium, High, or Critical.
- To show live weather data using Open-Meteo API.
- To display flood risk on a map.
- To save prediction history locally and in Supabase.
- To send Telegram and email alerts for High and Critical risk.
- To generate downloadable PDF flood risk reports.
- To provide Gemini AI explanation for prediction results.
- To create an admin dashboard for monitoring predictions.
- To compare multiple ML models and select the best model.

---

## 6. Existing System

In many cases, flood alerts are provided through government disaster management systems, weather departments, or river monitoring agencies. These systems may be official and accurate, but they are sometimes difficult for normal users to understand.

Some systems only show rainfall or river data separately. Users may need to manually compare different values to understand flood risk.

---

## 7. Proposed System

The proposed system is an AI-based flood prediction dashboard. It combines weather data, river-level data, machine learning prediction, map visualization, alert system, database storage, AI explanation, and report generation into one application.

The user can enter or search a location, load live weather data, select a river station, and predict flood risk. The system then shows the risk level, confidence score, explanation, map, safety suggestion, and downloadable report.

---

## 8. System Architecture

The system architecture follows this flow:

```text
User
 ↓
Streamlit Dashboard
 ↓
Location Search using Open-Meteo Geocoding API
 ↓
Live Weather Data using Open-Meteo Forecast API
 ↓
River Level Station Data
 ↓
Machine Learning Model
 ↓
Flood Risk Prediction
 ↓
Map + Alerts + PDF Report + Supabase Storage + Admin Dashboard
```

---

## 9. Technologies Used

| Category             | Technology                   |
| -------------------- | ---------------------------- |
| Programming Language | Python                       |
| Dashboard            | Streamlit                    |
| Data Processing      | Pandas, NumPy                |
| Machine Learning     | Scikit-learn                 |
| Model Saving         | Joblib                       |
| Weather API          | Open-Meteo API               |
| Location Search      | Open-Meteo Geocoding API     |
| Map                  | Folium, Streamlit-Folium     |
| Database             | Supabase                     |
| Alerts               | Telegram Bot API, Email SMTP |
| PDF Report           | ReportLab                    |
| AI Explanation       | Gemini API                   |
| Deployment           | Streamlit Community Cloud    |

---

## 10. Dataset Description

The dataset contains flood-related input features and a target output column.

### Input Features

| Feature           | Description                       |
| ----------------- | --------------------------------- |
| rainfall_24h      | Rainfall in the last 24 hours     |
| rainfall_3d       | Rainfall in the last 3 days       |
| river_level       | Current river water level         |
| humidity          | Air humidity percentage           |
| temperature       | Temperature in Celsius            |
| elevation         | Height from sea level             |
| distance_to_river | Distance from river in kilometers |
| past_flood_count  | Number of previous flood events   |

### Output Feature

| Column     | Description                                      |
| ---------- | ------------------------------------------------ |
| risk_level | Flood risk category: Low, Medium, High, Critical |

---

## 11. Machine Learning Model

The system compares multiple machine learning models:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* K-Nearest Neighbors

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

The best model is selected based on F1 Score and saved as:

```text
models/flood_model.pkl
```

The Streamlit dashboard loads this saved model to make flood risk predictions.

---

## 12. Implementation Modules

### Module 1: Data Collection

The system uses CSV data, Open-Meteo weather data, and river station data.

### Module 2: Data Preprocessing

The dataset is cleaned by checking missing values, removing duplicates, and separating input/output columns.

### Module 3: Model Training

Multiple ML models are trained and compared. The best model is saved using Joblib.

### Module 4: Streamlit Dashboard

The dashboard allows users to enter data, search locations, load weather data, select river stations, and predict flood risk.

### Module 5: Live Weather API

Open-Meteo API is used to fetch live temperature, humidity, and rainfall data.

### Module 6: Flood Risk Map

Folium map shows the selected location and flood risk marker.

### Module 7: Prediction History

Predictions are saved in local CSV and Supabase database.

### Module 8: Alerts

Telegram and email alerts are sent when flood risk is High or Critical.

### Module 9: PDF Report

The system generates a downloadable PDF flood risk report.

### Module 10: Gemini AI Explanation

Gemini API explains the prediction result in simple language.

### Module 11: Admin Dashboard

The admin dashboard shows total predictions, risk counts, high-risk records, charts, and downloadable admin data.

### Module 12: Model Comparison

Different machine learning models are trained and compared using evaluation metrics. The best model is selected automatically.

---

## 13. Output Screens

The project output includes:

* AI Flood Predictor dashboard
* Location search section
* Live weather data section
* River station selection
* Input data table
* Flood risk prediction result
* Gemini AI explanation
* Flood risk map
* Prediction history table
* Analytics dashboard
* Admin dashboard
* PDF report download
* Telegram/email alert confirmation
* Supabase saved prediction table
* Model comparison table

---

## 14. Result and Discussion

The system successfully predicts flood risk based on environmental data. It classifies risk into Low, Medium, High, and Critical levels.

The model comparison feature helps identify the best-performing ML algorithm. The dashboard makes the prediction easy to understand by showing charts, maps, explanation, and reports.

The system also improves disaster decision support by sending alerts for dangerous risk levels.

---

## 15. Advantages

* Easy-to-use dashboard
* Uses live weather data
* Supports location search
* Provides flood risk map
* Saves prediction history
* Supports Supabase online database
* Sends Telegram and email alerts
* Generates PDF report
* Provides AI explanation
* Includes admin monitoring dashboard
* Compares multiple ML models
* Supports river station data

---

## 16. Limitations

* The current dataset is a prototype dataset.
* Real-time river-level data is currently sample/station CSV based.
* The model accuracy depends on dataset quality.
* This system is not an official government warning system.
* For real-world use, official hydrological and disaster data must be integrated.
* Flood prediction requires high-quality real-time data for accurate public safety usage.

---

## 17. Future Scope

Future improvements can include:

* Real official river-level API integration
* Satellite rainfall data integration
* Flood zone map layers
* Evacuation route suggestion
* Shelter location recommendation
* Mobile app version
* SMS alert system
* More advanced ML/deep learning models
* Automatic model retraining
* Multi-user login system
* Government disaster API integration
* IoT-based water-level sensor integration

---

## 18. Conclusion

AI Flood Predictor is a machine learning-based flood risk prediction and disaster decision support system. It combines weather data, river data, ML prediction, map visualization, alerts, reports, AI explanation, and admin monitoring.

The system helps users understand flood risk in a simple way and supports early decision-making. It is a useful prototype for flood risk awareness and disaster management support.

Important note: This project is only an AI decision-support prototype and not an official government flood warning system.
