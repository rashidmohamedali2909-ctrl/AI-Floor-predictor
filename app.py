import streamlit as st
import pandas as pd
import joblib
import os
import requests
import folium
from streamlit_folium import st_folium
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from supabase import create_client
from google import genai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import base64
import logging

# Helper to read secrets safely and avoid crashes if config is missing
def get_secret_safe(key, default=""):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

def is_supabase_configured():
    url = get_secret_safe("SUPABASE_URL", "")
    key = get_secret_safe("SUPABASE_KEY", "")
    if not url or not key or "paste_your" in url or "paste_your" in key or url.strip() == "" or key.strip() == "":
        return False
    return True

# Page setup
st.set_page_config(
    page_title="AI Flood Predictor",
    page_icon="🌊",
    layout="wide"
)

def set_water_background(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()

        page_bg_img = f"""
        <style>
        .stApp {{
            background-image: linear-gradient(
                rgba(5, 18, 32, 0.50),
                rgba(5, 18, 32, 0.65)
            ), url("data:image/jpg;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        [data-testid="stSidebar"] {{
            background: rgba(8, 20, 35, 0.78);
            backdrop-filter: blur(12px);
            border-right: 1px solid rgba(255, 255, 255, 0.12);
        }}

        [data-testid="stHeader"] {{
            background: rgba(0, 0, 0, 0);
        }}

        .block-container {{
            background: rgba(6, 18, 34, 0.72);
            backdrop-filter: blur(10px);
            border-radius: 22px;
            padding: 2rem;
            margin-top: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35);
        }}

        h1, h2, h3, h4, h5, h6, p, label, span {{
            color: #f4fbff !important;
        }}

        .stAlert {{
            border-radius: 14px;
        }}

        .stButton > button {{
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.25);
            background: linear-gradient(135deg, #0284c7, #06b6d4);
            color: white;
            font-weight: 700;
            transition: 0.2s ease-in-out;
        }}

        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 8px 24px rgba(6, 182, 212, 0.35);
        }}

        .stDownloadButton > button {{
            border-radius: 12px;
            background: linear-gradient(135deg, #0ea5e9, #22d3ee);
            color: white;
            font-weight: 700;
        }}

        [data-testid="stMetricValue"] {{
            color: #7dd3fc !important;
        }}

        [data-testid="stDataFrame"] {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
        }}

        input, textarea {{
            background-color: rgba(8, 20, 35, 0.85) !important;
            color: white !important;
            border-radius: 10px !important;
        }}

        .stSelectbox div[data-baseweb="select"] > div {{
            background-color: rgba(8, 20, 35, 0.85) !important;
            color: white !important;
            border-radius: 10px !important;
        }}
        </style>
        """

        st.markdown(page_bg_img, unsafe_allow_html=True)

    except FileNotFoundError:
        st.warning("Background image not found. Please add assets/water_wave_background.jpg")

set_water_background("assets/water_wave_background.jpg")

st.title("🌊 AI Flood Predictor")
st.write(
    "Machine Learning based flood risk prediction dashboard with live weather API, map view, alerts, reports, AI explanation, and admin monitoring."
)

st.warning(
    "This is an AI decision-support prototype, not an official government flood warning system."
)

st.info(
    "Alerts System: Telegram and Email alerts are sent only when flood risk is High or Critical."
)

# Check if trained model exists
model_path = "models/flood_model.pkl"

if not os.path.exists(model_path):
    st.error("Model file not found. Please run: python train_model.py")
    st.stop()

# Load trained ML model
model = joblib.load(model_path)
st.success("Trained ML model loaded successfully.")

# Display best model info if available
best_model_info_path = "models/best_model_info.json"

if os.path.exists(best_model_info_path):
    with open(best_model_info_path, "r") as file:
        best_model_info = json.load(file)

    st.info(
        f"🏆 Best ML Model: {best_model_info['best_model_name']} "
        f"| F1 Score: {best_model_info['best_f1_score']}"
    )


# Function to load river level CSV station data
def load_river_level_data():
    river_file = "data/river_levels.csv"

    if not os.path.exists(river_file):
        return pd.DataFrame()

    return pd.read_csv(river_file)


# Function to evaluate river level status
def get_river_status(river_level, warning_level, danger_level):
    if river_level >= danger_level:
        return "Danger"
    elif river_level >= warning_level:
        return "Warning"
    else:
        return "Normal"


import math

def calculate_distance(lat1, lon1, lat2, lon2):
    # Haversine formula
    R = 6371.0  # Earth radius in km
    try:
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
    except Exception:
        return 9999.0  # Safe fallback high distance


def find_nearest_station(lat, lon, river_df):
    if river_df.empty:
        return None, 9999.0
    
    min_dist = float("inf")
    nearest_row = None
    
    for _, row in river_df.iterrows():
        try:
            r_lat = float(row["latitude"])
            r_lon = float(row["longitude"])
            dist = calculate_distance(lat, lon, r_lat, r_lon)
            if dist < min_dist:
                min_dist = dist
                nearest_row = row
        except (ValueError, TypeError):
            continue
            
    return nearest_row, min_dist



# Function to search location name using Open-Meteo Geocoding API
def search_location_by_name(location_name):
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search"

        params = {
            "name": location_name,
            "count": 100,
            "language": "en",
            "format": "json",
            "countryCode": "IN"
        }

        response = requests.get(url, params=params, timeout=20)

        if response.status_code != 200:
            return None

        data = response.json()

        if "results" not in data or len(data["results"]) == 0:
            return None

        results = data["results"]

        # Prefer India result
        india_results = [
            item for item in results
            if item.get("country_code") == "IN" or item.get("country") == "India"
        ]

        if india_results:
            first_result = india_results[0]
        else:
            first_result = results[0]

        location_data = {
            "name": first_result.get("name", ""),
            "country": first_result.get("country", ""),
            "country_code": first_result.get("country_code", ""),
            "admin1": first_result.get("admin1", ""),
            "latitude": first_result.get("latitude"),
            "longitude": first_result.get("longitude")
        }

        return location_data

    except Exception:
        return None




# Function to get live weather from Open-Meteo
def get_live_weather(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation,rain",
        "hourly": "precipitation,temperature_2m,relative_humidity_2m",
        "past_days": 3,
        "forecast_days": 1,
        "timezone": "auto"
    }

    response = requests.get(url, params=params, timeout=20)

    if response.status_code != 200:
        raise Exception("Failed to fetch weather data from Open-Meteo")

    data = response.json()

    current = data["current"]
    hourly = data["hourly"]

    hourly_df = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "precipitation": hourly["precipitation"]
    })

    current_time = pd.to_datetime(current["time"])

    last_24_hours = current_time - pd.Timedelta(hours=24)
    last_72_hours = current_time - pd.Timedelta(hours=72)

    rainfall_24h = hourly_df[
        (hourly_df["time"] > last_24_hours) &
        (hourly_df["time"] <= current_time)
    ]["precipitation"].sum()

    rainfall_3d = hourly_df[
        (hourly_df["time"] > last_72_hours) &
        (hourly_df["time"] <= current_time)
    ]["precipitation"].sum()

    # Ensure 3-day rainfall is at least equal to 24h rainfall
    rainfall_3d = max(rainfall_3d, rainfall_24h)

    weather_data = {
        "latitude": latitude,
        "longitude": longitude,
        "temperature": current["temperature_2m"],
        "humidity": current["relative_humidity_2m"],
        "current_precipitation": current["precipitation"],
        "current_rain": current["rain"],
        "rainfall_24h": round(rainfall_24h, 2),
        "rainfall_3d": round(rainfall_3d, 2)
    }

    return weather_data



def get_marker_color(prediction):
    if prediction == "Low":
        return "green"
    elif prediction == "Medium":
        return "orange"
    elif prediction == "High":
        return "red"
    elif prediction == "Critical":
        return "darkred"
    else:
        return "blue"


def show_risk_map(latitude, longitude, prediction):
    marker_color = get_marker_color(prediction)

    flood_map = folium.Map(
        location=[latitude, longitude],
        zoom_start=11
    )

    folium.Marker(
        location=[latitude, longitude],
        popup=f"Flood Risk: {prediction}",
        tooltip=f"Flood Risk: {prediction}",
        icon=folium.Icon(color=marker_color, icon="info-sign")
    ).add_to(flood_map)

    # Use returned_objects=[] to prevent map interactions from triggering a page rerun
    st_folium(flood_map, width=900, height=500, returned_objects=[])


def save_prediction_history(
    latitude,
    longitude,
    rainfall_24h,
    rainfall_3d,
    river_level,
    humidity,
    temperature,
    elevation,
    distance_to_river,
    past_flood_count,
    prediction,
    confidence
):
    history_file = "data/prediction_history.csv"

    os.makedirs("data", exist_ok=True)

    new_record = pd.DataFrame({
        "prediction_time": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "latitude": [latitude],
        "longitude": [longitude],
        "rainfall_24h": [rainfall_24h],
        "rainfall_3d": [rainfall_3d],
        "river_level": [river_level],
        "humidity": [humidity],
        "temperature": [temperature],
        "elevation": [elevation],
        "distance_to_river": [distance_to_river],
        "past_flood_count": [past_flood_count],
        "risk_level": [prediction],
        "confidence": [round(confidence, 2)]
    })

    if os.path.exists(history_file):
        old_data = pd.read_csv(history_file)
        updated_data = pd.concat([old_data, new_record], ignore_index=True)
    else:
        updated_data = new_record

    updated_data.to_csv(history_file, index=False)


def get_supabase_client():
    try:
        supabase_url = st.secrets.get("SUPABASE_URL", "")
        supabase_key = st.secrets.get("SUPABASE_KEY", "")

        if not supabase_url or not supabase_key or "paste_your" in supabase_url:
            return None

        supabase = create_client(supabase_url, supabase_key)
        return supabase

    except Exception:
        return None


def save_prediction_to_supabase(
    latitude,
    longitude,
    rainfall_24h,
    rainfall_3d,
    river_level,
    humidity,
    temperature,
    elevation,
    distance_to_river,
    past_flood_count,
    prediction,
    confidence
):
    supabase = get_supabase_client()

    if supabase is None:
        return False

    try:
        record = {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "rainfall_24h": float(rainfall_24h),
            "rainfall_3d": float(rainfall_3d),
            "river_level": float(river_level),
            "humidity": float(humidity),
            "temperature": float(temperature),
            "elevation": float(elevation),
            "distance_to_river": float(distance_to_river),
            "past_flood_count": int(past_flood_count),
            "risk_level": str(prediction),
            "confidence": float(round(confidence, 2))
        }

        supabase.table("prediction_history").insert(record).execute()
        return True

    except Exception:
        return False


def load_prediction_history_from_supabase():
    supabase = get_supabase_client()

    if supabase is None:
        return pd.DataFrame()

    try:
        response = (
            supabase
            .table("prediction_history")
            .select("*")
            .order("prediction_time", desc=True)
            .limit(50)
            .execute()
        )

        return pd.DataFrame(response.data)

    except Exception:
        return pd.DataFrame()



def show_prediction_analytics():
    history_file = "data/prediction_history.csv"

    if not os.path.exists(history_file):
        st.info("No analytics available yet. Make some predictions first.")
        return

    history_data = pd.read_csv(history_file)

    if history_data.empty:
        st.info("Prediction history is empty.")
        return

    st.subheader("📈 Prediction Analytics Dashboard")

    history_data["prediction_time"] = pd.to_datetime(
        history_data["prediction_time"],
        errors="coerce"
    )

    total_predictions = len(history_data)
    latest_prediction = history_data.iloc[-1]["risk_level"]
    average_confidence = history_data["confidence"].mean()

    high_risk_count = history_data[
        history_data["risk_level"].isin(["High", "Critical"])
    ].shape[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Predictions", total_predictions)
    col2.metric("Latest Risk", latest_prediction)
    col3.metric("Average Confidence", f"{average_confidence:.2f}%")
    col4.metric("High/Critical Count", high_risk_count)

    st.divider()

    st.subheader("📊 Risk Level Count")

    risk_counts = history_data["risk_level"].value_counts()
    st.bar_chart(risk_counts)

    st.subheader("📉 Confidence Trend")

    confidence_trend = history_data[["prediction_time", "confidence"]].dropna()
    confidence_trend = confidence_trend.set_index("prediction_time")

    st.line_chart(confidence_trend)

    st.subheader("🧾 Recent Predictions")

    st.dataframe(history_data.tail(10))


def send_telegram_alert(message):
    try:
        bot_token = get_secret_safe("TELEGRAM_BOT_TOKEN", "")
        chat_id = get_secret_safe("TELEGRAM_CHAT_ID", "")

        if (
            not bot_token 
            or not chat_id 
            or "paste_your" in bot_token 
            or "paste_your" in chat_id
            or bot_token.strip() == ""
            or chat_id.strip() == ""
        ):
            return False, "Telegram alert is not configured yet."

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

        payload = {
            "chat_id": chat_id,
            "text": message
        }

        response = requests.post(url, data=payload, timeout=20)

        if response.status_code == 200:
            return True, "Telegram alert sent successfully."
        else:
            return False, "Telegram alert service is temporarily unavailable. Please check your credentials."

    except Exception:
        return False, "Failed to connect to Telegram. Please check your network connection."


def create_flood_alert_message(
    latitude,
    longitude,
    rainfall_24h,
    rainfall_3d,
    river_level,
    humidity,
    temperature,
    elevation,
    distance_to_river,
    past_flood_count,
    prediction,
    confidence
):
    message = f"""
🚨 AI Flood Predictor Alert

Flood Risk Level: {prediction}
Model Confidence: {confidence:.2f}%

Location:
Latitude: {latitude}
Longitude: {longitude}

Environmental Data:
Rainfall 24h: {rainfall_24h} mm
Rainfall 3 days: {rainfall_3d} mm
River Level: {river_level} meters
Humidity: {humidity}%
Temperature: {temperature} Celsius
Elevation: {elevation} meters
Distance to River: {distance_to_river} km
Past Flood Count: {past_flood_count}

Suggested Action:
Please monitor this location carefully and follow official disaster management alerts.

Important Note:
This is an AI decision-support prototype, not an official government flood warning system.
"""
    return message



def generate_flood_report(
    latitude,
    longitude,
    rainfall_24h,
    rainfall_3d,
    river_level,
    humidity,
    temperature,
    elevation,
    distance_to_river,
    past_flood_count,
    prediction,
    confidence
):
    buffer = BytesIO()

    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, y, "AI Flood Predictor Report")

    y -= 35

    pdf.setFont("Helvetica", 10)
    pdf.drawString(
        50,
        y,
        "This is an AI decision-support prototype, not an official government flood warning system."
    )

    y -= 40

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "Prediction Summary")

    y -= 30

    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, y, f"Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 25
    pdf.drawString(50, y, f"Flood Risk Level: {prediction}")
    y -= 25
    pdf.drawString(50, y, f"Model Confidence: {confidence:.2f}%")

    y -= 40

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "Location Details")

    y -= 30

    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, y, f"Latitude: {latitude}")
    y -= 25
    pdf.drawString(50, y, f"Longitude: {longitude}")

    y -= 40

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "Environmental Data")

    y -= 30

    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, y, f"Rainfall in Last 24 Hours: {rainfall_24h} mm")
    y -= 25
    pdf.drawString(50, y, f"Rainfall in Last 3 Days: {rainfall_3d} mm")
    y -= 25
    pdf.drawString(50, y, f"River Level: {river_level} meters")
    y -= 25
    pdf.drawString(50, y, f"Humidity: {humidity}%")
    y -= 25
    pdf.drawString(50, y, f"Temperature: {temperature} Celsius")
    y -= 25
    pdf.drawString(50, y, f"Elevation: {elevation} meters")
    y -= 25
    pdf.drawString(50, y, f"Distance to River: {distance_to_river} km")
    y -= 25
    pdf.drawString(50, y, f"Past Flood Count: {past_flood_count}")

    y -= 40

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "Suggested Action")

    y -= 30

    pdf.setFont("Helvetica", 12)

    if prediction == "Low":
        action = "Continue normal monitoring."
    elif prediction == "Medium":
        action = "Stay alert and monitor rainfall and river level frequently."
    elif prediction == "High":
        action = "Alert local authorities and prepare emergency support."
    else:
        action = "Immediate disaster response and evacuation planning may be required."

    pdf.drawString(50, y, action)

    y -= 40

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "Important Note")

    y -= 30

    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, y, "Always follow official alerts from disaster management authorities.")

    pdf.save()

    buffer.seek(0)
    return buffer


def generate_gemini_explanation(
    latitude,
    longitude,
    rainfall_24h,
    rainfall_3d,
    river_level,
    humidity,
    temperature,
    elevation,
    distance_to_river,
    past_flood_count,
    prediction,
    confidence
):
    try:
        gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
        # Define logging configuration if not done
        logging.basicConfig(level=logging.INFO)
        app_logger = logging.getLogger(__name__)
        app_logger.info("Generating Gemini content explanation...")

        if (
            not gemini_api_key
            or "paste_your" in gemini_api_key
            or gemini_api_key.strip() == ""
        ):
            return (
                "Gemini is not connected yet. Add your real GEMINI_API_KEY in .streamlit/secrets.toml."
            )

        client = genai.Client(api_key=gemini_api_key)

        prompt = f"""
You are an AI flood risk explanation assistant.

Explain this flood prediction in simple beginner-friendly English.

Important rules:
- Do not say this is an official government warning.
- Say this is only an AI decision-support explanation.
- Keep it short and clear.
- Give safety advice, but tell the user to follow official disaster management alerts.

Prediction details:
- Flood Risk Level: {prediction}
- Model Confidence: {confidence:.2f}%
- Latitude: {latitude}
- Longitude: {longitude}
- Rainfall in last 24 hours: {rainfall_24h} mm
- Rainfall in last 3 days: {rainfall_3d} mm
- River Level: {river_level} meters
- Humidity: {humidity}%
- Temperature: {temperature} Celsius
- Elevation: {elevation} meters
- Distance to River: {distance_to_river} km
- Past Flood Count: {past_flood_count}

Give output in this format:

1. Simple Explanation:
2. Main Reasons:
3. Suggested Action:
4. Safety Note:
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception:
        return (
            "Gemini is not connected yet. Add your real GEMINI_API_KEY in .streamlit/secrets.toml."
        )


def load_admin_prediction_data():
    # First try to load from Supabase
    try:
        supabase_data = load_prediction_history_from_supabase()

        if supabase_data is not None and not supabase_data.empty:
            return supabase_data

    except Exception:
        pass

    # If Supabase is not available, load from local CSV
    history_file = "data/prediction_history.csv"

    if os.path.exists(history_file):
        return pd.read_csv(history_file)

    return pd.DataFrame()


def show_admin_dashboard():
    st.header("🛡️ Admin Disaster Management Dashboard")

    admin_data = load_admin_prediction_data()

    if admin_data.empty:
        st.info("No prediction records available yet.")
        return

    # Convert prediction time if available
    if "prediction_time" in admin_data.columns:
        admin_data["prediction_time"] = pd.to_datetime(
            admin_data["prediction_time"],
            errors="coerce"
        )

    total_predictions = len(admin_data)

    low_count = admin_data[admin_data["risk_level"] == "Low"].shape[0]
    medium_count = admin_data[admin_data["risk_level"] == "Medium"].shape[0]
    high_count = admin_data[admin_data["risk_level"] == "High"].shape[0]
    critical_count = admin_data[admin_data["risk_level"] == "Critical"].shape[0]

    average_confidence = admin_data["confidence"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Predictions", total_predictions)
    col2.metric("Low Risk", low_count)
    col3.metric("Medium Risk", medium_count)
    col4.metric("High Risk", high_count)
    col5.metric("Critical Risk", critical_count)

    st.metric("Average Model Confidence", f"{average_confidence:.2f}%")

    st.divider()

    st.subheader("🚨 High and Critical Risk Locations")

    danger_data = admin_data[
        admin_data["risk_level"].isin(["High", "Critical"])
    ]

    if not danger_data.empty:
        st.dataframe(danger_data.tail(20))
    else:
        st.success("No High or Critical risk records found.")

    st.divider()

    st.subheader("📊 Risk Level Distribution")

    risk_counts = admin_data["risk_level"].value_counts()
    st.bar_chart(risk_counts)

    st.subheader("📉 Confidence Trend")

    if "prediction_time" in admin_data.columns:
        confidence_data = admin_data[["prediction_time", "confidence"]].dropna()

        if not confidence_data.empty:
            confidence_data = confidence_data.set_index("prediction_time")
            st.line_chart(confidence_data)

    st.divider()

    st.subheader("📁 Recent Prediction Records")

    st.dataframe(admin_data.tail(20))

    csv_data = admin_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Admin Prediction Data",
        data=csv_data,
        file_name="admin_prediction_data.csv",
        mime="text/csv"
    )


def send_email_alert(subject, message):
    try:
        email_sender = get_secret_safe("EMAIL_SENDER", "")
        email_password = get_secret_safe("EMAIL_PASSWORD", "")
        email_receiver = get_secret_safe("EMAIL_RECEIVER", "")
        smtp_server = get_secret_safe("SMTP_SERVER", "smtp.gmail.com")
        
        try:
            smtp_port = int(get_secret_safe("SMTP_PORT", "587"))
        except Exception:
            smtp_port = 587

        if (
            not email_sender 
            or not email_password 
            or not email_receiver
            or "your_email" in email_sender 
            or "your_email" in email_password
            or "receiver_email" in email_receiver
            or email_sender.strip() == ""
            or email_password.strip() == ""
            or email_receiver.strip() == ""
        ):
            return False, "Email alert is not configured yet."

        email = MIMEMultipart()
        email["From"] = email_sender
        email["To"] = email_receiver
        email["Subject"] = subject

        email.attach(MIMEText(message, "plain"))

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(email_sender, email_password)
        server.sendmail(email_sender, email_receiver, email.as_string())
        server.quit()

        return True, "Email alert sent successfully."

    except Exception:
        return False, "Failed to send email alert. Please check your network connection or SMTP settings."



# Sidebar location search
st.sidebar.header("🌍 Location Search")

# Default location values
if "selected_latitude" not in st.session_state:
    st.session_state["selected_latitude"] = 13.0827

if "selected_longitude" not in st.session_state:
    st.session_state["selected_longitude"] = 80.2707

if "selected_location_name" not in st.session_state:
    st.session_state["selected_location_name"] = "Chennai"

location_query = st.sidebar.text_input(
    "Enter city or location name",
    value=st.session_state["selected_location_name"]
)

if st.sidebar.button("Search Location"):
    location_result = search_location_by_name(location_query)

    if location_result:
        new_lat = float(location_result["latitude"])
        new_lon = float(location_result["longitude"])
        st.session_state["selected_latitude"] = new_lat
        st.session_state["selected_longitude"] = new_lon
        st.session_state["selected_location_name"] = location_result["name"]

        st.sidebar.success(
            f"Location found: {location_result['name']}, {location_result['admin1']}, {location_result['country']}"
        )
        
        # Auto-fetch weather for the searched location
        try:
            weather_data = get_live_weather(new_lat, new_lon)
            st.session_state["weather"] = weather_data
            st.sidebar.success("Live weather loaded automatically.")
        except Exception:
            if "weather" in st.session_state:
                st.session_state.pop("weather")
            st.sidebar.warning("Could not automatically load weather data. Please use 'Get Live Weather' button.")
    else:
        st.sidebar.error("Location not found. Please try another city name.")


st.sidebar.header("📍 Selected Coordinates")

latitude = st.sidebar.number_input(
    "Latitude",
    value=float(st.session_state["selected_latitude"]),
    format="%.4f"
)

longitude = st.sidebar.number_input(
    "Longitude",
    value=float(st.session_state["selected_longitude"]),
    format="%.4f"
)

st.sidebar.caption("You can search by city name or manually edit latitude and longitude.")

st.sidebar.header("🚨 Alerts System")

enable_telegram_alert = st.sidebar.checkbox(
    "Enable Telegram Alert",
    value=True
)

enable_email_alert = st.sidebar.checkbox(
    "Enable Email Alert",
    value=True
)

st.sidebar.caption("Alerts are sent only for High or Critical flood risk.")


# Sidebar flood input
st.sidebar.header("🌊 River Level Data")

river_data = load_river_level_data()

if not river_data.empty:
    station_options = ["Manual Input"] + river_data["station_name"].tolist()

    # Find nearest station to pre-select
    nearest_station_row, nearest_dist = find_nearest_station(latitude, longitude, river_data)
    default_index = 0
    if nearest_station_row is not None and nearest_dist <= 50.0:
        nearest_station_name = nearest_station_row["station_name"]
        if nearest_station_name in station_options:
            default_index = station_options.index(nearest_station_name)

    # Use coordinates in key so selectbox resets when location changes
    selectbox_key = f"river_station_select_{latitude}_{longitude}"
    
    selected_station = st.sidebar.selectbox(
        "Select River Station",
        station_options,
        index=default_index,
        key=selectbox_key
    )

    if selected_station != "Manual Input":
        selected_river_row = river_data[
            river_data["station_name"] == selected_station
        ].iloc[0]

        # Calculate actual distance to selected station
        selected_dist = calculate_distance(
            latitude,
            longitude,
            float(selected_river_row["latitude"]),
            float(selected_river_row["longitude"])
        )

        river_level_default = float(selected_river_row["river_level"])
        warning_level = float(selected_river_row["warning_level"])
        danger_level = float(selected_river_row["danger_level"])

        river_status = get_river_status(
            river_level_default,
            warning_level,
            danger_level
        )

        st.sidebar.write("River:", selected_river_row["river_name"])
        st.sidebar.write("State:", selected_river_row["state"])
        st.sidebar.write("Latest Reading:", selected_river_row["reading_time"])
        st.sidebar.write(f"Distance to Station: {selected_dist:.2f} km")
        st.sidebar.write("River Status:", river_status)

        if selected_dist > 50.0:
            st.sidebar.warning(
                f"⚠️ Selected station is {selected_dist:.1f} km away. "
                "Consider entering manual river data."
            )
    else:
        river_level_default = 1.5
        warning_level = 5.0
        danger_level = 6.0
        river_status = "Normal"
        selected_dist = 0.0
        st.sidebar.info("No nearby monitoring station within 50 km. Enter river level manually below.")

else:
    river_level_default = 4.5
    warning_level = 5.0
    danger_level = 6.0
    river_status = "Normal"
    selected_station = "Manual River Level"
    selected_dist = 0.0
    st.sidebar.warning("river_levels.csv not found. Using manual river level.")

river_level = st.sidebar.number_input(
    "River Level (meters)",
    min_value=0.0,
    value=river_level_default
)


st.sidebar.header("🌦️ Environmental Data")

# Auto-fetch live weather if cache is empty or coordinates changed
if "weather" not in st.session_state or (
    round(st.session_state["weather"].get("latitude", 0.0), 4) != round(latitude, 4)
    or round(st.session_state["weather"].get("longitude", 0.0), 4) != round(longitude, 4)
):
    try:
        with st.sidebar.spinner("Fetching live weather..."):
            weather_data = get_live_weather(latitude, longitude)
            st.session_state["weather"] = weather_data
    except Exception:
        if "weather" in st.session_state:
            st.session_state.pop("weather")

weather = None
if "weather" in st.session_state:
    weather = st.session_state["weather"]
else:
    st.sidebar.warning("⚠️ Live weather API is currently offline. Using default fallback values.")

if st.sidebar.button("Get Live Weather"):
    try:
        weather_data = get_live_weather(latitude, longitude)
        st.session_state["weather"] = weather_data
        st.sidebar.success("Live weather loaded successfully.")
        st.rerun()
    except Exception:
        st.sidebar.error("Could not load weather data. Please check your internet connection or try again later.")

if weather:
    rainfall_24h_default = float(weather["rainfall_24h"])
    rainfall_3d_default = float(weather["rainfall_3d"])
    humidity_default = float(weather["humidity"])
    temperature_default = float(weather["temperature"])
else:
    rainfall_24h_default = 0.0
    rainfall_3d_default = 0.0
    humidity_default = 60.0
    temperature_default = 25.0

rainfall_24h = st.sidebar.number_input(
    "Rainfall in last 24 hours (mm)",
    min_value=0.0,
    value=rainfall_24h_default
)

rainfall_3d = st.sidebar.number_input(
    "Rainfall in last 3 days (mm)",
    min_value=0.0,
    value=rainfall_3d_default
)

# Enforce that 3-day rainfall is at least equal to 24h rainfall
if rainfall_3d < rainfall_24h:
    st.sidebar.warning("⚠️ 3-day rainfall cannot be less than 24-hour rainfall. Adjusting 3-day rainfall to match.")
    rainfall_3d = rainfall_24h

humidity = st.sidebar.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=humidity_default
)

temperature = st.sidebar.number_input(
    "Temperature (°C)",
    min_value=0.0,
    value=temperature_default
)

elevation = st.sidebar.number_input(
    "Elevation from Sea Level (meters)",
    min_value=0.0,
    value=35.0
)

# Set distance_to_river_default to the calculated distance if within reasonable range (<= 50km) and a station is selected
if not river_data.empty and selected_station != "Manual Input" and selected_dist <= 50.0:
    distance_to_river_default = round(selected_dist, 2)
else:
    distance_to_river_default = 2.0

distance_to_river = st.sidebar.number_input(
    "Distance to River (km)",
    min_value=0.0,
    value=distance_to_river_default
)

past_flood_count = st.sidebar.number_input(
    "Past Flood Count",
    min_value=0,
    value=2
)


# Show selected location
st.subheader("📍 Selected Location")
st.write(f"Location Name: {st.session_state.get('selected_location_name', 'Custom Location')}")
st.write(f"Latitude: {latitude}")
st.write(f"Longitude: {longitude}")


# Show river level dashboard information
st.subheader("🌊 River Level Information")

if not river_data.empty:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Selected Station", selected_station)
    if selected_station != "Manual Input":
        col1.caption(f"Distance: {selected_dist:.2f} km from selected location")
    else:
        col1.caption("Manual coordinate input mode")
    
    col2.metric("River Level", f"{river_level} m")
    col3.metric("Warning Level", f"{warning_level} m")
    col4.metric("Danger Level", f"{danger_level} m")

    if river_status == "Normal":
        st.success("River Status: Normal")
    elif river_status == "Warning":
        st.warning("River Status: Warning Level Reached")
    elif river_status == "Danger":
        st.error("River Status: Danger Level Reached")
else:
    st.info("River level is currently entered manually.")


# Show live weather result
if weather:
    st.subheader("🌦️ Live Weather Data from Open-Meteo")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Temperature", f"{weather['temperature']} °C")
    col2.metric("Humidity", f"{weather['humidity']} %")
    col3.metric("Rainfall 24h", f"{weather['rainfall_24h']} mm")
    col4.metric("Rainfall 3 days", f"{weather['rainfall_3d']} mm")

else:
    st.info("Click **Get Live Weather** in the sidebar to load live weather data.")


# Create input data for model
input_data = pd.DataFrame({
    "rainfall_24h": [rainfall_24h],
    "rainfall_3d": [rainfall_3d],
    "river_level": [river_level],
    "humidity": [humidity],
    "temperature": [temperature],
    "elevation": [elevation],
    "distance_to_river": [distance_to_river],
    "past_flood_count": [past_flood_count]
})

st.subheader("📊 Input Data Sent to ML Model")
st.dataframe(input_data)


# Prediction
if st.button("Predict Flood Risk"):
    prediction = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data).max() * 100

    save_prediction_history(
        latitude,
        longitude,
        rainfall_24h,
        rainfall_3d,
        river_level,
        humidity,
        temperature,
        elevation,
        distance_to_river,
        past_flood_count,
        prediction,
        confidence
    )

    st.success("Prediction saved successfully in history.")

    if is_supabase_configured():
        supabase_saved = save_prediction_to_supabase(
            latitude,
            longitude,
            rainfall_24h,
            rainfall_3d,
            river_level,
            humidity,
            temperature,
            elevation,
            distance_to_river,
            past_flood_count,
            prediction,
            confidence
        )

        if supabase_saved:
            st.success("Prediction saved successfully to Supabase database.")
        else:
            st.warning("Prediction was not saved to Supabase. Check Supabase URL, key, and table.")
    else:
        st.info("Supabase is not connected yet. Local CSV history is active.")

    st.subheader("🚨 Alerts System")

    if prediction in ["High", "Critical"]:
        alert_message = create_flood_alert_message(
            latitude,
            longitude,
            rainfall_24h,
            rainfall_3d,
            river_level,
            humidity,
            temperature,
            elevation,
            distance_to_river,
            past_flood_count,
            prediction,
            confidence
        )

        if enable_telegram_alert:
            telegram_sent, telegram_message = send_telegram_alert(alert_message)

            if telegram_sent:
                st.success(telegram_message)
            else:
                st.warning(telegram_message)
        else:
            st.info("Telegram alert is disabled.")

        if enable_email_alert:
            email_subject = f"🚨 AI Flood Alert - {prediction} Risk"

            email_sent, email_message = send_email_alert(
                email_subject,
                alert_message
            )

            if email_sent:
                st.success(email_message)
            else:
                st.warning(email_message)
        else:
            st.info("Email alert is disabled.")

    else:
        st.info("No alert sent because flood risk is Low or Medium.")


    st.subheader("🚨 Prediction Result")

    if prediction == "Low":
        st.success(f"Flood Risk Level: {prediction}")
    elif prediction == "Medium":
        st.warning(f"Flood Risk Level: {prediction}")
    elif prediction == "High":
        st.error(f"Flood Risk Level: {prediction}")
    elif prediction == "Critical":
        st.error(f"Flood Risk Level: {prediction}")

    st.write(f"Model Confidence: {confidence:.2f}%")

    st.subheader("🧠 Reasoning")

    if prediction == "Low":
        st.write("Rainfall and river level are currently low. Flood risk is low.")
    elif prediction == "Medium":
        st.write("Rainfall or river level is increasing. Stay alert and monitor updates.")
    elif prediction == "High":
        st.write("Heavy rainfall, high river level, or flood history indicates high flood risk.")
    elif prediction == "Critical":
        st.write("Very heavy rainfall and dangerous conditions indicate critical flood risk.")

    st.subheader("✅ Suggested Action")

    if prediction == "Low":
        st.write("Continue normal monitoring.")
    elif prediction == "Medium":
        st.write("Prepare emergency contacts and monitor rainfall/river levels.")
    elif prediction == "High":
        st.write("Alert local authorities and prepare evacuation support.")
    elif prediction == "Critical":
        st.write("Immediate disaster response and evacuation planning may be required.")

    st.subheader("🤖 Gemini AI Explanation")

    gemini_explanation = generate_gemini_explanation(
        latitude,
        longitude,
        rainfall_24h,
        rainfall_3d,
        river_level,
        humidity,
        temperature,
        elevation,
        distance_to_river,
        past_flood_count,
        prediction,
        confidence
    )

    st.write(gemini_explanation)

    st.subheader("🗺️ Flood Risk Map")
    show_risk_map(latitude, longitude, prediction)

    st.subheader("📄 Flood Risk Report")

    pdf_report = generate_flood_report(
        latitude,
        longitude,
        rainfall_24h,
        rainfall_3d,
        river_level,
        humidity,
        temperature,
        elevation,
        distance_to_river,
        past_flood_count,
        prediction,
        confidence
    )

    st.download_button(
        label="Download Flood Risk Report",
        data=pdf_report,
        file_name="flood_risk_report.pdf",
        mime="application/pdf"
    )


st.divider()

st.subheader("🤖 ML Model Comparison")

comparison_file = "models/model_comparison_results.csv"

if os.path.exists(comparison_file):
    comparison_data = pd.read_csv(comparison_file)
    st.dataframe(comparison_data)

    chart_data = comparison_data.set_index("model_name")[
        ["accuracy", "precision", "recall", "f1_score"]
    ]

    st.bar_chart(chart_data)
else:
    st.info("Model comparison results not found. Run python train_model.py first.")


st.divider()

st.subheader("📁 Prediction History")

history_file = "data/prediction_history.csv"

if os.path.exists(history_file):
    history_data = pd.read_csv(history_file)

    st.write("Previous flood risk predictions are saved below:")
    st.dataframe(history_data.tail(10))

    csv_data = history_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Prediction History CSV",
        data=csv_data,
        file_name="prediction_history.csv",
        mime="text/csv"
    )
else:
    st.info("No prediction history found yet. Click Predict Flood Risk to save your first prediction.")


st.divider()

st.subheader("☁️ Supabase Prediction History")

if is_supabase_configured():
    supabase_history = load_prediction_history_from_supabase()

    if not supabase_history.empty:
        st.write("Latest predictions saved in Supabase:")
        st.dataframe(supabase_history.head(10))
    else:
        st.info("No Supabase prediction history found yet.")
else:
    st.info("Supabase is not connected yet. Local CSV history is active.")


st.divider()
show_prediction_analytics()


st.divider()

st.header("🔐 Admin Panel")

admin_password_input = st.text_input(
    "Enter Admin Password",
    type="password"
)

correct_admin_password = get_secret_safe("ADMIN_PASSWORD", "admin123")

if admin_password_input:
    if admin_password_input == correct_admin_password:
        st.success("Admin access granted.")
        show_admin_dashboard()
    else:
        st.error("Incorrect admin password.")

