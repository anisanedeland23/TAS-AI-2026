import streamlit as st

# --- Kredensial MQTT ---
MQTT_BROKER = st.secrets["MQTT_BROKER"]
MQTT_PORT = st.secrets["MQTT_PORT"]
MQTT_USERNAME = st.secrets["MQTT_USERNAME"]
MQTT_PASSWORD = st.secrets["MQTT_PASSWORD"]
MQTT_TOPIC_SUHU = st.secrets["MQTT_TOPIC_SUHU"]
MQTT_TOPIC_KELEMBABAN = st.secrets["MQTT_TOPIC_KELEMBABAN"]

# --- Kredensial Supabase ---
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

# --- Groq Keys ---
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# --- Target & Start Date Recording ---
TARGET_DAYS = st.secrets["TARGET_DAYS"]
RECORDING_START_DATE = st.secrets["RECORDING_START_DATE"]
