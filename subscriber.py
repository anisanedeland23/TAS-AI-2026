import paho.mqtt.client as mqtt
from supabase import create_client
import config

# Konek ke Supabase
supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

# Fungsi ini jalan otomatis tiap kali ada data baru masuk dari MQTT
def on_message(client, userdata, msg):
    topic = msg.topic
    try:
        value = float(msg.payload.decode())
    except ValueError:
        print(f"Data tidak valid dari topik {topic}: {msg.payload}")
        return

    print(f"Diterima -> {topic}: {value}")

    if topic == config.MQTT_TOPIC_SUHU:
        supabase.table("sensor_data").insert({"suhu": value}).execute()
    elif topic == config.MQTT_TOPIC_KELEMBABAN:
        supabase.table("sensor_data").insert({"kelembaban": value}).execute()

def on_connect(client, userdata, flags, rc, properties=None):
    print("Berhasil konek ke broker MQTT dengan status:", rc)
    client.subscribe(config.MQTT_TOPIC_SUHU)
    client.subscribe(config.MQTT_TOPIC_KELEMBABAN)

# Setup client MQTT
client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(config.MQTT_USERNAME, config.MQTT_PASSWORD)
client.tls_set()  # aktifkan koneksi aman (SSL/TLS)

client.on_connect = on_connect
client.on_message = on_message

client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)

print("Mulai mendengarkan data MQTT... (tekan Ctrl+C untuk berhenti)")
client.loop_forever()