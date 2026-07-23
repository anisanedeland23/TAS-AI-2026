<div align="center">

# 🌡️ Smart Environment IoT — Visualisasi & Forecasting

**Real-time IoT monitoring dashboard dengan AI-powered forecasting & environmental insight**

Tugas Akhir Semester — Mata Kuliah *Artificial Intelligence* (BD002)
Universitas Kristen Satya Wacana · 2026

[![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E?logo=supabase&logoColor=white)](https://supabase.com)
[![Railway](https://img.shields.io/badge/Railway-Deployment-0B0D0E?logo=railway&logoColor=white)](https://railway.app)
[![Prophet](https://img.shields.io/badge/Prophet-Forecasting-1877F2)](https://facebook.github.io/prophet/)
[![Groq](https://img.shields.io/badge/Groq-LLM%20API-F55036)](https://groq.com)

[**🚀 Live Dashboard**](https://tas-ai-672023104.streamlit.app) · [Fitur](#-fitur-utama) · [Arsitektur](#️-arsitektur-sistem) · [Setup](#-cara-menjalankan)

</div>

---

## 📖 Tentang Proyek

Sistem end-to-end yang menangkap data suhu & kelembaban dari sensor IoT secara **real-time** melalui MQTT broker, merekamnya secara mandiri ke *cloud database* selama 7+ hari tanpa bergantung pada perangkat lokal, lalu menyajikannya dalam dashboard interaktif lengkap dengan **prediksi time series** dan **analisis cerdas berbasis LLM**.

> 💡 Dibangun dengan pendekatan *cloud-to-cloud*: tidak ada laptop yang perlu menyala 24/7 — seluruh pipeline berjalan mandiri di layanan cloud gratis.

## ✨ Fitur Utama

| Fitur | Deskripsi |
|---|---|
| 📡 **Recording Otomatis** | Subscriber Python di Railway menangkap data MQTT 24/7, tersimpan ke Supabase PostgreSQL |
| 📊 **Monitoring Real-Time** | Grafik historis, statistik deskriptif (mean/median/min/max/std), filter rentang waktu |
| 🔮 **AI Forecasting** | Model Prophet memprediksi suhu & kelembaban 6 jam ke depan, dengan mode validasi akurasi terpisah |
| ✅ **Evaluasi Model** | Metrik MAE, RMSE, MAPE dihitung otomatis dari data uji tersembunyi |
| 🤖 **AI Insight** | Groq LLM (`openai/gpt-oss-120b`) menganalisis kondisi lingkungan & memberi rekomendasi tindakan |
| 🗄️ **Data Explorer** | Jelajahi & filter seluruh data mentah langsung dari dashboard |

## 🏗️ Arsitektur Sistem

```
┌─────────────────┐     MQTT      ┌──────────────────┐     INSERT        ┌─────────────────┐
│   MQTT Broker    │──────────────▶│  Railway Cloud    │───────────────▶│  Supabase        │
│   (HiveMQ)       │   subscribe   │  subscriber.py    │    SQL          │  PostgreSQL      │
└─────────────────┘                └──────────────────┘                  └────────┬─────────┘
                                                                                   │
                                                                          pull data│(paginated)
                                                                                   ▼
                                                                        ┌─────────────────────┐
                                                                        │  Streamlit Dashboard │
                                                                        │  ─────────────────── │
                                                                        │  📊 Monitoring       │
                                                                        │  📈 Analytics        │
                                                                        │  🔮 AI Forecast      │
                                                                        │  🗄️ Database         │
                                                                        └──────────┬───────────┘
                                                                                   │
                                                                     ┌─────────────┴─────────────┐
                                                                     ▼                           ▼
                                                          ┌────────────────────┐      ┌────────────────────┐
                                                          │  Prophet Model     │       │  Groq LLM API     │
                                                          │  (forecasting)     │       │  (AI Insight)     │
                                                          └────────────────────┘      └────────────────────┘
```

## 🛠️ Tech Stack

<table>
<tr>
<td><b>Data Pipeline</b></td>
<td>Python · paho-mqtt · Supabase Python SDK</td>
</tr>
<tr>
<td><b>Storage</b></td>
<td>Supabase (PostgreSQL, cloud-hosted)</td>
</tr>
<tr>
<td><b>Deployment</b></td>
<td>Railway (recording service) · Streamlit Community Cloud (dashboard)</td>
</tr>
<tr>
<td><b>Forecasting</b></td>
<td>Facebook Prophet · scikit-learn (metrik evaluasi)</td>
</tr>
<tr>
<td><b>AI Insight</b></td>
<td>Groq API — model <code>openai/gpt-oss-120b</code></td>
</tr>
<tr>
<td><b>Dashboard</b></td>
<td>Streamlit</td>
</tr>
<tr>
<td><b>Analisis Data</b></td>
<td>Pandas · NumPy · Matplotlib · Plotly (Google Colab)</td>
</tr>
</table>

## 📂 Struktur Proyek

```
tas-ai-iot/
├── recording/
│   ├── subscriber.py        # MQTT subscriber → Supabase (deployed on Railway)
│   └── config.py            # Kredensial (tidak di-commit publik)
├── dashboard/
│   └── streamlit_app.py     # Dashboard utama: Monitoring, Analytics, AI Forecast, Database
├── forecasting/
│   └── README.md            # Catatan implementasi Prophet (terintegrasi di dashboard)
├── llm_integration/
│   └── README.md            # Catatan integrasi Groq LLM (terintegrasi di dashboard)
├── data/
│   └── sensor_data.csv      # Snapshot data historis hasil recording
└── requirements.txt
```

## 📊 Hasil & Performa Model

Model forecasting dievaluasi menggunakan skema *hold-out validation* (6 jam data terakhir disisihkan sebagai data uji):

| Parameter | MAE | RMSE | MAPE |
|---|---|---|---|
| 🌡️ Suhu | 0.25 | 0.31 | **1.03%** |
| 💧 Kelembaban | 4.80 | 5.20 | **7.40%** |

> MAPE di bawah 10% pada kedua parameter menunjukkan tingkat akurasi model yang tinggi untuk prediksi jangka pendek (6 jam ke depan).

**Data Recording:** 63.754+ titik data mentah, terkumpul kontinu selama **7,6 hari** (15–23 Juli 2026) via koneksi MQTT-TLS ke broker HiveMQ Cloud.

## 🚀 Cara Menjalankan

**1. Clone & install dependencies**
```bash
git clone https://github.com/anisanedeland23/tas-ai-iot.git
cd tas-ai-iot
pip install -r requirements.txt
```

**2. Konfigurasi kredensial**

Buat `config.py` di folder `recording/` dan `dashboard/`:
```python
MQTT_BROKER = "your-broker-url"
MQTT_PORT = 8883
MQTT_USERNAME = "..."
MQTT_PASSWORD = "..."
MQTT_TOPIC_SUHU = "..."
MQTT_TOPIC_KELEMBABAN = "..."

SUPABASE_URL = "https://xxxxx.supabase.co"
SUPABASE_KEY = "..."
GROQ_API_KEY = "..."
```

**3. Jalankan recording service**
```bash
python recording/subscriber.py
```

**4. Jalankan dashboard**
```bash
streamlit run dashboard/streamlit_app.py
```

## 👤 Author

**Anisa Virginia Shalomita Nedeland**
NIM 672023104 · Program Studi Teknik Informatika
Fakultas Teknologi Informasi — Universitas Kristen Satya Wacana

*Dosen Pengampu: Dr. Suryasatriya Trihandaru, M.Sc.nat*

---

<div align="center">

Dibuat dengan 🧠 + ☕ untuk Tugas Akhir Semester — Artificial Intelligence 2026

</div>
