import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from supabase import create_client
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import config

st.set_page_config(
    page_title="Smart Environment Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

# Optional config values (safe defaults so the app doesn't crash if not set in config.py)
TARGET_DAYS = getattr(config, "TARGET_DAYS", 7)
# Tanggal mulai perekaman data. Bisa dioverride lewat config.RECORDING_START_DATE;
# kalau tidak diset, fallback ke tanggal mulai perekaman yang diketahui.
RECORDING_START_DATE = getattr(config, "RECORDING_START_DATE", datetime(2026, 7, 15, 22, 4, 0))


# ============================================================
# STYLE — dark navy / glassmorphism sesuai blueprint
# ============================================================

def inject_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: radial-gradient(circle at 15% 0%, #10192f 0%, #0b1120 45%, #070b16 100%);
        }

        section[data-testid="stSidebar"] {
            background: #0b1120;
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        /* Header */
        .app-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 18px 24px;
            border-radius: 20px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            backdrop-filter: blur(12px);
            margin-bottom: 22px;
        }
        .app-title { font-size: 22px; font-weight: 800; color: #f4f6fb; margin: 0; }
        .app-subtitle { font-size: 13px; color: #8b93a7; margin: 0; }
        .pill-row { display: flex; gap: 8px; }
        .pill {
            font-size: 11px; font-weight: 600; padding: 5px 12px; border-radius: 999px;
            background: rgba(34,197,94,0.12); color: #4ade80; border: 1px solid rgba(74,222,128,0.25);
        }
        .pill.offline { background: rgba(239,68,68,0.12); color: #f87171; border-color: rgba(248,113,113,0.25); }

        /* Glass card */
        .glass-card {
            border-radius: 20px;
            padding: 20px 22px;
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.08);
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.25);
            transition: transform .15s ease, box-shadow .15s ease;
            margin-bottom: 18px;
        }
        .glass-card:hover { transform: translateY(-2px); box-shadow: 0 12px 28px rgba(0,0,0,0.32); }

        .metric-card { display: flex; flex-direction: column; gap: 6px; border-left: 4px solid; }
        .metric-icon { font-size: 22px; }
        .metric-label { font-size: 13px; color: #9aa3b8; font-weight: 500; }
        .metric-value { font-size: 28px; font-weight: 800; color: #f4f6fb; }
        .metric-delta { font-size: 12px; font-weight: 600; }
        .metric-delta.up { color: #4ade80; }
        .metric-delta.down { color: #f87171; }

        .accent-orange { border-color: #f97316; }
        .accent-blue   { border-color: #38bdf8; }
        .accent-green  { border-color: #22c55e; }
        .accent-purple { border-color: #a78bfa; }

        .section-title { font-size: 16px; font-weight: 700; color: #f4f6fb; margin-bottom: 12px; }
        .stat-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }
        .stat-box { text-align: center; padding: 10px 4px; border-radius: 14px; background: rgba(255,255,255,0.03); }
        .stat-box .v { font-size: 17px; font-weight: 700; color: #f4f6fb; }
        .stat-box .l { font-size: 11px; color: #8b93a7; }

        .ai-insight-text { font-size: 14px; line-height: 1.6; color: #dbe0ee; }

        .progress-wrap { background: rgba(255,255,255,0.06); border-radius: 999px; height: 10px; overflow: hidden; }
        .progress-bar { height: 100%; border-radius: 999px; background: linear-gradient(90deg,#a78bfa,#38bdf8); }

        div[data-testid="stMetricValue"] { color: #f4f6fb; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(icon, label, value, delta_text, accent, delta_up=True):
    delta_class = "up" if delta_up else "down"
    arrow = "▲" if delta_up else "▼"
    st.markdown(
        f"""
        <div class="glass-card metric-card accent-{accent}">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta {delta_class}">{arrow} {delta_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FUNGSI-FUNGSI UTILITAS DATA
# ============================================================

@st.cache_data(ttl=60)
def ambil_semua_data():
    semua_data = []
    batas = 1000
    halaman = 0
    while True:
        mulai = halaman * batas
        akhir = mulai + batas - 1
        response = supabase.table("sensor_data").select("*").range(mulai, akhir).execute()
        if not response.data:
            break
        semua_data.extend(response.data)
        if len(response.data) < batas:
            break
        halaman += 1
    df = pd.DataFrame(semua_data)
    df["waktu"] = pd.to_datetime(df["waktu"])
    df = df.sort_values("waktu")
    return df


def siapkan_untuk_chart(data):
    data_resampled = data.set_index("waktu").resample("20s").mean(numeric_only=True)
    data_resampled[["suhu", "kelembaban"]] = data_resampled[["suhu", "kelembaban"]].ffill()
    return data_resampled


def siapkan_untuk_model(data, kolom):
    resampled = data.set_index("waktu").resample("15min").mean(numeric_only=True).dropna()
    df_model = resampled.reset_index()[["waktu", kolom]].rename(columns={"waktu": "ds", kolom: "y"})
    df_model["ds"] = df_model["ds"].dt.tz_localize(None)
    return df_model


def hitung_metrik(y_asli, y_prediksi):
    mae = mean_absolute_error(y_asli, y_prediksi)
    rmse = np.sqrt(mean_squared_error(y_asli, y_prediksi))
    mape = np.mean(np.abs((y_asli - y_prediksi) / y_asli)) * 100
    return mae, rmse, mape


@st.cache_data(ttl=300)
def latih_dan_validasi(df_suhu, df_kelembaban, jam_testing=6):
    waktu_potong = df_suhu["ds"].max() - pd.Timedelta(hours=jam_testing)

    train_suhu = df_suhu[df_suhu["ds"] <= waktu_potong]
    test_suhu = df_suhu[df_suhu["ds"] > waktu_potong]
    train_kelembaban = df_kelembaban[df_kelembaban["ds"] <= waktu_potong]
    test_kelembaban = df_kelembaban[df_kelembaban["ds"] > waktu_potong]

    model_suhu = Prophet(daily_seasonality=True)
    model_suhu.fit(train_suhu)
    future_suhu = model_suhu.make_future_dataframe(periods=jam_testing, freq="h")
    forecast_suhu = model_suhu.predict(future_suhu)
    hasil_suhu = forecast_suhu[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(jam_testing)

    train_kelembaban = train_kelembaban.copy()
    train_kelembaban["cap"] = 100
    train_kelembaban["floor"] = 0
    model_kelembaban = Prophet(daily_seasonality=True, growth="logistic")
    model_kelembaban.fit(train_kelembaban)
    future_kelembaban = model_kelembaban.make_future_dataframe(periods=jam_testing, freq="h")
    future_kelembaban["cap"] = 100
    future_kelembaban["floor"] = 0
    forecast_kelembaban = model_kelembaban.predict(future_kelembaban)
    hasil_kelembaban = forecast_kelembaban[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(jam_testing)

    test_suhu_hourly = test_suhu.set_index("ds").resample("h").mean().dropna().reset_index()
    test_kelembaban_hourly = test_kelembaban.set_index("ds").resample("h").mean().dropna().reset_index()

    n = min(len(test_suhu_hourly), len(hasil_suhu))

    mae_suhu, rmse_suhu, mape_suhu = hitung_metrik(test_suhu_hourly["y"].values[:n], hasil_suhu["yhat"].values[:n])
    mae_kelembaban, rmse_kelembaban, mape_kelembaban = hitung_metrik(
        test_kelembaban_hourly["y"].values[:n], hasil_kelembaban["yhat"].values[:n]
    )

    return {
        "hasil_suhu": hasil_suhu, "hasil_kelembaban": hasil_kelembaban,
        "test_suhu_hourly": test_suhu_hourly, "test_kelembaban_hourly": test_kelembaban_hourly,
        "n": n,
        "metrik_suhu": (mae_suhu, rmse_suhu, mape_suhu),
        "metrik_kelembaban": (mae_kelembaban, rmse_kelembaban, mape_kelembaban),
    }


@st.cache_data(ttl=300)
def latih_dan_prediksi_masa_depan(df_suhu, df_kelembaban, jam_prediksi=6):
    model_suhu = Prophet(daily_seasonality=True)
    model_suhu.fit(df_suhu)
    future_suhu = model_suhu.make_future_dataframe(periods=jam_prediksi, freq="h")
    forecast_suhu = model_suhu.predict(future_suhu)
    hasil_suhu = forecast_suhu[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(jam_prediksi)

    df_kelembaban = df_kelembaban.copy()
    df_kelembaban["cap"] = 100
    df_kelembaban["floor"] = 0
    model_kelembaban = Prophet(daily_seasonality=True, growth="logistic")
    model_kelembaban.fit(df_kelembaban)
    future_kelembaban = model_kelembaban.make_future_dataframe(periods=jam_prediksi, freq="h")
    future_kelembaban["cap"] = 100
    future_kelembaban["floor"] = 0
    forecast_kelembaban = model_kelembaban.predict(future_kelembaban)
    hasil_kelembaban = forecast_kelembaban[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(jam_prediksi)

    return hasil_suhu, hasil_kelembaban


@st.cache_data(ttl=300)
def get_llm_comment(suhu, kelembaban, waktu, stat_suhu, stat_kelembaban):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    prompt = f"""
Anda adalah analis lingkungan berpengalaman. Analisis data sensor IoT berikut:

DATA TERKINI:
- Suhu: {suhu}°C
- Kelembaban: {kelembaban}%
- Waktu: {waktu}

DATA HISTORIS:
- Rata-rata suhu: {stat_suhu['avg']:.2f}°C (Min: {stat_suhu['min']:.2f}, Max: {stat_suhu['max']:.2f})
- Rata-rata kelembaban: {stat_kelembaban['avg']:.2f}% (Min: {stat_kelembaban['min']:.2f}, Max: {stat_kelembaban['max']:.2f})

Berikan output singkat (3-5 kalimat) yang memuat:
1. Analisis kondisi lingkungan saat ini (nyaman/tidak nyaman)
2. Insight menarik dari perbandingan data terkini vs historis
3. Rekomendasi tindakan jika diperlukan
"""
    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "system", "content": "Anda adalah analis lingkungan yang memberikan insight singkat dan actionable."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 300,
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ Gagal mengambil komentar AI: {e}"


# ============================================================
# APP
# ============================================================

inject_style()

df = ambil_semua_data()
data_terakhir = df.iloc[-1]
data_sebelumnya = df.iloc[-2] if len(df) > 1 else data_terakhir
data_ok = len(df) > 0

# ---- Sidebar navigasi (sesuai blueprint) ----
with st.sidebar:
    st.markdown("### 🌐 Smart Environment")
    halaman = st.radio(
        "Navigasi",
        ["📊 Dashboard", "📈 Analytics", "🔮 AI Forecast", "🗄️ Database"],
        label_visibility="collapsed",
    )
    st.divider()
    if st.button("🔄 Segarkan Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Data otomatis diperbarui tiap 60 detik.")

# ---- Header ----
status = "online" if data_ok else "offline"
pill_class = "" if data_ok else "offline"
st.markdown(
    f"""
    <div class="app-header">
        <div>
            <p class="app-title">🌡️ Smart Environment Monitoring</p>
            <p class="app-subtitle">Real-Time IoT Monitoring & AI Forecast</p>
        </div>
        <div class="pill-row">
            <span class="pill {pill_class}">MQTT</span>
            <span class="pill {pill_class}">Railway</span>
            <span class="pill {pill_class}">Supabase</span>
            <span class="pill {pill_class}">ESP32</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

suhu_now = df["suhu"].dropna().iloc[-1]
kelembaban_now = df["kelembaban"].dropna().iloc[-1]
suhu_prev = df["suhu"].dropna().iloc[-2] if len(df["suhu"].dropna()) > 1 else suhu_now
kelembaban_prev = df["kelembaban"].dropna().iloc[-2] if len(df["kelembaban"].dropna()) > 1 else kelembaban_now


# ============================================================
# HALAMAN: DASHBOARD
# ============================================================
if halaman == "📊 Dashboard":

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("🌡️", "Temperature", f"{suhu_now:.1f} °C", f"{abs(suhu_now - suhu_prev):.1f} °C", "orange", suhu_now >= suhu_prev)
    with c2:
        metric_card("💧", "Humidity", f"{kelembaban_now:.1f} %", f"{abs(kelembaban_now - kelembaban_prev):.1f} %", "blue", kelembaban_now >= kelembaban_prev)
    with c3:
        metric_card("📊", "Total Records", f"{len(df):,}", "data tersimpan", "green", True)
    with c4:
        # AI prediction jam berikutnya, dihitung otomatis (tanpa tombol)
        try:
            df_suhu_model = siapkan_untuk_model(df, "suhu")
            df_kelembaban_model = siapkan_untuk_model(df, "kelembaban")
            hasil_suhu_depan, _ = latih_dan_prediksi_masa_depan(df_suhu_model, df_kelembaban_model, jam_prediksi=1)
            pred = hasil_suhu_depan.iloc[0]
            confidence = max(0, 100 - (pred["yhat_upper"] - pred["yhat_lower"]) / max(pred["yhat"], 1) * 100)
            metric_card("🤖", "AI Prediction (1h)", f"{pred['yhat']:.1f} °C", f"confidence {confidence:.0f}%", "purple", True)
        except Exception:
            metric_card("🤖", "AI Prediction", "—", "menunggu data cukup", "purple", True)

    st.write("")
    col1, col2 = st.columns(2)
    df_chart = siapkan_untuk_chart(df)
    with col1:
        st.markdown('<div class="glass-card"><div class="section-title">📈 Temperature Trend</div>', unsafe_allow_html=True)
        st.area_chart(df_chart[["suhu"]], color="#f97316")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="glass-card"><div class="section-title">📈 Humidity Trend</div>', unsafe_allow_html=True)
        st.area_chart(df_chart[["kelembaban"]], color="#38bdf8")
        st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📐 Statistics</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#9aa3b8;font-size:13px;margin-top:-6px;">Suhu (°C)</p>', unsafe_allow_html=True)
        s = df["suhu"]
        st.markdown(
            f"""
            <div class="stat-grid">
                <div class="stat-box"><div class="v">{s.mean():.1f}</div><div class="l">Avg</div></div>
                <div class="stat-box"><div class="v">{s.median():.1f}</div><div class="l">Median</div></div>
                <div class="stat-box"><div class="v">{s.min():.1f}</div><div class="l">Min</div></div>
                <div class="stat-box"><div class="v">{s.max():.1f}</div><div class="l">Max</div></div>
                <div class="stat-box"><div class="v">{s.std():.1f}</div><div class="l">Std Dev</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<p style="color:#9aa3b8;font-size:13px;margin-top:14px;">Kelembaban (%)</p>', unsafe_allow_html=True)
        k = df["kelembaban"]
        st.markdown(
            f"""
            <div class="stat-grid">
                <div class="stat-box"><div class="v">{k.mean():.1f}</div><div class="l">Avg</div></div>
                <div class="stat-box"><div class="v">{k.median():.1f}</div><div class="l">Median</div></div>
                <div class="stat-box"><div class="v">{k.min():.1f}</div><div class="l">Min</div></div>
                <div class="stat-box"><div class="v">{k.max():.1f}</div><div class="l">Max</div></div>
                <div class="stat-box"><div class="v">{k.std():.1f}</div><div class="l">Std Dev</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🤖 AI Insight</div>', unsafe_allow_html=True)
        stat_suhu = {"avg": s.mean(), "min": s.min(), "max": s.max()}
        stat_kelembaban = {"avg": k.mean(), "min": k.min(), "max": k.max()}
        with st.spinner("Menganalisis pola data..."):
            komentar = get_llm_comment(
                suhu=suhu_now, kelembaban=kelembaban_now,
                waktu=data_terakhir["waktu"].strftime("%Y-%m-%d %H:%M:%S"),
                stat_suhu=stat_suhu, stat_kelembaban=stat_kelembaban,
            )
        st.markdown(f'<p class="ai-insight-text">{komentar}</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🕒 Recent Activity</div>', unsafe_allow_html=True)
        st.dataframe(
            df[["waktu", "suhu", "kelembaban"]].tail(10).sort_values("waktu", ascending=False),
            use_container_width=True, hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📅 Recording Progress</div>', unsafe_allow_html=True)

        hari_mulai = pd.Timestamp(RECORDING_START_DATE)
        sekarang = pd.Timestamp.now()
        # samakan timezone-awareness antara hari_mulai dan sekarang supaya tidak error
        if df["waktu"].dt.tz is not None:
            if hari_mulai.tzinfo is None:
                hari_mulai = hari_mulai.tz_localize(df["waktu"].dt.tz)
            sekarang = pd.Timestamp.now(tz=df["waktu"].dt.tz)

        durasi_berjalan = sekarang - hari_mulai
        hari_berjalan = max(durasi_berjalan.days, 0)
        jam_sisa = int(durasi_berjalan.seconds // 3600)

        total_jam_target = TARGET_DAYS * 24
        total_jam_berjalan = durasi_berjalan.total_seconds() / 3600
        persentase = min(100, (total_jam_berjalan / total_jam_target) * 100) if total_jam_target else 0
        estimasi_selesai = hari_mulai + pd.Timedelta(days=TARGET_DAYS)

        st.markdown(
            f"""
            <div class="progress-wrap"><div class="progress-bar" style="width:{persentase:.0f}%;"></div></div>
            <p style="color:#9aa3b8;font-size:13px;margin-top:10px;">
                Mulai: {hari_mulai.strftime('%d %b %Y, %H:%M')} &nbsp;•&nbsp;
                Hari ke-{hari_berjalan} ({hari_berjalan} hari {jam_sisa} jam berjalan) dari target {TARGET_DAYS} hari<br>
                {persentase:.1f}% selesai &nbsp;•&nbsp; Estimasi selesai: {estimasi_selesai.strftime('%d %b %Y')}
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HALAMAN: ANALYTICS (dulu bagian validasi model)
# ============================================================
elif halaman == "📈 Analytics":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">✅ Validasi Model — akurasi 6 jam terakhir</div>', unsafe_allow_html=True)
    df_suhu_model = siapkan_untuk_model(df, "suhu")
    df_kelembaban_model = siapkan_untuk_model(df, "kelembaban")

    with st.spinner("Melatih & menguji model..."):
        hasil_validasi = latih_dan_validasi(df_suhu_model, df_kelembaban_model, jam_testing=6)

    mae_s, rmse_s, mape_s = hasil_validasi["metrik_suhu"]
    mae_k, rmse_k, mape_k = hasil_validasi["metrik_kelembaban"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE Suhu", f"{mae_s:.2f}")
    c2.metric("RMSE Suhu", f"{rmse_s:.2f}")
    c3.metric("MAE Kelembaban", f"{mae_k:.2f}")
    c4.metric("RMSE Kelembaban", f"{rmse_k:.2f}")
    st.caption(f"MAPE Suhu: {mape_s:.2f}% • MAPE Kelembaban: {mape_k:.2f}%")

    n = hasil_validasi["n"]
    chart_suhu = pd.DataFrame({
        "Aktual": hasil_validasi["test_suhu_hourly"]["y"].values[:n],
        "Prediksi": hasil_validasi["hasil_suhu"]["yhat"].values[:n],
    }, index=hasil_validasi["hasil_suhu"]["ds"].values[:n])
    chart_kelembaban = pd.DataFrame({
        "Aktual": hasil_validasi["test_kelembaban_hourly"]["y"].values[:n],
        "Prediksi": hasil_validasi["hasil_kelembaban"]["yhat"].values[:n],
    }, index=hasil_validasi["hasil_kelembaban"]["ds"].values[:n])

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Suhu: Aktual vs Prediksi**")
        st.line_chart(chart_suhu)
    with col2:
        st.write("**Kelembaban: Aktual vs Prediksi**")
        st.line_chart(chart_kelembaban)
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HALAMAN: AI FORECAST (dulu bagian prediksi masa depan)
# ============================================================
elif halaman == "🔮 AI Forecast":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔮 Prediksi 6 Jam ke Depan</div>', unsafe_allow_html=True)
    st.caption("Model dilatih otomatis dengan seluruh data yang ada, memprediksi 6 jam setelah data terakhir masuk.")

    df_suhu_model = siapkan_untuk_model(df, "suhu")
    df_kelembaban_model = siapkan_untuk_model(df, "kelembaban")

    with st.spinner("Melatih model & membuat prediksi..."):
        hasil_suhu_depan, hasil_kelembaban_depan = latih_dan_prediksi_masa_depan(
            df_suhu_model, df_kelembaban_model, jam_prediksi=6
        )

    tabel_prediksi = pd.DataFrame({
        "Jam": hasil_suhu_depan["ds"].dt.strftime("%Y-%m-%d %H:%M"),
        "Prediksi Suhu (°C)": hasil_suhu_depan["yhat"].round(2),
        "Prediksi Kelembaban (%)": hasil_kelembaban_depan["yhat"].round(2),
    })
    st.dataframe(tabel_prediksi, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Grafik Prediksi Suhu**")
        st.line_chart(hasil_suhu_depan.set_index("ds")[["yhat"]])
    with col2:
        st.write("**Grafik Prediksi Kelembaban**")
        st.line_chart(hasil_kelembaban_depan.set_index("ds")[["yhat"]])
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HALAMAN: DATABASE
# ============================================================
elif halaman == "🗄️ Database":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🗄️ Filter & Jelajahi Data</div>', unsafe_allow_html=True)
    tanggal_min = df["waktu"].min().date()
    tanggal_max = df["waktu"].max().date()
    rentang = st.date_input(
        "Pilih rentang tanggal", value=(tanggal_min, tanggal_max),
        min_value=tanggal_min, max_value=tanggal_max, key="filter_db",
    )
    if len(rentang) == 2:
        mulai, akhir = rentang
        df_filtered = df[(df["waktu"].dt.date >= mulai) & (df["waktu"].dt.date <= akhir)]
    else:
        df_filtered = df
    st.dataframe(df_filtered.sort_values("waktu", ascending=False), use_container_width=True, hide_index=True)
    st.caption(f"{len(df_filtered):,} baris ditampilkan dari total {len(df):,} baris.")
    st.markdown("</div>", unsafe_allow_html=True)
