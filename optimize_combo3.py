import os
import pickle
import numpy as np
from datetime import datetime, timedelta
from meteostat import Point, Hourly
import pytz
from tensorflow.keras.models import load_model
import math
import json

# ===================== CONFIG =====================
MODEL_FILE = "lstm_karachi_multi_v1.h5"
FEATURE_SCALER_FILE = "feature_scaler_multi_v1.pkl"

KARACHI_LAT, KARACHI_LON = 24.8608, 67.0104
KARACHI_TZ = pytz.timezone('Asia/Karachi')

SEQUENCE_LENGTH = 120
FEATURES = ['temp', 'dwpt', 'coco', 'hour', 'rhum', 'pres']

PANEL_KW = 5.0
SYSTEM_EFFICIENCY = 0.90

BATTERY_FILE = "battery.json"

# Karachi sun hours (realistic)
MONTHLY_SUN_HOURS = {
    1: 4.0, 2: 4.5, 3: 5.5, 4: 6.5,
    5: 7.5, 6: 7.8, 7: 6.5, 8: 6.2,
    9: 6.0, 10: 5.5, 11: 4.5, 12: 4.0
}

# KE tariff simple
def get_unit_price(units):
    if units < 100:
        return 25
    elif units < 300:
        return 40
    return 55

# ===================== BATTERY =====================
def load_battery():
    if os.path.exists(BATTERY_FILE):
        return json.load(open(BATTERY_FILE))
    return {"stored_watts": 0}

def save_battery(b):
    json.dump(b, open(BATTERY_FILE, "w"))

# ===================== DATA =====================
def get_last_sequence():
    with open(FEATURE_SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)

    end = datetime.now(KARACHI_TZ).astimezone(pytz.UTC).replace(tzinfo=None)
    start = end - timedelta(hours=SEQUENCE_LENGTH + 24)

    df = Hourly(Point(KARACHI_LAT, KARACHI_LON), start, end).fetch()
    df.index = df.index.tz_localize(pytz.UTC).tz_convert(KARACHI_TZ)

    df["hour"] = df.index.hour
    for f in FEATURES:
        if f not in df:
            df[f] = 0

    df = df[FEATURES].dropna()
    return scaler.transform(df[-SEQUENCE_LENGTH:]), df

# ===================== SOLAR PREDICTION (FIXED REALISTIC) =====================
def predict_solar():
    model = load_model(MODEL_FILE, compile=False)
    x, _ = get_last_sequence()

    with open(FEATURE_SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)

    pred = model.predict(x.reshape(1, SEQUENCE_LENGTH, len(FEATURES)), verbose=0)
    pred = scaler.inverse_transform(pred)

    temp = float(pred[0][0])
    cloud = int(round(pred[0][2]))

    base = PANEL_KW * SYSTEM_EFFICIENCY

    # FIXED realistic scaling (important correction)
    cloud_factor = 1 - (cloud * 0.06)
    cloud_factor = max(0.5, cloud_factor)

    temp_factor = 1 - max(0, (temp - 32) * 0.002)

    atmospheric_loss = 0.93

    solar_kw = base * cloud_factor * temp_factor * atmospheric_loss
    solar_kw = max(0.5, min(solar_kw, PANEL_KW))

    print("\n🌡 Temp:", round(temp,2))
    print("☁ Cloud:", cloud)
    print("☀ Solar:", round(solar_kw,2), "kW")

    return solar_kw

# ===================== PSO (ON/OFF ONLY) =====================
def decide(appliances, available_watts):
    priority = {"fan": 5, "lights": 5, "fridge": 4, "AC": 2, "motor": 3}

    items = sorted(appliances.items(),
                   key=lambda x: priority.get(x[0], 1),
                   reverse=True)

    result = {}
    used = 0

    for name, data in items:
        load = data["qty"] * data["watts"]

        if used + load <= available_watts:
            result[name] = "ON"
            used += load
        else:
            result[name] = "OFF"

    return result, used

# ===================== BATTERY =====================
def battery_update(solar, used, battery):
    excess = solar - used

    if excess > 0:
        battery["stored_watts"] += excess
        print(f"🔋 Battery +{excess:.0f}W store")
    else:
        battery["stored_watts"] = max(0, battery["stored_watts"] + excess)
        print(f"🔋 Battery -{abs(excess):.0f}W used")

    save_battery(battery)
    return battery

# ===================== MAIN =====================
if __name__ == "__main__":

    appliances = {
        "fan": {"qty": 4, "watts": 80},
        "lights": {"qty": 8, "watts": 20},
        "fridge": {"qty": 1, "watts": 150},
        "AC": {"qty": 1, "watts": 1200},
        "motor": {"qty": 1, "watts": 750}
    }

    battery = load_battery()

    solar_kw = predict_solar()
    solar_watts = solar_kw * 1000

    decision, used = decide(appliances, solar_watts)

    battery = battery_update(solar_watts, used, battery)

    # ===================== OUTPUT =====================
    print("\n⚡ APPLIANCE STATUS")
    for k, v in decision.items():
        print(k, ":", v)

    print("\n☀ Solar Power:", round(solar_watts), "W")
    print("⚡ Used:", round(used), "W")

    # ===================== DAILY ESTIMATION (RESTORED) =====================
    month = datetime.now(KARACHI_TZ).month
    sun_hours = MONTHLY_SUN_HOURS[month]

    daily_units = (
        PANEL_KW *
        sun_hours *
        SYSTEM_EFFICIENCY *
        0.9   # real world loss
    )

    monthly_units = daily_units * 30
    price = get_unit_price(monthly_units)
    savings = daily_units * price

    print(f"\n📈 Aaj expected units: ~{daily_units:.2f} kWh")
    print(f"💰 Savings: ~Rs {savings:.0f} / day")

    print("\n🔋 Battery Stored:", round(battery["stored_watts"]), "W")