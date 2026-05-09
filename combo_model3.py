import os
import pickle
import numpy as np
from datetime import datetime, timedelta
from meteostat import Point, Hourly
import pytz
from tensorflow.keras.models import load_model
import math

# ===================== CONFIG =====================
MODEL_FILE = "lstm_karachi_multi_v1.h5"
FEATURE_SCALER_FILE = "feature_scaler_multi_v1.pkl"

KARACHI_LAT, KARACHI_LON = 24.8608, 67.0104
KARACHI_TZ = pytz.timezone('Asia/Karachi')

SEQUENCE_LENGTH = 120
FEATURES = ['temp', 'dwpt', 'coco', 'hour', 'rhum', 'pres']

PANEL_KW = 3.0
SYSTEM_EFFICIENCY = 0.90


# ===================== DATA =====================
def get_last_sequence():

    with open(FEATURE_SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)

    end_utc = datetime.now(KARACHI_TZ).astimezone(pytz.UTC).replace(tzinfo=None)
    start_utc = end_utc - timedelta(hours=SEQUENCE_LENGTH + 24)

    df = Hourly(Point(KARACHI_LAT, KARACHI_LON), start_utc, end_utc).fetch()
    df.index = df.index.tz_localize(pytz.UTC).tz_convert(KARACHI_TZ)

    df['hour'] = df.index.hour

    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0

    df = df[FEATURES].copy().dropna()

    seq = df[-SEQUENCE_LENGTH:]
    scaled = scaler.transform(seq.to_numpy())

    return scaled.reshape(1, SEQUENCE_LENGTH, len(FEATURES))


# ===================== SOLAR MODEL =====================
def predict_next_hour():

    model = load_model(MODEL_FILE, compile=False)
    x = get_last_sequence()

    pred = model.predict(x, verbose=0)

    with open(FEATURE_SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)

    pred_actual = scaler.inverse_transform(pred)

    temp = pred_actual[0][0]
    coco = round(pred_actual[0][2])

    next_time = datetime.now(KARACHI_TZ) + timedelta(hours=1)
    day = next_time.timetuple().tm_yday
    hour = next_time.hour

    solar_decl = -23.44 * math.cos(math.radians(360/365*(day+10)))
    lat = math.radians(KARACHI_LAT)
    ha = math.radians(15*(hour-12))

    cosz = (math.sin(lat)*math.sin(math.radians(solar_decl)) +
            math.cos(lat)*math.cos(math.radians(solar_decl))*math.cos(ha))

    cosz = max(cosz, 0)

    if cosz <= 0:
        ghi = 0
    else:
        solar_const = 1367
        dist = 1 + 0.033*math.cos(math.radians(360*day/365))
        ghi = solar_const * dist * cosz
        air = 1/cosz
        ghi *= 0.7 ** (air ** 0.678)

    cloud_map = {1:0.0,2:0.1,3:0.3,4:0.5,5:0.7,6:0.9,7:1.0}
    cloud = cloud_map.get(min(max(coco,1),7),0.8)

    solar_after_cloud = ghi*(1-cloud) + ghi*cloud*0.3

    temp_corr = 1 - 0.004*max(temp-25,0)

    solar_kw = (solar_after_cloud/1000)*PANEL_KW*SYSTEM_EFFICIENCY*temp_corr
    solar_kw = max(0, min(solar_kw, PANEL_KW))

    print("\n🌡️ Temp:", round(temp,2))
    print("☁️ Cloud:", coco)
    print("☀️ Solar Output:", round(solar_kw,2),"kW")

    print("🔹 GHI:", round(ghi))
    print("🔹 After Cloud:", round(solar_after_cloud))
    print("🔹 Temp Factor:", round(temp_corr,2))

    return solar_kw


# ===================== PSO =====================
def pso_optimize(appliances, solar_watts, critical_load):

    flexible = [n for n,d in appliances.items() if d['type']=='flexible']
    n = len(flexible)

    particles = [np.random.randint(0,2,n) for _ in range(30)]
    velocity = [np.zeros(n) for _ in range(30)]

    pbest = particles.copy()
    pbest_score = [-1e9]*30

    gbest = None
    gbest_score = -1e9

    available = max(0, solar_watts - critical_load)

    def fitness(x):

        total = 0

        for i in range(n):
            if x[i] == 1:
                total += appliances[flexible[i]]['qty'] * appliances[flexible[i]]['watts']

        if total <= available:
            return total

        return available - (total - available)*2

    w,c1,c2 = 0.5,1.5,1.5

    for _ in range(40):

        for i in range(30):

            score = fitness(particles[i])

            if score > pbest_score[i]:
                pbest[i] = particles[i].copy()
                pbest_score[i] = score

            if score > gbest_score:
                gbest = particles[i].copy()
                gbest_score = score

        for i in range(30):

            r1,r2 = np.random.rand(n),np.random.rand(n)

            velocity[i] = (
                w*velocity[i] +
                c1*r1*(pbest[i]-particles[i]) +
                c2*r2*(gbest-particles[i])
            )

            prob = 1/(1+np.exp(-velocity[i]))

            particles[i] = np.array([
                1 if np.random.rand()<prob[j] else 0
                for j in range(n)
            ])

    return {flexible[i]: bool(gbest[i]) for i in range(n)}


# ===================== OPTIMIZER =====================
def optimize_next_hour(solar_watts, appliances):

    critical = sum(d['qty']*d['watts']
                   for d in appliances.values()
                   if d['type']=='critical')

    best = pso_optimize(appliances, solar_watts, critical)

    available = max(0, solar_watts - critical)

    used = 0
    result = []

    for name, state in best.items():

        power = appliances[name]['qty'] * appliances[name]['watts']

        if state and available >= power:
            result.append(f"✅ {name.upper()} full ON ({power}W)")
            available -= power
            used += power

        elif state:
            result.append(f"⚡ {name.upper()} PARTIAL")

        else:
            result.append(f"❌ {name.upper()} OFF")

    grid = max(0, critical + used - solar_watts)

    return {
        "critical": critical,
        "recommendations": result,
        "grid": grid,
        "available": available
    }


# ===================== MAIN =====================
if __name__ == "__main__":

    appliances = {
        'fridge': {'qty':1,'watts':150,'type':'flexible'},
        'fan': {'qty':2,'watts':80,'type':'critical'},
        'lights': {'qty':5,'watts':20,'type':'critical'},
        'AC': {'qty':1,'watts':1200,'type':'flexible'},
        'washing_machine': {'qty':1,'watts':500,'type':'flexible'},
        'water_pump': {'qty':1,'watts':750,'type':'flexible'},
    }

    solar_kw = predict_next_hour()
    solar_watts = solar_kw * 1000

    result = optimize_next_hour(solar_watts, appliances)

    print("\n⚡ Critical Load:", result['critical'])

    print("\n🔋 Flexible Recommendations:")
    for r in result['recommendations']:
        print("  ", r)

    print("\n🔌 Grid Usage:", result['grid'], "W")
    print("🔋 Remaining Solar:", result['available'], "W")