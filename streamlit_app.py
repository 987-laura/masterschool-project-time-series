# importe
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# App-Konfiguration
st.set_page_config(page_title="LSTM Forecast Guayas", layout="centered")
st.title("Vorhersagemodell für die Guayas Region")
st.write("Wähle mit dem Slider die Anzahl der Tage aus, die vorhergesagt werden sollen.")

MODEL_PATH = "/Users/laurapeters/Desktop/VS-Time-series-Projekt/model/lstm_model.h5"
DATA_PATH  = "/Users/laurapeters/Desktop/VS-Time-series-Projekt/model/lstm_test_data.pkl"
SCALER_PATH = "/Users/laurapeters/Desktop/VS-Time-series-Projekt/model/lstm_scaler.pkl"

# Modell und Daten laden
model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
X_test, y_test, dates_test = joblib.load(DATA_PATH)

# Sicherstellen, dass dates_test ein DatetimeIndex ist
dates_test = pd.to_datetime(dates_test)

# Slider
max_days = min(30, len(X_test))
n_days = st.slider("Anzahl der Tage für Vorhersage:", 1, max_days, value=max_days)

X_n = X_test[:n_days]
y_n = y_test[:n_days]
dates_n = dates_test[:n_days]

# Modell
y_pred = model.predict(X_n).flatten()

# Rücktransformation der Zielwerte
def inverse_scale_target(scaled_values, scaler, target_col=0):
    dummy = np.zeros((len(scaled_values), scaler.n_features_in_))
    dummy[:, target_col] = scaled_values.flatten()
    inversed = scaler.inverse_transform(dummy)
    return inversed[:, target_col]

y_n_orig = inverse_scale_target(y_n, scaler)
y_pred_orig = inverse_scale_target(y_pred, scaler)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dates_n, y_n_orig, label="Ist-Werte", marker="o")
ax.plot(dates_n, y_pred_orig, label="Vorhersage", marker="x", color="red")

ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Wichtige Ticks automatisch setzen
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # Format z.B. 2014-12-31
fig.autofmt_xdate(rotation=45)  # Datumslabels drehen


ax.set_title(f"Vorhersage für {n_days} Tage")
ax.set_xlabel("Datum")
ax.set_ylabel("Unit Sales")
ax.legend()
ax.grid(True)
fig.autofmt_xdate()
st.pyplot(fig)

