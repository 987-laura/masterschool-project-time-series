# 📈 Projekt: Nachfrageprognose im Einzelhandel – Region Guayas

## 🧭 Projektübersicht

Dieses Projekt beschäftigt sich mit der Vorhersage von Nachfrage auf Basis historischer Zeitreihendaten. Es kombiniert klassische Machine-Learning-Modelle (XGBoost) mit Deep-Learning-Ansätzen (LSTM), um präzise Prognosen für verschiedene Regionen und Produkte zu erstellen.

Die App bietet:
- 📅 Prognosen für mehrere Tage (N-Tage)
- 📊 Visualisierung von tatsächlicher vs. prognostizierter Nachfrage
- 🧠 Einsatz eines trainierten Modells (LSTM oder XGBoost)

---

## 🧠 Modellwahl & Leistung

Zwei Modelle wurden getestet:
- **XGBoost**: Schnell, interpretierbar und robust für tabellarische Zeitreihendaten
- **LSTM**: Deep-Learning-Modell für sequenzielle Muster

## Daten

Das LSTM-Modell sowie der zugehörige Scaler befinden sich im Ordner `model/`. Eine große Datei (`lstm_test_data.pkl`) ist extern gespeichert und kann [hier heruntergeladen werden](https://drive.google.com/file/d/1t9uBMEeVVr3g-y7YyH_HsqBcLCHfTxl8/view?usp=share_link).

---

## 📁 Projektstruktur

```
retail_demand_forecast/
│
├── app/
│   ├── streamlit_app.py         # Haupt-UI der Streamlit-App
│
├── model/
│   ├── lstm_model.h5            # Trainiertes LSTM-Modell
│   ├── lstm_scaler.pkl          # Scaler für LSTM-Eingaben
│   ├── xgb12monate.pkl          # Trainiertes XGBoost-Modell
│   └── lstm_test_data.pkl       # [Externe Datei – siehe Link oben]
│
├── notebooks/
│   ├── 1-Time-series-Project-data-prep.ipynb
│   ├── 2-Time-Series-Project-feature_eng.ipynb
│   ├── 3-Time-Series-Project-xgboost.ipynb
│   ├── 4-Time-Series-Project-lstm.ipynb
│   └── plots/                   # Visualisierungen
│
├── files/                       # # [Externe Dateien – siehe Link unten]
│   ├── holidays_events.csv
│   ├── items.csv
│   ├── oil.csv
│   ├── stores.csv
│   ├── train.csv
│   ├── df_train_clean.parquet
│   ├── df_test_clean.parquet
│   ├── df_train_ML_ready.pkl
│   ├── df_test_ML_ready.pkl
│   └── df_train_test.pkl
│
├── requirements.txt             # Python-Abhängigkeiten
├── README.md                    # Diese Datei
```

📂 Zusätzliche Daten (CSV, Parquet, PKL) befinden sich hier: [Google Drive Ordner](https://drive.google.com/drive/folders/1NAQd17M4ce-c7iTSYXrK_teYiuWFWleb?usp=share_link).

---

## 🚀 Lokale Ausführung der App

1. **Repository klonen**  
   ```bash
   git clone https://github.com/deinusername/retail_demand_forecast.git
   cd retail_demand_forecast
   ```

2. **Abhängigkeiten installieren**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Streamlit-App starten**  
   ```bash
   streamlit run app/streamlit_app.py
   ```

4. **Optional**: Lade `lstm_test_data.pkl` herunter und platziere sie im `model/`-Ordner, falls du das LSTM-Modell verwenden möchtest.

---

## ⚠️ Hinweise

- Die **Streamlit-App ist zeitbedingt sehr minimalistisch gehalten**, aber grundsätzlich funktional.
- Die **Notebooks** enthalten teilweise Code, der schöner aufgeräumt werden könnte (ebenfalls Zeitgründe). Des weiteren wäre es notwendig die Features weiter zu bearbeiten, damit das Modell Ausreißer besser einfangen bzw. erkennen kann.
- Große Dateien wurden **nicht ins Git-Repository aufgenommen**, sondern extern gespeichert.
- MLflow-Tracking-Dateien sind der Übersichtlichkeit halber in dieser Version nicht enthalten, könnten nachgereicht oder selbst generiert werden.
- Der Zeitraum der Modelle wurde auf ein Jahr erhöht, um saisonale Schwankungen besser berücksichtigen zu können.
- Die Präsentation fand im Rahmen der Live-Session statt: https://masterschool.zoom.us/rec/play/94lySMVfVYHIW0IwBf4-oncPfzCmVVdP7rnm7tGw7giTNEPRCTBRaMHCEIETmes6sR1gBWMzl3d7WCQV.BIOJRY33sYqAsh4k

---

## 📸 Screenshots

### Streamlit-App
<img width="400" height="auto" alt="Screenshot-streamlit-app" src="https://github.com/user-attachments/assets/81588650-f98a-4472-af3b-04a2c492a438" />

### XGBoost-Modell-Plot
<img width="300" height="auto" alt="forecast_XGB_12Months" src="https://github.com/user-attachments/assets/f11d87a8-85bf-4f1f-92b3-4b02cd9feed8" />

### LSTM-Modell-Plot
<img width="300" height="auto" alt="forecast_LSTM_12Months" src="https://github.com/user-attachments/assets/1a601092-6d4f-42b2-8ee3-cac6201020bc" />

---
