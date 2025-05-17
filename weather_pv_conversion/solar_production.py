import datetime
import joblib   
import numpy as np
import pandas as pd
import openmeteo_requests # pip install openmeteo-requests
import requests_cache     # pip install requests-cache requests-cache[all]
from retry_requests import retry # pip install retry-requests
import os # Per os.path.join e os.path.exists

# Non abbiamo più bisogno di creare un modello dummy qui, se il file è corretto
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler


class SolarProductionPredictor:
    def __init__(self, model_path="output/model/best_estimator_model.pkl",
                 scaler_path="output/model/scaler.pkl"):
        """
        Inizializza il predittore caricando il modello e lo scaler.

        Args:
            model_path (str): Percorso al file del modello serializzato (preferibilmente .joblib o .pkl).
            scaler_path (str): Percorso al file dello scaler serializzato (preferibilmente .joblib o .pkl).
        """
        self.model = None
        self.scaler = None
        self.model_features_names = None # Nomi delle feature che il modello si aspetta

        # 1. Carica il modello
        if not os.path.exists(model_path):
            print(f"ERRORE CRITICO: File del modello non trovato in '{os.path.abspath(model_path)}'. Impossibile procedere.")
            raise FileNotFoundError(f"File del modello non trovato: {model_path}")
        
        try:
            self.model = joblib.load(model_path) # Usa joblib.load
            print(f"Modello caricato con successo da: {os.path.abspath(model_path)}")
            
            if hasattr(self.model, 'feature_names_in_'):
                self.model_features_names = list(self.model.feature_names_in_) # Converti in lista
                print(f"Il modello si aspetta le seguenti feature: {self.model_features_names}")
            else:
                print("ATTENZIONE: Il modello caricato non ha l'attributo 'feature_names_in_'. "
                      "Assicurarsi che la preparazione dei dati corrisponda esattamente all'addestramento del modello.")
                # Se feature_names_in_ non è disponibile, dovrai definire self.model_features_names
                # manualmente o assicurarti che il codice in predict() sia corretto.
                # Per il nostro caso SVR, dovrebbe averlo.
        except Exception as e:
            print(f"ERRORE CRITICO: Errore durante il caricamento del modello da '{os.path.abspath(model_path)}': {e}")
            raise

        # 2. Carica lo scaler
        if not os.path.exists(scaler_path):
            print(f"ATTENZIONE: File dello scaler non trovato in '{os.path.abspath(scaler_path)}'. "
                  "Se il modello è stato addestrato su dati scalati, le previsioni saranno inaccurate senza lo scaler.")
            # Non sollevare un'eccezione qui, ma il modello potrebbe non funzionare correttamente.
        else:
            try:
                self.scaler = joblib.load(scaler_path)
                print(f"Scaler caricato con successo da: {os.path.abspath(scaler_path)}")
            except Exception as e:
                print(f"ATTENZIONE: Errore durante il caricamento dello scaler da '{os.path.abspath(scaler_path)}': {e}")
                self.scaler = None # Assicura che sia None se il caricamento fallisce


    def fetch_weather_data(self, start_date_api, end_date_api):
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        url = "https://archive-api.open-meteo.com/v1/archive"
        # Definisci le feature richieste all'API. Devono includere quelle usate dal modello.
        api_required_features = [
            "temperature_2m", "precipitation", "wind_speed_10m", "snowfall", "rain", 
            "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", 
            "shortwave_radiation", "direct_radiation", "diffuse_radiation", 
            "direct_normal_irradiance", "global_tilted_irradiance", "terrestrial_radiation"
        ]
        # Se self.model_features_names è definito, potremmo usarlo per essere più precisi,
        # ma per ora richiediamo un set standard noto.
        
        params = {
            "latitude": 45.9,
            "longitude": 11.9,
            "start_date": start_date_api,
            "end_date": end_date_api,
            "hourly": api_required_features,
            "timezone": "Europe/Berlin"
        }
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
        hourly_data = {"date": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s", utc = True).tz_convert(params["timezone"]),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True).tz_convert(params["timezone"]),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )}
        for i, var_name in enumerate(params["hourly"]):
            hourly_data[var_name] = hourly.Variables(i).ValuesAsNumpy()
        hourly_dataframe = pd.DataFrame(data = hourly_data)
        hourly_dataframe = hourly_dataframe.set_index("date")
        return hourly_dataframe

    def predict(self, start_date_str, start_hour=0, end_date_str=None, end_hour=23):
        if self.model is None:
            print("ERRORE: Nessun modello caricato. Impossibile effettuare previsioni.")
            return pd.DataFrame(columns=['predicted_production'])

        if end_date_str is None:
            end_date_str = start_date_str
        try:
            start_dt_str = f"{start_date_str} {start_hour:02d}:00:00"
            end_dt_str = f"{end_date_str} {end_hour:02d}:00:00"
            start_datetime_filter = pd.Timestamp(start_dt_str, tz='Europe/Berlin')
            end_datetime_filter = pd.Timestamp(end_dt_str, tz='Europe/Berlin')

            api_start_date = start_date_str
            api_end_date = end_date_str
            
            print(f"Recupero dati meteo da {api_start_date} a {api_end_date}...")
            weather_df = self.fetch_weather_data(api_start_date, api_end_date)

            if weather_df.empty:
                print("Nessun dato meteo recuperato.")
                return pd.DataFrame(columns=['predicted_production'])

            filtered_df = weather_df[
                (weather_df.index >= start_datetime_filter) &
                (weather_df.index <= end_datetime_filter)
            ]

            if filtered_df.empty:
                print(f"Nessun dato meteo per la finestra: {start_datetime_filter} a {end_datetime_filter}")
                return pd.DataFrame(columns=['predicted_production'])
            
            print(f"Dati filtrati per la previsione: {len(filtered_df)} righe.")

            # --- 4. Prepara le feature per il modello ---
            if self.model_features_names is None:
                print("ERRORE CRITICO: i nomi delle feature del modello non sono definiti (model_features_names è None). "
                      "Impossibile selezionare le feature corrette. "
                      "Verificare il caricamento del modello o definire manualmente le feature attese.")
                return pd.DataFrame(columns=['predicted_production'])

            features_to_use = self.model_features_names

            # Seleziona solo le colonne necessarie dal filtered_df
            # NON aggiungere 'hour', 'month', ecc. qui, perché il modello SVR
            # è stato addestrato con le 15 feature meteorologiche dirette.
            
            # Verifica se tutte le feature necessarie sono presenti nei dati recuperati
            missing_cols = [col for col in features_to_use if col not in filtered_df.columns]
            if missing_cols:
                print(f"ERRORE: Feature richieste dal modello mancanti nei dati recuperati: {missing_cols}")
                print(f"Colonne disponibili: {filtered_df.columns.tolist()}")
                return pd.DataFrame(columns=['predicted_production'])

            X_predict_raw = filtered_df[features_to_use] # DataFrame con le sole feature richieste, nell'ordine corretto se features_to_use lo è
            
            # --- SCALATURA ---
            X_for_prediction_final = X_predict_raw.copy() 
            if self.scaler:
                print("Applicazione dello scaler caricato...")
                # Le colonne da scalare sono tutte tranne 'precipitation' (come nel notebook)
                cols_to_scale = [col for col in X_predict_raw.columns if col != 'precipitation']
                
                if cols_to_scale: # Procedi solo se ci sono colonne da scalare
                    try:
                        # Applica la trasformazione solo alle colonne selezionate
                        scaled_values = self.scaler.transform(X_predict_raw[cols_to_scale])
                        # Aggiorna X_for_prediction_final con i valori scalati
                        X_for_prediction_final[cols_to_scale] = scaled_values
                        print("Dati scalati con successo.")
                    except ValueError as ve:
                        print(f"ERRORE durante la scalatura: {ve}")
                        print("Questo può accadere se lo scaler è stato fittato su un numero/nomi di feature diversi "
                              "da quelle ora presenti in cols_to_scale, o se l'ordine delle colonne è diverso.")
                        print(f"Colonne che si è tentato di scalare: {cols_to_scale}")
                        print("Impossibile procedere con la previsione a causa dell'errore di scalatura.")
                        return pd.DataFrame(columns=['predicted_production'])
                else:
                    print("Nessuna colonna (esclusa 'precipitation') trovata da scalare.")
            else:
                print("ATTENZIONE: Nessuno scaler caricato. Si utilizzeranno le feature grezze per la previsione. "
                      "Questo è un ERRORE se il modello è stato addestrato su dati scalati.")
                # Se il modello è un SVR (come nel tuo caso), si aspetta quasi certamente dati scalati.
                # Senza lo scaler, le previsioni saranno errate.

            # --- 5. Esegui previsioni ---
            print(f"Esecuzione previsioni con {len(X_for_prediction_final.columns)} feature: {X_for_prediction_final.columns.tolist()}")
            predictions_array = self.model.predict(X_for_prediction_final)

            results_df = pd.DataFrame(
                data={'predicted_production': predictions_array},
                index=X_for_prediction_final.index
            )
            return results_df

        except Exception as e:
            print(f"Si è verificato un errore durante la previsione: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(columns=['predicted_production'])

# --- Esempio di Utilizzo ---
if __name__ == '__main__':
    # Assicurati che i file 'best_estimator_model.pkl' e 'scaler.pkl' esistano nei percorsi specificati.
    # E che 'scaler.pkl' sia stato salvato dal tuo notebook di addestramento:
    # Esempio nel notebook: joblib.dump(scaler, "output/model/scaler.pkl")

    # Percorsi relativi alla directory da cui esegui questo script (es. AI_for_social_good/)
    # Se questo script è in weather_pv_conversion/, i percorsi dovrebbero essere:
    # model_file_path = os.path.join("..", "output", "model", "best_estimator_model.pkl")
    # scaler_file_path = os.path.join("..", "output", "model", "scaler.pkl")
    # Per ora, assumo che lo script sia nella directory radice AI_for_social_good/
    
    base_output_dir = "output/model" # Modifica se la tua struttura è diversa
    model_file_name = "best_estimator_model.pkl"
    scaler_file_name = "scaler.pkl"

    model_file_path = os.path.join(base_output_dir, model_file_name)
    scaler_file_path = os.path.join(base_output_dir, scaler_file_name)
    
    # Controlla se i file esistono prima di creare l'istanza
    if not os.path.exists(model_file_path):
        print(f"ERRORE: File del modello non trovato in {os.path.abspath(model_file_path)}")
        print("Assicurati che il modello sia stato salvato correttamente dal notebook di addestramento.")
        exit()
    # Non uscire per lo scaler mancante, ma il predittore darà un avviso e le previsioni saranno probabilmente errate.
    if not os.path.exists(scaler_file_path):
        print(f"ATTENZIONE: File dello scaler non trovato in {os.path.abspath(scaler_file_path)}")
        print("Le previsioni potrebbero essere inaccurate. Salva lo scaler dal notebook.")
        
    try:
        predictor = SolarProductionPredictor(model_path=model_file_path, scaler_path=scaler_file_path)

        # Test case 1
        print("\n--- Test Case 1: Singolo giorno, ore specifiche ---")
        # Date API valide: "2024-01-01" a "2025-02-28" (dal tuo notebook)
        start_date_test = "2024-07-10" 
        predictions = predictor.predict(start_date_test, start_hour=8, end_date_str=start_date_test, end_hour=17)
        if not predictions.empty:
            print(f"Previsioni per {start_date_test} dalle 8:00 alle 17:00:")
            print(predictions)
        else:
            print("Nessuna previsione restituita per Test Case 1.")

        # Test case 2
        print("\n--- Test Case 2: Giorni multipli, ore predefinite (00:00 a 23:00) ---")
        start_date_2_test = "2024-07-10"
        end_date_2_test = "2024-07-11"
        predictions_multi_day = predictor.predict(start_date_2_test, start_hour=0, end_date_str=end_date_2_test, end_hour=23)
        if not predictions_multi_day.empty:
            print(f"Previsioni da {start_date_2_test} 00:00 a {end_date_2_test} 23:00:")
            print(predictions_multi_day)
        else:
            print("Nessuna previsione restituita per Test Case 2.")

    except FileNotFoundError as fnf_error:
        # Gestisce specificamente FileNotFoundError sollevato da __init__
        print(f"Esecuzione interrotta a causa di file mancante: {fnf_error}")
    except Exception as main_exception:
        print(f"ERRORE CRITICO nell'esecuzione principale: {main_exception}")
        import traceback
        traceback.print_exc()