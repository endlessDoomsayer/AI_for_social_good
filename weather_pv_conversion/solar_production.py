import datetime
import joblib
import numpy as np
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import os
import traceback # Keep for error reporting

class SolarProductionPredictor:
    def __init__(self, model_path="output/model/best_estimator_model.pkl",
                 scaler_path="output/model/scaler.pkl"):
        """
        Initializes the predictor by loading the trained model and scaler.
        """
        self.model = None
        self.scaler = None
        self.model_features_names = None # Expected feature names by the SVR model

        # Load the SVR model
        if not os.path.exists(model_path):
            print(f"CRITICAL ERROR: Model file not found at '{os.path.abspath(model_path)}'. Cannot proceed.")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded successfully from: {os.path.abspath(model_path)}")
            if hasattr(self.model, 'feature_names_in_'):
                self.model_features_names = list(self.model.feature_names_in_)
                print(f"Model expects features (order from model.feature_names_in_): {self.model_features_names}")
            else:
                # This case should ideally not happen if the model is a recent scikit-learn SVR
                print("WARNING: Loaded model does not have 'feature_names_in_'. "
                      "Data preparation must precisely match model training.")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load model from '{os.path.abspath(model_path)}': {e}")
            raise

        # Load the scaler
        if not os.path.exists(scaler_path):
            print(f"WARNING: Scaler file not found at '{os.path.abspath(scaler_path)}'. "
                  "Predictions will be inaccurate if the model was trained on scaled data.")
        else:
            try:
                self.scaler = joblib.load(scaler_path)
                print(f"Scaler loaded successfully from: {os.path.abspath(scaler_path)}")
                if hasattr(self.scaler, 'feature_names_in_'):
                    print(f"Scaler was fit on features (order from scaler.feature_names_in_): {list(self.scaler.feature_names_in_)}")
            except Exception as e:
                print(f"WARNING: Failed to load scaler from '{os.path.abspath(scaler_path)}': {e}")
                self.scaler = None # Ensure scaler is None if loading fails

    def fetch_weather_data(self, start_date_api, end_date_api):
        # Fetches weather data from Open-Meteo API
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        url = "https://archive-api.open-meteo.com/v1/archive"
        # Standard set of weather features our SVR model (and its scaler) were trained on
        api_required_features = [
            "temperature_2m", "precipitation", "wind_speed_10m", "snowfall", "rain",
            "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
            "shortwave_radiation", "direct_radiation", "diffuse_radiation",
            "direct_normal_irradiance", "global_tilted_irradiance", "terrestrial_radiation"
        ]
        params = {
            "latitude": 45.9, "longitude": 11.9,
            "start_date": start_date_api, "end_date": end_date_api,
            "hourly": api_required_features, "timezone": "Europe/Berlin"
        }
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
        hourly_data = {"date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True).tz_convert(params["timezone"]),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True).tz_convert(params["timezone"]),
            freq=pd.Timedelta(seconds=hourly.Interval()), inclusive="left"
        )}
        for i, var_name in enumerate(params["hourly"]):
            hourly_data[var_name] = hourly.Variables(i).ValuesAsNumpy()
        hourly_dataframe = pd.DataFrame(data=hourly_data)
        hourly_dataframe = hourly_dataframe.set_index("date")
        return hourly_dataframe

    def predict(self, start_date_str, start_hour=0, end_date_str=None, end_hour=23):
        if self.model is None:
            print("ERROR: Model not loaded. Cannot make predictions.")
            return pd.DataFrame(columns=['predicted_production'])

        if end_date_str is None:
            end_date_str = start_date_str
        try:
            start_dt_str = f"{start_date_str} {start_hour:02d}:00:00"
            end_dt_str = f"{end_date_str} {end_hour:02d}:00:00"
            start_datetime_filter = pd.Timestamp(start_dt_str, tz='Europe/Berlin')
            end_datetime_filter = pd.Timestamp(end_dt_str, tz='Europe/Berlin')

            print(f"Fetching weather data from {start_date_str} to {end_date_str}...")
            weather_df = self.fetch_weather_data(start_date_str, end_date_str)

            if weather_df.empty:
                print("No weather data fetched.")
                return pd.DataFrame(columns=['predicted_production'])

            filtered_df = weather_df[
                (weather_df.index >= start_datetime_filter) &
                (weather_df.index <= end_datetime_filter)
            ]

            if filtered_df.empty:
                print(f"No weather data for the specified window: {start_datetime_filter} to {end_datetime_filter}")
                return pd.DataFrame(columns=['predicted_production'])
            
            print(f"Filtered data for prediction: {len(filtered_df)} rows.")

            if self.model_features_names is None:
                print("CRITICAL ERROR: Model feature names (model_features_names) are not defined.")
                return pd.DataFrame(columns=['predicted_production'])

            # SVR model expects features in this specific order
            svr_expected_features_ordered = self.model_features_names

            missing_cols_for_svr = [col for col in svr_expected_features_ordered if col not in filtered_df.columns]
            if missing_cols_for_svr:
                print(f"ERROR: SVR model required features missing from fetched data: {missing_cols_for_svr}")
                print(f"Available columns: {list(filtered_df.columns)}")
                return pd.DataFrame(columns=['predicted_production'])

            # Raw features, ordered as expected by the SVR model
            X_predict_raw = filtered_df[svr_expected_features_ordered]
            
            # This will hold the final data for prediction (scaled or unscaled)
            X_for_prediction_final = X_predict_raw.copy()
            
            if self.scaler:
                print("Applying loaded scaler...")
                
                if not hasattr(self.scaler, 'feature_names_in_'):
                    print("CRITICAL ERROR: Scaler does not have 'feature_names_in_'. Cannot determine correct column order for scaling.")
                    return pd.DataFrame(columns=['predicted_production'])

                # Features expected by the scaler, in the correct order
                scaler_expected_features_ordered = list(self.scaler.feature_names_in_)

                missing_for_scaler = [col for col in scaler_expected_features_ordered if col not in X_predict_raw.columns]
                if missing_for_scaler:
                    print(f"ERROR: Scaler required features missing from X_predict_raw: {missing_for_scaler}")
                    return pd.DataFrame(columns=['predicted_production'])

                if scaler_expected_features_ordered: # Proceed only if scaler expects features
                    try:
                        # Prepare data for scaler: select columns in the order scaler expects
                        data_to_transform = X_predict_raw[scaler_expected_features_ordered]
                        

                        scaled_values = self.scaler.transform(data_to_transform)
                        
                        # Update the final prediction DataFrame with scaled values
                        X_for_prediction_final.loc[:, scaler_expected_features_ordered] = scaled_values
                        print("Data scaled successfully.")
                    except ValueError as ve:
                        print(f"ERROR during scaling: {ve}")
                        print(f"  Attempted to scale columns (ordered by scaler.feature_names_in_): {scaler_expected_features_ordered}")
                        if hasattr(self.scaler, 'n_features_in_'):
                             print(f"  (Debug) Scaler expects {self.scaler.n_features_in_} features.")
                        print("Cannot proceed with prediction due to scaling error.")
                        return pd.DataFrame(columns=['predicted_production'])
                else:
                    print("WARNING: scaler.feature_names_in_ is empty. No columns will be scaled.")
            else:
                print("WARNING: No scaler loaded. Using raw features. "
                      "This is an ERROR if the SVR model was trained on scaled data.")

            # Prepare final data for SVR model, ensuring correct feature order
            final_data_for_svr = X_for_prediction_final[svr_expected_features_ordered]
            
            print(f"Making predictions with {len(final_data_for_svr.columns)} features (SVR model order): {list(final_data_for_svr.columns)}")
            predictions_array = self.model.predict(final_data_for_svr)

            results_df = pd.DataFrame(
                data={'predicted_production': predictions_array},
                index=final_data_for_svr.index
            )
            return results_df

        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            traceback.print_exc() # Print full traceback for unexpected errors
            return pd.DataFrame(columns=['predicted_production'])
    
    def sum_predicted_production(predictions_df: pd.DataFrame) -> float:
        """
        Calculates the sum of predicted solar production.

        Args:
            predictions_df (pd.DataFrame): A DataFrame with a 'predicted_production' column
                                        and a DatetimeIndex. The sum is of the production
                                        values themselves (e.g., Watts). If you want energy (e.g., Wh),
                                        you'd need to multiply by the time interval.

        Returns:
            float: The total sum of 'predicted_production' values.
                Returns 0.0 if the DataFrame is empty or the column is missing.
        """
        if predictions_df.empty or 'predicted_production' not in predictions_df.columns:
            print("Input DataFrame is empty or 'predicted_production' column is missing. Returning 0.0.")
            return 0.0
        
        # Sum the 'predicted_production' column
        total_production = predictions_df['predicted_production'].sum()
        
        return total_production

# --- Example Usage ---
if __name__ == '__main__':
    # Adjust base_output_dir if your script is not in the project root (AI_for_social_good)
    # e.g., if in weather_pv_conversion/: base_output_dir = os.path.join("..", "output", "model")
    base_output_dir = "weather_pv_conversion/output/model" 
    
    model_file_name = "best_estimator_model.pkl"
    scaler_file_name = "scaler.pkl"

    model_file_path = os.path.join(base_output_dir, model_file_name)
    scaler_file_path = os.path.join(base_output_dir, scaler_file_name)
    
    if not os.path.exists(model_file_path):
        print(f"ERROR: Model file not found at {os.path.abspath(model_file_path)}. Please ensure it's saved correctly.")
        exit()
    if not os.path.exists(scaler_file_path):
        print(f"WARNING: Scaler file not found at {os.path.abspath(scaler_file_path)}. Predictions may be inaccurate.")
        
    try:
        predictor = SolarProductionPredictor(model_path=model_file_path, scaler_path=scaler_file_path)

        start_date_test = "2024-07-10" 
        print(f"\n--- Test Case 1: {start_date_test} ---")
        predictions = predictor.predict(start_date_test, start_hour=8, end_date_str=start_date_test, end_hour=17)
        if not predictions.empty:
            print(f"Predictions for {start_date_test}:\n{predictions}")
        else:
            print(f"No predictions returned for {start_date_test}.")

        start_date_2_test = "2024-07-10"
        end_date_2_test = "2024-07-11"
        print(f"\n--- Test Case 2: {start_date_2_test} to {end_date_2_test} ---")
        predictions_multi_day = predictor.predict(start_date_2_test, start_hour=0, end_date_str=end_date_2_test, end_hour=23)
        if not predictions_multi_day.empty:
            print(f"Predictions from {start_date_2_test} to {end_date_2_test}:\n{predictions_multi_day}")
        else:
            print(f"No predictions returned for {start_date_2_test} to {end_date_2_test}.")

    except FileNotFoundError as fnf_error: # Handles critical FileNotFoundError from __init__
        print(f"Execution stopped due to missing file: {fnf_error}")
    except Exception as main_exception:
        print(f"CRITICAL ERROR in main execution: {main_exception}")
        traceback.print_exc()