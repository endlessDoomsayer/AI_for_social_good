from data_classes import *
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import openmeteo_requests
import requests_cache
from retry_requests import retry
import numpy as np
import pandas as pd
import abc # Abstract Base Classes

# --- Weather API Integration Framework ---

class WeatherProvider(abc.ABC):
    """Abstract Base Class for weather data providers."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    @abc.abstractmethod
    def get_weather_data(self, location: Location, start_time_utc: datetime, end_time_utc: datetime | None = None) -> pd.DataFrame | None:
        """
        Fetches weather data for a given location and time range.

        Args:
            location: The Location object.
            start_time_utc: The start timestamp (UTC).
            end_time_utc: The end timestamp (UTC). If None, fetch for single time point (start_time_utc).
            granularity: The desired time step ('hourly', 'daily').
            
        Returns:
            A list of WeatherData objects, one for each time step returned by the API.
            Returns an empty list if data cannot be fetched or an error occurs.

        Raises:
            NotImplementedError: If the concrete class doesn't implement this.
            requests.exceptions.RequestException: For network/API request errors.
            ValueError: For invalid input or API response format issues.
            KeyError: If expected data fields are missing in the API response.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def requires_api_key(self) -> bool:
        """Indicates if this provider typically requires an API key."""
        raise NotImplementedError
    
    def _check_api_key_available(self):
        """
        Helper method used by concrete classes before making API calls
        that require authentication. Checks if an API key is needed and if it exists.
        """
        if self.requires_api_key() and not self.api_key:
            raise ValueError(f"{self.__class__.__name__} requires an API key for this operation, but none was provided.")


# --- Concrete Implementation Example (Placeholder - Open-Meteo) ---
# Open-Meteo is great because it often doesn't require an API key for basic use.

class OpenMeteoProvider(WeatherProvider):
    """Fetches weather data using the Open-Meteo API, supporting hourly/daily."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    # --- Define requested variables for Open-Meteo ---
    # Using instantaneous versions where requested and available hourly
    HOURLY_VARS = [
        "temperature_2m",
        "precipitation_probability",
        "wind_speed_10m",
        "shortwave_radiation_instant",    # Instant GHI
        "diffuse_radiation_instant",      # Instant DHI
        "direct_normal_irradiance_instant", # Instant DNI
        "cloud_cover",
        "precipitation"
    ]
    
    def __init__(self, api_key: str | None = None):
        super().__init__(api_key)
        # Setup the Open-Meteo client (same as before)
        self.cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        self.retry_session = retry(self.cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=self.retry_session)
        print("OpenMeteo client initialized.")

    def requires_api_key(self) -> bool:
        return False # Standard Open-Meteo hourly use often doesn't need a key

    def get_weather_data(self,
                         location: Location,
                         start_time_utc: datetime,
                         end_time_utc: datetime | None = None,
                         ) -> pd.DataFrame | None:
        """Fetches weather data from Open-Meteo using the specified parameters."""

        self._check_api_key_available()
        
        # Input validation
        if not isinstance(start_time_utc, datetime) or not start_time_utc.tzinfo:
             raise ValueError("start_time_utc must be a timezone-aware datetime object.")
        if end_time_utc and (not isinstance(end_time_utc, datetime) or not end_time_utc.tzinfo):
             raise ValueError("end_time_utc must be a timezone-aware datetime object if provided.")
        if start_time_utc >= end_time_utc:
             raise ValueError("end_time_utc must be after start_time_utc.")
        
        self._check_api_key_available()

        target_end_time = end_time_utc or start_time_utc
        start_date_str = start_time_utc.astimezone(timezone.utc).strftime('%Y-%m-%d')
        end_date_str = target_end_time.astimezone(timezone.utc).strftime('%Y-%m-%d')

        params = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "timezone": "UTC",
            "hourly": self.HOURLY_VARS,
            "start_date": start_date_str,
            "end_date": end_date_str,
        }

        print(f"start time: {start_date_str}, end time: {end_date_str}")
        
        # --- Make API Call ---
        try:
            print(f"Requesting Open-Meteo (Hourly) via library: {params}")
            responses = self.openmeteo.weather_api(self.BASE_URL, params=params)
        except Exception as e:
            print(f"Error fetching hourly data via Open-Meteo library: {e}")
            import traceback
            traceback.print_exc()
            return None

        if not responses:
            print("Received no response from Open-Meteo library.")
            return None
        response = responses[0]
        # print(f"Coordinates: {response.Latitude():.2f}°N {response.Longitude():.2f}°E") # Debug

        # --- Process Hourly Data ---
        hourly = response.Hourly()
        if hourly is None:
            print("No hourly data block received.")
            return None

        hourly_data = {"date": pd.date_range(
        	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        	freq = pd.Timedelta(seconds = hourly.Interval()),
        	inclusive = "left"
        )}
        
        # Add other variables, checking if they exist in the response
        # Variables are indexed 0, 1, 2,... based on the request order
        for i, var_name in enumerate(self.HOURLY_VARS):
            var_data = hourly.Variables(i)
            if var_data is not None:
                # Get data as numpy array (library handles potential single value case)
                values_np = var_data.ValuesAsNumpy()
                # Assign to dictionary - pandas handles NaN conversion later if needed
                hourly_data[var_name] = values_np
            else:
                print(f"Warning: Data for variable '{var_name}' (index {i}) not found in response.")
                # Optionally add a column of NaNs: hourly_data[var_name] = np.nan

        try:
            hourly_dataframe = pd.DataFrame(data=hourly_data)
        except ValueError as e:
            print(f"Error creating DataFrame. Possible length mismatch? Error: {e}")
            # Debug: Print lengths of arrays in hourly_data
            # for k, v in hourly_data.items():
            #      print(f"  Length of {k}: {len(v) if hasattr(v, '__len__') else 'N/A'}")
            return None

        # Set timestamp as index
        hourly_dataframe = hourly_dataframe.set_index("date")

        # --- Filter DataFrame to precise start/end times ---
        # Note: Index comparison works with timezone-aware datetimes
        hourly_dataframe = hourly_dataframe[
            (hourly_dataframe.index >= start_time_utc) &
            (hourly_dataframe.index <= end_time_utc)
        ]

        if hourly_dataframe.empty:
            print("Warning: No data found within the specified precise time range after filtering.")
            # Return the empty DataFrame or None? Let's return the empty DF for consistency.
            # return None

        return hourly_dataframe
        
        

def get_weather_provider(provider_name: str, api_key: str | None = None) -> WeatherProvider:
    """Returns an instance of the specified weather provider."""
    name_lower = provider_name.lower()
    if name_lower == "open-meteo":
        return OpenMeteoProvider(api_key)
    # Add other providers here
    # elif name_lower == "tomorrow.io":
    #    return TomorrowIOProvider(api_key) # Assuming you create this class
    else:
        raise ValueError(f"Unknown weather provider: {provider_name}")