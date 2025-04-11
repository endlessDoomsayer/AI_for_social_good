from data_classes import *
from dataclasses import dataclass, field
from datetime import datetime, timezone
import abc # Abstract Base Classes

# --- Weather API Integration Framework ---

class WeatherProvider(abc.ABC):
    """Abstract Base Class for weather data providers."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    @abc.abstractmethod
    def get_weather_data(self, location: Location, start_time_utc: datetime, end_time_utc: datetime | None = None, granularity: TimeGranularity = 'hourly') -> list[WeatherData]:
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
    HOURLY_VARS = (
        "temperature_2m,"
        "precipitation_probability,"
        "wind_speed_10m,"
        "shortwave_radiation_instant,"    # Instant GHI
        "diffuse_radiation_instant,"      # Instant DHI
        "direct_normal_irradiance_instant," # Instant DNI
        "cloud_cover,"
        "precipitation"
    )
    # Daily equivalents (means/sums)
    DAILY_VARS = (
        "temperature_2m_mean,"
        "precipitation_probability_mean,"
        "wind_speed_10m_mean,"
        "shortwave_radiation_sum," # Daily GHI Sum (e.g., Wh/m²/day)
        # DNI/DHI sums are less common/useful for simple models, omit unless needed
        "cloud_cover_mean,"
        "precipitation_sum"
    )
    
    def __init__(self, api_key: str | None = None):
        # Open-Meteo doesn't strictly require an API key for non-commercial use
        super().__init__(api_key) # Pass key anyway if provided/needed later

    def requires_api_key(self) -> bool:
        # Basic Open-Meteo usage typically does not require a key.
        # Set to True if using commercial features or if they change policy.
        return False

    def get_weather_data(self,
                         location: Location,
                         start_time_utc: datetime,
                         end_time_utc: datetime | None = None,
                         granularity: TimeGranularity = 'hourly'
                         ) -> list[WeatherData]:
        """Fetches weather data from Open-Meteo using the specified parameters."""

        # Input validation
        if not isinstance(start_time_utc, datetime) or not start_time_utc.tzinfo:
             raise ValueError("start_time_utc must be a timezone-aware datetime object.")
        if end_time_utc and (not isinstance(end_time_utc, datetime) or not end_time_utc.tzinfo):
             raise ValueError("end_time_utc must be a timezone-aware datetime object if provided.")
        if granularity not in ['hourly', 'daily']:
             raise ValueError(f"Invalid granularity specified: {granularity}")

        # Check for API key if needed (using the correct method name from your base class)
        # self._check_api_key() # OR
        self._check_api_key_available() # Use the one defined in your WeatherProvider

        target_end_time = end_time_utc or start_time_utc
        # Use UTC dates for the API request
        start_date_str = start_time_utc.astimezone(timezone.utc).strftime('%Y-%m-%d')
        end_date_str = target_end_time.astimezone(timezone.utc).strftime('%Y-%m-%d')

        params = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "timezone": "UTC", # IMPORTANT: Request data in UTC time
            "start_date": start_date_str,
            "end_date": end_date_str,
        }

        if granularity == 'hourly':
            params["hourly"] = self.HOURLY_VARS
        elif granularity == 'daily':
            params["daily"] = self.DAILY_VARS
            params["precipitation_unit"] = "mm" # Ensure consistency

        try:
            print(f"Requesting Open-Meteo ({granularity}): {self.BASE_URL} with params {params}")
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            # print(f"Open-Meteo Response: {json.dumps(data, indent=2)}") # Debug

        except requests.exceptions.RequestException as e:
            print(f"Error fetching {granularity} data from Open-Meteo: {e}")
            if response is not None: print(f"Response content: {response.text}")
            return []
        except json.JSONDecodeError:
            print(f"Error decoding {granularity} JSON response from Open-Meteo.")
            return []

        # --- Parse the response ---
        weather_data_list = []
        try:
            data_block = data.get(granularity)
            if not data_block: raise KeyError(f"'{granularity}' data block missing.")

            times = data_block.get("time")
            if not times: raise KeyError("'time' array missing.")
            num_steps = len(times)

            # --- Fetch arrays based on requested variables ---
            # Hourly/Instantaneous or Daily Means/Sums
            var_prefix = "" if granularity == 'hourly' else "_" + granularity # Adjust if needed by API
            temps = data_block.get("temperature_2m" if granularity == 'hourly' else "temperature_2m_mean")
            precip_probs = data_block.get("precipitation_probability" if granularity == 'hourly' else "precipitation_probability_mean")
            winds = data_block.get("wind_speed_10m" if granularity == 'hourly' else "windspeed_10m_mean")
            cloud_covers = data_block.get("cloud_cover" if granularity == 'hourly' else "cloud_cover_mean")
            precips = data_block.get("precipitation" if granularity == 'hourly' else "precipitation_sum")

            # Irradiance - fetch correct names
            ghis_instant = data_block.get("shortwave_radiation_instant") # Hourly only
            dhis_instant = data_block.get("diffuse_radiation_instant")   # Hourly only
            dnis_instant = data_block.get("direct_normal_irradiance_instant") # Hourly only
            ghi_sums_daily = data_block.get("shortwave_radiation_sum") # Daily only

            # Basic length validation (optional)
            # ...

            # --- Loop through time steps ---
            for i in range(num_steps):
                ts_str = times[i]
                if granularity == 'hourly':
                    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00')).astimezone(timezone.utc)
                else: # daily
                    ts = datetime.strptime(ts_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)

                # --- Process Irradiance ---
                current_ghi_inst = None
                current_dni_inst = None
                current_dhi_inst = None

                if granularity == 'hourly':
                    current_ghi_inst = float(ghis_instant[i]) if ghis_instant and ghis_instant[i] is not None else None
                    current_dni_inst = float(dnis_instant[i]) if dnis_instant and dnis_instant[i] is not None else None
                    current_dhi_inst = float(dhis_instant[i]) if dhis_instant and dhis_instant[i] is not None else None
                elif granularity == 'daily':
                    # No instant values daily. Could estimate average GHI from sum if needed.
                     if ghi_sums_daily and ghi_sums_daily[i] is not None:
                         try: # Basic conversion MJ/day -> W/m² average
                             # Or Wh/day if API provides that: Wh / 24h = W avg
                             avg_ghi_power = (float(ghi_sums_daily[i]) * 1_000_000) / (24 * 3600) # Assuming MJ input
                             # Store average power in ghi_instant for consistency, or create new field?
                             # Let's keep instant fields None for daily for clarity.
                             pass # Or store avg_ghi_power in a different field if needed
                         except (ValueError, TypeError): pass # Ignore conversion errors


                irradiance = IrradianceComponents(
                    ghi=current_ghi_inst,
                    dni=current_dni_inst,
                    dhi=current_dhi_inst,
                )

                # --- Process Conditions ---
                current_temp = float(temps[i]) if temps and temps[i] is not None else 0.0
                current_wind = float(winds[i]) if winds and winds[i] is not None else None
                current_precip_prob = float(precip_probs[i]) if precip_probs and precip_probs[i] is not None else None
                current_cloud = float(cloud_covers[i]) if cloud_covers and cloud_covers[i] is not None else None
                current_precip = float(precips[i]) if precips and precips[i] is not None else None


                conditions = WeatherConditions(
                    ambient_temperature=current_temp,
                    wind_speed=current_wind,
                    precipitation=current_precip,
                    precipitation_probability=current_precip_prob,
                    cloud_cover=current_cloud
                )

                # --- Append WeatherData ---
                weather_data_list.append(WeatherData(
                    timestamp_utc=ts,
                    location=location,
                    irradiance=irradiance,
                    conditions=conditions,
                    granularity=granularity
                ))

        except (KeyError, ValueError, TypeError) as e:
            print(f"Error parsing Open-Meteo {granularity} response data: {e}")
            import traceback
            traceback.print_exc()
            return []

        # --- Filtering by precise start/end times ---
        final_list = [
             wd for wd in weather_data_list
             if wd.timestamp_utc >= start_time_utc and wd.timestamp_utc <= target_end_time
        ]
        # Handle request for single point in time
        if end_time_utc is None and final_list:
             closest_entry = min(final_list, key=lambda wd: abs(wd.timestamp_utc - start_time_utc))
             return [closest_entry]
        return final_list


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