from dataclasses import dataclass, field
from datetime import datetime
import abc # Abstract Base Classes
import requests # For making API calls
import json # For handling API responses
import os # For potential API key management
from typing import Literal # To specify allowed granularity values
# --- Core Parameter Classes ---

@dataclass
class Location:
    """Represents the geographical location."""
    latitude: float  # Degrees North (negative for South)
    longitude: float # Degrees East (negative for West)
    altitude: float = 0.0 # Meters above sea level (optional, minor effect)

    def __str__(self) -> str:
        alt_str = f", Alt: {self.altitude:.1f}m" if self.altitude else "Alt: not defined"
        return f"Lat: {self.latitude:.4f}, Lon: {self.longitude:.4f}{alt_str}"

@dataclass
class TimestampInfo:
    """Represents the time of calculation."""
    timestamp_utc: datetime # Use timezone-aware datetime, preferably UTC
    
    def __str__(self) -> str:
        return f"Time: {self.timestamp_utc.isoformat()}"
    
@dataclass
class PanelOrientation:
    """Represents the orientation of the solar panel."""
    surface_tilt: float    # Degrees from horizontal (0=flat, 90=vertical)
    surface_azimuth: float # Degrees clockwise from North (0=N, 90=E, 180=S, 270=W)

    def __str__(self) -> str:
        return f"Tilt: {self.surface_tilt:.1f}°, Azimuth: {self.surface_azimuth:.1f}°"
    
@dataclass
class PanelSpecs:
    """Represents the technical specifications of the solar panel model."""
    rated_power: float      # Wp (Watt-peak) - Nominal power at STC
    temp_coefficient_power: float # %/°C or fraction/°C (usually negative)
    noct: float             # °C (Nominal Operating Cell Temperature)
    module_area: float | None = None # m^2 (Optional, useful for efficiency calcs)
    # Add other relevant specs if needed (e.g., voltage, current at Pmax for more detailed models)

    def __str__(self) -> str:
        area_str = f", Area: {self.module_area:.2f}m²" if self.module_area else ""
        # Handle display if coefficient is fraction or percentage
        temp_coeff_display = self.temp_coefficient_power
        unit = "/°C"
        if abs(temp_coeff_display) > 0.1: # Heuristic: likely percentage
           temp_coeff_display *= 100
           unit = "%/°C"

        return f"Pmax: {self.rated_power:.1f}Wp, TempCoeff: {temp_coeff_display:.3f}{unit}, NOCT: {self.noct:.1f}°C{area_str}"

@dataclass
class SystemParameters:
    """Represents system-level characteristics and losses."""
    system_losses: float = 0.85 # Combined derate factor (e.g., 0.85 = 15% loss)
                                # Includes soiling, wiring, mismatch, degradation etc.
    inverter_efficiency: float = 0.96 # DC/AC conversion efficiency factor

    def __str__(self) -> str:
        loss_perc = (1 - self.system_losses) * 100
        inv_eff_perc = self.inverter_efficiency * 100
        return f"System Losses: {loss_perc:.1f}% (Factor: {self.system_losses:.3f}), Inverter Eff: {inv_eff_perc:.1f}% (Factor: {self.inverter_efficiency:.3f})"
    

@dataclass
class IrradianceComponents:
    """Represents the different components of solar irradiance. Values are average power (W/m²) over the period unless noted."""
    ghi: float | None # W/m^2 (Global Horizontal Irradiance) - Can be None
    dni: float | None # W/m^2 (Direct Normal Irradiance) - Can be None
    dhi: float | None # W/m^2 (Diffuse Horizontal Irradiance) - Can be None
    # Note: Daily APIs often return sums (Wh/day), requiring conversion or careful interpretation.
    #       We store None if the value isn't directly available as W/m^2 average power.

    def __str__(self) -> str:
        ghi_str = f"GHI: {self.ghi:.1f}" if self.ghi is not None else "GHI: N/A"
        dni_str = f"DNI: {self.dni:.1f}" if self.dni is not None else "DNI: N/A"
        dhi_str = f"DHI: {self.dhi:.1f}" if self.dhi is not None else "DHI: N/A"
        return f"{ghi_str}, {dni_str}, {dhi_str} W/m²"
    
@dataclass
class WeatherConditions:
    """Represents relevant atmospheric conditions (instantaneous or average/total over the period)."""
    ambient_temperature: float # °C (instantaneous or average)
    wind_speed: float | None = None # m/s (instantaneous or average)
    precipitation: float | None = None # mm (total amount in the period)
    # rain: float | None = None # mm (total rain amount in the period)
    # snowfall: float | None = None # mm (total snow amount in the period)

    precipitation_probability: float | None = None # % (Probability 0-100)
    cloud_cover: float | None = None # % (Total cloud cover 0-100)

    def __str__(self) -> str:
        temp_str = f"Temp: {self.ambient_temperature:.1f}°C"
        wind_str = f", Wind: {self.wind_speed:.1f} m/s" if self.wind_speed is not None else ", Wind: N/A"
        precip_str = f", Precip: {self.precipitation:.2f}mm" if self.precipitation is not None and self.precipitation > 0 else ""
        prob_str = f", PrecipProb: {self.precipitation_probability:.0f}%" if self.precipitation_probability is not None else ""
        cloud_str = f", CloudCover: {self.cloud_cover:.0f}%" if self.cloud_cover is not None else ""

        return f"{temp_str}{wind_str}{cloud_str}{prob_str}{precip_str}"

# --- Combined Weather Data Container ---

# Define allowed granularity levels
TimeGranularity = Literal['hourly', 'daily'] # Extend later if needed e.g., 'minutely'

@dataclass
class WeatherData:
    """Container for all weather-related data for a specific time period and location."""
    timestamp_utc: datetime
    location: Location
    irradiance: IrradianceComponents
    conditions: WeatherConditions
    granularity: Literal['hourly', 'daily']

    def __str__(self) -> str:
        return (f"WeatherData ({self.granularity} @ {self.timestamp_utc.isoformat()} for {self.location}):\n"
                f"  Irradiance: {self.irradiance}\n"
                f"  Conditions: {self.conditions}")