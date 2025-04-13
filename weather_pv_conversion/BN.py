# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import pvlib
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.continuous import LinearGaussianCPD
# Inference might require sampling or manual calculation for continuous
# from pgmpy.sampling import BayesianModelSampling

# --- Assume helper functions from previous answers are available ---
# calculate_solar_details, calculate_aoi, calculate_poa,
# calculate_cell_temp (will be adapted/simplified inside BN),
# calculate_consistency_metric

# --- Define Node Names ---
# External Evidence Nodes
E_CLOUD_COVER = 'E_CLOUD_COVER'
E_PRECIP_AMOUNT = 'E_PRECIP_AMOUNT'
E_SOLAR_ZENITH = 'E_SOLAR_ZENITH'
E_AOI = 'E_AOI'
E_TEMP_AIR = 'E_TEMP_AIR'
E_WIND_SPEED = 'E_WIND_SPEED'
E_IRRAD_CONSISTENCY = 'E_IRRAD_CONSISTENCY' # Provide this calculated metric

# Intermediate Nodes
I_IRRAD_LEVEL = 'I_IRRAD_LEVEL'         # Simplified POA proxy
I_IRRAD_VARIABILITY = 'I_IRRAD_VARIABILITY' # Factor 0+
I_SOILING_FACTOR = 'I_SOILING_FACTOR'     # Factor ~1
I_CELL_TEMP = 'I_CELL_TEMP'             # Estimated °C

# Output Node
O_PVLIB_ACCURACY = 'O_PVLIB_ACCURACY'     # Score 0-1

# --- Define Bayesian Network Structure ---
model = DiscreteBayesianNetwork([
    # Weather influences
    (E_CLOUD_COVER, I_IRRAD_LEVEL),
    (E_SOLAR_ZENITH, I_IRRAD_LEVEL),
    (E_CLOUD_COVER, I_IRRAD_VARIABILITY),
    (E_PRECIP_AMOUNT, I_SOILING_FACTOR),

    # Cell temp influences
    (I_IRRAD_LEVEL, I_CELL_TEMP),
    (E_TEMP_AIR, I_CELL_TEMP),
    (E_WIND_SPEED, I_CELL_TEMP),

    # Accuracy influences
    (I_IRRAD_LEVEL, O_PVLIB_ACCURACY),
    (I_IRRAD_VARIABILITY, O_PVLIB_ACCURACY),
    (I_SOILING_FACTOR, O_PVLIB_ACCURACY),
    (I_CELL_TEMP, O_PVLIB_ACCURACY),
    (E_AOI, O_PVLIB_ACCURACY),
    (E_IRRAD_CONSISTENCY, O_PVLIB_ACCURACY)
])

# --- Define Linear Gaussian CPDs ---
# ****** CRITICAL: All beta weights and variances are PLACEHOLDERS ******
# ****** You MUST tune these based on data or expert knowledge! ******

# --- Intermediate Nodes CPDs ---

# I_IRRAD_LEVEL ~ Normal(intercept + w_cloud*Cloud + w_zenith*cos(Zenith), var)
# Approximating POA: Max possible (~1000 W/m2), reduced by clouds and low sun angle (high zenith)
# Using cos(zenith) as a proxy for solar angle effect.
cpd_irrad_level = LinearGaussianCPD(
    variable=I_IRRAD_LEVEL,
    beta=[ # [Intercept, w_cloud, w_zenith]
        1100, # Intercept: Theoretical max possible irradiance slightly > 1000
        -8.0, # Weight Cloud Cover (%): Higher cloud -> lower level
        -300.0 # Weight Solar Zenith (deg): Higher zenith -> lower level (rough approximation)
               # Note: A cos(zenith) term would be better but requires non-linear CPD
    ],
    variance=200*200 # High variance, this is a rough estimate
)

# I_IRRAD_VARIABILITY ~ Normal(intercept + w_cloud*Cloud, var)
# Factor: 0 = no variability, higher = more variability. Assume linear with cloud cover.
cpd_irrad_variability = LinearGaussianCPD(
    variable=I_IRRAD_VARIABILITY,
    beta=[ # [Intercept, w_cloud]
        0.1,  # Intercept: Base variability factor (even clear sky fluctuates slightly)
        0.015 # Weight Cloud Cover (%): Higher clouds -> higher variability factor
    ],
    variance=0.3*0.3 # Moderate variance
)

# I_SOILING_FACTOR ~ Normal(intercept + w_precip*Precip, var)
# Factor: 1.0 = standard loss, >1.0 cleaner (less loss), <1.0 dirtier (more loss)
cpd_soiling = LinearGaussianCPD(
    variable=I_SOILING_FACTOR,
    beta=[ # [Intercept, w_precip]
        0.98, # Intercept: Baseline assumes slightly dirty panels (e.g., 2% loss)
        0.015 # Weight Precip Amount (mm): Rain washes, increasing factor towards/above 1
    ],
    variance=0.005 # Low variance, effect of rain is somewhat direct
)

# I_CELL_TEMP ~ Normal(intercept + w_irrad*Irrad + w_temp*TempAir + w_wind*Wind, var)
# Linear approximation of cell temperature rise.
cpd_cell_temp = LinearGaussianCPD(
    variable=I_CELL_TEMP,
    beta=[ # [Intercept, w_irrad, w_temp_air, w_wind]
        2.0,  # Intercept: Small base difference?
        0.03, # Weight Irradiance Level (W/m2): Higher irradiance -> higher temp increase (NOCT-like slope)
        1.0,  # Weight Temp Air (°C): Cell temp strongly follows air temp (base)
        -1.5  # Weight Wind Speed (m/s): Wind cools the panel
    ],
    variance=3*3 # Moderate variance reflecting model simplification & other factors
)


# --- Output Node CPD ---

# O_PVLIB_ACCURACY ~ Normal(intercept + w_irrad*Irrad + w_var*Variability + ..., var)
# Score 0-1, where 1 is perfect accuracy. Factors reduce it from an optimistic baseline.
cpd_accuracy = LinearGaussianCPD(
    variable=O_PVLIB_ACCURACY,
    beta=[
        # Intercept: Optimistic base accuracy (e.g., 1.0 or slightly higher)
        1.05,

        # Weights for parents (mostly negative, reducing accuracy from baseline)
        -0.00005, # I_IRRAD_LEVEL: Very low irradiance slightly reduces confidence
        -0.1,     # I_IRRAD_VARIABILITY: Higher variability reduces accuracy
        -0.2,     # I_SOILING_FACTOR: Deviation |Soiling-1.0| reduces accuracy. Linear approx: Assume values further from 1 (e.g. <0.95 or >1.05) are worse. Weight applied directly to factor for simplicity. Need better model ideally.
        -0.0015,  # I_CELL_TEMP: Extreme temps |Temp-25| reduce accuracy. Linear approx: weight applies directly to temp.
        -0.002,   # E_AOI: High AOI reduces accuracy
        -0.001    # E_IRRAD_CONSISTENCY: Higher inconsistency metric reduces accuracy
    ],
    variance=0.03*0.03 # Low base variance for the final accuracy estimate itself
)

# --- Add CPDs to the model ---
model.add_cpds(cpd_irrad_level, cpd_irrad_variability, cpd_soiling, cpd_cell_temp, cpd_accuracy)

# --- Check Model Validity ---
print("Checking model structure validity:", model.check_model())

# --- Function to Estimate Continuous Accuracy ---

def estimate_accuracy_continuous_weather(lat, lon, surface_tilt, surface_azimuth, dt_utc, temp_air, wind_speed, ghi, dhi, dni, cloud_cover, precip_amount):
    """
    Calculates external conditions and estimates continuous accuracy using the BN.

    Returns:
        A dictionary {'mean': float, 'std_dev': float} for Pvlib_Accuracy,
        or None if calculation fails.
    """
    # 1. Perform External Calculations
    try:
        solar_details = calculate_solar_details(lat, lon, dt_utc)
        solar_zenith = solar_details['apparent_zenith']

        # Handle Nighttime
        if solar_zenith >= 89:
            print("Nighttime. Assuming high accuracy for zero prediction.")
            return {'mean': 1.0, 'std_dev': 0.01}

        aoi = calculate_aoi(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)
        irrad_consistency = calculate_consistency_metric(ghi, dni, dhi, solar_zenith)

    except Exception as e:
        print(f"Error during external calculations: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 2. Prepare Evidence for BN (External Nodes)
    evidence_dict = {
        E_CLOUD_COVER: cloud_cover,
        E_PRECIP_AMOUNT: precip_amount,
        E_SOLAR_ZENITH: solar_zenith,
        E_AOI: aoi,
        E_TEMP_AIR: temp_air,
        E_WIND_SPEED: wind_speed,
        E_IRRAD_CONSISTENCY: irrad_consistency
    }
    print(f"\nEvidence values fed into BN: {evidence_dict}")

    # 3. Calculate Expected Mean and Variance Manually (Approximate)
    # This requires propagating through the network based on Linear Gaussian equations.
    try:
        # E[I_IRRAD_LEVEL | Evidence]
        weights_irrad = cpd_irrad_level.beta
        mean_irrad = weights_irrad[0] + weights_irrad[1]*evidence_dict[E_CLOUD_COVER] + weights_irrad[2]*evidence_dict[E_SOLAR_ZENITH]
        # Clamp irradiance to realistic bounds (0+)
        mean_irrad = max(0, mean_irrad)
        var_irrad = cpd_irrad_level.variance

        # E[I_IRRAD_VARIABILITY | Evidence]
        weights_var = cpd_irrad_variability.beta
        mean_var = weights_var[0] + weights_var[1]*evidence_dict[E_CLOUD_COVER]
        mean_var = max(0, mean_var) # Variability factor shouldn't be negative
        var_var = cpd_irrad_variability.variance

        # E[I_SOILING_FACTOR | Evidence]
        weights_soil = cpd_soiling.beta
        mean_soil = weights_soil[0] + weights_soil[1]*evidence_dict[E_PRECIP_AMOUNT]
        # Clamp soiling factor to reasonable range (e.g., 0.8 to 1.1)
        mean_soil = max(0.8, min(1.1, mean_soil))
        var_soil = cpd_soiling.variance

        # E[I_CELL_TEMP | Evidence] - depends on E[I_IRRAD_LEVEL]
        weights_cell = cpd_cell_temp.beta
        mean_cell = weights_cell[0] + weights_cell[1]*mean_irrad + weights_cell[2]*evidence_dict[E_TEMP_AIR] + weights_cell[3]*evidence_dict[E_WIND_SPEED]
        var_cell_base = cpd_cell_temp.variance
        # Approx variance propagation (ignoring covariance)
        var_cell = var_cell_base + (weights_cell[1]**2)*var_irrad

        # E[O_PVLIB_ACCURACY | Evidence] - depends on upstream means
        weights_acc = cpd_accuracy.beta
        parent_means = [
            mean_irrad, mean_var, mean_soil, mean_cell,
            evidence_dict[E_AOI], evidence_dict[E_IRRAD_CONSISTENCY]
        ]
        parent_vars_approx = [ # Approximate variances for calculation below
             var_irrad, var_var, var_soil, var_cell,
             30*30, 50*50 # Placeholder variances for AOI, Consistency evidence
        ]

        if len(weights_acc) != len(parent_means) + 1:
             raise ValueError("Mismatch weights/parents for Accuracy CPD.")

        mean_accuracy = weights_acc[0] # Intercept
        for i in range(len(parent_means)):
            mean_accuracy += weights_acc[i+1] * parent_means[i]

        # Approximate variance propagation (ignoring covariance terms)
        var_accuracy_base = cpd_accuracy.variance
        var_accuracy_prop = 0
        for i in range(len(parent_means)):
            var_accuracy_prop += (weights_acc[i+1]**2) * parent_vars_approx[i]

        total_variance_approx = var_accuracy_base + var_accuracy_prop
        std_dev_accuracy = np.sqrt(max(1e-6, total_variance_approx)) # Ensure non-negative, non-zero

        # Clamp final accuracy score
        mean_accuracy = max(0.0, min(1.0, mean_accuracy))

        print(f"Calculated Internals: Mean Irrad={mean_irrad:.1f}, Mean Var={mean_var:.2f}, Mean Soil={mean_soil:.3f}, Mean CellT={mean_cell:.1f}")
        print(f"Calculated Accuracy: Mean={mean_accuracy:.3f}, Approx StdDev={std_dev_accuracy:.3f}")

        return {'mean': mean_accuracy, 'std_dev': std_dev_accuracy}

    except Exception as e:
        print(f"Error during BN calculation: {e}")
        import traceback
        traceback.print_exc()
        return None


