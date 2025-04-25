import pandas as pd
import numpy as np
from datetime import datetime, timezone
import pvlib
# Use the specific class for Linear Gaussian models
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.factors.continuous import LinearGaussianCPD
# Inference might require sampling or manual calculation for continuous
# from pgmpy.sampling import BayesianModelSampling
import traceback # Import traceback for detailed error info

# --- Assume helper functions from previous answers are available ---
# Since they are needed for the script to run, I'll add placeholder implementations.
# REPLACE THESE WITH YOUR ACTUAL IMPLEMENTATIONS if they are complex.

def calculate_solar_details(lat, lon, dt_utc):
    """Placeholder: Calculates solar position details."""
    # Use pvlib for a realistic placeholder
    try:
        location = pvlib.location.Location(lat, lon)
        # Ensure timezone is correctly handled by pvlib location
        # get_solarposition expects a DatetimeIndex
        times_index = pd.DatetimeIndex([pd.Timestamp(dt_utc)]).tz_convert(location.tz) # Wrap in DatetimeIndex

        # Try using the keyword argument 'times=' again, but with the DatetimeIndex
        solpos = location.get_solarposition(times=times_index) # CHANGED BACK TO KEYWORD

        # Return dict matching expected usage - access the first (and only) value
        return {
            'apparent_zenith': solpos['apparent_zenith'].iloc[0],
            'solar_azimuth': solpos['azimuth'].iloc[0]
        }
    except Exception as e:
        print(f"Placeholder solar details calculation failed: {e}")
        # traceback.print_exc() # Uncomment for more detailed error during placeholder execution
        raise # Re-raise the exception

def calculate_aoi(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth):
    """Placeholder: Calculates Angle of Incidence."""
    # Use pvlib for a realistic placeholder
    try:
        aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)
        # pvlib.irradiance.aoi can return a series, return scalar if input was scalar
        if isinstance(aoi, pd.Series):
             return aoi.iloc[0]
        return aoi # Should be a single float
    except Exception as e:
         print(f"Placeholder AOI calculation failed: {e}")
         # traceback.print_exc() # Uncomment for more detailed error during placeholder execution
         raise # Re-raise the exception


def calculate_consistency_metric(ghi, dni, dhi, solar_zenith):
    """Placeholder: Calculates an irradiance consistency metric."""
    # Simple placeholder: GHI vs (DNI*cos(zenith) + DHI) ratio, higher deviation is worse
    if solar_zenith >= 89.0:
        return 0.0 # Consistency is moot if sun is down, metric is zero (high consistency implied for zero)
    cos_zenith = np.cos(np.deg2rad(solar_zenith))

    # Handle potential negative or near-zero modeled GHI in edge cases
    modeled_ghi = dni * cos_zenith + dhi
    modeled_ghi = max(0.0, modeled_ghi) # Clamp modeled GHI to non-negative

    # If measured GHI is very low, check if modeled GHI is also low
    if ghi < 50.0: # Threshold for "very low" GHI (tune this)
         if modeled_ghi < 50.0:
              return 0.0 # High consistency if both actual and model are low
         else:
              # If actual is low but model is high, high inconsistency
              # Metric could be the absolute difference or a scaled version
              return abs(ghi - modeled_ghi) # Use absolute difference as a metric

    # For higher irradiance levels, use relative difference scaled
    if modeled_ghi <= 10.0: # Avoid division by zero or near zero
         if ghi > 10.0:
              # Inconsistency is the magnitude of difference when model is low but actual is high
              return abs(ghi - modeled_ghi)
         else:
              # Both low, or both zero, high consistency
              return 0.0

    # Calculate relative difference for meaningful irradiance
    relative_diff = abs(ghi - modeled_ghi) / modeled_ghi
    metric = relative_diff * 100.0 # Scale to make it a larger number, e.g., percentage points difference

    # Clamp to a plausible range, e.g., 0 to 200
    return min(200.0, max(0.0, metric))


# --- Define Node Names ---
# External Evidence Nodes
E_CLOUD_COVER = 'E_CLOUD_COVER'             # Input: % (0-100)
E_PRECIP_AMOUNT = 'E_PRECIP_AMOUNT'         # Input: Recent accumulated mm
E_SOLAR_ZENITH = 'E_SOLAR_ZENITH'           # Input: Degrees (0-180)
E_AOI = 'E_AOI'                             # Input: Degrees (0-90)
E_TEMP_AIR = 'E_TEMP_AIR'                   # Input: °C
E_WIND_SPEED = 'E_WIND_SPEED'               # Input: m/s
E_IRRAD_CONSISTENCY = 'E_IRRAD_CONSISTENCY' # Input: Metric (e.g., 0+)

# Intermediate Nodes
I_IRRAD_LEVEL = 'I_IRRAD_LEVEL'             # Estimated POA proxy (W/m2)
I_IRRAD_VARIABILITY = 'I_IRRAD_VARIABILITY' # Estimated Factor (0+)
I_SOILING_FACTOR = 'I_SOILING_FACTOR'       # Estimated Factor (~1.0)
I_CELL_TEMP = 'I_CELL_TEMP'                 # Estimated °C

# Output Node
O_PVLIB_ACCURACY = 'O_PVLIB_ACCURACY'       # Score (0-1)

# --- Define Bayesian Network Structure ---
# Use LinearGaussianBayesianNetwork for Linear Gaussian CPDs
model = LinearGaussianBayesianNetwork([
    # Weather influences on intermediates
    (E_CLOUD_COVER, I_IRRAD_LEVEL),
    (E_SOLAR_ZENITH, I_IRRAD_LEVEL),
    (E_CLOUD_COVER, I_IRRAD_VARIABILITY),
    (E_PRECIP_AMOUNT, I_SOILING_FACTOR),

    # Cell temp influences
    (I_IRRAD_LEVEL, I_CELL_TEMP),
    (E_TEMP_AIR, I_CELL_TEMP),
    (E_WIND_SPEED, I_CELL_TEMP),

    # Accuracy influences (Intermediate Nodes and External Evidence)
    (I_IRRAD_LEVEL, O_PVLIB_ACCURACY),
    (I_IRRAD_VARIABILITY, O_PVLIB_ACCURACY),
    (I_SOILING_FACTOR, O_PVLIB_ACCURACY),
    (I_CELL_TEMP, O_PVLIB_ACCURACY),
    (E_AOI, O_PVLIB_ACCURACY), # AOI is an external input directly affecting accuracy
    (E_IRRAD_CONSISTENCY, O_PVLIB_ACCURACY) # Consistency is an external input directly affecting accuracy
])

# --- Define Linear Gaussian CPDs ---
# ****** CRITICAL: All beta weights and variances are PLACEHOLDERS ******
# ****** You MUST tune these based on data or expert knowledge! ******
# Remember variance must be > 0. Use small positive values if uncertain.

# NOTE for recent pgmpy: Using positional arguments for LinearGaussianCPD
# The likely signature is:
# LinearGaussianCPD(variable_name, beta_list, variance_value, evidence_list)

# --- Intermediate Nodes CPDs ---

# I_IRRAD_LEVEL ~ Normal(intercept + w_cloud*Cloud + w_zenith*Zenith, var)
cpd_irrad_level = LinearGaussianCPD(
    I_IRRAD_LEVEL, # 1. variable name
    [ # 2. beta list [Intercept, w_cloud, w_zenith]
        1100.0,
        -8.0,
        -8.0
    ],
    200.0*200.0, # 3. variance value
    evidence=[E_CLOUD_COVER, E_SOLAR_ZENITH] # 4. evidence list (parents)
)

# I_IRRAD_VARIABILITY ~ Normal(intercept + w_cloud*Cloud, var)
cpd_irrad_variability = LinearGaussianCPD(
    I_IRRAD_VARIABILITY, # 1. variable name
    [ # 2. beta list [Intercept, w_cloud]
        0.1,
        0.015
    ],
    0.3*0.3, # 3. variance value
    evidence=[E_CLOUD_COVER] # 4. evidence list (parents)
)

# I_SOILING_FACTOR ~ Normal(intercept + w_precip*Precip, var)
cpd_soiling = LinearGaussianCPD(
    I_SOILING_FACTOR, # 1. variable name
    [ # 2. beta list [Intercept, w_precip]
        0.98,
        0.005
    ],
    0.01*0.01, # 3. variance value
    evidence=[E_PRECIP_AMOUNT] # 4. evidence list (parents)
)

# I_CELL_TEMP ~ Normal(intercept + w_irrad*Irrad + w_temp*TempAir + w_wind*Wind, var)
cpd_cell_temp = LinearGaussianCPD(
    I_CELL_TEMP, # 1. variable name
    [ # 2. beta list [Intercept, w_irrad, w_temp_air, w_wind]
        2.0,
        0.03,
        1.0,
        -1.5
    ],
    3.0*3.0, # 3. variance value
    evidence=[I_IRRAD_LEVEL, E_TEMP_AIR, E_WIND_SPEED] # 4. evidence list (parents)
)

# --- Output Node CPD ---

# O_PVLIB_ACCURACY ~ Normal(intercept + w_irrad*Irrad + w_var*Variability + w_soil*Soiling + w_cell*CellTemp + w_aoi*AOI + w_consistency*Consistency, var)
cpd_accuracy = LinearGaussianCPD(
    O_PVLIB_ACCURACY, # 1. variable name
    [ # 2. beta list - MUST match parent order in evidence list!
        1.05,     # Intercept
        -0.00002, # I_IRRAD_LEVEL weight
        -0.1,     # I_IRRAD_VARIABILITY weight
        -0.2,     # I_SOILING_FACTOR weight
        -0.0015,  # I_CELL_TEMP weight
        -0.002,   # E_AOI weight
        -0.003    # E_IRRAD_CONSISTENCY weight
    ],
    0.03*0.03, # 3. variance value
    evidence=[ # 4. evidence list (parents) - Order MUST match beta weights[1:]
        I_IRRAD_LEVEL, I_IRRAD_VARIABILITY, I_SOILING_FACTOR, I_CELL_TEMP, E_AOI, E_IRRAD_CONSISTENCY
    ]
)

# --- Add CPDs to the model ---
# This should work with LinearGaussianBayesianNetwork if it accepts LGC PDs
model.add_cpds(cpd_irrad_level, cpd_irrad_variability, cpd_soiling, cpd_cell_temp, cpd_accuracy)

# --- Check Model Validity ---
# This checks if the CPDs are correctly defined for the structure
print("Checking model structure validity:", model.check_model())
if not model.check_model():
    print("Model is not valid. Please check CPD definitions.")
    # Consider exiting or raising an error if the model isn't valid.
    # For this example, we'll proceed, but in production, you'd want it valid.

# --- Function to Estimate Continuous Accuracy ---

def estimate_accuracy_continuous_weather(lat, lon, surface_tilt, surface_azimuth, dt_utc, temp_air, wind_speed, ghi, dhi, dni, cloud_cover, precip_amount):
    """
    Calculates external conditions and estimates continuous accuracy using the BN.

    Uses manual approximate inference for Linear Gaussian Network.
    Note: This manual method ignores covariance terms that arise from shared
    ancestors (e.g., Cloud Cover affecting both Irrad Level and Variability,
    both of which affect Accuracy). This means the estimated standard deviation
    is likely an UNDERESTIMATE of the true uncertainty.

    Returns:
        A dictionary {'mean': float, 'std_dev': float} for Pvlib_Accuracy,
        or None if calculation fails significantly.
    """
    # 1. Perform External Calculations
    try:
        # Ensure dt_utc is timezone-aware (e.g., UTC)
        if dt_utc.tzinfo is None:
             # Assume UTC if no timezone is provided, or handle as appropriate
             print("Warning: Input datetime is naive. Assuming UTC.")
             dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        elif dt_utc.tzinfo != timezone.utc:
             # Convert to UTC if not already
             dt_utc = dt_utc.astimezone(timezone.utc)


        solar_details = calculate_solar_details(lat, lon, dt_utc) # Call the corrected function
        solar_zenith = solar_details.get('apparent_zenith')
        solar_azimuth = solar_details.get('solar_azimuth')

        if solar_zenith is None or solar_azimuth is None:
             raise ValueError("Could not retrieve solar zenith or azimuth.")

        # Handle Nighttime (Sun below horizon)
        if solar_zenith >= 89.0:
            print("Nighttime (Zenith >= 89 deg). Assuming high accuracy for zero prediction.")
            if ghi < 10.0:
                 return {'mean': 1.0, 'std_dev': 0.01}
            else:
                 return {'mean': 0.1, 'std_dev': 0.1}


        # Calculate AOI
        aoi = calculate_aoi(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)

        # Calculate Irradiance Consistency Metric
        irrad_consistency = calculate_consistency_metric(ghi, dni, dhi, solar_zenith)

    except Exception as e:
        print(f"Error during external calculations: {e}")
        traceback.print_exc()
        return None

    # 2. Prepare Evidence for BN (External Nodes)
    # Ensure all evidence values are floats
    evidence_dict = {
        E_CLOUD_COVER: float(cloud_cover),
        E_PRECIP_AMOUNT: float(precip_amount),
        E_SOLAR_ZENITH: float(solar_zenith),
        E_AOI: float(aoi),
        E_TEMP_AIR: float(temp_air),
        E_WIND_SPEED: float(wind_speed),
        E_IRRAD_CONSISTENCY: float(irrad_consistency)
    }
    # Basic clamping/validation for evidence ranges (optional but good practice)
    evidence_dict[E_CLOUD_COVER] = max(0.0, min(100.0, evidence_dict[E_CLOUD_COVER]))
    evidence_dict[E_PRECIP_AMOUNT] = max(0.0, evidence_dict[E_PRECIP_AMOUNT])
    evidence_dict[E_SOLAR_ZENITH] = max(0.0, min(180.0, evidence_dict[E_SOLAR_ZENITH]))
    evidence_dict[E_AOI] = max(0.0, min(90.0, evidence_dict[E_AOI]))
    evidence_dict[E_WIND_SPEED] = max(0.0, evidence_dict[E_WIND_SPEED])
    evidence_dict[E_IRRAD_CONSISTENCY] = max(0.0, evidence_dict[E_IRRAD_CONSISTENCY]) # Consistency is >= 0 by definition

    print(f"\nEvidence values fed into BN: {evidence_dict}")

    # 3. Calculate Expected Mean and Variance Manually (Approximate Inference)
    # This propagates means and variances through the network layer by layer.
    # CRITICAL LIMITATION: This ignores covariance terms that arise from shared
    # ancestors (e.g., Cloud Cover affecting both Irrad Level and Variability,
    # both of which affect Accuracy). This means the estimated standard deviation
    # is likely an UNDERESTIMATE of the true uncertainty.
    try:
        # --- Layer 1: Intermediate Nodes dependent on External Evidence ---

        # E[I_IRRAD_LEVEL | Evidence] = intercept + w_cloud*E[Cloud] + w_zenith*E[Zenith]
        # Since Evidence nodes are observed, E[Node] = observed_value, Var(Node) = 0
        weights_irrad = cpd_irrad_level.beta
        # Assuming order [intercept, E_CLOUD_COVER, E_SOLAR_ZENITH] based on evidence list
        if len(weights_irrad) != 3: raise ValueError("Irrad CPD beta wrong size")
        mean_irrad = weights_irrad[0] + \
                     weights_irrad[1]*evidence_dict[E_CLOUD_COVER] + \
                     weights_irrad[2]*evidence_dict[E_SOLAR_ZENITH]
        # Clamp irradiance to realistic bounds (0 to ~1200)
        mean_irrad = max(0.0, min(1200.0, mean_irrad))
        # Variance of I_IRRAD_LEVEL given observed parents is just its inherent variance
        var_irrad = cpd_irrad_level.variance


        # E[I_IRRAD_VARIABILITY | Evidence] = intercept + w_cloud*E[Cloud]
        weights_var = cpd_irrad_variability.beta
        # Assuming order [intercept, E_CLOUD_COVER] based on evidence list
        if len(weights_var) != 2: raise ValueError("Variability CPD beta wrong size")
        mean_var = weights_var[0] + \
                   weights_var[1]*evidence_dict[E_CLOUD_COVER]
        mean_var = max(0.0, mean_var) # Variability factor shouldn't be negative
        var_var = cpd_irrad_variability.variance


        # E[I_SOILING_FACTOR | Evidence] = intercept + w_precip*E[Precip]
        weights_soil = cpd_soiling.beta
        # Assuming order [intercept, E_PRECIP_AMOUNT] based on evidence list
        if len(weights_soil) != 2: raise ValueError("Soiling CPD beta wrong size")
        mean_soil = weights_soil[0] + \
                    weights_soil[1]*evidence_dict[E_PRECIP_AMOUNT]
        # Clamp soiling factor to reasonable range (e.g., 0.8 to 1.1)
        mean_soil = max(0.8, min(1.1, mean_soil))
        var_soil = cpd_soiling.variance


        # --- Layer 2: Intermediate Node dependent on Intermediate Means/Variances and Evidence ---

        # E[I_CELL_TEMP | Evidence] = intercept + w_irrad*E[I_IRRAD_LEVEL|Ev] + w_temp*E[TempAir|Ev] + w_wind*E[WindSpeed|Ev]
        weights_cell = cpd_cell_temp.beta
        # Assuming order [intercept, I_IRRAD_LEVEL, E_TEMP_AIR, E_WIND_SPEED] based on evidence list
        if len(weights_cell) != 4: raise ValueError("CellTemp CPD beta wrong size")
        mean_cell = weights_cell[0] + \
                    weights_cell[1]*mean_irrad + \
                    weights_cell[2]*evidence_dict[E_TEMP_AIR] + \
                    weights_cell[3]*evidence_dict[E_WIND_SPEED]
        # Clamp cell temp to a plausible range (e.g., -40 to 100)
        mean_cell = max(-40.0, min(100.0, mean_cell))

        # Approximate Variance of I_CELL_TEMP given Evidence:
        # Var(I_CELL_TEMP | Ev) = Var(intercept + w_irrad*I_IRRAD_LEVEL + w_temp*E_TEMP_AIR + w_wind*E_WIND_SPEED | Ev) + cpd_cell_temp.variance
        # Var(.) | Ev = Var(w_irrad*I_IRRAD_LEVEL | Ev) + Var(w_temp*E_TEMP_AIR | Ev) + Var(w_wind*E_WIND_SPEED | Ev) + 2*Cov(...)
        # Since E_TEMP_AIR and E_WIND_SPEED are evidence, Var(...) = 0 and Cov(...) = 0 for terms involving them.
        # Approx Var(I_CELL_TEMP | Ev) = (w_irrad)^2 * Var(I_IRRAD_LEVEL | Ev) --- IGNORING COVARIANCE between parents of CellTemp (none in this structure)
        #                               + cpd_cell_temp.variance # Add base variance
        var_cell = (weights_cell[1]**2) * var_irrad + \
                   cpd_cell_temp.variance


        # --- Layer 3: Output Node dependent on Intermediate Means/Variances and Evidence ---

        # E[O_PVLIB_ACCURACY | Evidence] = intercept + sum(w_i * E[Parent_i | Ev])
        weights_acc = cpd_accuracy.beta
        # Match the order of parents defined in the evidence list for applying weights
        # Parents: [I_IRRAD_LEVEL, I_IRRAD_VARIABILITY, I_SOILING_FACTOR, I_CELL_TEMP, E_AOI, E_IRRAD_CONSISTENCY]
        # Check number of parents matches beta size (plus intercept)
        expected_acc_beta_size = len(cpd_accuracy.evidence) + 1 # Use evidence list length
        if len(weights_acc) != expected_acc_beta_size:
             raise ValueError(f"Accuracy CPD beta wrong size. Expected {expected_acc_beta_size}, got {len(weights_acc)}.")

        mean_accuracy = weights_acc[0] + \
                        weights_acc[1]*mean_irrad + \
                        weights_acc[2]*mean_var + \
                        weights_acc[3]*mean_soil + \
                        weights_acc[4]*mean_cell + \
                        weights_acc[5]*evidence_dict[E_AOI] + \
                        weights_acc[6]*evidence_dict[E_IRRAD_CONSISTENCY]

        # Approximate Variance of O_PVLIB_ACCURACY given Evidence:
        # Var(O_PVLIB_ACCURACY | Ev) = Var(E[Acc | Pa(Acc)]) + E[Var(Acc | Pa(Acc))]
        #                          = Var(intercept + sum w_i Parent_i) + cpd_accuracy.variance
        # The parents are: [I_IRRAD_LEVEL, I_IRRAD_VARIABILITY, I_SOILING_FACTOR, I_CELL_TEMP, E_AOI, E_IRRAD_CONSISTENCY]
        # Variances given Evidence are: [var_irrad, var_var, var_soil, var_cell, 0, 0] (Variance of evidence nodes is 0)
        # Approx Var(...) = sum(w_i^2 * Var(Parent_i | Ev)) --- IGNORING ALL COVARIANCE terms between parents.
        parent_weights_for_variance = weights_acc[1:] # Exclude intercept weight
        parent_vars_approx = [
            var_irrad,       # Variance of I_IRRAD_LEVEL given Evidence
            var_var,         # Variance of I_IRRAD_VARIABILITY given Evidence
            var_soil,        # Variance of I_SOILING_FACTOR given Evidence
            var_cell,        # Variance of I_CELL_TEMP given Evidence
            0.0,             # Variance of E_AOI is 0 because it's evidence
            0.0              # Variance of E_IRRAD_CONSISTENCY is 0 because it's evidence
        ]

        if len(parent_weights_for_variance) != len(parent_vars_approx):
             # This should not happen if beta and parent list order match model structure, but check defensively
             print("Error: Mismatch in parent weights and variance list size for Accuracy CPD during manual calculation.")
             # Attempt to calculate anyway, might fail with index error
             # Consider raising a specific error here instead.


        var_accuracy_prop = 0.0 # Initialize as float
        # Sum of (weight^2 * variance_of_parent_given_evidence) for direct parents
        # This loop still does NOT account for covariance between parents, e.g. Cov(I_IRRAD_LEVEL, I_IRRAD_VARIABILITY)
        for i in range(len(parent_weights_for_variance)):
             # Add term only if parent_vars_approx[i] is not None and > a tiny value
             if parent_vars_approx[i] is not None and parent_vars_approx[i] > 1e-12: # Use smaller epsilon
                var_accuracy_prop += (parent_weights_for_variance[i]**2) * parent_vars_approx[i]


        total_variance_approx = cpd_accuracy.variance + var_accuracy_prop # Add the accuracy node's inherent variance
        # Ensure variance is non-negative before taking sqrt (due to approximations)
        std_dev_accuracy = np.sqrt(max(1e-9, total_variance_approx)) # Use epsilon


        # Clamp final accuracy score to the 0-1 range
        mean_accuracy = max(0.0, min(1.0, mean_accuracy))

        print(f"Calculated Internals (Mean): Irrad={mean_irrad:.1f}, Var={mean_var:.2f}, Soil={mean_soil:.3f}, CellT={mean_cell:.1f}")
        print(f"Calculated Accuracy: Mean={mean_accuracy:.3f}, Approx StdDev (Ignoring Covariance)={std_dev_accuracy:.3f}")

        return {'mean': mean_accuracy, 'std_dev': std_dev_accuracy}

    except Exception as e:
        print(f"Error during manual BN calculation: {e}")
        traceback.print_exc()
        return None