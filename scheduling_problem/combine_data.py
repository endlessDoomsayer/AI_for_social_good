import random
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nilm.dataset_functions import Dataset, plot_data
from weather_pv_conversion.solar_production import SolarProductionPredictor 


# TODO bisogna vedere unita di misura pannelli e macchine

def get_data():
    
    data = {}
    
    solar_predictor = SolarProductionPredictor(
    model_path="../weather_pv_conversion/output/model/best_estimator_model.pkl",
    scaler_path="../weather_pv_conversion/output/model/scaler.pkl"
    )


    datasetjson_path = '../nilm/output/IMDELD.json'
    datasetjson = Dataset('IMDELD', datasetjson_path)
    datasetjson.load()

    # 1. NILM DATA
    print("--------------------------------------\nNILM DATA\n")

    machine_names = datasetjson.get_machine_names()
    machine_names = machine_names[:-2] # delete last 2 machines since we don't have any data on the power
    print(f"Machine names: {machine_names}")
    print(f"Machine number: {len(machine_names)}")

    MACHINES = len(machine_names) # Number of machines
    MAX_JOB_N = 21  # Maximum number of jobs per machine
    T_MAX = 24  # Number of time periods (e.g., 48 half-hours in a day)
    
    print("Max jobs per machine:", MAX_JOB_N)
    print("Max time period:", T_MAX)
    
    data["T_MAX"] = T_MAX
    data["MACHINES"] = MACHINES

    # Define sets
    I = list(range(1, MACHINES + 1 )) # machines 
    T = list(range(1, T_MAX + 1)) # time periods TODO
    J = list(range(1, MAX_JOB_N + 1)) # jobs TODO
    
    data["T"] = T
    data["I"] = I
    data["J"] = J
    
    # Energy parameters
    # e_i: energy consumption when machine i is running
    start_time, end_time = datasetjson.get_start_end_time()
    day = pd.Timestamp("2017-12-12")  # it is an example
    day_data = datasetjson.get_data_day(day)


    '''
    print(f"Day data for {day.date()}, number of elements: {len(day_data)}")
    for timestamp, machines_at_hour in day_data.items():
        print(timestamp)
        for machine_name, stats in machines_at_hour.items():
            print((machine_name, stats))
        print("---")
    '''

    e = {}
    for idx, machine_name in enumerate(machine_names, start=1):
        avg_power = datasetjson.get_average_power_usage(machine_name, min(day_data.keys()), max(day_data.keys()))
        e[idx] = avg_power
    print("e =", e)
    data["e"] = e

    # f_i: additional energy consumed when machine i starts TODO
    f = {i: int(e[i] * (1/6)) for i in e}
    print("f =", f)
    data["f"] = f
    
    # d_i: duration of job i on machine i
    def get_job_duration_per_machine(idx, day):
        #data = datasetjson.get_data_day(day)
        if (i <= 2):
            return 8
        elif (i <= 4):
            return 12
        return 10
        """ 
        for timestamp in data.keys():
            machine_data = data[timestamp].get(machine_name)
            if machine_data and machine_data['power_apparent'] > 0:
                total_duration += 1

        if(total_duration != 0): total_duration  = total_duration - 5   
        """
        
        return total_duration

    d = {}
    for i in I:
        machine_name = machine_names[i - 1]
        duration = get_job_duration_per_machine(i, day)
        d[i] = duration
    print("d =", d)
    data["d"] = d
    
    
    # 2. SOLAR PANELS
    # p_t: energy produced at time t by one unit of power
    print("\n--------------------------------------\nSOLAR PANELS\n")
    predictions_df = solar_predictor.predict(start_date_str="2023-07-25")
    p = {}
    for idx, (timestamp, row) in enumerate(predictions_df.iterrows(), start=1):
        p[idx] = float(row['predicted_production'])
        if (p[idx] < 0): p[idx] = 0
        
    data["p"] = p

    print("\n\nSolar production in the day 2023-07-25:", p)
    

    # 3. GENERATED DATA
    # Cost parameters
    c_b = 6500  # Cost per battery
    c_p = 8000   # Cost per unit of power
    c_e = 2.56 # Cost of energy
    B = 5000 # B: battery capacity
    
    data["c_b"] = c_b
    data["c_p"] = c_p
    data["B"] = B

    # m_t: maximum energy available at time t TODO
    mmm = {t: random.randint(800000, 2000000) for t in T}
    data["mmm"] = mmm

    # n_i: number of jobs required for machine i TODO
    n_jobs = {i: 1 for i in I } 
    data["n_jobs"] = n_jobs

    # c_i: cooldown period for machine i
    c = {i: 1 for i in I}
    data["c"] = c
    
    # THRESHOLD_FOR_JOB_J
    THRESHOLD_FOR_JOB_J_AND_I = {(i,j): 24 for i in I for j in J} # Time limit for each job
    data["THRESHOLD_FOR_JOB_J_AND_I"] = THRESHOLD_FOR_JOB_J_AND_I


    # Sets of dependencies and shared resources
    M_dependencies = []  # Pairs of machines where the second depends on the first
    M_shared = []  # Groups of machines that share resources and cannot run simultaneously
    silent_periods = {}  # Periods when certain machines must be off
    
    data["M_dependencies"] = M_dependencies
    data["M_shared"] = M_shared
    data["silent_periods"] = silent_periods
    
    # Cost of energy if taken from outside
    c_e = 0.1
    data["c_e"] = c_e

    # Print the generated data for reference
    print("\n--------------------------------------\nGenerated Parameters: \n")
    print(f"Cost per battery (c_b): {c_b}")
    print(f"Cost per power unit (c_p): {c_p}")
    print(f"Cost of energy taken from outside (c_e): {c_e}")
    print(f"Battery capacity (B): {B}")
    print(f"Max energy available (mmm): {mmm}")
    print(f"Job durations (d): {d}")
    print(f"Required jobs per machine (n_jobs): {n_jobs}")
    print(f"Machine cooldown periods (c): {c}")
    print(f"Machine dependencies: {M_dependencies}")
    print(f"Shared resource groups: {M_shared}")
    print(f"Silent periods: {silent_periods}")
    print("--------------------------------------\n")

    return data
