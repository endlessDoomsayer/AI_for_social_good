import random
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nilm.dataset_functions import Dataset, plot_data
from weather_pv_conversion.solar_production import SolarProductionPredictor 


def get_data(number_of_days = 2, day = pd.Timestamp("2018-02-18")):
    
    end_date = day+pd.to_timedelta(number_of_days-1, unit='D')
    
    print(f"Number of days: {number_of_days}")
    print(f"Start date: {day}")
    print(f"End date: {end_date}")
    
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
    

    MACHINES = len(machine_names) # Number of machines
    MAX_JOB_N = number_of_days*2  # Maximum number of jobs per machine
    T_MAX = 24*number_of_days-1  # Number of time periods (e.g., 48 half-hours in a day) TODO: try for 5 days and for all the seasons
    
    print(f"Machine number:", MACHINES)
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
    #start_time, end_time = datasetjson.get_start_end_time()

    day_data = datasetjson.get_data_start_end(day, end_date)

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
    d = {}
    d[1] = 6
    d[2] = 6
    d[3] = 6
    d[4] = 6
    d[5] = 12
    d[6] = 12
    print("d =", d)
    data["d"] = d

 
    
    # 2. SOLAR PANELS
    # p_t: energy produced at time t by one unit of power
    print("\n--------------------------------------\nSOLAR PANELS\n")
    
    
    predictions_df = solar_predictor.predict(start_date_str=day.strftime("%Y-%m-%d"), end_date_str=end_date.strftime("%Y-%m-%d"))
    p = {}
    for idx, (timestamp, row) in enumerate(predictions_df.iterrows(), start=1):
        p[idx] = float(row['predicted_production'])
        if (p[idx] < 0): p[idx] = 0
        
    data["p"] = p
    print("\n\nSolar production in the days from 2018-01-01 to", end_date, ":", p)
    

    # 3. GENERATED DATA
    # Cost parameters
    # https://www.mosaikosrl.net/batterie-accumulo-fotovoltaico/
    B = 5000 # B: battery capacity  
    c_b = 2530 # Cost per battery
    # https://www.ecodirect.com/Canadian-Solar-CS6X-300P-300W-36V-PV-Panel-p/canadian-solar-cs6x-300p.htm?srsltid=AfmBOor_kd4mknwa-Am9K9m7VYG55_jnXMM3QTP7aTw2Y2qCChJ9GuL7
    c_p = 290   # Cost per unit of power: Canadian Solar CS6X-300P (inverter SMA_America_SB7000TL_US240V cost neglected)
    # TODO: ogni 25 pannelli solari c'è un inverter
    c_i = 3000 # Cost per inverter
    # https://tariffe.segugio.it/guide-e-strumenti/domande-frequenti/quanto-costa-un-kwh-di-energia-elettrica.aspx#:~:text=Il%20prezzo%20dell'energia%20elettrica%20oggi%20%C3%A8%20pari%20a%200,si%20applica%20ai%20clienti%20vulnerabili.
    c_e = 0.16053/1000 # Cost of energy 
    

    data["B"] = B
    data["c_b"] = c_b
    data["c_p"] = c_p
    data["c_e"] = c_e
    

    # m_t: maximum energy available at time t TODO
    mmm = {t: random.randint(800000, 1000000) for t in T}
    data["mmm"] = mmm

    # n_i: number of jobs required for machine i 
    n_jobs = {}
    n_jobs[1] = 2*number_of_days
    n_jobs[2] = 2*number_of_days
    n_jobs[3] = 1*number_of_days
    n_jobs[4] = 1*number_of_days
    n_jobs[5] = 1*number_of_days
    n_jobs[6] = 1*number_of_days
    data["n_jobs"] = n_jobs

    # c_i: cooldown period for machine i TODO
    c = {i: 1 for i in I}
    data["c"] = c
    
    # THRESHOLD_FOR_JOB_J
    THRESHOLD_FOR_JOB_J_AND_I = {(i,j): 24*(j) for i in I for j in J} # Time limit for each job TODO
    data["THRESHOLD_FOR_JOB_J_AND_I"] = THRESHOLD_FOR_JOB_J_AND_I


    # Sets of dependencies and shared resources
    M_dependencies =[]# [(1,3)]  # Pairs of machines where the second depends on the first
    M_shared =[]#[(3,4)]  # Groups of machines that share resources and cannot run simultaneously
    silent_periods ={}# {1:[25,26,27,28,29,30,31,32,33], 5:[32,33,34,35,36,37,38], 6:[24,25,26,27,28,29,30]}  # Periods when certain machines must be off
    
    data["M_dependencies"] = M_dependencies
    data["M_shared"] = M_shared
    data["silent_periods"] = silent_periods
    
    
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
    print(f"Time limit for each job (THRESHOLD_FOR_JOB_J_AND_I): {THRESHOLD_FOR_JOB_J_AND_I}")
    print("--------------------------------------\n")

    return data


def get_modified_data(number_of_days=2, day=pd.Timestamp("2018-01-02")):
    end_date = day + pd.to_timedelta(number_of_days - 1, unit='D')

    print(f"Number of days: {number_of_days}")
    print(f"Start date: {day}")
    print(f"End date: {end_date}")

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
    machine_names = machine_names[:-2]  # delete last 2 machines since we don't have any data on the power
    print(f"Machine names: {machine_names}")

    MACHINES = len(machine_names)  # Number of machines
    MAX_JOB_N = number_of_days * 2  # Maximum number of jobs per machine
    T_MAX = 24 * number_of_days - 1  # Number of time periods (e.g., 48 half-hours in a day) TODO: try for 5 days and for all the seasons

    print(f"Machine number:", MACHINES)
    print("Max jobs per machine:", MAX_JOB_N)
    print("Max time period:", T_MAX)

    data["T_MAX"] = T_MAX
    data["MACHINES"] = MACHINES

    # Define sets
    I = list(range(1, MACHINES + 1))  # machines
    T = list(range(1, T_MAX + 1))  # time periods TODO
    J = list(range(1, MAX_JOB_N + 1))  # jobs TODO

    data["T"] = T
    data["I"] = I
    data["J"] = J

    # Energy parameters
    # e_i: energy consumption when machine i is running
    # start_time, end_time = datasetjson.get_start_end_time()

    day_data = datasetjson.get_data_start_end(day, end_date)

    e = {}
    for idx, machine_name in enumerate(machine_names, start=1):
        avg_power = datasetjson.get_average_power_usage(machine_name, min(day_data.keys()), max(day_data.keys()))
        e[idx] = avg_power
    print("e =", e)
    data["e"] = e

    # f_i: additional energy consumed when machine i starts TODO
    f = {i: int(e[i] * (1 / 6)) for i in e}
    print("f =", f)
    data["f"] = f

    # d_i: duration of job i on machine i
    d = {}
    d[1] = 6
    d[2] = 6
    d[3] = 6
    d[4] = 6
    d[5] = 12
    d[6] = 12
    print("d =", d)
    data["d"] = d

    # 2. SOLAR PANELS
    # p_t: energy produced at time t by one unit of power
    print("\n--------------------------------------\nSOLAR PANELS\n")

    predictions_df = solar_predictor.predict(start_date_str=day.strftime("%Y-%m-%d"),
                                             end_date_str=end_date.strftime("%Y-%m-%d"))
    p = {}
    for idx, (timestamp, row) in enumerate(predictions_df.iterrows(), start=1):
        p[idx] = float(row['predicted_production'])
        if (p[idx] < 0): p[idx] = 0

    data["p"] = p
    print("\n\nSolar production in the days from 2018-01-01 to", end_date, ":", p)

    # 3. GENERATED DATA
    # Cost parameters
    # https://www.mosaikosrl.net/batterie-accumulo-fotovoltaico/
    B = 5000  # B: battery capacity
    c_b = 2530  # Cost per battery
    # https://www.ecodirect.com/Canadian-Solar-CS6X-300P-300W-36V-PV-Panel-p/canadian-solar-cs6x-300p.htm?srsltid=AfmBOor_kd4mknwa-Am9K9m7VYG55_jnXMM3QTP7aTw2Y2qCChJ9GuL7
    c_p = 290  # Cost per unit of power: Canadian Solar CS6X-300P (inverter SMA_America_SB7000TL_US240V cost neglected)
    # TODO: ogni 25 pannelli solari c'è un inverter
    c_i = 3000  # Cost per inverter
    # https://tariffe.segugio.it/guide-e-strumenti/domande-frequenti/quanto-costa-un-kwh-di-energia-elettrica.aspx#:~:text=Il%20prezzo%20dell'energia%20elettrica%20oggi%20%C3%A8%20pari%20a%200,si%20applica%20ai%20clienti%20vulnerabili.
    c_e = 0.16053 / 1000  # Cost of energy

    data["B"] = B
    data["c_b"] = c_b
    data["c_p"] = c_p
    data["c_e"] = c_e

    # m_t: maximum energy available at time t TODO
    mmm = {t: random.randint(800000, 1000000) for t in T}
    data["mmm"] = mmm

    # n_i: number of jobs required for machine i
    n_jobs = {}
    n_jobs[1] = 2 * number_of_days
    n_jobs[2] = 2 * number_of_days
    n_jobs[3] = 1 * number_of_days
    n_jobs[4] = 1 * number_of_days
    n_jobs[5] = 1 * number_of_days
    n_jobs[6] = 1 * number_of_days
    data["n_jobs"] = n_jobs

    # c_i: cooldown period for machine i TODO
    c = {i: 1 for i in I}
    data["c"] = c

    # THRESHOLD_FOR_JOB_J
    THRESHOLD_FOR_JOB_J_AND_I = {(i, j): 24 * (j) for i in I for j in J}  # Time limit for each job TODO
    data["THRESHOLD_FOR_JOB_J_AND_I"] = THRESHOLD_FOR_JOB_J_AND_I

    # Sets of dependencies and shared resources
    M_dependencies = [(1,3)]  # Pairs of machines where the second depends on the first
    M_shared = [(3,4)]  # Groups of machines that share resources and cannot run simultaneously
    silent_periods = {1:[25,26,27,28,29,30,31,32,33], 5:[32,33,34,35,36,37,38], 6:[24,25,26,27,28,29,30]}  # Periods when certain machines must be off

    data["M_dependencies"] = M_dependencies
    data["M_shared"] = M_shared
    data["silent_periods"] = silent_periods

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
    print(f"Time limit for each job (THRESHOLD_FOR_JOB_J_AND_I): {THRESHOLD_FOR_JOB_J_AND_I}")
    print("--------------------------------------\n")

    return data


def get_data_house():
    
    data = {}
    
    solar_predictor = SolarProductionPredictor(
    model_path="../weather_pv_conversion/output/model/best_estimator_model.pkl",
    scaler_path="../weather_pv_conversion/output/model/scaler.pkl"
    )


    # 1. https://www.expressvpn.com/it/blog/il-consumo-energetico-della-tecnologia/?srsltid=AfmBOorBF1XynXRfEISxCWEpMMz4LLAI-2oxi2d1u-fjlvkUJGdnhW8D

    machine_energy_usage = [
        ("Portable Heater", 1.5*1000, 2),
        ("Air Conditioner (240V)", 1.8*1000, 4),
        ("Pedestal Fan", 0.03*1000, 8),
        ("Ceiling Fan", 0.05*1000, 6),
        ("Incandescent Bulbs (60W)", 0.75*1000, 8),
        ("Clock", 0.1*1000, 24),

        # Bathroom
        ("Electric Water Heater", 0.52*1000, 3),
        ("Electric Toothbrush", 0.003*1000, 5/60),
        ("Hair Dryer", 1.5*1000, 20/60),
        ("Curling Iron", 0.15*1000, 10/60),

        # Bedroom
        ("Electric Blanket (Queen)", 0.09*1000, 3),
        ("Night Light (4W)", 0.12*1000, 10),
        ("Smartphone Charger", 0.001*1000, 24),
        ("Alarm Clock", 0.008*1000, 24),

        # Kitchen
        ("Oven", 2.3*1000, 0.5),
        ("Cooktop", 1.5*1000, 0.5),
        ("Microwave", 1.44*1000, 0.25),
        ("Kettle", 0.11*1000, 10/60),
        ("Coffee Maker/Warmer", 0.4*1000, 1),
        ("Dishwasher (normal cycle)", 1.1*1000, 2),
        ("Toaster Oven", 0.75*1000, 5/60),
        ("Refrigerator + Freezer (17 cu. ft.)", 0.04*1000, 24),
        ("Air Fryer (1500W)", 1.5*1000, 20/60),

        # Laundry
        ("Dryer", 3.0*1000, 2/7),
        ("Conventional Washing Machine", 2.3*1000, 2/7),
        ("Vacuum Cleaner", 0.75*1000, 2/7),
        ("Robot Vacuum", 0.007*1000, 45/60),
        ("Iron", 1.08*1000, 40/60/7),

        # Living Room
        ("LED TV (average)", 0.095*1000, 3),  # average between 0.071 and 0.12
        ("Cable Box", 0.01*1000, 3),
        ("Media Player (Roku)", 0.002*1000, 3),
        ("Game Console", 0.07*1000, 2),
        ("Speakers (2 x 25W)", 0.05*1000, 3),
        ("Stereo", 0.05*1000, 2),
        ("Radio/CD Player", 0.02*1000, 1.5),
        ("Halogen Floor Lamp", 0.02*1000, 6),

        # Home Office
        ("Wi-Fi Router", 0.01*1000, 24),
        ("Desktop Computer", 0.06*1000, 2),
        ("Laptop", 0.05*1000, 2),
        ("LCD Monitor (17\")", 0.04*1000, 2),
        ("Printer", 0.001*1000, 10/60/7),

        # Outdoor/Garage
        ("Pool Pump", 0.56*1000, 6),
        ("Hot Tub (1500W)", 1.5*1000, 0.25),
        ("Electric Car (charging)", 0.49*1000, 7.5),
        ("Electric Bike (500Wh) (charging)", 0.37*1000, 3.5)  # Assuming 1h charge/day
    ]

    print("Machine names:")
    for i, (name, _, _) in enumerate(machine_energy_usage, start=1):
        print(f"{i}. {name}")
    

    MACHINES = len(machine_energy_usage) # Number of machines
    MAX_JOB_N = 24  # Maximum number of jobs per machine
    T_MAX = 24  # Number of time periods (e.g., 48 half-hours in a day)
    
    print(f"Machine number:", MACHINES)
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
    
    e = {}
    for i, (_, energy, _) in enumerate(machine_energy_usage, start=1):
        e[i] = energy
    print("e =", e)
    data["e"] = e

    # f_i: additional energy consumed when machine i starts TODO
    f = {i: int(e[i] * (1/6)) for i in e}
    print("f =", f)
    data["f"] = f

    # d_i: duration of job i on machine i
    d = {}
    for i, (_, _, usage) in enumerate(machine_energy_usage, start=1):
        d[i] = usage
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
    # https://www.mosaikosrl.net/batterie-accumulo-fotovoltaico/
    B = 5000 # B: battery capacity  
    c_b = 2530 # Cost per battery
    # https://www.ecodirect.com/Canadian-Solar-CS6X-300P-300W-36V-PV-Panel-p/canadian-solar-cs6x-300p.htm?srsltid=AfmBOor_kd4mknwa-Am9K9m7VYG55_jnXMM3QTP7aTw2Y2qCChJ9GuL7
    c_p = 290   # Cost per unit of power: Canadian Solar CS6X-300P (inverter SMA_America_SB7000TL_US240V cost neglected)
    # https://tariffe.segugio.it/guide-e-strumenti/domande-frequenti/quanto-costa-un-kwh-di-energia-elettrica.aspx#:~:text=Il%20prezzo%20dell'energia%20elettrica%20oggi%20%C3%A8%20pari%20a%200,si%20applica%20ai%20clienti%20vulnerabili.
    c_e = 0.16053/1000 # Cost of energy 
    

    data["B"] = B
    data["c_b"] = c_b
    data["c_p"] = c_p
    data["c_e"] = c_e
    

    # m_t: maximum energy available at time t TODO
    mmm = {t: random.randint(800000, 1000000) for t in T}
    data["mmm"] = mmm

    # n_i: number of jobs required for machine i 
    n_jobs = {}
    for i, (_, _, _) in enumerate(machine_energy_usage, start=1):
        n_jobs[i] = 1
    data["n_jobs"] = n_jobs

    # c_i: cooldown period for machine i TODO
    c = {i: 1 for i in I}
    data["c"] = c
    
    # THRESHOLD_FOR_JOB_J
    THRESHOLD_FOR_JOB_J_AND_I = {(i,j): 24 for i in I for j in J} # Time limit for each job TODO
    data["THRESHOLD_FOR_JOB_J_AND_I"] = THRESHOLD_FOR_JOB_J_AND_I


    # Sets of dependencies and shared resources
    M_dependencies = []  # Pairs of machines where the second depends on the first
    M_shared = []  # Groups of machines that share resources and cannot run simultaneously
    silent_periods = {}  # Periods when certain machines must be off
    
    data["M_dependencies"] = M_dependencies
    data["M_shared"] = M_shared
    data["silent_periods"] = silent_periods
    
    
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


"""
def get_job_duration_per_machine(idx, day):
    #data = datasetjson.get_data_day(day)
    if (i <= 2):
        return 8
    elif (i <= 4):
        return 12
    return 10

    # in piu
    for timestamp in data.keys():
        machine_data = data[timestamp].get(machine_name)
        if machine_data and machine_data['power_apparent'] > 0:
            total_duration += 1

    if(total_duration != 0): total_duration  = total_duration - 5   

    
    return total_duration

d = {}
for i in I:
    machine_name = machine_names[i - 1]
    duration = get_job_duration_per_machine(i, day)
    d[i] = duration
print("d =", d)
data["d"] = d
"""