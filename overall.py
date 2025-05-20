from weather_pv_conversion.solar_production import SolarProductionPredictor
from nilm.dataset_functions import Dataset, plot_data
import pandas as pd
import matplotlib.pyplot as plt

# Take the info of the machines from nilm
datasetjson_path = 'nilm/output/IMDELD.json'
datasetjson = Dataset('IMDELD', datasetjson_path)
datasetjson.load()

# Take the names of the machines and map them to I
machine_names = datasetjson.get_machine_names()
print(machine_names)
# TODO: map to I

# Get the start and end time of the dataset (to understand bounds)
start, end = datasetjson.get_start_end_time()

# Get the whole data (plot it just so yeah it's cool)
whole_data = datasetjson.get_data_start_end(start, end)
plot_data(machine_names, whole_data)

# Pick a day (in this case the second day of the dataset)
day = start + pd.Timedelta(days=1)

# Get the data for that day
hour_datas = datasetjson.get_data_day(day)
#daily data is a dictionary with the hour as key and the data as value
#each hour is a dictionary with the machine as key and the data as value
#each machine is a dictionary with 3 elements: 'power_apparent', 'voltage' and 'current'

plot_data(machine_names, hour_datas)

# Get data of a week (there is also month)
day = start + pd.Timedelta(days=5)
week_data = datasetjson.get_data_week(day)


# Then take the solar production prediction for a certain day or sequence of days.
predictor = SolarProductionPredictor()

start_date = "2024-07-10"
predictions = predictor.predict(start_date_str=start_date) # this gives every hour as a pd dataframe
print(predictions)

end_date = "2024-07-15"
predictions = predictor.predict(start_date_str=start_date, end_date_str=end_date) # this gives every hour as a pd dataframe
print(predictions)

# And now do the magic with our models