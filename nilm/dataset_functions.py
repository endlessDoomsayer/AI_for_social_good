import json
import pandas as pd
import matplotlib.pyplot as plt

def plot_data(machine_names, data_slice):
    #start, *_, end = data_slice.keys()
    powers = {}
    time_index = []
    for machine_name in machine_names:
        powers[machine_name] = []

    plt.figure(figsize=(20, 10))
    for hour_data in data_slice.items():
        time_index.append(hour_data[0])
        for machine_data in hour_data[1].items():
            #get the data of the machine
            data = machine_data[1]
            #get the power data
            power = data['power_apparent']
            powers[machine_data[0]].append(power)
    for machine, power in powers.items():
        #plot the data
        plt.plot(time_index, power, marker='o', linestyle='None', label=machine)
    plt.legend(loc='best')

class Dataset:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.data = None

    def load(self):
        '''
        Load the json dataset

        Reading hour by hour, it creates a dictionary of the type: (k:v) -> (pd.Timestamp : dict). each dict is of the type: (k:v) -> (machine_name : dict). each dict is of the type: (k:v) -> (stat : value). stats are 'power_apparent', 'current', 'voltage'.
        If in a row (hour) there is no data for a machine, the respective dictionary is created and every stat is set to 0.
        '''
        #load the json file, in order to have a dictionary where the keys are Timestamps and the values are dictionaries with the machine name as key and the power usage as value
        with open(self.path, 'r') as f:
            data = json.load(f)
        self.data = {}
        #convert the keys to datetime objects
        for key in data.keys():
            key_ts = pd.Timestamp(key)
            #convert the key to a datetime object
            self.data[key_ts] = {}
            for k,v in data[key].items():
                #convert the value to a dictionary
                self.data[key_ts][k] = {}
                #convert string v to a dictionary
                if v is None: v = {'power_apparent': 0, 'current': 0, 'voltage': 0}
                else: v = json.loads(v)

                self.data[key_ts][k] = v
        print(f"Loaded dataset {self.name} with {len(self.data)} entries")

    def get_machine_names(self):
        #get the machine names from the dataset
        machine_names = []
        for key in self.data.keys():
            for k in self.data[key].keys():
                if k not in machine_names:
                    machine_names.append(k)
        return machine_names

    def get_start_end_time(self):
        #get the start and end time of the dataset
        start_time = min([key for key in self.data.keys()])
        end_time = max([key for key in self.data.keys()])
        return start_time, end_time
    
    def get_data_start_end(self, start_time: pd.Timestamp, end_time: pd.Timestamp):
        #get the data in the given time interval
        start_time = start_time.tz_localize(None)
        end_time = end_time.tz_localize(None)
        data = {}
        for key in self.data.keys():
            key2 = key.tz_localize(None)
            if key2 >= start_time and key2 <= end_time:
                data[key] = self.data[key]
        return data
    
    def get_data_day(self, day: pd.Timestamp):
        #get the data for a given day
        data = {}
        for key in self.data.keys():
            if key.date() == day.date():
                data[key] = self.data[key]
        return data
    
    def get_data_week(self, day: pd.Timestamp):
        #get the data for a given week
        week = day.isocalendar()[1]
        data = {}
        for key in self.data.keys():
            if key.isocalendar()[1] == week:
                data[key] = self.data[key]
        return data
    
    def get_data_month(self, day: pd.Timestamp):
        #month = day.isocalendar().
        #get the data for a given month
        data = {}
        for key in self.data.keys():
            if key.month == day.month:
                data[key] = self.data[key]
        return data
    
    def get_average_power_usage(self, machine_name: str, start_time: pd.Timestamp, end_time: pd.Timestamp):
        #get the average power usage for a given machine in a given time interval
        data = self.get_data_start_end(start_time, end_time)
        total_power = 0
        hours = 0
        for key in data.keys():
            if data[key][machine_name] is not None:
                total_power += data[key][machine_name][('power_apparent')]
                hours += 1
        if hours == 0:
            return 0
        return total_power / hours