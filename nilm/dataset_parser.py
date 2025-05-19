from nilmtk import DataSet
from nilmtk.elecmeter import ElecMeterID
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os

def load_dataset(path_to_dataset):
    ds = DatasetHandler(path_to_dataset)

    ds.set_start_time(pd.Timestamp('2017-12-11'))
    ds.set_end_time(pd.Timestamp('2018-04-01'))
    ds.set_useful_meters([3,4,5,6,7,8,10,11])
    ds.set_useful_columns([('power', 'apparent'), ('current', ''), ('voltage','')])
    print("Loading dataset...")
    ds.init()

    return ds


class DatasetHandler:
    def __init__(self, path):
        self.path = path
        self.nilm_dataset = DataSet(path)

    def set_start_time(self, start_time):
        self.start_time = start_time

    def set_end_time(self, end_time):
        self.end_time = end_time
    
    def set_useful_meters(self, useful_meters):
        self.useful_meters = [ElecMeterID(instance = id, building = 1, dataset = self.nilm_dataset.metadata['name']) for id in useful_meters]

    def set_useful_columns(self, useful_columns):
        self.useful_columns = useful_columns
    
    def init(self):
        self.nilm_dataset.set_window(start=self.start_time, end=self.end_time)
        self.elec_meters = self.nilm_dataset.buildings[1].elec

        self.loaded_data = {}
        for meter in self.useful_meters:
            temp_data = next(self.elec_meters[meter].load())
            # Create hourly markers within the original data interval
            hourly_markers = pd.date_range(start=temp_data.index[0].ceil('H'),
                                           end=temp_data.index[-1].floor('H'),
                                           freq='H')
            selected_rows = []
            new_index = []
            for marker in hourly_markers:
                # Find the index with minimal time difference to the marker (within 30 minutes)
                diff = np.abs(temp_data.index - marker)
                diff = diff[diff < pd.Timedelta(minutes=30)]
                if len(diff) == 0:
                    continue
                closest_index = temp_data.index[np.argmin(diff)]
                selected_rows.append(temp_data.loc[closest_index])
                new_index.append(marker)
            
            # Create a new DataFrame with the precise hourly markers as index
            temp_data = pd.DataFrame(selected_rows, index=new_index)
            
            # Then shift the indices by -12 hours
            temp_data.index = temp_data.index - pd.Timedelta(hours=12)
            
            # Change the columns using the new names
            
            self.loaded_data[meter] = temp_data[self.useful_columns]
            self.loaded_data[meter].columns = ['power_apparent', 'current', 'voltage']
            print(f"Loaded data for meter {meter} with shape {self.loaded_data[meter].shape}")
            print(self.loaded_data[meter].head())
        self.json_dataset = None

    def get_machines_ids(self):
        return [self.elec_meters[id].identifier for id in self.useful_meters]
    
    def get_machine_name(self, machine_id):
        return self.elec_meters[machine_id.instance].label()
    
    def get_power_usage(self, machine_id, start, end):
        """
        This function retrieves the power usage in a given time interval.
        :param start: start time of the interval
        :param end: end time of the interval
        :return: power usage in the given time interval
        """
        #filter the data by the given time interval
        df = df[(df.index >= start) & (df.index <= end)]
        #return the power usage
        return df
    
    def get_loaded_data(self):
        """
        This function retrieves the loaded data.
        :return: loaded data
        """
        return self.loaded_data
    
    def convert_nilm_to_json(self):
        
        print("Converting nilm to json...")

        all_keys = pd.DatetimeIndex([])
        for meter_id in self.useful_meters:
            current_index = pd.to_datetime(self.loaded_data[meter_id].index)
            all_keys = all_keys.union(current_index)
            
        all_keys = all_keys.sort_values()
        print("-------------------------")

        self.json_dataset = {}
        for key in all_keys:
            # Ensure key is used as a Timestamp object
            self.json_dataset[str(key)] = {}
            for meter in self.useful_meters:
                json_value_for_meter = None
                machine_name = self.get_machine_name(meter)

                meter_series = self.loaded_data[meter]
                #print(meter_series.head())
                # Use key directly as a Timestamp object
                try:
                    data_at_key = meter_series.loc[key]
                except KeyError:
                    # Handle the case where the key is not found in the series
                    data_at_key = None
                
                #print(key, data_at_key, "\n")
                print(type(data_at_key))
                if data_at_key is not None:
                    if isinstance(data_at_key, pd.Series):
                        json_value_for_meter = data_at_key.to_json()
                # Add the data to the dictionary
                self.json_dataset[str(key)][machine_name] = json_value_for_meter
        print("Finished converting nilm to json")

    def save_dataset_to_json(self, path):
        """
        This function saves the dataset to an xml file.
        :param path: path to the xml file
        """
        if not self.json_dataset:
            print("json dataset is empty, converting nilm to json")
            self.convert_nilm_to_json()
        
        #if path folder does not exist, create it
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        #save the json dataset to a file
        with open(path, 'w') as f:
            json.dump(self.json_dataset, f, indent=4)