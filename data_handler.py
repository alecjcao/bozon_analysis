import datetime
import os
import glob
import re
from dateutil.parser import parse
import h5py

import platform
import platform
name = platform.node()
if name == 'GLaDOS':
    ROOT_DIR = 'A:'
elif name == 'burrito':
    ROOT_DIR = '/mnt/heap'
else:
    ROOT_DIR = 'A:\\heap'
RAW_DATA_FOLDER = 'Raw Data'
DATE_STR_FORMAT = '%y%m%d'

class DataHandler():
    def __init__(self):
        self._date = self.get_most_recent_date()
        self._file = self.get_most_recent_file()
        
        self.current_data_loaded = False
        self.data = None

    @property
    def date(self):
        return self._date
    
    @date.setter
    def date(self, new_date):
        """
        If valid date, set the date attribute to the input date.
        """
        if isinstance(new_date, datetime.date):
            self._date = new_date
        elif isinstance(new_date, str):
            try:
                self._date = datetime.datetime.strptime(new_date, DATE_STR_FORMAT)
            except ValueError as e:
                raise ValueError("Invalid date string. Please use 'YYMMDD'.")
        else:
            raise ValueError("Invalid date type. Please use a valid string or datetime.date object.")
        self.current_data_loaded = False

    @property
    def file(self):
        return self._file
    
    @file.setter
    def file(self, new_file):
        """
        If valid file number, set the file attribute to the input file number.
        """
        try:
            self._file = int(new_file)
        except ValueError:
            raise ValueError("Invalid file number. Please use a valid integer.")
        self.current_data_loaded = False


    def get_most_recent_date(self):
        """
        Find and return the most recent date with data in the Raw Data folder
        """
        dirs = next(os.walk(ROOT_DIR))[1]
        most_recent_day = datetime.date.today()
        while True:
            most_recent_day_str = most_recent_day.strftime(DATE_STR_FORMAT)
            if most_recent_day_str in dirs:
                # check if there is a raw data folder with data
                data_address = os.path.join(ROOT_DIR, most_recent_day_str, RAW_DATA_FOLDER)
                if os.path.exists(data_address):
                    # check that there is data in that folder
                    if os.listdir(data_address) != []:
                        return most_recent_day
            most_recent_day = most_recent_day - datetime.timedelta(days=1)

    def get_most_recent_file(self):
        """
        Find and return the most recent file number in the current date's Raw Data folder.
        """
        data_folder = os.path.join(ROOT_DIR, self.date.strftime(DATE_STR_FORMAT), RAW_DATA_FOLDER)
        most_recent_file = sorted(glob.glob(data_folder+'\\data_*.h5'), key=os.path.getmtime)[-1]
        return int(re.search('data_(.*).h5', most_recent_file).group(1))
    
    def load_data(self):
        data_path = os.path.join(ROOT_DIR, self.date.strftime(DATE_STR_FORMAT), RAW_DATA_FOLDER, f'data_{self.file}.h5')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        return h5py.File(data_path, 'r')
    
    def get_data(self):
        if not self.current_data_loaded:
            self.current_data = self.load_data()
            self.current_data_loaded = True
        return self.current_data
        
