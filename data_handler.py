import datetime
import os
import glob
import re
from dateutil.parser import parse
import h5py
import xarray as xr
import logging

from PyQt5.QtCore import QObject, pyqtSignal

import platform
name = platform.node()
if name == 'GLaDOS':
    ROOT_DIR = 'A:'
    BACKUP_DIR = 'S:'
elif name == 'burrito':
    ROOT_DIR = '/mnt/heap'
    BACKUP_DIR = '/mnt/jilafile/strontium_archive_uncompressed/Raw Data'
elif name == 'PAL9000':
    ROOT_DIR = 'A:\\heap'
    BACKUP_DIR = 'S:\\kaufman\\archive\\strontium_archive_uncompressed\\Raw Data'
else:
    ROOT_DIR = 'A:\\heap'
    BACKUP_DIR = 'S:\\archive\\strontium_archive_uncompressed\\Raw Data'
RAW_DATA_FOLDER = 'Raw Data'
PROCESSED_DATA_FOLDER = 'Processed Data'
SAVE_FOLDER = 'autoanalysis_results'
DATE_STR_FORMAT = '%y%m%d'

class DataHandler(QObject):
    date_updated = pyqtSignal(datetime.datetime)
    file_updated = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._date = self.get_most_recent_date()
        self._file = self.get_most_recent_file()
        
        self.raw_data_loaded = False
        self.raw_data_path = None
        self.raw_data = None
        self.processed_data_loaded = False
        self.processed_data_path = None
        self.processed_data = None

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
        self.date_updated.emit(self.date)
        self.raw_data_loaded = False
        self.processed_data_loaded = False

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
        self.file_updated.emit(self.file)
        self.raw_data_loaded = False
        self.processed_data_loaded = False


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
    
    def __load_raw_data(self):
        data_path = os.path.join(ROOT_DIR, self.date.strftime(DATE_STR_FORMAT), 
            RAW_DATA_FOLDER, f'data_{self.file}.h5')
        if not os.path.exists(data_path):
            data_path = os.path.join(BACKUP_DIR, self.date.strftime('%Y'), self.date.strftime('%y%m'),
                self.date.strftime(DATE_STR_FORMAT), RAW_DATA_FOLDER, f'data_{self.file}.h5')
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at {data_path}")
        self.current_data_path = data_path
        self.raw_data = h5py.File(self.current_data_path, 'r')
        self.raw_data_loaded = True
    
    def get_raw_data(self):
        if not self.raw_data_loaded:
            self.__load_raw_data()
        return self.raw_data
    
    def save_processed_dataset(self, ds):
        processed_data_path = os.path.join(ROOT_DIR, SAVE_FOLDER, self.date.strftime('%Y'), self.date.strftime('%y%m'),
            self.date.strftime(DATE_STR_FORMAT), PROCESSED_DATA_FOLDER, f'data_{self.file}.nc')
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        ds.to_netcdf(processed_data_path)
        self.processed_data = ds
        self.processed_data_path = processed_data_path
        self.processed_data_loaded = True

    def save_image_processing_fig(self, fig):
        fig_path = os.path.join(ROOT_DIR, SAVE_FOLDER, self.date.strftime('%Y'), self.date.strftime('%y%m'),
            self.date.strftime(DATE_STR_FORMAT), "image_processing_summary", f'summary_{self.file}.pdf')
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path)

    def __load_processed_data(self):
        processed_data_path = os.path.join(ROOT_DIR, SAVE_FOLDER, self.date.strftime('%Y'), self.date.strftime('%y%m'),
            self.date.strftime(DATE_STR_FORMAT), PROCESSED_DATA_FOLDER, f'data_{self.file}.nc')
        if not os.path.exists(processed_data_path):
            raise FileNotFoundError(f"Processed data file not found at {processed_data_path}")
        self.processed_data = xr.open_dataset(processed_data_path)
        self.processed_data_loaded = True

    def get_processed_data(self):
        if not self.processed_data_loaded:
            self.__load_processed_data()
        return self.processed_data
    
    def save_analysis_fig(self, fig):
        fig.suptitle(self.date.strftime(DATE_STR_FORMAT) + f' File {self.file}')
        fig.tight_layout()
        fig_path = os.path.join(ROOT_DIR, SAVE_FOLDER, self.date.strftime('%Y'), self.date.strftime('%y%m'),
            self.date.strftime(DATE_STR_FORMAT), "analysis_summary", f'summary_{self.file}.pdf')
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path)
