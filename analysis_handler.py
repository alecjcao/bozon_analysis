import os
import importlib.util
import sys
from pathlib import Path
from PyQt5.QtCore import QObject, pyqtSignal

SCRIPTS_DIR = Path(os.path.join(os.path.dirname(__file__), "analysis_scripts"))

class AnalysisHandler(QObject):
    module_updated = pyqtSignal(str)

    def __init__(self, data_handler):
        super().__init__()
        self.all_modules = {}
        self.__load_all_modules()
        self._module_name = None
        self.module = None

        self.data_handler = data_handler
        
    @property
    def module_name(self):
        return self._module_name
    
    @module_name.setter
    def module_name(self, new_module):
        new_module = Path(new_module).stem
        if new_module == self.module_name:
            return
        if new_module is None:
            self._module_name = None
            self.module = None
            self.module_updated.emit('')
            return
        if new_module in self.all_modules.keys():
            self._module_name = new_module
            self.module = self.all_modules[new_module]
            self.module_updated.emit(self.module_name)
            return
        else:
            raise ValueError(f'{new_module} not found in analysis_scripts folder.')

    def __load_all_modules(self):
        self.all_modules = {}
        for script in SCRIPTS_DIR.glob("*.py"):
            module_name = script.stem  # Extract script name without extension
            # Load the script as a module
            spec = importlib.util.spec_from_file_location(module_name, script)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module  # Add to system modules
            spec.loader.exec_module(module)  # Execute the script
            self.all_modules[module_name] = module  # Store in dictionary

    def run_analysis_script(self):
        result = {}
        if self.module is not None:
            data = self.data_handler.get_processed_data()
            result, figure = self.module.main(data)
            self.data_handler.save_analysis_fig(figure)
            if result is None:
                result = {}
        return result