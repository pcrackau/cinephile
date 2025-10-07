import configparser
from typing import List
from pathlib import Path


class Config:
    def __init__(self, config=None):
        if isinstance(config, configparser.ConfigParser):
            for section in config._sections:
                self.__dict__[section] = Config.__dict__[section](config)                
        else:
            self.Responder = self.Responder()
            self.Pipeline = self.Pipeline()

    def set_option_from_config(sectionObj, config):
        if isinstance(config, configparser.ConfigParser):
            sect = config._sections[type(sectionObj).__name__]
            for option in sect:
                sett = config._sections[type(sectionObj).__name__][option]
                setattr(sectionObj, option, sett)                
            return True
        return False
    
    class Paths:
        def __init__(self, config=None):
            self.film_data = "datasets/"
            self.models = "models/"
            self.obj_detect_model = "models/yolov8x.pt"
            Config.set_option_from_config(self, config)

    class Responder:
        def __init__(self, config=None):
            self.llm = "mistral:latest"
            Config.set_option_from_config(self, config)

    class Pipeline:
        def __init__(self, config=None):
            self.shottypes = ['close_up', 'extreme_close_up', 'long_shot', 'medium_long_shot', 'medium_shot']
            # Overwrite from config if available
            Config.set_option_from_config(self, config)



