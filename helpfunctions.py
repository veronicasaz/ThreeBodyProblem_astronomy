import numpy as np
import json

def load_json(filepath):
    """
    load json file as dictionary
    """
    with open(filepath) as jsonFile:
        data = json.load(jsonFile)
    jsonFile.close()
    return data
