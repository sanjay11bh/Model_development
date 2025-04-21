# Target is to combine both model2 and preprocessing.py into one file
# and make it a single file for deployment
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model2 import ReadingData , Models , OptunaTuner ,  ModelSelector
from sklearn.metrics import mean_squared_error
from preprocessing import SpectralData

class FinalModel:
    