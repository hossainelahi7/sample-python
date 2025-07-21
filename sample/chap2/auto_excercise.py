import pandas as pd
import os
import matplotlib.pyplot as plt

file_path = os.path.abspath('Auto.csv')

auto = pd.read_csv(file_path)
print(auto.describe())