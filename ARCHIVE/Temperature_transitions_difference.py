"""
Temperature_transitions.py
Maggie Jacoby 2020-04-19



"""

import os
import sys
import csv
import json
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import matplotlib.dates as mdates



class TempstoDF():
    def __init__(self, path, w_name):
        self.root_dir = path
        self.weather_file = os.path.join(self.root_dir, w_name)
        
    def read_T(self, file_p):
        df = pd.read_csv(file_p)
        df.Temperature_degC = df.Temperature_degC.round()
        df.Hour = pd.to_datetime(df.Hour).dt.strftime('%H:%M')
        return df
    
    def get_dist(self, df):
        all_H = df.Hour.unique()
        max_diff = df['Temperature_degC'].diff(periods=-1).max()
        min_diff = df['Temperature_degC'].diff(periods=-1).min()
        cols = [t for t in range(int(min_diff), int(max_diff)+1)]
        
        Temp_T = {}
        for h in all_H:
            p = []
            for t in range(int(min_diff), int(max_diff)+1):
                p.append(len(df.loc[(df['Temperature_degC'].diff(periods=-1)==t) & (df['Hour'] == h)]))
            new_p = [i/sum(p) for i in p]
            Temp_T[h] = new_p
        self.write_csv(Temp_T, cols)
        return Temp_T
 
    
    def write_csv(self, dist_dict, cols):
        csv_file = os.path.join(self.root_dir, 'To-transitions.csv')
        with open(csv_file, 'w') as f:
            f.write(','+f'{cols}'.strip("[]")+'\n')
            for hour in dist_dict:
                f.write(f'{hour}' +','+ f'{dist_dict[hour]}'.strip("[]") + '\n')
                        
            
    def main(self):
        df = self.read_T(self.weather_file)
        self.dist = self.get_dist(df)
 
 
if __name__ == '__main__':
    # w_path = '/Users/maggie/Documents/Github/DMU-Final-Project/boulder_hourlyTemps_jan.csv'
    # path = '/Users/maggie/Documents/Github/DMU-Final-Project/TransitionProbabilties/'

    if len(sys.argv) > 1:
        r_path = sys.argv[1]
    else:
        r_path = '/Users/maggie/Desktop/DMU_project_test/data_from_simulink'
    
    w_path = '/Users/maggie/Desktop/DMU_project_test/T_matrices'

    T = TempstoDF(r_path, w_path)
    T.main()