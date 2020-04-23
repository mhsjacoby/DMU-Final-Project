"""
TemperatureOut_transitions.py
Maggie Jacoby 2020-04-21

This file takes an outdooe TMY3 type file and creates temperture transitions P(s'|s)
transitions are created per hour, and output to csvs for each hour. 
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



class ToTransitionsCSV():
    def __init__(self, r_path, w_name):
        self.root_dir = r_path
        self.weather_file = os.path.join(self.root_dir, w_name)
        
    def read_T(self, file_p):
        df = pd.read_csv(file_p)
        df.Temperature_degC = df.Temperature_degC.round()
        df.Hour = pd.to_datetime(df.Hour).dt.strftime('%H:%M')
        return df
    
    def ind(self, t):
        return int(t-self.minT)
    
    def get_dist(self, df):
        all_H = df.Hour.unique()
        self.minT = df['Temperature_degC'].min()
        self.maxT = df['Temperature_degC'].max()
        allT = [t for t in range(int(self.minT), int(self.maxT)+1)]
        self.write_allT(allT)
        df['Tprime'] = df['Temperature_degC'].shift(-1)
        df = df.drop(df.tail(1).index)
        Temp_T = {}
        for h in all_H:
            transT_h = np.zeros((len(allT), len(allT)))
            for index, row in df[df.Hour == h].iterrows():
                i, j = self.ind(row['Temperature_degC']), self.ind(row['Tprime'])
                transT_h[i,j] += 1
            transT_h = transT_h/transT_h.sum()
            Temp_T[h] = transT_h
        return Temp_T
        
    def write_allT(self, allT):
        T_ind = [self.ind(t) for t in allT]
        csv_file = os.path.join(self.root_dir, f'temperature_indices.csv')
        np.savetxt(csv_file, np.column_stack((T_ind, allT)), fmt='%10.1f', delimiter=',', header='index,temperature', comments='')
    
    def write_csv(self, dist_dict):
        for h in dist_dict:       
            csv_file = os.path.join(self.root_dir, 'TransitionProbabilties', 'outdoorT', f'To_{h[0:2]}.csv')
            np.savetxt(csv_file, dist_dict[h], fmt='%10.5f', delimiter=',')

            
    def main(self):
        df = self.read_T(self.weather_file)
        T = self.get_dist(df)
        self.write_csv(T)


 
if __name__ == '__main__':
    # w_path = '/Users/maggie/Documents/Github/DMU-Final-Project/boulder_hourlyTemps_jan.csv'
    # path = '/Users/maggie/Documents/Github/DMU-Final-Project/TransitionProbabilties/'

    # if len(sys.argv) > 1:
    #     read_path = sys.argv[1]
    # else:
    #     read_path = '/Users/maggie/Documents/Github/DMU-Final-Project/boulder_hourlyTemps_jan.csv'

    write_path = '/Users/maggie/Documents/Github/DMU-Final-Project/'
    weather_file = 'boulder_hourlyTemps_jan.csv'

    t = ToTransitionsCSV(write_path, weather_file)
    t.main()


    

