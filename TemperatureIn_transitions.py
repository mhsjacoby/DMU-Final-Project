"""
TemperatureIn_transitions.py
Maggie Jacoby 2020-04-23

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



class TiTransitionsCSV():
    def __init__(self, r_path):
        self.root_dir = r_path
        
    def mylistdir(self, directory, bit='', end=True):
        filelist = os.listdir(directory)
        if end:
            return [x for x in filelist if x.endswith(f'{bit}') and not x.startswith('.') and not 'Icon' in x]
        else:
             return [x for x in filelist if x.startswith(f'{bit}') and not x.startswith('.') and not 'Icon' in x]

    def make_storage_directory(self, target_dir):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        return target_dir        
          
        
    def read_T(self, file_p):
        df = pd.read_csv(file_p)
        columns = ['Ti', 'Ti_prime', 'To']
        for col in columns:
            df[col] = df[col].round()
        df['To_prime'] = df['To'].shift(-1)
        df = df.drop(df.tail(1).index)
        return df

    
    def get_dist(self, df, rf_num):
        minTi, maxTi = df['Ti'].min(), df['Ti'].max()
        minTo, maxTo = df['To'].min(), df['To'].max()
        allTi = [t for t in range(int(minTi), int(maxTi)+1)]
        allTo = [t for t in range(int(minTo), int(maxTo)+1)]

        all_pairs = [(x,y) for x in allTo for y in allTi]
        inds = {x:i for i, x in enumerate(all_pairs)}
        self.write_inds(inds, rf_num)
    
        Temp_T = {}
        transT = np.zeros((len(allTi)*len(allTo), len(allTi)*len(allTo)))
        for index, row in df.iterrows():
            i = inds[(int(row.To), int(row.Ti))]
            j = inds[(int(row.To_prime), int(row.Ti_prime))]            
            transT[i,j] += 1
          
        transT = transT/transT.sum()
        return transT


    def write_inds(self, inds_dict, rf):
        csv_file = os.path.join(self.root_dir,  f'pair_inds-{rf}.csv')
        np.savetxt(csv_file, np.column_stack(([k for k in inds_dict.keys()], [v for v in inds_dict.values()])), 
                   fmt='%10.1f', delimiter=',', header='index,temp out,temp in', comments='')

    
    def write_csv(self, transitionT, rf_num):    
        
        T_store = self.make_storage_directory(os.path.join(self.root_dir, 'TransitionProbabilties'))
        csv_file = os.path.join(T_store, f'To_Ti-{rf_num}.csv')
        np.savetxt(csv_file, transitionT, fmt='%10.5f', delimiter=',')

            
    def main(self):
        for f in self.mylistdir(os.path.join(self.root_dir, 'data_from_simulink'), bit='.csv'):
            rf_num = f.split('_')[1].strip('.csv')
            weather_file = os.path.join(self.root_dir, 'data_from_simulink', f)
            df = self.read_T(weather_file)
            T = self.get_dist(df, rf_num)
            self.write_csv(T, rf_num)

 
if __name__ == '__main__':
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = '/Users/maggie/Desktop/DMU_project_test/'

    t = TiTransitionsCSV(root_dir)
    t.main()


    

