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



class ToTiTransitionsCSV():
    def __init__(self, root_path, rf, action):
        self.root_dir = root_path
        self.results_file = rf
        self.action = action
        
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
    
    def get_dist(self, df):
        minTi, maxTi = df['Ti'].min(), df['Ti'].max()
        minTo, maxTo = df['To'].min(), df['To'].max()
        self.allTi = [t for t in range(int(minTi), int(maxTi)+1)]
        self.allTo = [t for t in range(int(minTo), int(maxTo)+1)]

        all_pairs = [(x,y) for x in self.allTo for y in self.allTi]
        pair_inds = {x:i for i, x in enumerate(all_pairs)}
        Ti_inds = {x:int(x-minTi) for x in self.allTi}

        self.write_inds(pair_inds, 'ToTi', 'index,temp out,temp in')
        self.write_inds(Ti_inds, 'Ti', 'index,temp in')

        transT = np.zeros((len(self.allTi)*len(self.allTo), len(self.allTi)))
        df = df[df.a==self.action]

        print(f'action {self.action} length {len(df)}')
        for index, row in df.iterrows():
            i = pair_inds[(int(row.To), int(row.Ti))]
            j = Ti_inds[(int(row.Ti_prime))]            
            transT[i,j] += 1
        transT = transT/transT.sum()
        
        ToTi_Ti = pd.DataFrame(data=transT, index=pair_inds.keys(), columns=Ti_inds.keys())

        df_ext = ToTi_Ti
        for i in range(1, len(self.allTo)):
            df_ext = pd.concat([df_ext, ToTi_Ti], axis=1, sort=False)
        # df_ext.index = [df_ext.index.map('({0[0]},{0[1]})'.format)]
        # print("before", len(df_ext), len(df_ext.columns), len(df_ext.index))
        df_ext.index = df_ext.index.to_flat_index()
        # print("aftre", len(df_ext), len(df_ext.columns), len(df_ext.index))

        return df_ext

    def write_inds(self, inds_dict, fname, header):
        csv_file = os.path.join(self.root_dir,  f'{fname}-inds-{self.rf_num}.csv')
        np.savetxt(csv_file, np.column_stack(([v for v in inds_dict.values()], [k for k in inds_dict.keys()])), 
                   fmt='%10.1f', delimiter=',', header=header, comments='')

            
    def main(self):
        self.rf_num = self.results_file.split('_')[1].strip('.csv')
        file_path = os.path.join(self.root_dir, 'data_from_simulink', self.results_file)
        self.df = self.read_T(file_path)
        self.TransitionMatrix = self.get_dist(self.df)



class ToTransitionsCSV():
    def __init__(self, r_path, w_name, h):
        self.root_dir = r_path
        self.weather_file = os.path.join(self.root_dir, w_name)
        self.hour = h
        
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
        df = df[df['Hour'] == self.hour]

        transT = np.zeros((len(allT), len(allT)))
        for index, row in df.iterrows():
            i, j = self.ind(row['Temperature_degC']), self.ind(row['Tprime'])
            transT[i,j] += 1
        transT = transT/transT.sum()
        To_To = pd.DataFrame(data=transT, index=allT, columns=allT)
        return To_To
        
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
        self.T = self.get_dist(df)


if __name__ == '__main__':
    def make_storage_dir(target_dir):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        return target_dir

    def generate_ToTi(a):
        t = ToTiTransitionsCSV(root_dir, results_file, actions[a])
        t.main()
        storage_directory = make_storage_dir(os.path.join(root_dir, 'TransitionProbabilties'))
        t.TransitionMatrix.to_csv(os.path.join(storage_directory,f'indoor_temp-A{a}.csv'))
        return t.TransitionMatrix, len(t.allTi)


    root_dir = '/Users/maggie/Desktop/DMU_project_test/'
    results_file = 'resultFile_2.csv'
    weather_file = 'boulder_hourlyTemps_jan.csv'

    hours = [str(x).zfill(2) + ':00' for x in range(0,24)]
    actions = {0:0, 1:3500, 2:6000}
    
    TT_0, L = generate_ToTi(0)
    TT_1, L = generate_ToTi(1)
    TT_2, L = generate_ToTi(2)

    action_dfs = [(TT_0, 0), (TT_1, 1), (TT_2,2)]

    for h in hours:
        to = ToTransitionsCSV(root_dir, weather_file, h)
        to.main()
        cols_orig = [x for x in to.T.columns]
        cols_new = np.repeat(cols_orig, L)

        for col in to.T.columns:
            for i in reversed(range(1, L)):
                to.T.insert(to.T.columns.get_loc(col), f'{col}-{i}', to.T[col])
        to.T = to.T.iloc[np.arange(len(to.T)).repeat(L)]

        for TT in action_dfs:
            TT_df = TT[0].copy(deep=True)

            new_col = [f'({x},{y})' for x, y in zip(cols_new, TT_df.columns)]
            new_ind = [f'[{y},{x}]' for x, y in zip(to.T.index, TT_df.index)]

            to.T.columns, TT_df.columns = new_col, new_col
            to.T['nindex'], TT_df['nindex'] = new_ind, new_ind
            to.T = to.T.set_index('nindex')
            TT_df = TT_df.set_index('nindex')

            new_df = TT_df*to.T
            new_df.to_csv(os.path.join(root_dir, f'TransitionProbabilties/h{h[0:2]}-A{TT[1]}.csv'))
