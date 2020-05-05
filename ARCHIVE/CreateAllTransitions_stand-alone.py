"""
CreateAllTransitions.py
Maggie Jacoby 2020-04-25

This code combines the two temperature transition classes and imports the occupancy class
Combines all results files to create full transitions
When run alone this outputs a .npy file containing the full transition matrix

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

import MyFunctions as my        # Contains mylistidir and make_storage_directory functions
import OccupancyCalcs as occ   # module that creates and plots occupancy data


class ToTiTransitionsCSV():
    def __init__(self, root_path, action, results_dir='data_from_simulink', rf='x'):
        self.root_dir = root_path
        self.results_file_path = os.path.join(self.root_dir, results_dir)
        self.action = action
        self.rf_num=rf
        
    def read_T(self, file_p):
        df = pd.read_csv(file_p)
        df.columns = ['Ti', 'Ti_prime', 'To', 'a']
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

        df = df[df.a==self.action]
        self.len_rfa = len(df)
        transT = np.zeros((len(self.allTi)*len(self.allTo), len(self.allTi)))

        for index, row in df.iterrows():
            i = pair_inds[(int(row.To), int(row.Ti))]
            j = Ti_inds[(int(row.Ti_prime))] 
            transT[i,j] += 1

        replacedT = self.impute_missing_T(transT)
        ToTi_Ti = pd.DataFrame(data=replacedT, index=pair_inds.keys(), columns=Ti_inds.keys())
        normedT = ToTi_Ti.div(ToTi_Ti.sum(axis=1), axis=0)

        df_ext = normedT.copy(deep=True)
        for i in range(1, len(self.allTo)):
            df_ext = pd.concat([df_ext, normedT], axis=1, sort=False)
        df_ext.index = df_ext.index.to_flat_index()
        return df_ext
     
    def impute_missing_T(self, df):
        missing = []
        replaced_inds = []
        for i, row in enumerate(df):
            if row.sum() != 0:
                saved_r = row
            else:
                try:
                    df[i] = saved_r
                    replaced_inds.append(i)
                except NameError:
                    missing.append(i)
        for i in missing:
            df[i] = saved_r
        return df


    def read_in_results(self, f):
        file_path = os.path.join(self.results_file_path, f)
        print(f'> Reading from {file_path}...')
        df = self.read_T(os.path.join(self.results_file_path, f))
        return df


    def main(self):
        results_files = my.mylistdir(self.results_file_path, bit='.csv')
        df1 = self.read_in_results(results_files[0])
        
        if len(results_files) > 1:
            for f in results_files[1:]:
                dfx = self.read_in_results(f)
                df1 = pd.concat([df1, dfx], axis=0)

        self.df=df1
        self.TransitionMatrix = self.get_dist(self.df)
        self.len_df = len(self.TransitionMatrix)


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
        self.minT, self.maxT = df['Temperature_degC'].min(), df['Temperature_degC'].max()
        allT = [t for t in range(int(self.minT), int(self.maxT)+1)]
        df['Tprime'] = df['Temperature_degC'].shift(-1)
        df = df.drop(df.tail(1).index)
        df = df[df['Hour'] == self.hour]

        transT = np.zeros((len(allT), len(allT)))
        print('dims: ', len(allT), len(allT))
        for index, row in df.iterrows():
            i, j = self.ind(row['Temperature_degC']), self.ind(row['Tprime'])
            transT[i,j] += 1.0

        To_To = pd.DataFrame(data=transT, index=allT, columns=allT)
        To_To_normed = To_To.div(To_To.sum(axis=1), axis=0).fillna(value=0.0)
        return To_To_normed

    def main(self):
        df = self.read_T(self.weather_file)
        self.T = self.get_dist(df)


# Some general functions for file reading and writing, specific to transition functions
def generate_ToTi(a, Rdir):                       # Given the results file from simulink, create TiTo->Ti' transitions (by action)
    t = ToTiTransitionsCSV(Rdir, actions[a])
    t.main()
    # storage_directory = my.make_storage_directory(os.path.join(Rdir, 'TransitionProbabilties'))
    return t.TransitionMatrix, len(t.allTi), t.len_rfa


def combine_occ_data(h, occD, tempDF):
    home_home, home_away = occD.Home[h][0], occD.Home[h][1]         # Get probs of o'=home, o'=leave, for each hour, given o=home
    away_home, away_away = occD.notHome[h][0], occD.notHome[h][1]   # Get probs of o'=home, o'=not home, for each hour, given o=not home

    df_HH = tempDF*home_home                                        # Create top-left df
    df_HA = tempDF*home_away                                        # Create top-right
    df_givenH = pd.concat([df_HH, df_HA], axis = 1)                 # Combine top half

    df_AH = tempDF*away_home                                        # Create bottom-left df
    df_AA = tempDF*away_away                                        # Create bottom-right df
    df_givenA = pd.concat([df_AH, df_AA], axis = 1)                 # Combine bottom half 

    full_df = pd.concat([df_givenH, df_givenA], axis = 0)           # Create full Occ inlcuded df
    print(full_df.index)
    return full_df

    

if __name__ == '__main__':
    root_dir = '/Users/maggie/Desktop/DMU_project_test/'            # Dicrectory to write to
    # results_file = 'resultFileLowHeat_3.csv'                               # Simulink output file
    weather_file = 'boulder_hourlyTemps_jan.csv'                    # TMY3 file to create T_outdoor transitions

    occupancy_path = '/Users/maggie/Documents/Github/DMU-Final-Project/Occupancy'   # Location of original occupancy files
    occupancy = occ.HomeOccupancy(occupancy_path, root_dir)         # using Occupancy file, read in raw data and create df for the home
    occupancy.main()                                            
    full_df = occupancy.df_hr                                       # Output of full occupancy df
    occ_probs = occ.GetProbs(full_df, 'Will', root_dir)             # Get transition prbabilties for one resident
    occ_probs.main()

    hours = [str(x).zfill(2) + ':00' for x in range(0,24)]          # 24 hours to use     
    actions = {0:0, 1:3500, 2:6000}                                 # Actions used by simulink
    
    TT_0, L0, l0 = generate_ToTi(0, root_dir)                       # Create ToTi->Ti' for each action
    TT_1, L1, l1 = generate_ToTi(1, root_dir)
    TT_2, L2, l2 = generate_ToTi(2, root_dir)

    print(f'Total length of results file: {l0 + l1 + l2}, dimesions: {L0, L1, L2}')
    L = max(L0, L1, L2)

    action_dfs = [(TT_0, 0), (TT_1, 1), (TT_2,2)]                   # Group all 3 action Transition matrices
  
    hours_list_lists = []
    for h in hours:
        actions_list_dfs = []                                       # Create merged transition matrices from ToTi->Ti' (by action) and To->To' (by hour) 

        to = ToTransitionsCSV(root_dir, weather_file, h)            # Generate To->To' transition matrix for the hour
        to.main()
        cols_orig = [x for x in to.T.columns]                       # List of columns names from To->To' matrix
        cols_new = np.repeat(cols_orig, L)                          # Generate full column list to be used for ToTi->To'Ti' matrix

        for col in to.T.columns:                                    # Copy columns and insert into To->To' matrix (depends on cols in ToTi->Ti')
            for i in reversed(range(1, L)):
                to.T.insert(to.T.columns.get_loc(col), f'{col}-{i}', to.T[col])
        to.T = to.T.iloc[np.arange(len(to.T)).repeat(L)]            # Copy rows and insert into To->To' matrix (depends on len of ToTi->Ti')

        for TT in action_dfs:
            TT_df = TT[0].copy(deep=True)
            new_col = [f'({x},{y})' for x, y in zip(cols_new, TT_df.columns)]   # Combined column names
            new_ind = [f'{y}' for x, y in zip(to.T.index, TT_df.index)]         # Combined indices

            to.T.columns, TT_df.columns = new_col, new_col                      # Rename columns for both matrices
            to.T['nindex'], TT_df['nindex'] = new_ind, new_ind                  # Add new column of indices for both matrices
            to.T = to.T.set_index('nindex')                                     # Set indices to new inds
            TT_df = TT_df.set_index('nindex')

            new_df = TT_df*to.T       
            ToTiOcc_df = combine_occ_data(int(h[0:2]), occ_probs, new_df)
            np_df = ToTiOcc_df.to_numpy()
            print(f'action: {actions[TT[1]]}, hour: {h}, df nans: {ToTiOcc_df.isnull().sum().sum()},  Any numpy nans? (any number means no,"NaN" means yes): {np_df.sum()}')
            


            actions_list_dfs.append(np_df)
        hours_list_lists.append(actions_list_dfs)

    final_df = np.asarray(hours_list_lists)
    print('final lengths', len(final_df), len(final_df[0]), len(final_df[0][0]), final_df.sum())

    np.save('/Users/maggie/Desktop/TransitionMatrix_1.npy', final_df)                 # Save full output to a 4-dimensional numpy array