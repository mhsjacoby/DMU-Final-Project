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

# Reads in basics about the home, contains basic functions
class HomeData():
    def __init__(self, path, write):
        self.root_dir = path
        self.write_dir = write
        self.home = path.split('/')[-1].split('-')[-2]
        self.system = path.split('/')[-1].split('-')[-1]
    
    
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


# reads in occupancy files and creates dfs for all occupants
class HomeOccupancy(HomeData):
    def __init__(self, path, write):     
        HomeData.__init__(self, path, write) 
        self.ground_path = os.path.join(self.root_dir, 'GroundTruth')
        self.occupant_names = []
        self.average_length = 60
        self.start_date = '2019-02-04'
        self.end_date = '2019-04-08'

    def get_ground_truth(self):
        occupant_files = self.mylistdir(self.ground_path, '.csv')
        occupants = {}
        enter_times, exit_times = [], []
        
        for occ in occupant_files:
            occupant_name = occ.strip('.csv').split('-')[1]
            self.occupant_names.append(occupant_name)
            ishome = []
            with open(os.path.join(self.ground_path, occ)) as csv_file:
                csv_reader, line_count = csv.reader(csv_file, delimiter=','), 0
                for row in csv_reader:
                    status, when = row[1], row[2].split('at')
                    dt_day = datetime.strptime(str(when[0] + when[1]), '%B %d, %Y  %I:%M%p')
                    ishome.append((status, dt_day))
                    if line_count == 0:
                        enter_times.append(dt_day)
                    line_count += 1
                exit_times.append(dt_day)
                
            occupants[occupant_name] = ishome        
        self.first_last = (sorted(enter_times)[0], sorted(exit_times)[-1])
        print(self.occupant_names)
        return occupants
    
    def create_occupancy_df(self, occupants, frequency):
        occ_range = pd.date_range(start=self.first_last[0], end=self.first_last[1], freq=frequency)    
        occ_df = pd.DataFrame(index=occ_range)
        
        for occ in occupants:
            occ_df[occ] = 99
            s1 = 'exited'
            for r in occupants[occ]:
                date = r[1]
                s2 = r[0]                
                occ_df.loc[(occ_df.index < date) & (occ_df[occ]==99) & (s1 == 'exited') & (s2 == 'entered'), occ] =  0
                occ_df.loc[(occ_df.index < date) & (occ_df[occ]==99) & (s1 == 'entered') & (s2 == 'exited'), occ] =  1
                s1 = s2               
            occ_df.loc[(occ_df.index >= date) & (occ_df[occ] == 99) & (s1 == 'entered'), occ] = 1
            occ_df.loc[(occ_df.index >= date) & (occ_df[occ] == 99) & (s1 == 'exited'), occ] = 0
        
        occ_df['day'] = occ_df.index.weekday
        occ_df['weekend'] = 1
        occ_df['day_name'] = occ_df.index.day_name()
        occ_df.loc[occ_df.day < 5, 'weekend'] = 0 
        return (occ_df)
    
    def average_df(self, df):
        time_series = []
        for group, df_chunk in df.groupby(np.arange(len(df))//self.average_length):
            df_max = df_chunk.max()
            df_index = df_chunk.iloc[-1]
            time_series.append(df_index.name)
            df_summary = df_max.to_frame().transpose() 
            new_df = df_summary if group == 0 else pd.concat([new_df, df_summary])
        new_df.index = time_series  
        return new_df
          
    def write_occupancy_csv(self, df, fname):#, write_dir):   
        target_dir = self.make_storage_directory(os.path.join(self.write_dir, 'Occupancy_CSVs'))
        fname = os.path.join(target_dir, fname)
        df.to_csv(fname, index = True)
        print(fname + ': Write Successful!')
            
    def main(self):
        self.occupant_status = self.get_ground_truth()  
        df_hr = self.create_occupancy_df(self.occupant_status, frequency='1H')
        self.df_hr = df_hr.loc[(df_hr.index >= self.start_date) & (df_hr.index < self.end_date)] 
        self.write_occupancy_csv(self.df_hr, f'{self.home}-{self.system}-Occupancy_df.csv')#, self.write_dir)

# takes dfs for all occupants are create probabilites for transitions into/out of occupancy states for one person
class GetProbs(HomeData):
    def __init__(self, df, name, write):
        self.name = name
        self.all_H = [x for x in range(0,24)]
        self.df = df
        self.write_dir = write

    def get_occ(self, df):    
        hours_occ = {}
        hours_unocc = {}
        for h in self.all_H:
            occ_to_un = len(df.loc[(df.index.hour == h) & (df.diff(periods=-1).index.hour==h) & (df.diff(periods=-1)==1)])            # ~A|B
            same_occ  = len(df.loc[(df.index.hour == h) & (df.diff(periods=-1).index.hour==h) & (df.diff(periods=-1)==0)& (df==1)])   #  A|B
            
            un_to_occ = len(df.loc[(df.index.hour == h) & (df.diff(periods=-1).index.hour==h) & (df.diff(periods=-1)==-1)])           #  A|~B
            same_un   = len(df.loc[(df.index.hour == h) & (df.diff(periods=-1).index.hour==h) & (df.diff(periods=-1)==0)& (df==0)])   # ~A|~B
            
            t_occ = occ_to_un+same_occ
            t_un = un_to_occ+same_un
            
            p_leave = occ_to_un/t_occ if t_occ > 0 else 0.0
            p_arrive = un_to_occ/t_un if t_un > 0 else 0.0
            hours_occ[h] = (p_leave, 1-p_leave)      # if home: (probability leave, probability stay home)
            hours_unocc[h] = (p_arrive, 1-p_arrive)  # if gone: (probability arrive, probability stay out)

        return hours_occ, hours_unocc

    def write_occ(self, occ_probs, fname, cols):
        store_dir = self.make_storage_directory(os.path.join(self.write_dir, 'TransitionProbabilties'))
        csv_file = os.path.join(store_dir, fname)
        with open(csv_file, 'w') as f:
            f.write(f'{cols[0]}, {cols[1]}, {cols[2]}\n')
            for hour in occ_probs:
                f.write(f'{hour}:00, {occ_probs[hour][0]:.4f}, {occ_probs[hour][1]:.4f}\n')        
        print(csv_file + ': Write Successful!')


    def main(self):
        Home, notHome = self.get_occ(self.df[self.name].loc[self.df.weekend==0])
        self.write_occ(Home, f'{name}-given-occupied.csv', ['hour', 'prob leave', 'prob no leave'])
        self.write_occ(notHome, f'{name}-given-unoccupied.csv', ['hour', 'prob arrive', 'prob no arrive'])


# creates plot of occupancy for one person
class PlotOcc(HomeData):
    def __init__(self, df, name, write, D=21):
        self.df = df
        self.days = D
        self.name = name
        self.write_dir = write
    
    def write_occ_asPD(self, df):
        df['Date'] = pd.to_datetime(df.index)         
        df = df.set_index('Date')
        dfs = []
        day1 = df.index[0]
        day_start = df.day[0]
        num_weeks = int(np.ceil(len(df)/(24*self.days)))
        print('{} time periods of {} days'.format(num_weeks, self.days))
        
        if day_start > 0:
            print(int(24*(7-day_start)))
            day1 = day1 + timedelta(days = int(self.days-day_start))
            df1 = df.loc[(df.index <= day1)]
            dfs.append(df1)
        
        for i in range(num_weeks):
            dayf = day1 + timedelta(days = self.days*i)
            dayn = day1 + timedelta(days = self.days*(i+1))
            print(dayf, dayn)
            dfn = df.loc[(df.index >= dayf) & (df.index < dayn)]
            dfs.append(dfn)
        return dfs

    def highlight_weekend(self, D, df, ax):
        for i in range(int(self.days/7)):
            start = df[(df.weekend > 0)].index[48*i]
            end = start + timedelta(days=2)
            ax.axvspan(start, end, facecolor='pink', edgecolor='none', alpha=0.6)
        return ax

    def plot_occ_all(self, dfs, n=5, scale=1.2, height=4):
        for x, df in enumerate(dfs[0:n]):
            L = np.floor(len(df)/24)*scale
            ax = df.plot(y=name, title = self.name, figsize = (L,height), legend=False)
            ax = self.highlight_weekend(self.days, df, ax)
            save_dir = self.make_storage_directory(os.path.join(self.write_dir, 'Occ_Figs'))
            plt.savefig(os.path.join(save_dir, f'{self.name}_{x}.png'))       

    def main(self):
        dfs = self.write_occ_asPD(self.df)
        self.plot_occ_all(dfs)



        

if __name__ == '__main__':

    write_path =  '/Users/maggie/Documents/Github/DMU-Final-Project'

    if len(sys.argv) > 1:
        root_path = sys.argv[1]
        name = sys.argv[2]
    else:
        root_path = '/Users/maggie/Desktop/HPD_mobile_data/HPD-env-summaries/HPD_mobile-H1/H1-black'
        name = 'Will'

    print(f'Getting occupancy data from {root_path} ...')
    H1_occ = HomeOccupancy(root_path, write_path)
    H1_occ.main()
    print(f'Full occupancy df created!')
    full_df = H1_occ.df_hr

    print(f'Getting probabilties for {name} ...')
    H1_p1_probs = GetProbs(full_df, name, write_path)
    H1_p1_probs.main()

    print(f'Saving figures for {name} ...')
    p = PlotOcc(full_df, name, write_path)
    p.main()
    

