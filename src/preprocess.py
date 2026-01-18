import pandas as pd
import numpy as np
import os

def load_data(file_path):
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = [f's_{i}' for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names
    
    df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)
    return df

def add_remaining_useful_life(df):
    max_cycle = df.groupby('unit_nr')['time_cycles'].transform('max')
    
    df['RUL'] = max_cycle - df['time_cycles']
    return df

if __name__ == "__main__":
    train_file = "../data/train_FD001.txt"
    
    if os.path.exists(train_file):
        df = load_data(train_file)
        df = add_remaining_useful_life(df)
        
        print("✅ Data Loaded and RUL Calculated!")
        print(df[['unit_nr', 'time_cycles', 'RUL']].head())

        df.to_csv("../data/processed_train.csv", index=False)
    else:
        print("❌ Still can't find the data! Make sure the .txt files are inside the 'data' folder.")