'''
Filename: c:\ConwaysTexture\2sql.py
Path: c:\ConwaysTexture
Created Date: Thursday, April 28th 2022, 4:27:01 pm
Author: Ilan Valencius

Copyright (c) 2022 Boston College

Saves the results of a series of runs into one sql database.
'''

import pandas as pd
import sqlite3
import glob
import os

read_folder = 'C:\\ConwaysTexture\\stats_tests\\'
table_name = '1600x1600_stats.db'
csv_list = glob.glob(os.path.join(read_folder, '*.csv'))
conn = sqlite3.connect(table_name)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS stats (Run, Mean,PScore,Variance,Skewness,Kurtosises,Entropy,Contrast,Homogeneity,Probability)')
conn.commit()

for csv in csv_list:
    #csv = csv_list[0]
    basename = os.path.basename(csv).strip('.csv')
    df = pd.read_csv(csv)
    df['Probability'] = basename
    #print(df)
    df.to_sql('stats', conn, if_exists='append', index = False)