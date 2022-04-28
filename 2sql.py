import pandas as pd
import sqlite3
import glob
import os

read_folder = 'C:\\ConwaysTexture\\stats_tests\\'
table_name = '100x100_stats.db'
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