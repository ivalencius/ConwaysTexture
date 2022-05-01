import pandas as pd
import sqlite3
import glob
import os
#read_folder = 'C:\\ConwaysTexture\\'
table_name = '1600x1600_stats.db'
conn = sqlite3.connect(table_name)
c = conn.cursor()
query = 'SELECT * FROM stats WHERE Skewness>0.2 AND Variance>0.005 AND Entropy>0.5 AND Mean>64 AND Mean<192'
c.execute(query)
conn.commit()

for row in c:
    print(row)