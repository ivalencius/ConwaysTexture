from unittest import result
import pandas as pd
import sqlite3

#read_folder = 'C:\\ConwaysTexture\\'
table_name = '1600x1600_stats.db'
conn = sqlite3.connect(table_name)
c = conn.cursor()
query = 'SELECT * FROM stats WHERE PScore>0.05 AND RUN>0 AND Skewness<0.05 AND Entropy>0.05'
c.execute(query)
conn.commit()
results_list = []
for row in c:
    #print(row)
    results_list.append(row)
results = pd.DataFrame(results_list)
results.columns = ['Run', 'Mean','PScore','Variance','Skewness','Kurtosises','Entropy','Contrast','Homogeneity','Probability']
print(results)
