'''
Filename: c:\ConwaysTexture\query_plots.py
Path: c:\ConwaysTexture
Created Date: Monday, May 2nd 2022, 2:28:24 pm
Author: Ilan Valencius

Copyright (c) 2022 Boston College

For running a query and visualizing the results.
'''

import pandas as pd
import sqlite3
from texture import statistics, glcm_stats, normalRank
from gol import GameOfLife, downsample
import matplotlib.pyplot as plt
import numpy as np

table_name = '1600x1600_stats.db'
conn = sqlite3.connect(table_name)
c = conn.cursor()
query = 'SELECT * FROM stats WHERE Run > 0 and Pscore > .9'
c.execute(query)
conn.commit()
results_list = []
for row in c:
    results_list.append(row)
results = pd.DataFrame(results_list)
results.columns = ['Run', 'Mean','PScore','Variance','Skewness','Kurtosises','Entropy','Contrast','Homogeneity','Probability']
print(results)
run_number = results['Run']

# Manually grabbing first runs
# our formatting is weird but hopefully since it's in a 


final = []
names = ['Run', 'Mean','PScore','Variance','Skewness','Kurtosises','Entropy','Contrast','Homogeneity']

prob_29 = pd.read_csv("originals/0.29.csv")
df = pd.DataFrame(data=prob_29)
first = df.iloc[0]
first_list = [i for i in first]
# for i in range(len(first_list)-1): 
#    first_list[i] = [names[i],first_list[i]] 
final.append([0.29, first_list])

prob_30 = pd.read_csv("originals/0.3.csv")
df = pd.DataFrame(data=prob_30)
first = df.iloc[0]
first_list = [i for i in first]
# for i in range(len(first_list)-1): 
#    first_list[i] = [names[i],first_list[i]] 
final.append([0.3, first_list])



prob_33 = pd.read_csv("originals/0.33.csv")
df = pd.DataFrame(data=prob_33)
first = df.iloc[0]
first_list = [i for i in first]
# for i in range(len(first_list)-1): 
#    first_list[i] = [names[i],first_list[i]] 
final.append([0.33, first_list])


prob_34 = pd.read_csv("originals/0.34.csv")
df = pd.DataFrame(data=prob_34)
first = df.iloc[0]
first_list = [i for i in first]
# for i in range(len(first_list)-1): 
#    first_list[i] = [names[i],first_list[i]] 
final.append([0.34, first_list])


prob_38 = pd.read_csv("originals/0.38.csv")
df = pd.DataFrame(data=prob_38)
first = df.iloc[0]
first_list = [i for i in first]
# for i in range(len(first_list)-1): 
#    first_list[i] = [names[i],first_list[i]] 
final.append([0.38, first_list])


prob_40 = pd.read_csv("originals/0.4.csv")
df = pd.DataFrame(data=prob_40)
first = df.iloc[0]
first_list = [i for i in first]
# for i in range(len(first_list)-1): 
#    first_list[i] = [names[i],first_list[i]] 
final.append([0.40, first_list])

prob_42 = pd.read_csv("originals/0.42.csv")
df = pd.DataFrame(data=prob_42)
first = df.iloc[0]
first_list = [i for i in first]
# for i in range(len(first_list)-1): 
#    first_list[i] = [names[i],first_list[i]] 
final.append([0.42, first_list])


prob_43 = pd.read_csv("originals/0.43.csv")
df = pd.DataFrame(data=prob_43)
first = df.iloc[0]
first_list = [i for i in first]
# for i in range(len(first_list)-1): 
#    first_list[i] = [names[i],first_list[i]] 
final.append([0.43, first_list])

prob_45 = pd.read_csv("originals/0.45.csv")
df = pd.DataFrame(data=prob_45)
first = df.iloc[0]
first_list = [i for i in first]
# for i in range(len(first_list)-1): 
#    first_list[i] = [names[i],first_list[i]] 
final.append([0.45, first_list])

### Make Plots ###
n_plots = len(results_list)
# Subplots are organized in a Rows x Cols Grid
# Tot and Cols are known
Tot = n_plots # number_of_subplots
Cols = 3# number_of_columns

# Compute Rows required
Rows = Tot // Cols 
Rows += Tot % Cols
Position = range(1,Tot + 1)

plot_name = 'Statistics for Near Normal Runs'
#basename = os.path.basename(file).strip('.csv')
fig = plt.figure(figsize=(17,17))
fig.subplots_adjust(hspace=.5)

xlabels = ['Mean','PScore','Var','Skew','Kurt','Energy','Homogen']
font = {#'family': 'serif',
        #'color':  'darkred',
        #'weight': 'normal',
        'size': 8,
        }
for k in range(n_plots):
    run_stats = list(results_list[k])
    prob = run_stats.pop()
    stats = final[final[:][0]==prob][1].copy()
    '''b = GameOfLife(1600)
    b.set_board_rand(float(probability))
    down = downsample(b.get_board())
    stats2 = list(statistics(down))
    additional_stats = list(glcm_stats(down)[1:])
    run0_stats = [stats2[0]] + [normalRank(down)]+ stats2[1:]+ additional_stats'''
    ax = fig.add_subplot(Rows,Cols,Position[k])
    # Get rid of run metric
    run = str(int(run_stats.pop(0))) # run
    stats.pop(0) #run
    stats.pop(6) # contrast
    run_stats.pop(6) # contrast
    # Normalize mean and variance
    stats[0] = stats[0]/256
    stats[2] = stats[2]/256

    run_stats[0] = run_stats[0]/256
    run_stats[2] = run_stats[2]/256
    #print(run_stats)
    #print(stats)
    # Indexes for placement
    x = np.arange(len(xlabels))
    # Width of a bar 
    width = 0.3
    ax.bar(x - width/2,  stats, width, label='Run 0')
    ax.bar(x + width/2, run_stats, width, label='Run '+run)
    #ax.set_xlabel('Metric')
    ax.set_ylabel('Metric Value', fontdict=font)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontdict = font)
    ax.legend()
    handles, _ = ax.get_legend_handles_labels()
    ax.set_title(prob)
labels = ['Initial Run', 'GOL']
fig.legend(handles, labels, loc='right')
fig.suptitle(plot_name)
plt.show()
    


### Format: [   [Probability. [Run, Mean, PScore, ..., Homogeneity]],   [Probability, [Run, mean, PScore, ..., Homogeneity]] ...   ]

### Low homogeneity we want: EVerything different
### Skewness and kutosis near 0
### Variance increasing in all cases
