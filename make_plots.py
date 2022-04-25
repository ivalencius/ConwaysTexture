'''
Filename: c:\ConwaysTexture\make_plots.py
Path: c:\ConwaysTexture
Created Date: Monday, April 25th 2022, 5:21:21 pm
Author: Ilan Valencius

Copyright (c) 2022 Boston College
'''
import pandas as pd
import os
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt

read_folder = 'C:\\ConwaysTexture\\p_dependency\\'
plot_name = 'Probability Stats vs. Run #'

sort_lam = lambda x : float(os.path.basename(x).strip('.csv'))
file_list = sorted(glob.glob(os.path.join(read_folder, '*.csv')), key=sort_lam)

# Subplots are organized in a Rows x Cols Grid
# Tot and Cols are known
Tot = 8 # number_of_subplots
Cols = 3# number_of_columns

# Compute Rows required
Rows = Tot // Cols 
Rows += Tot % Cols

# Create a Position index
Position = range(1,Tot + 1)

fig = plt.figure(1)
fig.subplots_adjust(hspace=.5)
for file, k in zip(file_list, range(Tot)):
    basename = os.path.basename(file).strip('.csv')
    df = pd.read_csv(file)
    x_coords = df['Run']
    df_cols = list(df)
    df_cols.pop(0) # get rid of Run #
    ax = fig.add_subplot(Rows,Cols,Position[k])
    for col in df_cols:
        #plt.plot(df[col], label=col)
        ax.plot(df[col], label=col)
        handles, labels = ax.get_legend_handles_labels()
        ax.set_title(basename)
fig.legend(handles, labels, loc='right')
fig.suptitle(plot_name)
plt.show()
    