'''
Filename: c:\ConwaysTexture\make_plots.py
Path: c:\ConwaysTexture
Created Date: Monday, April 25th 2022, 5:21:21 pm
Author: Ilan Valencius

Copyright (c) 2022 Boston College

Creates plots for visualization
'''
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

one_plot = False
set_plots = True
noise_plot = False
multi_plots = False

if one_plot:
    # For a single plot
    file = r'C:\ConwaysTexture\stats_tests\0.8.csv'
    plot_name = 'Stats for 100 trials'
    #basename = os.path.basename(file).strip('.csv')
    df = pd.read_csv(file)
    x_coords = df['Run']
    df_cols = list(df)
    df_cols.pop(0) # get rid of Run #
    df_cols.pop(6) # get rid of contrast
    for col in df_cols:
        #plt.plot(df[col], label=col)
        plt.plot(df[col], label=col)
    plt.xlabel('Run #')
    plt.ylabel('Metric Value')
    plt.title(plot_name)
    plt.legend(loc='right')
    plt.show()

if set_plots:
    # For a single plot
    files = [r'C:\ConwaysTexture\stats_tests\0.1.csv',
             r'C:\ConwaysTexture\stats_tests\0.2.csv',
             r'C:\ConwaysTexture\stats_tests\0.3.csv',
             r'C:\ConwaysTexture\stats_tests\0.4.csv',
             r'C:\ConwaysTexture\stats_tests\0.5.csv',
             r'C:\ConwaysTexture\stats_tests\0.6.csv',
             r'C:\ConwaysTexture\stats_tests\0.7.csv',
             r'C:\ConwaysTexture\stats_tests\0.8.csv',
             r'C:\ConwaysTexture\stats_tests\0.9.csv']
    n_plots = len(files)
        # Subplots are organized in a Rows x Cols Grid
    # Tot and Cols are known
    Tot = n_plots # number_of_subplots
    Cols = 3# number_of_columns

    # Compute Rows required
    Rows = Tot // Cols 
    Rows += Tot % Cols
    Position = range(1,Tot + 1)
    
    plot_name = 'Stats for 100 trials'
    #basename = os.path.basename(file).strip('.csv')
    fig = plt.figure(1)
    fig.subplots_adjust(hspace=0.8)
    for file, k in zip(files, range(n_plots)):
        basename = os.path.basename(file).strip('.csv')
        df = pd.read_csv(file)
        x_coords = df['Run']
        df.rename(columns={'Entropy':'Energy'}, inplace=True)
        df_cols = list(df)
        df_cols.pop(0) # get rid of Run #
        df_cols.pop(6) # get rid of contrast
        df['Mean'] = df['Mean']/256.0
        df['Variance'] = df['Variance']/256.0
        ax = fig.add_subplot(Rows,Cols,Position[k])
        for col in df_cols:
            #plt.plot(df[col], label=col)
            ax.plot(df[col], label=col)
            handles, labels = ax.get_legend_handles_labels()
            ax.set_xlabel('Run #')
            ax.set_ylabel('Metric Value')
            ax.set_title(basename)
    fig.legend(handles, labels, loc='right')
    fig.suptitle(plot_name)
    plt.show()
    '''df = pd.read_csv(file)
    x_coords = df['Run']
    df_cols = list(df)
    df_cols.pop(0) # get rid of Run #
    df_cols.pop(6) # get rid of contrast
    for col in df_cols:
        #plt.plot(df[col], label=col)
        plt.plot(df[col], label=col)
    plt.xlabel('Run #')
    plt.ylabel('Metric Value')
    plt.title(plot_name)
    plt.legend(loc='right')
    plt.show()'''

if noise_plot:
    n = 100
    gauss = np.random.randn(n,n)
    max_val = np.max(gauss)
    min_val = np.min(gauss)
    uniform = np.random.uniform(min_val, max_val,n*n).reshape((n,n))
    fig, ax = plt.subplots(1,2,figsize=(20,10))
    fig.suptitle('Noise Types', fontsize=16)
    ax[0].imshow(gauss)
    ax[0].set_title('Gaussian')
    ax[1].imshow(uniform)
    ax[1].set_title('White')
    plt.show()
    
if multi_plots:
    # For Multiple Plots
    read_folder = 'C:\\ConwaysTexture\\trial_dependency\\'
    plot_name = 'Trial Stats vs. Run #'

    sort_lam = lambda x : float(os.path.basename(x).strip('.csv'))
    file_list = sorted(glob.glob(os.path.join(read_folder, '*.csv')), key=sort_lam)

    # Subplots are organized in a Rows x Cols Grid
    # Tot and Cols are known
    Tot = 4 # number_of_subplots
    Cols = 2# number_of_columns

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
        