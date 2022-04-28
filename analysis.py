'''
Filename: c:\ConwaysTexture\analysis.py
Path: c:\ConwaysTexture
Created Date: Monday, April 25th 2022, 4:45:25 pm
Author: Ilan Valencius

Copyright (c) 2022 Boston College
'''
import pandas as pd

from gol import GameOfLife, downsample
from texture import texture_fxn, normalRank, histogram, statistics, glcm_stats
import os
from tqdm import tqdm

# Need to determine number of trials, what is E[X] of game of life, use Markov or something
TRIALS = 10 # default number of times to run each set of parameters
N = 1000 # default size of NxN board
PROB = 0.5 # default board initialization prob

def texture_trials(b,trials, down=True):
    """Returns a dict of image stats for board b after trial runs.

    Args:
        b (GameOfLife): Game board.
        trials (n): Number of runs.
        down (bool, optional): Whether to downsample the board. Defaults to True.

    Returns:
        dict: Dictionary of stats per run.
    """
    entropies = []
    contrasts = []
    homogeneities = []
    runs = [i for i in range(trials+1)]
    for _ in range(trials+1):
        if down:
            board = downsample(b.get_board())
        else:
            board = b.get_board()
        _, entropy, contrast, homogeneity = texture_fxn(board)
        entropies.append(entropy)
        contrasts.append(contrast)
        homogeneities.append(homogeneity)
        b.step() # First time through loop is base board
    return_dict = {
        'Run':runs,
        'Entropy':entropies,
        'Contrast':contrasts,
        'Homogeneity':homogeneities
    }
    return return_dict
    
def p_dependence(write_folder):
    """Determines the stats of initializing a board with given probability.

    Args:
        write_folder (string): Directory to write csv files too.
    """
    probabilities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print('Testing probabilities')
    for prob in tqdm(probabilities):
        b = GameOfLife(N)
        b.set_board_rand(prob)
        texture_dict = texture_trials(b, TRIALS)
        df = pd.DataFrame(texture_dict)
        df.to_csv(os.path.join(write_folder, str(prob)+'.csv'), index=False)       
    
def size_dependence(write_folder):
    """Determines the stats of initializing a board with given size.

    Args:
        write_folder (string): Directory to write csv files too.
    """
    size= [5, 25, 50, 100, 250, 500, 750, 1000]
    print('Testing probabilities')
    for sz in tqdm(size):
        b = GameOfLife(sz)
        b.set_board_rand(PROB)
        texture_dict = texture_trials(b, TRIALS)
        df = pd.DataFrame(texture_dict)
        df.to_csv(os.path.join(write_folder, str(sz)+'.csv'), index=False)   

def trial_dependence(write_folder):
    """Determines the stats of running a given board trial times.

    Args:
        write_folder (string): Directory to write csv files too.
    """
    trials = [5, 25, 50, 100]
    print('Testing probabilities')
    for trial in tqdm(trials):
        b = GameOfLife(N)
        b.set_board_rand(PROB)
        texture_dict = texture_trials(b, trial)
        df = pd.DataFrame(texture_dict)
        df.to_csv(os.path.join(write_folder, str(trial)+'.csv'), index=False)

def stats_trials(b,trials, down=True):
    """Returns a dict of image stats for board b after trial runs.

    Args:
        b (GameOfLife): Game board.
        trials (n): Number of runs.
        down (bool, optional): Whether to downsample the board. Defaults to True.

    Returns:
        dict: Dictionary of stats per run.
    """
    pscores = []
    means = []
    variances = []
    skewnesses = []
    kurtosises = []
    runs = [i for i in range(trials+1)]
    for _ in range(trials+1):
        if down:
            board = downsample(b.get_board())
        else:
            board = b.get_board()
        p_val = normalRank(board)
        pscores.append(p_val)
        mean, variance, skewness, kurtosis = statistics(board)
        means.append(mean)
        variances.append(variance)
        skewnesses.append(skewness)
        kurtosises.append(kurtosis)
        b.step() # First time through loop is base board
    return_dict = {
        'Run':runs,
        'P-Score':pscores,
        'Mean Value': means,
        'Variance':variances,
        'Skewness':skewnesses,
        'Kurtosises':kurtosises
    }
    return return_dict

def stats_test(write_folder):
    #probabilities = [i/100 for i in range(1,100)]
    probabilities = [0.8]
    print('Starting Testing')
    for prob in tqdm(probabilities):
        b = GameOfLife(N)
        b.set_board_rand(prob)
        normal_dict = stats_trials(b, TRIALS)
        df = pd.DataFrame(normal_dict)
        df.to_csv(os.path.join(write_folder, str(prob)+'.csv'), index=False)   
        
if __name__ == "__main__":
    # p_dependence('C:\\ConwaysTexture\\p_dependency\\')
    # size_dependence('C:\\ConwaysTexture\\size_dependency\\')
    # trial_dependence('C:\\ConwaysTexture\\trial_dependency\\')
    stats_test('C:\\ConwaysTexture\\stats_tests\\')
    