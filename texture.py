'''
Filename: c:\ConwaysTexture\texture.py
Path: c:\ConwaysTexture
Created Date: Monday, April 25th 2022, 12:22:17 pm
Author: Ilan Valencius

Copyright (c) 2022 Boston College
'''

from cmath import nan
import numpy as np
import matplotlib.pyplot as plt
import functools
import math
from skimage import feature
from sklearn.metrics import homogeneity_completeness_v_measure
from scipy import stats
import matplotlib.animation as animation

from gol import GameOfLife, downsample

# Inspiration https://engineering.purdue.edu/kak/Tutorials/TextureAndColor.pdf (section 3.3)
# https://www.mathworks.com/help/images/texture-analysis-using-the-gray-level-co-occurrence-matrix-glcm.html

# https://www.wikiwand.com/en/Image_noise
# The noise caused by quantizing the pixels of a sensed image to a number 
# of discrete levels is known as quantization noise. 
# It has an approximately uniform distribution. 
# Though it can be signal dependent, 
# it will be signal independent if other noise sources are big enough to cause dithering, 
# or if dithering is explicitly applied.

def normalRank(image):
    """Returns the p value for the null hypothesis that the values in image are normally distributed.

    Args:
        image (np.array): board array

    Returns:
        float: p value
    """     
    image = image.flatten()
    analysis = stats.normaltest(image)
    p_val = analysis[1]
    '''if p_val > .9:   
        return("High Normal Distr. similarity: ", analysis[1])
    elif p_val < .9 and p_val > .75: 
        return("Mid Normal Distr. similarity: ", analysis[1])
    elif p_val < .75 and p_val > .5: 
        return("Low Normal Distr. similarity: ", analysis[1])
    else: 
        return("No Normal Distr. similarity: ", analysis[1])'''
    return p_val


# From https://www.wikiwand.com/en/White_noise
# Any distribution of values is possible (although it must have zero DC component). 
# Even a binary signal which can only take on the values 1 or 0 will be white 
# if the sequence is statistically uncorrelated.
def histogram(image, n=None):
    """Displays a histogram for the values of a board.

    Args:
        image (np.array): Game of life board.
    """
    levels = np.unique(image)
    fig, ax = plt.subplots(1,2,figsize=(20,10))
    fig.suptitle('Quantized', fontsize=16)
    ax[0].imshow(image, interpolation='nearest')
    ax[0].set_title('Downsampled')
    ax[1].hist(image.flatten(), bins=len(levels))
    if n != None:
        ymin, ymax = ax[1].get_ylim()
        ax[1].axvline(n*256, ymin, ymax, color='red', label='Center for Initialization Prob')
    ax[1].set_title('Unique Values')
    plt.show()
    
def glcm_stats(board):
    """Determines statistics of image via skimage.

    Args:
        board (ndarray): Board.

    Returns:
        tuple: Tuple of glcm, __statistics__
    """
    levels = 256 # Indicate the number of gray-levels counted [0, levels-1]
    num_unique = len(np.unique(board))
    # fig, ax = plt.subplots(1,2,figsize=(20,10))
    # fig.suptitle('Type translation', fontsize=16)
    # ax[0].imshow(board, interpolation='nearest')
    # ax[0].set_title('float32')
    # Need to convert to unsigned integer
    board2 = (board*float(num_unique)).astype('uint8') 
    
    # ax[1].imshow(board2, interpolation='nearest')
    # ax[1].set_title('uint8')
    # plt.show()
    
    glcm = feature.graycomatrix(board2, [1], [0], levels=levels, symmetric=False, normed=True)
    entropy = feature.graycoprops(glcm, 'energy')[0][0] # ???
    contrast = feature.graycoprops(glcm, 'contrast')[0][0]
    homogeneity = feature.graycoprops(glcm, 'homogeneity')[0][0]
    '''print("\nTexture attributes: ") #(D29)
    print(" entropy: %f" % entropy) #(D30)
    print(" contrast: %f" % contrast) #(D31)
    print(" homogeneity: %f" % homogeneity) #(D32)'''
    return glcm.squeeze(), entropy, contrast, homogeneity

def statistics(image):
    """Returns the statistics of a nxn np.array.

    Args:
        image (np.array): board array

    Returns:
        tuple: assosiated stats
    """    
    _, _, mean, variance, skewness, kurtosis = stats.describe(image, axis=None)
    return mean, variance, skewness, kurtosis


def animate_step(frameNum, board,ax1, ax2,n):
    """Helper function used to animate a downsampled GOL board.

    Args:
        frameNum (int): Ignored.
        board (np.array): Game board.
        img (matplotlib.axes._subplots.AxesSubplot): Board image.

    Returns:
        matplotlib.axes._subplots.AxesSubplot: New board image.
    """
    image = downsample(board.get_board())
    ax1.imshow(image)
    ax1.set_title('Downsampled Board')
    if frameNum % 5 ==0 and frameNum>1:
        ax2.clear()
    ax2.hist(image.flatten(), bins=len(np.unique(image)))
    ymin, ymax = ax2.get_ylim()
    ax2.axvline(n*256, ymin, ymax, color='red', label='Center for Initialization Prob')
    ax2.set_title('Unique Values')
    board.step()
    

def animate_histogram(b,n):
    """Animates game of life progression.

    Args:
        b (Board): GOL board.
    """
    # create a figure with two subplots
    image = downsample(b.get_board())
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,8))
    ax1.imshow(image)
    ax2.hist(image.flatten(), bins=len(np.unique(image)))
    ymin, ymax = ax2.get_ylim()
    ax2.axvline(n*256, ymin, ymax, color='red', label='Center for Initialization Prob')
    
    ani = animation.FuncAnimation(fig, func=animate_step, 
                                  fargs=(b,ax1, ax2, n),
                                  frames = 33,
                                  interval=500,
                                  save_count=50)
    writergif = animation.PillowWriter(fps=3) 
    ani.save('figs\\0.33_hist.gif', writer=writergif)
    
    #plt.show()
if __name__ == "__main__":
    N = 1600
    b = GameOfLife(N)
    b.set_board_rand(0.33)
    animate_histogram(b, n=0.33)
    #histogram(downsample(b.get_board()))
    #print(statistics(downsample(b.get_board())))
    #print(normalRank(downsample(b.get_board())))
    #statistics(downsample(b.get_board()))
    #glcm, _, _, _ = glcm2(downsample(b.get_board()))
    #for _ in range(5):
    #   b.step()
    #histogram(downsample(b.get_board()), n=0.1)
    '''fig, ax = plt.subplots(1,2,figsize=(20,10))
    fig.suptitle('After 10 runs', fontsize=16)
    ax[0].imshow(downsample(b.get_board()), interpolation='nearest')
    ax[0].set_title('Raw Board '+str(b.get_board().shape))
    #glcm, _, _, _ = texture_fxn(downsample(b.get_board()))
    glcm, _, _, _ = glcm2(downsample(b.get_board()))
    ax[1].imshow(glcm, interpolation='nearest')
    ax[1].set_title('GLCM of Board '+str(glcm.shape))
    plt.show()'''