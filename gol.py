'''
Filename: c:\ConwaysTexture\gol.py
Path: c:\ConwaysTexture
Created Date: Thursday, April 21st 2022, 8:55:06 pm
Author: Ilan Valencius

Copyright (c) 2022 Boston College
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d
from numpy.fft import fft2, ifft2

def fft_convolve2d(x,y):
    """
    2D convolution, using FFT
    From: https://github.com/thearn/game-of-life
    """
    fr = fft2(x)
    fr2 = fft2(np.flipud(np.fliplr(y)))
    m,n = fr.shape
    cc = np.real(ifft2(fr*fr2))
    cc = np.roll(cc, - int(m / 2) + 1, axis=0)
    cc = np.roll(cc, - int(n / 2) + 1, axis=1)
    return cc

class GameOfLife:
    def __init__(self, _n):
        """Sets up a new game with blank board.

        Args:
            _n (int): Size of NxN board.
        """
        self.N = _n
        self.board = np.zeros((self.N, self.N), dtype=int)
        
    def set_board(self, arr):
        """Give the game a set board.

        Args:
            arr (np.array): A NxN board.
        """
        assert(self.shape == arr.shape())
        self.board = arr
    
    def set_board_rand(self, p, seed=2223):
        """Generate a board with live cells being set with probability p.

        Args:
            p (float): probability.
            seed (int, optional): Seed for random generation. Defaults to 2223.
        """
        # Issue --> this relies on randomness to begin with
        # Figure out method to initialize using non-random process
        np.random.seed(seed)
        new_board = np.random.rand(self.N, self.N)
        self.board[new_board <= p] = 1
        self.board[new_board > p] = 0
            
    def get_neighbors(self, r, c):
        """Returns the number of neighbors of a cell at [r,c]. Assumes a torodial boundary conditon

        Args:
            r (int): Row.
            c (int): Column.

        Returns:
            int: Number of neighbors.
        """
        neighbors = []
        # torodial boundary conditions
        for i in [r-1, r, r+1]:
            for j in [c-1, c, c+1]:
                if i==r and j==c:
                    continue
                try:
                    neighbors.append(self.board[i%self.N, j%self.N])
                except:
                    pass
        return sum(neighbors) # returns sum of the live neighbors
    
    def step(self):
        """Updates the board according to the rules of Conway's game of life.
        """
        # Convolution method
        m, n = self.board.shape
        k = np.zeros((m, n))
        k[int(m/2-1):int(m/2+2), int(n/2-1):int(n/2+2)] = np.array([[1,1,1],[1,0,1],[1,1,1]])
        #k = np.array([[1,1,1],[1,0,1],[1,1,1]])
        b = fft_convolve2d(self.board,k).round()
        c = np.zeros(b.shape)

        c[np.where((b == 2) & (self.board == 1))] = 1
        c[np.where((b == 3) & (self.board == 1))] = 1

        c[np.where((b == 3) & (self.board == 0))] = 1
        
        self.board = c
        # Classical Method
        '''for row in range(self.N):
            for col in range(self.N):
                cell = self.board[row, col]
                next_cell = 1
                neighbors = self.get_neighbors(row, col)
                if cell == 1:
                    if neighbors < 2:
                        next_cell = 0
                    elif neighbors > 3:
                        next_cell = 0
                if cell == 0:
                    if neighbors != 3: next_cell = 0
                self.board[row, col] = next_cell'''
        
    def get_board(self):
        """Returns the game board.

        Returns:
            np.array: Board.
        """
        return self.board

def animate_step(frameNum, board, img):
    """Helper function used to animate GOL board.

    Args:
        frameNum (int): Ignored.
        board (np.array): Game board.
        img (matplotlib.axes._subplots.AxesSubplot): Board image.

    Returns:
        matplotlib.axes._subplots.AxesSubplot: New board image.
    """
    board.step()
    img.set_data(board.get_board())
    return img
from skimage.util.shape import view_as_windows

def strided4D(arr,arr2,s):
    return view_as_windows(arr, arr2.shape, step=s)

def stride_conv_strided(arr,arr2,s):
    arr4D = strided4D(arr,arr2,s=s)
    return np.tensordot(arr4D, arr2, axes=((2,3),(0,1)))

def downsample(board, n=16):
    """Downsamples a GOL board by a (n,n) kernel.

    Args:
        board (np.array): Board.
        n (int, optional): Kernel size. Defaults to 11.

    Returns:
        np.array: Downsampled board.
    """
    kernel = np.ones((n, n))
    #convolved = convolve2d(board, kernel, mode='same', boundary='wrap')
    #return convolved/(n**2)
    return stride_conv_strided(board, kernel, n)

def animate_step_downsample(frameNum, board, img):
    """Helper function used to animate a downsampled GOL board.

    Args:
        frameNum (int): Ignored.
        board (np.array): Game board.
        img (matplotlib.axes._subplots.AxesSubplot): Board image.

    Returns:
        matplotlib.axes._subplots.AxesSubplot: New board image.
    """
    board.step()
    img.set_data(downsample(board.get_board()))
    return img

def test_board(b):
    """Animates game of life progression.

    Args:
        b (Board): GOL board.
    """
    fig, ax = plt.subplots(figsize=(10,10))
    img = ax.imshow(b.get_board(), interpolation='nearest')

    ani = animation.FuncAnimation(fig, func=animate_step, 
                                  fargs=(b, img),
                                  frames = 10,
                                  interval=500,
                                  save_count=50)
    plt.show()
    
def test_board_downsample(b):
    """Plots effect of downsampling GOL board.

    Args:
        b (Board): GOL board.
    """
    for i in range(10):
        b.step()
    fig, ax = plt.subplots(1,2,figsize=(20,10))
    fig.suptitle('After 10 runs', fontsize=16)
    
    ax[0].imshow(b.get_board(), interpolation='nearest')
    ax[0].set_title('Raw Board '+str(b.get_board().shape))
    down = downsample(b.get_board())
    ax[1].imshow(down, interpolation='nearest')
    ax[1].set_title('Downsampled Board '+str(down.shape))
    plt.show()
    
if __name__ == "__main__":
    N = 1000
    b = GameOfLife(N)
    b.set_board_rand(0.3)
    
    ### Functions for Testing ###
    test_board_downsample(b)
    #test_board(b)
    