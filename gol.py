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
        for row in range(self.N):
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
                self.board[row, col] = next_cell
        
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

def downsample(board, n=11):
    """Downsamples a GOL board by a (n,n) kernel.

    Args:
        board (np.array): Board.
        n (int, optional): Kernel size. Defaults to 11.

    Returns:
        np.array: Downsampled board.
    """
    kernel = np.ones((n, n))
    convolved = convolve2d(board, kernel, mode='valid')
    board = convolved[::n, ::n] / n
    return board

def animate_downsample(frameNum, board, img):
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
    N = 100
    b = GameOfLife(N)
    b.set_board_rand(0.5)
    
    # Functions for Testing
    # test_board_downsample(b)
    # test_board(b)
    