"""
Continuous version of the Game of Life.
Inspired by ScienceEtonnante's video: https://www.youtube.com/watch?v=PlzV4aJ7iMI&t=21s
Based on Lenia: https://www.lenia.org/
"""

import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from tqdm import tqdm

HEIGHT = 100
WIDTH = 100
TIME_DELAY_MS = 100
NB_OF_GENERATIONS = 200

# Growth function parameters
MU = 0.15
SIG = 0.015
OFFSET = 1


def initialize_board(height=HEIGHT, width=WIDTH):
    board = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            board[i][j] = random.random()

    return board


def update_board(board):
    height = board.shape[0]
    width = board.shape[1]
    new_board = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            new_value = update_cell_value(board, i, j)
            if new_value < 0:
                new_value = 0
            elif new_value > 1:
                new_value = 1
            new_board[i][j] = new_value

    return new_board


def count_direct_neighbors(board, row, col):

    count  = board[(row+1)%WIDTH][col]
    count += board[(row-1)%WIDTH][col]
    count += board[(row+1)%WIDTH][(col+1)%HEIGHT]
    count += board[(row-1)%WIDTH][(col+1)%HEIGHT]
    count += board[(row+1)%WIDTH][(col-1)%HEIGHT]
    count += board[(row-1)%WIDTH][(col-1)%HEIGHT]
    count += board[(row)][(col+1)%HEIGHT]
    count += board[(row)][(col-1)%HEIGHT]

    return count


def gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x-mu)/sigma)**2)


def growth_function(x, mu=MU, sig=SIG):
    """
    Calculate the value of the growth function at a given point.

    Parameters:
    x (float): The input value (aka nb of neighbors).
    mu (float): The mean of the growth function.
    sig (float): The standard deviation of the growth function.

    Returns:
    float: The value of the growth function (aka growth rate) at the given point.
    """
    return -OFFSET + 2 * gauss(x,mu,sig)


def plot_growth_function():
    x = np.linspace(0, 0.3, 200)
    y = growth_function(x)
    plt.plot(x, y)
    plt.axhline(0, linestyle='--', color='red')
    plt.show()


def get_cell_growth_rate(board, row, col):
    return growth_function(count_direct_neighbors(board, row, col), 2, 0.5)


def update_cell_value(board, row, col):
    return board[row][col] + get_cell_growth_rate(board, row, col)


def animate_board(board):
    fig = plt.figure()
    ims = []
    with tqdm(total=NB_OF_GENERATIONS) as pbar:
        for i in range(NB_OF_GENERATIONS):
            im = plt.imshow(board, cmap="Blues", animated=True)
            ims.append([im])
            board = update_board(board)
            pbar.update(1)
    ani = animation.ArtistAnimation(fig, ims, interval=TIME_DELAY_MS, blit=True, repeat = False)
    plt.show()


if __name__ == "__main__":

    board = initialize_board()

    animate_board(board)
