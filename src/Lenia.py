"""
Continuous version of the Game of Life.

Inspired by ScienceEtonnante's video: https://www.youtube.com/watch?v=PlzV4aJ7iMI&t=21s
                                      https://github.com/scienceetonnante/lenia/tree/main

Based on Lenia: https://www.lenia.org/
                https://github.com/Chakazul/Lenia
"""

import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy.signal as signal
from tqdm import tqdm

# Board parameters
HEIGHT = 128
WIDTH = 128

# Animation parameters
FPS = 15
TIME_DELAY_MS = 1000 // FPS
DURATION_S = 12
NB_OF_GENERATIONS = FPS * DURATION_S

# Growth function parameters
MU = 0.2
SIG = 0.03

# Filter parameters
FILTER_SIZE = 21
FILTER_RADIUS = FILTER_SIZE // 2
FILTER_MU = 0.5
FILTER_SIG = 0.15

def init_random_board(height=HEIGHT, width=WIDTH):
    board = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            board[i][j] = random.random()

    return board

def init_board(height=HEIGHT, width=WIDTH):
    """
    The board is initialized with a gaussian spot in the middle.
    """
    board = np.ones((height, width))
    radius = 36
    y, x = np.ogrid[-height//2:height//2, -width//2:width//2]
    board = np.exp(-0.5 * (x*x + y*y) / (radius*radius))

    return board


def update_board(board, _filter):
    dt = 0.1 # time step to smooth the animation
    neighbors_array = get_neighbors_array(board, _filter)
    growthrate_array = get_growthrate_array(board, neighbors_array)

    board = board + dt * growthrate_array
    return np.clip(board, 0, 1)

def init_ring_filter():
    """
    The filter is a ring with a gaussian profile that cuts off beyond a certain radius.
    """

    ring_filter = np.zeros((FILTER_SIZE, FILTER_SIZE))

    for i in range(FILTER_SIZE):
        for j in range(FILTER_SIZE):
            distance = np.sqrt((i - FILTER_RADIUS)**2 + (j - FILTER_RADIUS)**2) / FILTER_RADIUS # distance from center for each pixel
            if distance <= 1:
                ring_filter[i][j] = gauss(distance, FILTER_MU, FILTER_SIG)
            else:
                ring_filter[i][j] = 0

    # Normalize the values of the filter
    ring_filter = ring_filter / np.sum(ring_filter)

    return ring_filter

def get_neighbors_array(board, _filter):
    """
    Count the number of each cell using a ring filter.
    """
    neighbors_array = signal.convolve2d(board, _filter, mode="same", boundary="wrap")
    return neighbors_array

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
    return -1 + 2 * gauss(x,mu,sig)

def get_growthrate_array(board, neighbors_array):
    height = board.shape[0]
    width = board.shape[1]
    growthrate_array = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            growthrate_array[i][j] = growth_function(neighbors_array[i][j])

    return growthrate_array


def animate_board(board, _filter):
    fig = plt.figure()
    ims = []
    with tqdm(total=NB_OF_GENERATIONS) as pbar:
        for i in range(NB_OF_GENERATIONS):
            im = plt.imshow(board, cmap="Blues", animated=True)
            ims.append([im])
            board = update_board(board, _filter)
            pbar.update(1)
    ani = animation.ArtistAnimation(fig, ims, interval=TIME_DELAY_MS, blit=True, repeat = False)
    plt.show()




if __name__ == "__main__":

    board = init_random_board()
    # board = init_board()
    _filter = init_ring_filter()
    animate_board(board, _filter)

    # Plot the growth function and the filter
    plt.subplot(1, 2, 1)
    x = np.linspace(0, 0.3, 200)
    y = growth_function(x)
    plt.xlabel("nb of neighbors")
    plt.ylabel("growthrate")
    plt.title("growth function")
    plt.plot(x, y)
    plt.axhline(0, linestyle='--', color='red')

    plt.subplot(1, 2, 2)
    plt.title("filter")
    plt.imshow(_filter, cmap="Blues")

    plt.show()
