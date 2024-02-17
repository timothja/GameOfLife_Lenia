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
HEIGHT = 64
WIDTH = 64

# Animation parameters
FPS = 24
TIME_DELAY_MS = 1000 // FPS
DURATION_S = 10
NB_OF_GENERATIONS = FPS * DURATION_S

# Growth function parameters
MU = 0.15
SIG = 0.015

# Filter parameters
FILTER_SIZE = 26
FILTER_RADIUS = FILTER_SIZE // 2
FILTER_MU = 0.5
FILTER_SIG = 0.15

def init_board_random(height=HEIGHT, width=WIDTH):
    """
    The board is initialized with random values between 0 and 1.
    """
    board = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            board[i][j] = random.random()

    return board

def init_board_gaussian_spot(height=HEIGHT, width=WIDTH):
    """
    The board is initialized with a gaussian spot in the middle.
    """
    board = np.ones((height, width))
    radius = 36
    y, x = np.ogrid[-height//2:height//2, -width//2:width//2]
    board = np.exp(-0.5 * (x*x + y*y) / (radius*radius))

    return board

def init_board_orbium(height=HEIGHT, width=WIDTH):
    orbium = np.array([[0,0,0,0,0,0,0.1,0.14,0.1,0,0,0.03,0.03,0,0,0.3,0,0,0,0], 
                       [0,0,0,0,0,0.08,0.24,0.3,0.3,0.18,0.14,0.15,0.16,0.15,0.09,0.2,0,0,0,0], 
                       [0,0,0,0,0,0.15,0.34,0.44,0.46,0.38,0.18,0.14,0.11,0.13,0.19,0.18,0.45,0,0,0], 
                       [0,0,0,0,0.06,0.13,0.39,0.5,0.5,0.37,0.06,0,0,0,0.02,0.16,0.68,0,0,0], 
                       [0,0,0,0.11,0.17,0.17,0.33,0.4,0.38,0.28,0.14,0,0,0,0,0,0.18,0.42,0,0], 
                       [0,0,0.09,0.18,0.13,0.06,0.08,0.26,0.32,0.32,0.27,0,0,0,0,0,0,0.82,0,0], 
                       [0.27,0,0.16,0.12,0,0,0,0.25,0.38,0.44,0.45,0.34,0,0,0,0,0,0.22,0.17,0], 
                       [0,0.07,0.2,0.02,0,0,0,0.31,0.48,0.57,0.6,0.57,0,0,0,0,0,0,0.49,0], 
                       [0,0.59,0.19,0,0,0,0,0.2,0.57,0.69,0.76,0.76,0.49,0,0,0,0,0,0.36,0], 
                       [0,0.58,0.19,0,0,0,0,0,0.67,0.83,0.9,0.92,0.87,0.12,0,0,0,0,0.22,0.07], 
                       [0,0,0.46,0,0,0,0,0,0.7,0.93,1,1,1,0.61,0,0,0,0,0.18,0.11], 
                       [0,0,0.82,0,0,0,0,0,0.47,1,1,0.98,1,0.96,0.27,0,0,0,0.19,0.1], 
                       [0,0,0.46,0,0,0,0,0,0.25,1,1,0.84,0.92,0.97,0.54,0.14,0.04,0.1,0.21,0.05], 
                       [0,0,0,0.4,0,0,0,0,0.09,0.8,1,0.82,0.8,0.85,0.63,0.31,0.18,0.19,0.2,0.01], 
                       [0,0,0,0.36,0.1,0,0,0,0.05,0.54,0.86,0.79,0.74,0.72,0.6,0.39,0.28,0.24,0.13,0], 
                       [0,0,0,0.01,0.3,0.07,0,0,0.08,0.36,0.64,0.7,0.64,0.6,0.51,0.39,0.29,0.19,0.04,0], 
                       [0,0,0,0,0.1,0.24,0.14,0.1,0.15,0.29,0.45,0.53,0.52,0.46,0.4,0.31,0.21,0.08,0,0], 
                       [0,0,0,0,0,0.08,0.21,0.21,0.22,0.29,0.36,0.39,0.37,0.33,0.26,0.18,0.09,0,0,0], 
                       [0,0,0,0,0,0,0.03,0.13,0.19,0.22,0.24,0.24,0.23,0.18,0.13,0.05,0,0,0,0], 
                       [0,0,0,0,0,0,0,0,0.02,0.06,0.08,0.09,0.07,0.05,0.01,0,0,0,0,0]])
    
    board = np.zeros((height, width))
    for i in range(20):
        for j in range(20):
            board[i + height // 6][j + width // 6] = orbium[i][j]

    return board



def update_board(board, _filter):
    """
    Update the board by applying the growth function to each cell.
    """
    dt = 0.1 # time step to smooth the animation
    neighbors_array = get_neighbors_array(board, _filter)
    growthrate_array = get_growthrate_array(board, neighbors_array)

    board = board + dt * growthrate_array
    return np.clip(board, 0, 1)

def init_ring_filter():
    """
    The filter is a ring with a gaussian profile that cuts off beyond a certain radius.
    """

    # Create distance matrix centered at (0, 0)
    y, x = np.ogrid[-FILTER_RADIUS:FILTER_RADIUS, -FILTER_RADIUS:FILTER_RADIUS]
    distance = np.sqrt((1+x)**2 + (1+y)**2) / FILTER_RADIUS

    # Calculate the filter values
    ring_filter = np.zeros((FILTER_SIZE, FILTER_SIZE))
    for i in range(FILTER_SIZE):
        for j in range(FILTER_SIZE):
            if distance[i, j] <= 1:
                ring_filter[i, j] = gauss(distance[i, j], FILTER_MU, FILTER_SIG)
            else:
                ring_filter[i, j] = 0

    # Normalize the values of the filter
    ring_filter = ring_filter / np.sum(ring_filter)

    return ring_filter

def get_neighbors_array(board, _filter):
    """
    Count the number of neighbors for each cell by applying a filter.
    """
    neighbors_array = signal.convolve2d(board, _filter, mode="same", boundary="wrap")
    return neighbors_array

def gauss(x, mu, sigma):
    """
    mu (float): The mean of the growth function.
    sig (float): The standard deviation of the growth function.
    """
    return np.exp(-0.5 * ((x-mu)/sigma)**2)

def growth_function(x, mu=MU, sig=SIG):
    """
    Calculate the growth value for a cell with x neighbors.
    """
    return -1 + 2 * gauss(x,mu,sig)

def get_growthrate_array(board, neighbors_array):
    """
    Calculate the growth rate for each cell based on the number of neighbors.
    """
    height = board.shape[0]
    width = board.shape[1]
    growthrate_array = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            growthrate_array[i][j] = growth_function(neighbors_array[i][j])

    return growthrate_array


def animate_board(board, _filter):
    """
    Animate the board using matplotlib's animation module.
    """
    fig = plt.figure()
    ims = []
    with tqdm(total=NB_OF_GENERATIONS) as pbar:
        for i in range(NB_OF_GENERATIONS):
            im = plt.imshow(board, cmap="Blues", animated=True)
            ims.append([im])
            board = update_board(board, _filter)
            pbar.update(1)
    ani = animation.ArtistAnimation(fig, ims, interval=TIME_DELAY_MS, blit=True, repeat = False)
    ani.save("orbium.mp4", writer="ffmpeg")
    plt.show()




if __name__ == "__main__":

    # board = init_board_random()
    # board = init_board_gaussian_spot()
    board = init_board_orbium()
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
