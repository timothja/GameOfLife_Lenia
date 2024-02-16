"""
Conway's Game of Life using matplotlib to display the board.
"""

import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

HEIGHT = 100
WIDTH = 100
TIME_DELAY_MS = 100
NB_OF_GENERATIONS = 200

def initialize_board(height=HEIGHT, width=WIDTH):
    board = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            board[i][j] = random.choice([0, 1])

    return board


def plot_board(board):
    plt.imshow(board, cmap="gray")
    plt.show()

def update_board(board):
    height = board.shape[0]
    width = board.shape[1]
    new_board = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            new_board[i][j] = update_cell(board, i, j)

    return new_board


def count_neighbors(board, row, col):
    count = 0

    count += board[(row+1)%WIDTH][col]
    count += board[(row-1)%WIDTH][col]
    count += board[(row+1)%WIDTH][(col+1)%HEIGHT]
    count += board[(row-1)%WIDTH][(col+1)%HEIGHT]
    count += board[(row+1)%WIDTH][(col-1)%HEIGHT]
    count += board[(row-1)%WIDTH][(col-1)%HEIGHT]
    count += board[(row)][(col+1)%HEIGHT]
    count += board[(row)][(col-1)%HEIGHT]

    return count

def update_cell(board, row, col):
    neighours = count_neighbors(board, row, col)

    if neighours == 3:
        return 1
    
    if board[row][col] == 1:
        if neighours > 3 or neighours < 2:
            return 0
        else:
            return 1
        
    return 0


def animate_board(board):
    fig = plt.figure()
    ims = []
    for i in range(NB_OF_GENERATIONS):
        im = plt.imshow(board, cmap="grey", animated=True)
        ims.append([im])
        board = update_board(board)
    ani = animation.ArtistAnimation(fig, ims, interval=TIME_DELAY_MS, blit=True, repeat = False)
    plt.show()


if __name__ == "__main__":

    board = initialize_board()

    animate_board(board)
        