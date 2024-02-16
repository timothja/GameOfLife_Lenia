"""
Conway's game of life using the terminal to display the board.
"""

import random
import time

HEIGHT = 40
WIDTH = 40
TIME_DELAY_S = 0.2
NB_OF_GENERATIONS = 200

def initialize_board(height=HEIGHT, width=WIDTH):
    board = []
    for i in range(height):
        row = []
        for j in range(width):
            row.append(random.choice([0, 1]))
        board.append(row)

    return board

def print_board(board):
    # clear the terminal
    print("\033c", end="")

    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == 1:
                print("◻️", end=" ")
            else:
                print(" ", end=" ")
        print()

def update_board(board):
    new_board = []

    for i in range(len(board)):
        row = []
        for j in range(len(board[i])):
            row.append(update_cell(board, i, j))
        new_board.append(row)

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
    count = count_neighbors(board, row, col)

    if count == 3:
        return 1
    
    if board[row][col] == 1:
        if count > 3 or count < 2:
            return 0
        else:
            return 1
    
    return 0




if __name__ == "__main__":

    board = initialize_board()

    for i in range(NB_OF_GENERATIONS):
        print_board(board)
        board = update_board(board)
        time.sleep(TIME_DELAY_S)
        