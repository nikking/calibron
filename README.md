[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/nikking/calibron/master)


# Solving Calibron Puzzle in Python

_Spoiler Alert:_ This Notebook does contain the puzzle solution below.

The Calibron 12-Block Puzzle is a devilishly simple puzzle invented by Theodore Edison. For a detailed history, check out http://www.pavelspuzzles.com/2010/08/the_calibron_12block_puzzle.html. 

To solve the puzzle, you must arrange the 12 pieces into a perfect rectangle. This is easier said than done as there are ~1.96 trillion possible piece permutations 
```
(12! * 2^12)
```
![](http://www.pavelspuzzles.com/images/calibron.jpg)

## General approach

Much like sudoku solving, there are probably many shortcuts that can be created to solve this puzzle. My goal was to see if I could write a mostly brute force algorithm that ran in a reasonable ammount of time.

0. Assumptions: Although there are other possible rectangular dimensions that have the correct area, this solution assumes a 56 x 56 square.
1. Create a board represented by a 56 x 56 Boolean coordinate system. All coordinates are False until occupied by a piece.
2. Intitialize the starting position of (0, 0), where the next piece will be placed. The position will always be the smallest x, y coordinate pair that is False.
3. Calculate the largest possible piece that could be place at the current position. Also calculate the smallest side for any remaining pieces.
4. Iterate though all pieces, checking if they are valid. 
5. If a piece can be placed, update the board, position, largest pieces, and smallest side. The new position is determine by finding the smallest (x, y) coordinate (smallest x then smallest y). Then repeat steps 2-4.
6. If a the board is no pieces can be placed in the current position, remove the last piece placed.
7. Return a solution once all pieces have been placed.

## Performance

Performance was greatly improved by using numpy arrays and numba just-in-time compilation. I also used %prun to profile the various methods and trim down execution time. A valid solution an be found in a matter of seconds and all solutions could be found in under two minutes on my 5 year old laptop (i7-4650U CPU @ 1.70GHz).

Space complexity should be O(n^2) where n is the number of pieces. All pieces swapping is done in-place and a copy of the piece list is created every time a new pieces is placed. To estimate the time complexity I think I'd have to run some different profiling with different sets of pieces.


```python
import numpy as np
from numba import jit
from copy import copy
from progressbar import ProgressBar
from math import factorial
from time import sleep
from matplotlib import pyplot as plt
from matplotlib import patches


CALIBRON_PIECES = [(21, 14),
                   (17, 14),
                   (21, 18),
                   (32, 10),
                   (21, 14),
                   (10, 7),
                   (14, 4),
                   (21, 18),
                   (28, 6),
                   (28, 14),
                   (32, 11),
                   (28, 7)]
WIDTH, HEIGHT = 56, 56
LENGTH = len(CALIBRON_PIECES) - 1
FACT = {x: factorial(x) * 2 ** x for x in range(0, LENGTH + 1)}
PERMUTATIONS = factorial(LENGTH + 1) * 2 ** (LENGTH + 1)


def new_board(width, height):
    '''Returns an empty board represented as a Boolean Numpy array.'''
    return np.array([[False for i in range(height)] for j in range(width)])       

    
def piece_valid(largest_piece, piece, smallest_side, position):
    '''Returns True if a this piece can be placed in the current position.'''
    return ((piece[0] <= largest_piece[0]) and
            (piece[1] <= largest_piece[1]) and 
            (HEIGHT - position[0] - piece[0] >= smallest_side or HEIGHT - position[0] - piece[0] == 0) and
            (WIDTH - position[1] - piece[1] >= smallest_side or WIDTH - position[1] - piece[1] == 0))


@jit(nopython=True)
def place_piece(board, position, piece):
    '''Places a given piece at the current position and returns the updated board.'''
    board[position[1]:position[1] + piece[1],
          position[0]:position[0] + piece[0]] = True
    return board

    
@jit(nopython=True)
def remove_piece(board, position, piece):
    '''Removes a given piece at the current position and returns the updated board.'''
    board[position[1]:position[1] + piece[1],
          position[0]:position[0] + piece[0]] = False
    return board

    
@jit(nopython=True)
def update_position(board, position, width, height):
    '''Returns the the updated position where the next piece will be placed.'''
    # first check current row
    i = position[1]
    for j in range(position[0], width):
        if not board[i][j]:
            return (j, i)
    # then check remaining rows
    for i in range(position[1] + 1, height):
        for j in range(width):
            if not board[i][j]:
                return (j, i)


@jit(nopython=True)
def update_largest_piece(width, height, board, position):
    '''Returns the dimensions of the largest piece that could be placed at the current position.'''
    piece_height = height - position[1]
    piece_width = 0
    for j in range(position[0], width):
        if not board[position[1]][j]:
            piece_width += 1
        else:
            break
    return (piece_width, piece_height)


def reverse(piece):
    '''This generator is used to iterate through both oreintations of a piece.'''
    yield(piece)
    yield((piece[1], piece[0]))


def update_progress(n):
    '''Update the progressbar by the ammount of permutations ruled out.'''
    global BAR
    global COUNT
    COUNT += FACT[n]
    if COUNT % 1000 == False:
        BAR.update(round(COUNT / 10 ** 9))

        
def is_end(i, length):
    '''Returns True if all peices have been placed.'''
    return i == length


def recurse(i, pieces, board, position):
    '''In-place algorithm to find valid pieces, place them, and step into a new board.'''
    largest_piece = update_largest_piece(WIDTH, HEIGHT, board, position)
    smallest_side = min((min(piece) for piece in pieces[i:LENGTH + 1]))
    j = LENGTH
    while i <= j:
        for pieces[i] in reverse(pieces[i]):
            if piece_valid(largest_piece, pieces[i], smallest_side, position):
                board = place_piece(board, position, pieces[i])
                if is_end(i, LENGTH):
                    return pieces
                solution = recurse(i + 1, copy(pieces), board, update_position(board, position, WIDTH, HEIGHT))
                if solution:
                    return solution
                else:
                    board = remove_piece(board, position, pieces[i])
            else:   
                update_progress(LENGTH - i)
        if i != j:
            pieces[i], pieces[j] = pieces[j], pieces[i]
        j -= 1
    return None
```

## Running the Algorithm

The Progress Bar shown below is in  billions, as there are 1.962 trillion possible solutions. Not all permutations are checked, but rather when a board is invalid the counter increments by however many permutations are ruled out. For example, if the second piece can't be placed the counter is incremented by 
```
11! * 2^11
```

With this algorithm there are multiple possible solutions because translations and rotations are considered unique. Not all permuations have to be calculated to find a valid solution.


```python
BAR = ProgressBar(max_value=round(PERMUTATIONS / 10 ** 9))
COUNT = 0
solution = recurse(0, CALIBRON_PIECES, new_board(WIDTH, HEIGHT), (0,0))
print("Found a valid solution.")
```

      3% (60 of 1962) |                      | Elapsed Time: 0:00:04 ETA:   0:01:52

    Found a valid solution.


## Plotting the solution

### Spoler Alert: Solution Below!


```python
def plot_solution(solution):
    fig = plt.figure(figsize=(6,6))
    ax1 = fig.add_subplot(111, aspect='equal')
    plt.ylim((0, HEIGHT))
    plt.xlim((0, WIDTH))
    plt.xticks([0, 10, 21, 28, 32, 38, 42, 56])
    plt.yticks([0, 14, 24, 28, 42, 49, 56])
    board = new_board(WIDTH, HEIGHT)
    colors = (x for x in ["#ef5350", "#ab47bc", "#7e57c2", "#5c6bc0", "#42a5f5", "#26c6da",
                          "#26a69a", "#66bb6a", "#9ccc65", "#d4e157", "#ffca28", "#ff7043"])
    position = (0, 0)
    for piece in solution:
        sleep(0.3)
        ax1.add_patch(patches.Rectangle(position, piece[0], piece[1], color=next(colors)))
        fig.canvas.draw()
        fig.canvas.flush_events()
        place_piece(board, position, piece)
        position = update_position(board, position, WIDTH, HEIGHT)
        
%matplotlib notebook
print(solution)
plot_solution(solution)
```
    [(21, 14), (21, 14), (14, 28), (32, 10), (6, 28), (4, 14), (11, 32), (21, 18), (18, 21), (17, 14), (10, 7), (28, 7)]
```

Run the notebook locally or with the Binder link to see a visualization of the solution.
