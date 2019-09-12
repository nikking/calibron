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



    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,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XqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUAXqGnIlhxg0ew4vIsuRqCVXL6NXZHsGrAanAogtUA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" width="600">
