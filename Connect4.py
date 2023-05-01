import time

import numpy as np
import random
import pygame
import sys
import math
from copy import deepcopy
from random import choice
import json
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
ROW_COUNT = 6
COLUMN_COUNT = 7
PLAYER = -1
AI = 1
EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2
WINDOW_LENGTH = 4
winner = 0

def create_board():
    board = np.zeros((ROW_COUNT, COLUMN_COUNT),dtype=int)
    return board


def drop_piece(board, row, col, piece):
    board[row][col] = piece


def is_valid_location(board, col):
    return board[ROW_COUNT - 1][col] == 0


def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r


def print_board(board):
    print(np.flip(board, 0))

#check win
def winning_move(board, piece):
    # Check horizontal locations for win
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][
                c + 3] == piece:
                return True

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][
                c] == piece:
                return True

    # Check positively sloped diaganols
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][
                c + 3] == piece:
                return True

    # Check negatively sloped diaganols
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][
                c + 3] == piece:
                return True


def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE
    if piece == PLAYER_PIECE:
        opp_piece = AI_PIECE

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 4
    return score


def score_position(board, piece):
    score = 0

    ## Score center column
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    ## Score Horizontal
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    ## Score Vertical
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    ## Score posiive sloped diagonal
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score


def is_terminal_node(board):
    global winner
    if winning_move(board, PLAYER_PIECE):
        winner = PLAYER
    elif winning_move(board, AI_PIECE):
        winner = AI
    else:
        winner = -1
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0


def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return (None, 100000000000000)
            elif winning_move(board, PLAYER_PIECE):
                return (None, -10000000000000)
            else:  # Game is over, no more valid moves
                return (None, 0)
        else:  # Depth is zero
            return (None, score_position(board, AI_PIECE))
    if maximizingPlayer:
        value = -math.inf
        p = AI_PIECE
        b = False
    else:
        value = math.inf
        p = PLAYER_PIECE
        b = True
    column = random.choice(valid_locations)
    for col in valid_locations:
        row = get_next_open_row(board, col)
        b_copy = board.copy()
        drop_piece(b_copy, row, col, p)
        new_score = minimax(b_copy, depth - 1, alpha, beta, b)[1]
        print(new_score)
        if maximizingPlayer:
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        else:
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
    return column,value

####################################for Q learning
class ComputerAgentQLearning():
    def __init__(self, alpha=0.1, epsilon=0.1, gamma=0.9):
        self.alpha = alpha  # Learning rate
        self.epsilon = epsilon  # Exploration rate
        self.gamma = gamma  # Discount factor
        self.q_table = {} # np.zeros((6, 7, 2))  # Q-table for each (state, action) pair

    def getQ(self, state, action):
        # encourage exploration; "optimistic" 1.0 initial values
        if self.q_table.get((state, action)) is None:
            self.q_table[(state, action)] = 1.0
        return self.q_table.get((state, action))

    def get_move(self,actions,board):
        current_state = get_state(board)
        if random.random() < self.epsilon: # explore!
            chosen_action = random.choice(actions)
            return chosen_action
        qs = [self.getQ(current_state, a) for a in actions]
        maxQ = max(qs)

        if qs.count(maxQ) > 1:
            # more than 1 best option; choose among them randomly
            best_options = [i for i in range(len(actions)) if qs[i] == maxQ]
            i = random.choice(best_options)
        else:
            i = qs.index(maxQ)

        return actions[i]

qplayer = ComputerAgentQLearning()  # Initialize the QPlayer object

def get_possible_moves(board):
    return [i for i in range(COLUMN_COUNT) if board[0][i] == 0]

def get_state(board):
    result = tuple(tuple(x) for x in board)
    return result

def make_move(column, piece,board):
    for row in range(5, -1, -1):  # start from bottom row and work upwards
        if board[row][column] == 0:
            board[row][column] = piece
            break
    else:
        raise ValueError("Invalid move")

def train_qlearing_agent(board):
    num_games = 1000  # Number of games to play
    for i in range(num_games):
        player = AI
        current_board = deepcopy(board)
        game_over = False
        # Play the game
        while not game_over:
            actions = get_possible_moves(current_board)
            move = qplayer.get_move(actions,current_board)
            previous_state = deepcopy(current_board)
            if player == AI:
                piece = AI_PIECE
            else:
                piece = PLAYER_PIECE
            make_move(move, piece,current_board)
            # Check if the game is over
            reward = 0
            if all(current_board[0]):
                game_over = True
            elif winning_move(current_board,piece):
                if piece == AI_PIECE:
                    reward = 1
                else:
                    reward = -2
                game_over = True

            # Update the QPlayer's Q-table
            prev_state = get_state(previous_state)
            prev = qplayer.getQ(prev_state, move)
            result_state = get_state(current_board)
            maxqnew = max([qplayer.getQ(result_state, a) for a in actions])
            qplayer.q_table[(prev_state, move)] = prev + qplayer.alpha * ((reward + qplayer.gamma*maxqnew) - prev)
            # Switch players
            player = -player
    return qplayer.q_table
######################################

def player_method(board):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if len(valid_locations) == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return (None, 100000000000000)
            elif winning_move(board, PLAYER_PIECE):
                return (None, -10000000000000)
            else:  # Game is over, no more valid moves
                return (None, 0)
        else:  # Depth is zero
            return (None, score_position(board, AI_PIECE))

    for col in valid_locations:
        row = get_next_open_row(board, col)
        b_copy = board.copy()
        #check player
        b_copy[row][col] = PLAYER_PIECE
        is_terminal = is_terminal_node(b_copy)
        if is_terminal:
            drop_piece(board, row, col, PLAYER_PIECE)
            return (None, -10000000000000)
        else:
            b_copy[row][col] = 0
        #check AI
        b_copy[row][col] = AI_PIECE
        is_terminal = is_terminal_node(b_copy)
        if is_terminal:
            drop_piece(board, row, col, PLAYER_PIECE)
            return
        else:
            b_copy[row][col] = 0

    col = choice(valid_locations)
    row = get_next_open_row(board, col)
    drop_piece(board, row, col, PLAYER_PIECE)

def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations


def pick_best_move(board, piece):
    valid_locations = get_valid_locations(board)
    best_score = -10000
    best_col = random.choice(valid_locations)
    for col in valid_locations:
        row = get_next_open_row(board, col)
        temp_board = board.copy()
        drop_piece(temp_board, row, col, piece)
        score = score_position(temp_board, piece)
        if score > best_score:
            best_score = score
            best_col = col
    return best_col

def draw_board(board):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (
            int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)

    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == PLAYER_PIECE:
                pygame.draw.circle(screen, RED, (
                int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
            elif board[r][c] == AI_PIECE:
                pygame.draw.circle(screen, YELLOW, (
                int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    pygame.display.update()


board = create_board()
train_qlearing_agent(board)
game_over = False

pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE / 2 - 5)
screen = pygame.display.set_mode(size)
draw_board(board)
pygame.display.update()

myfont = pygame.font.SysFont("monospace", 75)

turn = AI

# PLAYER = 0
# AI = 1

#use minimax method or not
mini = False

while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    if turn == PLAYER and not game_over:
        player_method(board)
        if winning_move(board, PLAYER_PIECE):
            label = myfont.render("Opponent wins!!", 1, YELLOW)
            screen.blit(label, (40, 10))
            game_over = True
        draw_board(board)
        turn = AI
        time.sleep(0.4)

    #AI turn
    if turn == AI and not game_over and mini == True:

        # col = random.randint(0, COLUMN_COUNT-1)
        # col = pick_best_move(board, AI_PIECE)
        col, minimax_score = minimax(board, 5, -math.inf, math.inf, True)

        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, AI_PIECE)

            if winning_move(board, AI_PIECE):
                label = myfont.render("AI wins!!", 1, YELLOW)
                screen.blit(label, (40, 10))
                game_over = True

            #print_board(board)
            draw_board(board)
            turn = PLAYER
            time.sleep(1)

    if turn == AI and not game_over and mini == False:

        board = np.flip(board, 0)
        actions = get_possible_moves(board)

        move = qplayer.get_move(actions, board)
        make_move(move, AI_PIECE, board)
        if winning_move(board, AI_PIECE):
            label = myfont.render("AI wins!!", 1, YELLOW)
            screen.blit(label, (40, 10))
            game_over = True
        board = np.flip(board, 0)
        draw_board(board)
        turn = PLAYER
        time.sleep(0.4)


    if game_over:
        print_board(board)
        pygame.time.wait(3000)
