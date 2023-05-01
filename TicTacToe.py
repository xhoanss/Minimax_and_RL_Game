import tkinter as tk
from random import choice
import time
from math import inf
import numpy as np


################################################## Draw game
# create main window
root = tk.Tk()
root.title("Tic Tac Toe")
# create
title_frame = tk.Frame(root)
title_frame.pack()
# create title
title_label = tk.Label(title_frame, text="Tic Tac Toe", font=("Arial", 24))
title_label.pack()
# creat Grid pack
grid_frame = tk.Frame(root)
grid_frame.pack()
grid_size = 200
board = [[0 for _ in range(3)] for _ in range(3)] # 创建一个3 x 3的二维数组来存储所有格子的信息
canvas_list=[]


for i in range(3):
    for j in range(3):
        x, y = i, j
        canvas = tk.Canvas(grid_frame, width=grid_size, height=grid_size, highlightthickness=0)
        canvas.grid(row=i, column=j)
        canvas.create_rectangle(0, 0, grid_size, grid_size, fill="white")
        canvas_list.append(canvas)

def draw_cross(index):
    canvas_list[index].create_line(10, 10, grid_size - 10, grid_size - 10, width=5, fill='red')
    canvas_list[index].create_line(10, grid_size - 10, grid_size - 10, 10, width=5, fill='red')
def draw_circle(index):
    x = y = grid_size // 2
    radius = grid_size // 3
    canvas_list[index].create_oval(x - radius, y - radius, x + radius, y + radius, outline="blue", width=5)


########################################### Play
Algorithm = +1
Opponent = -1
board = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]

def evaluate(state):
    if wins(state,Algorithm):
        score = 1
    elif wins(state,Opponent):
        score = -1
    else:
        score = 0
    return score

def wins(state, player):
    global AI_win,opponent_win
    win_state = [
        [state[0][0], state[0][1], state[0][2]],
        [state[1][0], state[1][1], state[1][2]],
        [state[2][0], state[2][1], state[2][2]],
        [state[0][0], state[1][0], state[2][0]],
        [state[0][1], state[1][1], state[2][1]],
        [state[0][2], state[1][2], state[2][2]],
        [state[0][0], state[1][1], state[2][2]],
        [state[2][0], state[1][1], state[0][2]],
    ]
    if [player, player, player] in win_state:
        if player == +1:
            title_label.config(text="Algorithm Wins!!")
        else:
            title_label.config(text="Opponent Wins!!")
        return True
    else:
        return False

def empty_cell(state):
    cells = []
    for x, row in enumerate(state):
        for y, cell in enumerate(row):
            if cell == 0:
                cells.append([x, y])
    return cells

def game_over(state):
    return wins(state, Algorithm) or wins(state, Opponent)

def valid_move(x, y):
    if [x, y] in empty_cell(board):
        return True
    else:
        return False

def action(x,y,player):
    if valid_move(x, y):
        board[x][y] = player
        index = x * 3 + y
        if player == Algorithm:
            draw_circle(index)
        else:
            draw_cross(index)
        return True
    else:
        return False

def opponent_mathod(board):
    for i in range(3):
        for j in range(3):
            if board[i][j]==0:
                board[i][j] = Opponent
                if game_over(board):
                    board[i][j] = 0
                    return [i,j]
                board[i][j] = Algorithm
                if game_over(board):
                    board[i][j] = 0
                    return [i,j]
                board[i][j] = 0

    for row in range(3):
        for col in range(3):
            if board[row][col] == 0:
                return row, col

def minimax(state,depth,player,alpha,beta):
    # two players start from the worst score
    if player == Algorithm:
        best = [-1,-1,-inf]
    else:
        best = [-1,-1,+inf]
    if depth == 0 or game_over(state):
        score = evaluate(state)
        return [-1, -1, score]
    for cell in empty_cell(state):
        x,y = cell[0],cell[1]
        state[x][y] = player
        score = minimax(state,depth-1,-player,alpha,beta)
        state[x][y] = 0
        score[0], score[1] = x, y
        if player == Algorithm:
            alpha = max(alpha,best[2])
            if beta<=alpha:
                break
            else:
                if score[2] > best[2]:
                    best = score
        else:
            if beta<=alpha:
                break
            else:
                if score[2] < best[2]:
                    best = score
    return best

class Qlearning:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.2):
        self.q_table = {}
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
    def get_action(self, state, valid_moves):
        if np.random.rand() < self.epsilon:
            # Choose a random action
            return tuple(valid_moves[np.random.choice(valid_moves.shape[0])])
        else:
            # Choose the best action according to the Q-table
            q_values = [self.q_table.get((state, tuple(move)), 0) for move in valid_moves]
            best_idx = np.argmax(q_values)
            return tuple(valid_moves[best_idx])
    def learn(self, state, act, reward, board):
        next_q_values = [self.q_table.get((get_state(board), tuple(move)), 0) for move in get_valid_moves(board)]
        target = reward + self.gamma * np.max(next_q_values)
        current_q = self.q_table.get((state, act), 0)
        self.q_table[(state, act)] = current_q + self.alpha * (target - current_q)

ql = Qlearning()

def get_state(board):
    return tuple(map(tuple, board))

def get_valid_moves(board):
    return np.argwhere(board == 0)

def QLear():
    rate = 0
    rate2 = 0
    for i in range(1000):
        board_array = np.array(board)
        done = False
        player = Algorithm
        while not done:
            state = get_state(board_array)
            valid_moves = get_valid_moves(board_array)
            act = ql.get_action(state, valid_moves)
            board_array[act[0],act[1]] = player
            b = board_array.tolist()
            if wins(b,player):
                if player == Algorithm:
                    rate+=1
                else:
                    rate2+=1
                done = True
            elif get_valid_moves(board_array).size == 0:
                done = True
            else:
                player = -player
            rewards = evaluate(b)
            ql.learn(state, act, rewards, np.array(board))
    print(rate/10000)
    print(rate2 / 10000)
    return ql.q_table

def Algorithm_turn(is_minimax):
    depth = len(empty_cell(board))
    if depth == 0 or game_over(board):
        return
    if is_minimax == True:
        if depth == 9:
            x = choice([0, 1, 2])
            y = choice([0, 1, 2])
        else:
            move = minimax(board, depth, Algorithm,-inf,+inf)
            x, y = move[0], move[1]
        action(x, y, Algorithm)
    else:
        board_array = np.array(board)
        state = get_state(board_array)
        valid_moves = get_valid_moves(board_array)
        act = ql.get_action(state, valid_moves)
        x,y = act[0],act[1]
        action(x, y, Algorithm)
    time.sleep(1)

def opponent_turn():
    depth = len(empty_cell(board))
    if depth == 0 or game_over(board):
        return
    move = opponent_mathod(board)
    x, y = move[0], move[1]
    action(x,y,Opponent)
    time.sleep(1)


def plot_game():
    AI_win = 0
    opponent_win = 0
    draw = 0
    for i in range(10000):
        while len(empty_cell(board)) > 0 and not game_over(board):
            Algorithm_turn(is_minimax=True)
            opponent_turn()
        if wins(board,Algorithm):
            AI_win+=1
        elif wins(board,Opponent):
            opponent_win+=1
        else:
            draw+=1
    print(AI_win)


def main_game():
    QLear()
    while len(empty_cell(board)) > 0 and not game_over(board):
        Algorithm_turn(is_minimax=False)
        title_label.config(text="Algorithm Turn")
        root.update()
        opponent_turn()
        title_label.config(text="Opponent Turn")
        root.update()

#plot_game()
main_game()
root.mainloop()




