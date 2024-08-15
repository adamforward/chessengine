import chess.pgn
import csv
import os
import torch
import numpy as np

print("running the updated script")
# Path to the PGN file
path_to_pgn = '../../T7/chess.pgn'
# Output file for intermediate batches
output_dir = '../../T7/batches'
os.makedirs(output_dir, exist_ok=True)

piece_to_index = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5
}


def board_to_tensor(board):
    tensor = np.zeros((6, 8, 8), dtype=int)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            index = piece_to_index[piece.symbol()]
            row = 7 - (square // 8)
            col = square % 8
            value = 1 if piece.color == chess.WHITE else -1
            tensor[index][row][col] = value
    return torch.tensor(tensor, dtype=torch.float16)


def extract_game_states(game, result):
    board = game.board()
    states = []
    for move in game.mainline_moves():
        board.push(move)
        states.append((board_to_tensor(board), result))
    return states


def save_batch(batch, batch_num):
    torch.save(batch, os.path.join(output_dir, f'batch_{batch_num}.pt'))


# Open the PGN file with latin1 encoding
batch_size = 1000
batch = []
batch_num = 0
with open(path_to_pgn, 'r', encoding='latin1') as pgn_file:
    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break
        result = game.headers["Result"]
        if result == "1-0":
            label = 1  # White wins
        elif result == "0-1":
            label = -1  # Black wins
        else:
            continue  # Ignore draws
        batch.extend(extract_game_states(game, label))
        if len(batch) >= batch_size:
            save_batch(batch, batch_num)
            batch = []
            batch_num += 1

# Save any remaining states in the last batch
if batch:
    save_batch(batch, batch_num)

print(f"Successfully processed and saved game states in batches.")

# Example usage of loading batches:
# loaded_batch = torch.load(os.path.join(output_dir, 'batch_0.pt'))
