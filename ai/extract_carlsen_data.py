import chess.pgn
import json

# Path to the PGN file
pgn_file_path = 'Carlsen.pgn'

# Output file to save the extracted moves
output_file_path = '/Users/adamforward/Desktop/chess/first_five_moves.json'


def extract_first_five_moves(pgn_file):
    first_five_moves = []
    with open(pgn_file, 'r', encoding='latin1') as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            moves = []
            try:
                for i, move in enumerate(game.mainline_moves()):
                    san_move = board.san(move)
                    board.push(move)
                    moves.append(san_move)
                    if i == 9:  # Only take the first 5 moves
                        break
                first_five_moves.append(moves)
            except (AssertionError, ValueError) as e:
                print(f"Skipping a game due to error: {e}")
                continue
    return first_five_moves


# Extract moves
first_five_moves = extract_first_five_moves(pgn_file_path)

# Save the moves to a JSON file
with open(output_file_path, 'w') as f:
    json.dump(first_five_moves, f)

print(f"First five moves extracted and saved to {output_file_path}")
