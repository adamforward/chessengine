import chess
import chess.syzygy
import json
import os
import itertools
import gc

# Path to the Syzygy endgame tablebase files
syzygy_path = '/Users/adamforward/Desktop/chess/syzygy/3-4-5'
tablebase = chess.syzygy.open_tablebase(syzygy_path)

# Define the output directory for the JSON files
output_dir = '/Users/adamforward/Desktop/chess/syzygy/json_files'
os.makedirs(output_dir, exist_ok=True)

# Function to generate all possible board configurations


def generate_boards(piece_list, batch_size=1000):
    boards = []
    squares = list(chess.SQUARES)
    for piece_positions in itertools.permutations(squares, len(piece_list)):
        board = chess.Board(fen=None)
        for square, piece_color in zip(piece_positions, piece_list):
            piece, color = piece_color
            if piece:
                board.set_piece_at(square, chess.Piece(piece, color))
        if board.is_valid() and not board.is_checkmate() and not board.is_stalemate():
            boards.append(board)
        if len(boards) >= batch_size:
            yield boards
            boards = []
    if boards:
        yield boards

# Function to query the Syzygy tablebase for the best move


def get_best_move(board):
    try:
        # Use probe_dtz for the best move suggestion
        best_move = tablebase.probe_dtz(board)
        if isinstance(best_move, chess.Move):
            return best_move.uci()  # Return the UCI notation of the best move
        elif isinstance(best_move, int):
            return None  # No valid move found, just DTZ value
        else:
            return None  # Handle any other unexpected return types
    except Exception as e:
        print(f"Error probing board {board.fen()}: {e}")
        return None

# Function to process a batch of boards


def process_board_batch(board_batch):
    results = {}
    for board in board_batch:
        best_move = get_best_move(board)
        if best_move:
            results[board.fen()] = best_move
    return results


# Replace this with your own piece list configurations
piece_list = [
    (chess.KING, chess.WHITE),
    (chess.KING, chess.BLACK),
    (chess.BISHOP, chess.WHITE),
    (chess.BISHOP, chess.WHITE),
    (chess.BISHOP, chess.WHITE)
]

# Iterate over generated boards and process them
best_moves = {}

for board_batch in generate_boards(piece_list):
    results = process_board_batch(board_batch)
    best_moves.update(results)

    # Release memory
    gc.collect()

# Save the mapping to a JSON file if there are any results
if best_moves:
    output_path = os.path.join(output_dir, "results.json")
    with open(output_path, 'w') as json_file:
        json.dump(best_moves, json_file, indent=4)
    print(f"Saved results to {output_path}")
else:
    print(f"No valid moves found, no file created.")

print("Completed generating JSON files.")
# Mapping of Syzygy types to piece lists
syzygy_types = {
    "KBBBvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.BISHOP, chess.WHITE), (chess.BISHOP, chess.WHITE)],
    "KBBNvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.BISHOP, chess.WHITE), (chess.KNIGHT, chess.WHITE)],
    "KBBPvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.BISHOP, chess.WHITE), (chess.PAWN, chess.WHITE)],
    "KBBvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.BISHOP, chess.WHITE)],
    "KBBvKB": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.BISHOP, chess.WHITE), (chess.BISHOP, chess.BLACK)],
    "KBBvKN": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.BISHOP, chess.WHITE), (chess.KNIGHT, chess.BLACK)],
    "KBBvKP": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.BISHOP, chess.WHITE), (chess.PAWN, chess.BLACK)],
    "KBBvKQ": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.BISHOP, chess.WHITE), (chess.QUEEN, chess.BLACK)],
    "KBBvKR": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.BISHOP, chess.WHITE), (chess.ROOK, chess.BLACK)],
    "KBNNvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.KNIGHT, chess.WHITE)],
    "KBNPvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.PAWN, chess.WHITE)],
    "KBNvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.KNIGHT, chess.WHITE)],
    "KBNvKB": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.BISHOP, chess.BLACK)],
    "KBNvKN": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.KNIGHT, chess.BLACK)],
    "KBNvKP": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.PAWN, chess.BLACK)],
    "KBNvKQ": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.QUEEN, chess.BLACK)],
    "KBNvKR": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.ROOK, chess.BLACK)],
    "KBPPvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.PAWN, chess.WHITE), (chess.PAWN, chess.WHITE)],
    "KBPvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.PAWN, chess.WHITE)],
    "KBPvKB": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.PAWN, chess.WHITE), (chess.BISHOP, chess.BLACK)],
    "KBPvKN": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.PAWN, chess.WHITE), (chess.KNIGHT, chess.BLACK)],
    "KBPvKP": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.PAWN, chess.WHITE), (chess.PAWN, chess.BLACK)],
    "KBPvKQ": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.PAWN, chess.WHITE), (chess.QUEEN, chess.BLACK)],
    "KBPvKR": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.PAWN, chess.WHITE), (chess.ROOK, chess.BLACK)],
    "KBvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE)],
    "KBvKB": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.BISHOP, chess.BLACK)],
    "KBvKN": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.KNIGHT, chess.BLACK)],
    "KBvKP": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.BISHOP, chess.WHITE), (chess.PAWN, chess.BLACK)],
    "KNNNvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.KNIGHT, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.KNIGHT, chess.WHITE)],
    "KNNPvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.KNIGHT, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.PAWN, chess.WHITE)],
    "KNNvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.KNIGHT, chess.WHITE), (chess.KNIGHT, chess.WHITE)],
    "KNNvKB": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.KNIGHT, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.BISHOP, chess.BLACK)],
    "KNNvKN": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.KNIGHT, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.KNIGHT, chess.BLACK)],
    "KNNvKP": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.KNIGHT, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.PAWN, chess.BLACK)],
    "KNNvKQ": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.KNIGHT, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.QUEEN, chess.BLACK)],
    "KNPPvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.KNIGHT, chess.WHITE), (chess.PAWN, chess.WHITE), (chess.PAWN, chess.WHITE)],
    "KNPvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.KNIGHT, chess.WHITE), (chess.PAWN, chess.WHITE)],
    "KNPvKB": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.KNIGHT, chess.WHITE), (chess.PAWN, chess.WHITE), (chess.BISHOP, chess.BLACK)],
    "KNPvKN": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.KNIGHT, chess.WHITE), (chess.PAWN, chess.WHITE), (chess.KNIGHT, chess.BLACK)],
    "KNPvKP": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.KNIGHT, chess.WHITE), (chess.PAWN, chess.WHITE), (chess.PAWN, chess.BLACK)],
    "KNPvKQ": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.KNIGHT, chess.WHITE), (chess.PAWN, chess.WHITE), (chess.QUEEN, chess.BLACK)],
    "KNPvKR": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.KNIGHT, chess.WHITE), (chess.PAWN, chess.WHITE), (chess.ROOK, chess.BLACK)],
    "KNvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.KNIGHT, chess.WHITE)],
    "KNvKB": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.KNIGHT, chess.WHITE), (chess.BISHOP, chess.BLACK)],
    "KNvKN": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.KNIGHT, chess.WHITE), (chess.KNIGHT, chess.BLACK)],
    "KNvKP": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.KNIGHT, chess.WHITE), (chess.PAWN, chess.BLACK)],
    "KPvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.PAWN, chess.WHITE)],
    "KPvKB": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.PAWN, chess.WHITE), (chess.BISHOP, chess.BLACK)],
    "KPvKN": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.PAWN, chess.WHITE), (chess.KNIGHT, chess.BLACK)],
    "KPvKP": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.PAWN, chess.WHITE), (chess.PAWN, chess.BLACK)],
    "KQBBvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.BISHOP, chess.WHITE), (chess.BISHOP, chess.WHITE)],
    "KQBNvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.BISHOP, chess.WHITE), (chess.KNIGHT, chess.WHITE)],
    "KQBPvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.BISHOP, chess.WHITE), (chess.PAWN, chess.WHITE)],
    "KQBvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.BISHOP, chess.WHITE)],
    "KQBvKB": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.BISHOP, chess.WHITE), (chess.BISHOP, chess.BLACK)],
    "KQBvKN": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.BISHOP, chess.WHITE), (chess.KNIGHT, chess.BLACK)],
    "KQBvKP": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.BISHOP, chess.WHITE), (chess.PAWN, chess.BLACK)],
    "KQBvKQ": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.BISHOP, chess.WHITE), (chess.QUEEN, chess.BLACK)],
    "KQBvKR": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.BISHOP, chess.WHITE), (chess.ROOK, chess.BLACK)],
    "KQNNvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.KNIGHT, chess.WHITE)],
    "KQNPvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.PAWN, chess.WHITE)],
    "KQNvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.KNIGHT, chess.WHITE)],
    "KQNvKB": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.BISHOP, chess.BLACK)],
    "KQNvKN": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.KNIGHT, chess.BLACK)],
    "KQNvKP": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.PAWN, chess.BLACK)],
    "KQNvKQ": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.QUEEN, chess.BLACK)],
    "KQNvKR": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.ROOK, chess.BLACK)],
    "KQPPvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.PAWN, chess.WHITE), (chess.PAWN, chess.WHITE)],
    "KQPvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.PAWN, chess.WHITE)],
    "KQPvKB": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.PAWN, chess.WHITE), (chess.BISHOP, chess.BLACK)],
    "KQPvKN": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.PAWN, chess.WHITE), (chess.KNIGHT, chess.BLACK)],
    "KQPvKP": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.PAWN, chess.WHITE), (chess.PAWN, chess.BLACK)],
    "KQPvKQ": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.PAWN, chess.WHITE), (chess.QUEEN, chess.BLACK)],
    "KQPvKR": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.PAWN, chess.WHITE), (chess.ROOK, chess.BLACK)],
    "KQQvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.QUEEN, chess.WHITE)],
    "KQQvKB": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.QUEEN, chess.WHITE), (chess.BISHOP, chess.BLACK)],
    "KQQvKN": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.QUEEN, chess.WHITE), (chess.KNIGHT, chess.BLACK)],
    "KQQvKP": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.QUEEN, chess.WHITE), (chess.PAWN, chess.BLACK)],
    "KQQvKQ": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.QUEEN, chess.WHITE), (chess.QUEEN, chess.BLACK)],
    "KQQvKR": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.QUEEN, chess.WHITE), (chess.ROOK, chess.BLACK)],
    "KQRBvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.ROOK, chess.WHITE), (chess.BISHOP, chess.WHITE)],
    "KQRNvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.ROOK, chess.WHITE), (chess.KNIGHT, chess.WHITE)],
    "KQRPvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.ROOK, chess.WHITE), (chess.PAWN, chess.WHITE)],
    "KQRvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.ROOK, chess.WHITE)],
    "KQRvKB": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.ROOK, chess.WHITE), (chess.BISHOP, chess.BLACK)],
    "KQRvKN": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.ROOK, chess.WHITE), (chess.KNIGHT, chess.BLACK)],
    "KQRvKP": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.ROOK, chess.WHITE), (chess.PAWN, chess.BLACK)],
    "KQRvKQ": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.ROOK, chess.WHITE), (chess.QUEEN, chess.BLACK)],
    "KQRvKR": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.QUEEN, chess.WHITE), (chess.ROOK, chess.WHITE), (chess.ROOK, chess.BLACK)],
    "KRBBvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.BISHOP, chess.WHITE), (chess.BISHOP, chess.WHITE)],
    "KRBNvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.BISHOP, chess.WHITE), (chess.KNIGHT, chess.WHITE)],
    "KRBPvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.BISHOP, chess.WHITE), (chess.PAWN, chess.WHITE)],
    "KRBvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.BISHOP, chess.WHITE)],
    "KRBvKB": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.BISHOP, chess.WHITE), (chess.BISHOP, chess.BLACK)],
    "KRBvKN": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.BISHOP, chess.WHITE), (chess.KNIGHT, chess.BLACK)],
    "KRBvKP": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.BISHOP, chess.WHITE), (chess.PAWN, chess.BLACK)],
    "KRBvKQ": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.BISHOP, chess.WHITE), (chess.QUEEN, chess.BLACK)],
    "KRBvKR": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.BISHOP, chess.WHITE), (chess.ROOK, chess.BLACK)],
    "KRNNvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.KNIGHT, chess.WHITE)],
    "KRNPvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.PAWN, chess.WHITE)],
    "KRNvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.KNIGHT, chess.WHITE)],
    "KRNvKB": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.BISHOP, chess.BLACK)],
    "KRNvKN": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.KNIGHT, chess.BLACK)],
    "KRNvKP": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.PAWN, chess.BLACK)],
    "KRNvKQ": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.QUEEN, chess.BLACK)],
    "KRNvKR": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.ROOK, chess.BLACK)],
    "KRPPvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.PAWN, chess.WHITE), (chess.PAWN, chess.WHITE)],
    "KRPvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.PAWN, chess.WHITE)],
    "KRPvKB": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.PAWN, chess.WHITE), (chess.BISHOP, chess.BLACK)],
    "KRPvKN": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.PAWN, chess.WHITE), (chess.KNIGHT, chess.BLACK)],
    "KRPvKP": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.PAWN, chess.WHITE), (chess.PAWN, chess.BLACK)],
    "KRPvKQ": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.PAWN, chess.WHITE), (chess.QUEEN, chess.BLACK)],
    "KRPvKR": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.PAWN, chess.WHITE), (chess.ROOK, chess.BLACK)],
    "KRRRvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.ROOK, chess.WHITE), (chess.ROOK, chess.WHITE)],
    "KRRvK": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.ROOK, chess.WHITE)],
    "KRRvKB": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.ROOK, chess.WHITE), (chess.BISHOP, chess.BLACK)],
    "KRRvKN": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.ROOK, chess.WHITE), (chess.KNIGHT, chess.BLACK)],
    "KRRvKP": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.ROOK, chess.WHITE), (chess.PAWN, chess.BLACK)],
    "KRRvKQ": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.ROOK, chess.WHITE), (chess.QUEEN, chess.BLACK)],
    "KRRvKR": [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK), (chess.ROOK, chess.WHITE), (chess.ROOK, chess.WHITE), (chess.ROOK, chess.BLACK)],
}

# Iterate over each Syzygy type
for syzygy_name, piece_list in syzygy_types.items():
    best_moves = {}

    for board_batch in generate_boards(piece_list):
        results = process_board_batch(board_batch)
        best_moves.update(results)

        # Release memory
        gc.collect()

    # Save the mapping to a JSON file if there are any results
    if best_moves:
        output_path = os.path.join(output_dir, f"{syzygy_name}.json")
        with open(output_path, 'w') as json_file:
            json.dump(best_moves, json_file, indent=4)
        print(f"Saved {syzygy_name} to {output_path}")
    else:
        print(f"No valid moves found for {syzygy_name}, no file created.")

print("Completed generating JSON files.")
