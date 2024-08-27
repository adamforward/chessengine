import chess


def validate_moves(fen, moves):
    """
    Validates each move in the list of moves against the legal moves in the given FEN.
    """
    board = chess.Board(fen)
    valid_moves = {move.uci() for move in board.legal_moves}

    # Check if all moves are valid and collect missing moves
    invalid_moves = []
    for move in moves.split(", "):
        if move not in valid_moves:
            invalid_moves.append(move)

    return invalid_moves


def parse_and_validate(file_path):
    count = 0
    with open(file_path, 'r') as file:
        while True:
            # Read Turn and FEN lines
            count += 1
            turn_line = file.readline().strip()
            if not turn_line:
                break

            # Split the turn line into Turn and FEN parts
            if "FEN:" in turn_line:
                turn_part, fen_part = turn_line.split("FEN:")
                turn = turn_part.replace("Turn: ", "").strip().lower()[0]
                fen = fen_part.strip()
            else:
                turn = turn_line.replace("Turn: ", "").strip().lower()[0]
                fen = file.readline().strip().replace("FEN: ", "")

            moves_line = file.readline().strip()

            print(f"\nValidating position {count}")
            print(f"Turn: {turn}")
            print(f"FEN: {fen}")
            print(f"Moves line: {moves_line}")

            # Construct the full FEN string
            fen = fen + f" {turn} - - 0 1"
            moves = moves_line.replace("Moves: ", "")

            print(f"Constructed FEN: {fen}")
            print(f"Moves to validate: {moves}")

            # Validate the moves
            try:
                invalid_moves = validate_moves(fen, moves)
                if invalid_moves:
                    print(
                        f"Invalid moves found for FEN {fen}: {invalid_moves}")
                else:
                    print(f"All moves are valid for FEN {fen}.")
            except ValueError as e:
                print(f"Error validating FEN {fen}: {e}")

            # Notify if there are missing moves (i.e., not enough moves were found)
            board = chess.Board(fen)
            expected_move_count = board.legal_moves.count()
            provided_move_count = len(moves.split(", "))
            if provided_move_count < expected_move_count:
                missing_count = expected_move_count - provided_move_count
                print(
                    f"Warning: {missing_count} move(s) are missing for FEN {fen}.")


# Usage
file_path = "../../rust_generated_moves.txt"
parse_and_validate(file_path)
