import os
import torch
import numpy as np
import chess
import chess.pgn
import chess.syzygy
import zstandard
import io
import gc
import boto3
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AWS S3 configuration
s3_bucket_name = 'chessengineaf'
s3 = boto3.client('s3')

# Paths to the PGN files in S3
pgn_files = [
    'lichess_db_standard_rated_2024-07.pgn.zst',
    'lichess_db_standard_rated_2024-06.pgn.zst'
]

# Directory to save batches locally before uploading to S3
local_output_dir = '/tmp/new_chess_batches_f32'
os.makedirs(local_output_dir, exist_ok=True)

# Specify the path to your Syzygy tablebases in S3
syzygy_path = '/tmp/syzygy'
os.makedirs(syzygy_path, exist_ok=True)

# Download Syzygy tablebase files from S3 to the local directory
syzygy_s3_path = 'syzygy/3-4-5/'
logger.info(
    f"Downloading Syzygy tablebase files from S3 bucket '{s3_bucket_name}'...")
for obj in s3.list_objects_v2(Bucket=s3_bucket_name, Prefix=syzygy_s3_path)['Contents']:
    s3.download_file(s3_bucket_name, obj['Key'], os.path.join(
        syzygy_path, os.path.basename(obj['Key'])))
logger.info("Syzygy tablebase files downloaded.")

# Create a Syzygy tablebase object
tablebases = chess.syzygy.open_tablebase(syzygy_path)

# Load all tablebases in the specified directory
tablebases.add_directory(syzygy_path)

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
    return torch.tensor(tensor, dtype=torch.float32)


def get_dtm(board, move_count):
    try:
        dtm_value = tablebases.probe_dtz(board)
    except KeyError:
        # Fallback if the tablebase does not contain this position
        dtm_value = move_count
    return torch.tensor([dtm_value], dtype=torch.float32)


def extract_game_states(game, result):
    board = game.board()
    states = []
    move_count = 0
    for move in game.mainline_moves():
        move_count += 1
        board.push(move)
        label = torch.tensor(
            [1.0 if result == 1 else 0.0], dtype=torch.float32)
        dtm_value = get_dtm(board, move_count)
        states.append((board_to_tensor(board), label, dtm_value))
    logger.debug(
        f"Extracted {len(states)} game states from game with result {result}.")
    return states


def save_batch(batch, batch_num):
    local_file_path = os.path.join(local_output_dir, f'batch_{batch_num}.pt')
    torch.save(batch, local_file_path)
    s3.upload_file(local_file_path, s3_bucket_name,
                   f'chess_batches_f32/batch_{batch_num}.pt')
    os.remove(local_file_path)  # Remove local file after upload to save space
    logger.info(f"Saved and uploaded batch {batch_num} to S3.")


def process_pgn_file(pgn_file_key, batch_num_start=135127, is_compressed=False):
    batch_size = 100
    batch = []
    batch_num = batch_num_start

    logger.info(f"Processing PGN file: {pgn_file_key}")
    response = s3.get_object(Bucket=s3_bucket_name, Key=pgn_file_key)
    if is_compressed:
        dctx = zstandard.ZstdDecompressor()
        reader = dctx.stream_reader(response['Body'])
        text_stream = io.TextIOWrapper(reader, encoding='utf-8')
        while True:
            line = text_stream.readline()
            if not line:
                break
            pgn_stream = io.StringIO(line)
            game = chess.pgn.read_game(pgn_stream)
            if game is None:
                continue
            result = game.headers["Result"]
            if result == "1-0":
                label = 1  # White wins
            elif result == "0-1":
                label = 0  # Black wins
            else:
                continue  # Ignore draws
            batch.extend(extract_game_states(game, label))
            if len(batch) >= batch_size:
                save_batch(batch, batch_num)
                batch = []
                batch_num += 1
                gc.collect()
    else:
        with io.TextIOWrapper(response['Body'], encoding='latin1') as pgn_file:
            while True:
                line = pgn_file.readline()
                if not line:
                    break
                pgn_stream = io.StringIO(line)
                game = chess.pgn.read_game(pgn_stream)
                if game is None:
                    continue
                result = game.headers["Result"]
                if result == "1-0":
                    label = 1  # White wins
                elif result == "0-1":
                    label = 0  # Black wins
                else:
                    continue  # Ignore draws
                batch.extend(extract_game_states(game, label))
                if len(batch) >= batch_size:
                    save_batch(batch, batch_num)
                    batch = []
                    batch_num += 1
                    gc.collect()

    # Save any remaining states in the last batch
    if batch:
        save_batch(batch, batch_num)
        gc.collect()

    logger.info(
        f"Finished processing PGN file: {pgn_file_key}, batches starting from {batch_num_start}.")


# Process each PGN file sequentially
for pgn_file in pgn_files:
    is_compressed = pgn_file.endswith('.zst')
    process_pgn_file(pgn_file, is_compressed=is_compressed)

print(f"Successfully processed and saved game states in batches.")
