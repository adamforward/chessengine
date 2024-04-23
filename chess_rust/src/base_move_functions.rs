use crate::base_functions::{primes, primes1};
use crate::types::{Board, Kind, Team};
pub fn generate_available_moves(board:&Board, row: usize, col: usize) -> Vec<usize> {
    let piece = &board.full_board[row][col];

    match piece.kind {
        Kind::Pawn => match piece.team {
            Team::B => generate_pawn_moves_black(board, row, col),
            Team::W => generate_pawn_moves_white(board, row, col),
            _ => Vec::new(),
        },
        Kind::Knight => generate_knight_moves(board, row, col),
        Kind::Bishop => generate_bishop_moves(board, row, col),
        Kind::Rook => generate_rook_moves(board, row, col),
        Kind::King => generate_king_moves(board, row, col),
        Kind::Queen => {
            let mut moves = generate_rook_moves(board, row, col);
            moves.extend(generate_bishop_moves(board, row, col));
            moves
        }
        _ => Vec::new(),
    }
}
fn generate_pawn_moves_black(board: &Board, row: usize, col: usize) -> Vec<usize> {
    let mut moves = Vec::new();

    // Move forward 1
    if board.full_board[row + 1][col].team == Team::N {
        moves.push((row + 1) * 10 + col);
        // Move forward 2 if on starting row
        if row != 6 {
            if row == 1 && board.full_board[row + 2][col].team == Team::N {
                moves.push(30 + col);
            }
        }
    }

    // Capture diagonally left
    if col > 0 {
        if board.full_board[row + 1][col - 1].team == Team::W {
            moves.push((row + 1) * 10 + col - 1);
        }
        if row == 4
            && board.full_board[4][col - 1].team == Team::W
            && board.full_board[4][col - 1].kind == Kind::Pawn
        {
            let index_ref = row + col;
            let pawn = board.black_i_to_p.get_piece(index_ref as usize).unwrap();
            let p1 = primes1(pawn);
            if board.white_prime % primes(col - 1) == 0 && board.black_prime1 % p1 == 0 {
                let ep = 20 + col - 1;
                moves.push(ep);
            }
        }
    }
    // Capture diagonally right
    if col < 7 {
        if board.full_board[row + 1][col + 1].team == Team::W {
            moves.push((row + 1) * 10 + col + 1);
        }
        if row == 4
            && board.full_board[4][col + 1].team == Team::W
            && board.full_board[4][col + 1].kind == Kind::Pawn
        {
            // Add the prime number checks here, similar to your Python code
            let index_ref = row + col;
            let pawn = board.black_i_to_p.get_piece(index_ref).unwrap();
            let p1 = primes1(pawn);
            if board.white_prime % primes(col + 1) == 0 && board.black_prime1 % p1 == 0 {
                let ep = 20 + col + 1;
                moves.push(ep);
            }
        }
    }
    return moves;
}
pub fn generate_pawn_moves_white(board: &Board, row: usize, col: usize) -> Vec<usize> {
    let mut moves = Vec::new();

    // Move forward 1
    if board.full_board[row - 1][col].team == Team::N {
        moves.push((row - 1) * 10 + col);
        // Move forward 2 if on starting row
        if row != 1 {
            if row == 6 && board.full_board[row - 2][col].team == Team::N {
                moves.push(40 + col);
            }
        }
    }

    // Capture diagonally left
    if col > 0 {
        if board.full_board[row - 1][col - 1].team == Team::B {
            moves.push((row - 1) * 10 + col - 1);
        }
        if row == 3
            && board.full_board[3][col - 1].team == Team::B
            && board.full_board[3][col - 1].kind == Kind::Pawn
        {
            let index_ref = row + col;
            let pawn = board.white_i_to_p.get_piece(index_ref).unwrap();
            let p1 = primes1(pawn);
            if board.black_prime % primes(col - 1) == 0 && board.white_prime1 % p1 == 0 {
                let ep = 20 + col - 1;
                moves.push(ep);
            }
        }
    }
    // Capture diagonally right
    if col < 7 {
        if board.full_board[row - 1][col + 1].team == Team::B {
            moves.push((row - 1) * 10 + col + 1);
        }
        if row == 3
            && board.full_board[3][col + 1].team == Team::B
            && board.full_board[3][col + 1].kind == Kind::Pawn
        {
            let index_ref = row + col;
            let pawn = board.white_i_to_p.get_piece(index_ref).unwrap();
            let p1 = primes1(pawn);
            // Add the prime number checks here, similar to your Python code
            if board.black_prime % primes(col + 1) == 0 && board.white_prime1 % p1 == 0 {
                let ep = 20 + col + 1;
                moves.push(ep);
            }
        }
    }
    return moves;
}
pub fn generate_knight_moves(board: &Board, row: usize, col: usize) -> Vec<usize> {
    let mut moves = Vec::new();
    let team = board.full_board[row][col].team;

    // Possible knight moves in L shape
    let knight_moves = [
        (1, 2),
        (1, -2),
        (-1, 2),
        (-1, -2),
        (2, 1),
        (2, -1),
        (-2, 1),
        (-2, -1),
    ];

    for (i, j) in knight_moves.iter() {
        let move_row = row as i32 + i;
        let move_col = col as i32 + j;

        // Check if the move is within the bounds of the board
        if move_row >= 0 && move_row < 8 && move_col >= 0 && move_col < 8 {
            let move_row = move_row as usize;
            let move_col = move_col as usize;

            // Check if the destination square is not occupied by a piece of the same team
            if board.full_board[move_row][move_col].team != team {
                moves.push(move_row * 10 + move_col as usize);
            }
        }
    }

    moves
}
pub fn generate_bishop_moves(board: &Board, row: usize, col: usize) -> Vec<usize> {
    let mut moves = Vec::new();
    let team = board.full_board[row][col].team;

    let directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

    for (dir_row, dir_col) in directions.iter() {
        let (mut move_row, mut move_col) = (row as i32 + dir_row, col as i32 + dir_col);

        // Traverse diagonally until you hit an obstacle
        while move_row >= 0 && move_row < 8 && move_col >= 0 && move_col < 8 {
            let (move_row_usize, move_col_usize) = (move_row as usize, move_col as usize);

            // If the square is occupied by a piece of the same team, break the loop
            if board.full_board[move_row_usize][move_col_usize].team == team {
                break;
            }

            // Add the move and check if the square is occupied by an opposing piece
            moves.push(move_row_usize * 10 + move_col_usize as usize);
            if board.full_board[move_row_usize][move_col_usize].team != Team::N {
                // If there's a piece, the bishop can't move past it
                break;
            }

            // Continue in the same diagonal direction
            move_row += dir_row;
            move_col += dir_col;
        }
    }

    moves
}
pub fn generate_king_moves(board: &Board, row: usize, col: usize) -> Vec<usize> {
    let mut moves = Vec::new();
    let team = board.full_board[row][col].team;
    let offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];

    for (offset_row, offset_col) in offsets.iter() {
        // Calculate potential move positions with signed arithmetic
        let move_row = row as i32 + offset_row;
        let move_col = col as i32 + offset_col;

        // Check if the move is within the bounds of the board using signed comparisons
        if move_row >= 0 && move_row < 8 && move_col >= 0 && move_col < 8 {
            // Safely convert back to usize for indexing
            let move_row_usize = move_row as usize;
            let move_col_usize = move_col as usize;

            // Check if the target square is not occupied by a piece from the same team
            if board.full_board[move_row_usize][move_col_usize].team != team {
                moves.push(move_row_usize * 8 + move_col_usize); // Assuming 8x8 board for index calculation
            }
        }
    }

    moves
}
pub fn generate_rook_moves(board: &Board, row: usize, col: usize) -> Vec<usize> {
    let mut moves = Vec::new();
    let team = board.full_board[row][col].team;

    // Vertical and horizontal moves
    let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)];

    for (dir_row, dir_col) in directions.iter() {
        let (mut move_row, mut move_col) = (row as i32, col as i32);

        // Traverse in each direction
        loop {
            move_row += dir_row;
            move_col += dir_col;

            // Check if the current position is within the bounds of the board
            if move_row < 0 || move_row > 7 || move_col < 0 || move_col > 7 {
                break;
            }

            let (move_row_usize, move_col_usize) = (move_row as usize, move_col as usize);

            // If the square is occupied by a piece of the same team, break the loop
            if board.full_board[move_row_usize][move_col_usize].team == team {
                break;
            }

            // Add the move and check if the square is occupied by an opposing piece
            moves.push(move_row_usize * 10 + move_col_usize);
            if board.full_board[move_row_usize][move_col_usize].team != Team::N {
                // If there's a piece, the rook can't move past it
                break;
            }
        }
    }

    moves
}
