use crate::base_move_functions::generate_knight_moves;
use crate::types::{Board, Kind, Piece, PieceId, Team};
pub fn king_move_eliminator_white(board: Board) {
    let mut king_moves=board.

        // Remove illegal moves
        king_moves.retain(|i| !to_remove.contains(i));
    }
}

pub fn king_move_eliminator_black(mut board: Board) {
    // Directly reference needed parts of the board to simplify access
    let white_i_to_p = &board.white_i_to_p; // Mapping from position to piece ID for black pieces
    let white_indexes = &board.white_indexes; // Positions of black pieces

    if let Some(king_moves) = board.black_available_moves.get_mut(&PieceId::K) {
        let mut to_remove = Vec::new(); // Collect moves to remove

        for &move_pos in king_moves.iter() {
            // Calculate potential threat positions
            let threat_pos_down_left = move_pos - 11;
            let threat_pos_down_right = move_pos - 9;
            let threat_pos_down_left = threat_pos_down_left as usize;
            let threat_pos_down_right = threat_pos_down_right as usize;
            // Check for pawn threats
            if let Some(&piece_id) = board.black_i_to_p.get(&threat_pos_down_left) {
                if matches!(
                    piece_id,
                    PieceId::P1
                        | PieceId::P2
                        | PieceId::P3
                        | PieceId::P4
                        | PieceId::P5
                        | PieceId::P6
                        | PieceId::P7
                        | PieceId::P8
                ) {
                    to_remove.push(move_pos);
                }
            }
            if let Some(&piece_id) = white_i_to_p.get(&threat_pos_down_right) {
                if matches!(
                    piece_id,
                    PieceId::P1
                        | PieceId::P2
                        | PieceId::P3
                        | PieceId::P4
                        | PieceId::P5
                        | PieceId::P6
                        | PieceId::P7
                        | PieceId::P8
                ) {
                    to_remove.push(move_pos);
                }
            }

            // Proximity to black king
            if let Some(&king_pos) = board.black_indexes.get(&PieceId::K) {
                let i32 
                if (king_pos % 10 - move_pos % 10).abs() <= 1
                    && (king_pos / 10 - move_pos / 10).abs() <= 1
                {
                    to_remove.push(move_pos);
                }
            }

            // Check knight threats
            let piece_pos_row = (move_pos / 10) as usize;
            let piece_pos_col = (move_pos % 10) as usize;
            if piece_pos_row < 8 {
                let moves = generate_knight_moves(board, piece_pos_row, piece_pos_col);
                for &knight_move in &moves {
                    if let Some(&k1_index) = white_indexes.get(&PieceId::K1) {
                        if knight_move == k1_index {
                            to_remove.push(move_pos);
                            break; // Avoid multiple insertions
                        }
                    }
                    if let Some(&k2_index) = white_indexes.get(&PieceId::K2) {
                        if knight_move == k2_index {
                            to_remove.push(move_pos);
                            break; // Avoid multiple insertions
                        }
                    }
                }
            }
        }

        // Remove identified threats from king's available moves
        king_moves.retain(|&pos| !to_remove.contains(&pos));
    }
}

fn w_rook_pinning(mut board: Board, pinning: PieceId, mut overlap: Vec<i32>) {
    let w_pos = board.white_indexes.get(&pinning).unwrap_or(&0);
    let w_row = w_pos / 10;
    let w_col = w_pos % 10;
    let k_pos = board.black_indexes.get(&PieceId::K).unwrap_or(&0);
    let k_row = k_pos / 10;
    let k_col = k_pos % 10;
    let mut p_indexes = -1;
    let mut move_vector: Vec<i32> = vec![*w_pos];

    if w_col == k_col {
        // Column alignment
        let magnitude = (k_row as i32 - w_row as i32).abs();
        let direction = if k_row > w_row { 1 } else { -1 };
        let mut done = false;

        for i in 1..=magnitude {
            let j = i * direction;
            match board
                .full_board
                .get((w_row as i32 + j) as usize)
                .and_then(|r| r.get(w_col as usize))
            {
                Some(piece) if piece.team == Team::W => {
                    if magnitude - i == 1 {
                        overlap.push((w_row as i32 + j) * 10 + w_col as i32);
                    }
                    return;
                }
                Some(piece) if piece.team == Team::B && done => {
                    return;
                }
                Some(piece) if piece.team == Team::B && !done => {
                    p_indexes = (w_row as i32 + j) * 10 + w_col as i32;
                    done = true;
                }
                _ => move_vector.push((w_row as i32 + j) * 10 + w_col as i32),
            }
        }

        if done {
            move_vector.push(p_indexes);
        }
    } else {
        // Row alignment
        let magnitude = (k_col as i32 - w_col as i32).abs();
        let direction = if k_col > w_col { 1 } else { -1 };
        let mut done = false;

        for i in 1..=magnitude {
            let j = i * direction;
            match board
                .full_board
                .get(w_row as usize)
                .and_then(|r| r.get((w_col as i32 + j) as usize))
            {
                Some(piece) if piece.team == Team::W => {
                    if magnitude - i == 1 {
                        overlap.push(w_row as i32 * 10 + w_col as i32 + j);
                    }
                    return;
                }
                Some(piece) if piece.team == Team::B && done => {
                    return;
                }
                Some(piece) if piece.team == Team::B && !done => {
                    p_indexes = w_row as i32 * 10 + w_col as i32 + j;
                    done = true;
                }
                _ => move_vector.push(w_row as i32 * 10 + w_col as i32 + j),
            }
        }

        if done {
            move_vector.push(p_indexes);
        }
    }

    // Assuming the `moveVector` somehow influences the white_available_moves
    // This could be a complex integration depending on your data structures
    // For simplicity, assuming we just update the moves for `pinning` PieceId
    if done{
        if let Some(moves) = board.white_available_moves.get_mut(&pinning) {
            *moves = move_vector;
        }
    }
}

// pub fn w_rook_pinning(mut board: Board, pinning: PieceId, mut overlap: Vec<i32>) {
//     let w_pos = board.white_indexes.get(&pinning).unwrap_or(&0);
//     let w_row = w_pos / 10;
//     let w_col = w_pos % 10;
//     let k_pos = board.black_indexes.get(&PieceId::K).unwrap_or(&0);
//     let k_row = k_pos / 10;
//     let k_col = k_pos % 10;
//     let mut p_index = -1;
//     let mut move_vector: Vec<i32> = vec![*w_pos];

//     if w_col == k_col {
//         // Column alignment
//         let magnitude = (k_row as i32 - w_row as i32).abs();
//         let direction = if k_row > w_row { 1 } else { -1 };
//         let mut done = false;

//         for i in 1..=magnitude {
//             let j = i * direction;
//             match board
//                 .full_board
//                 .get((w_row as i32 + j) as usize)
//                 .and_then(|r| r.get(w_col as usize))
//             {
//                 Some(piece) if piece.team == Team::W => {
//                     if magnitude - i == 1 {
//                         overlap.push((w_row as i32 + j) * 10 + w_col as i32);
//                     }
//                     return;
//                 }
//                 Some(piece) if piece.team == Team::B && done => {
//                     return;
//                 }
//                 Some(piece) if piece.team == Team::B && !done => {
//                     p_index = (w_row as i32 + j) * 10 + w_col as i32;
//                     done = true;
//                 }
//                 _ => move_vector.push((w_row as i32 + j) * 10 + w_col as i32),
//             }
//         }

//         if done {
//             move_vector.push(p_index);
//         }
//     } else {
//         // Row alignment
//         let magnitude = (k_col as i32 - w_col as i32).abs();
//         let direction = if k_col > w_col { 1 } else { -1 };
//         let mut done = false;

//         for i in 1..=magnitude {
//             let j = i * direction;
//             match board
//                 .full_board
//                 .get(w_row as usize)
//                 .and_then(|r| r.get((w_col as i32 + j) as usize))
//             {
//                 Some(piece) if piece.team == Team::W => {
//                     if magnitude - i == 1 {
//                         overlap.push(w_row as i32 * 10 + w_col as i32 + j);
//                     }
//                     return;
//                 }
//                 Some(piece) if piece.team == Team::B && done => {
//                     return;
//                 }
//                 Some(piece) if piece.team == Team::B && !done => {
//                     p_index = w_row as i32 * 10 + w_col as i32 + j;
//                     done = true;
//                 }
//                 _ => move_vector.push(w_row as i32 * 10 + w_col as i32 + j),
//             }
//         }

//         if done {
//             move_vector.push(p_index);
//         }
//     }

//     // Assuming the `moveVector` somehow influences the white_available_moves
//     // This could be a complex integration depending on your data structures
//     // For simplicity, assuming we just update the moves for `pinning` PieceId
//     if done {
//         if let Some(moves) = board.white_available_moves.get_mut(&pinning) {
//             *moves = move_vector;
//         }
//     }
// }

pub fn w_bishop_pinning(
    board: &mut Board,
    pinning: PieceId,
    overlap: &mut Vec<i32>,
) -> (Vec<i32>, Vec<i32>) {
    let w_pos = board.white_indexes.get(&pinning).unwrap();
    let w_row = w_pos / 10;
    let w_col = w_pos % 10;
    let k_pos = board.black_indexes.get(&PieceId::K).unwrap();
    let k_row = k_pos / 10;
    let k_col = k_pos % 10;

    if (w_row as isize - k_row as isize).abs() != (w_col as isize - k_col as isize).abs()
        || w_row == k_row
        || w_col == k_col
    {
        return (vec![], overlap.clone());
    }

    let mut p_indexes = -1;
    let mut move_vector = vec![*w_pos];
    let magnitude = (k_row as isize - w_row as isize).abs() as usize;
    let direction_r = if k_row > w_row { 1 } else { -1 };
    let direction_c = if k_col > w_col { 1 } else { -1 };
    let mut done = false;

    for i in 1..=magnitude {
        let r_inc = (i as isize) * direction_r;
        let c_inc = (i as isize) * direction_c;

        if let Some(piece) = board
            .full_board
            .get((w_row as isize + r_inc) as usize)
            .and_then(|row| row.get((w_col as isize + c_inc) as usize))
        {
            if piece.team == Team::W {
                if magnitude - i == 1 {
                    overlap.push(r_inc + c_inc);
                }
                return (vec![], overlap.clone());
            }

            if piece.team == Team::B && done {
                return (vec![], overlap.clone());
            } else if piece.team == Team::B && !done {
                p_indexes = 10 * (w_row as isize + r_inc) as i32 + (w_col as isize + c_inc) as i32;
                done = true;
            } else {
                move_vector
                    .push(10 * (w_row as isize + r_inc) as i32 + (w_col as isize + c_inc) as i32);
            }
        }
    }

    if !done {
        return (vec![], overlap.clone());
    }

    // Assuming black_i_to_p maps from position to PieceId for black pieces
    if let Some(&piece_id) = board.black_i_to_p.get(&p_indexes) {
        move_vector.push(piece_id as i32);
    }

    (move_vector, overlap.clone())
}

pub fn b_bishop_pinning(
    mut board: Board,
    pinning: PieceId,
    mut overlap: Vec<usize>,
) -> (Vec<usize>, Vec<usize>) {
    let b_pos = board.black_indexes.get(&pinning).unwrap();
    let b_row = b_pos / 10;
    let b_col = b_pos % 10;
    let k_pos = board.white_indexes.get(&PieceId::K).unwrap();
    let k_row = k_pos / 10;
    let k_col = k_pos % 10;

    if (b_row as isize - k_row as isize).abs() != (b_col as isize - k_col as isize).abs()
        || (b_col as isize - k_col as isize) == 0
        || (b_row as isize - k_row as isize) == 0
    {
        return (vec![], overlap.clone());
    }

    let mut p_index = -1;
    let mut move_vector = vec![*b_pos];
    let magnitude = (k_row as isize - b_row as isize).abs() as usize;
    let direction_r = if k_row > b_row { 1 } else { -1 };
    let direction_c = if k_col > b_col { 1 } else { -1 };
    let mut done = false;

    for i in 1..=magnitude {
        let r_inc = (i as isize) * direction_r;
        let c_inc = (i as isize) * direction_c;

        if let Some(piece) = board
            .full_board
            .get((b_row as isize + r_inc) as usize)
            .and_then(|row| row.get((b_col as isize + c_inc) as usize))
        {
            if piece.team == Team::B {
                if magnitude - i == 1 {
                    overlap.push((r_inc + c_inc as usize);
                }
                return (vec![], overlap.clone());
            }

            if piece.team == Team::W && done {
                return (vec![], overlap.clone());
            } else if piece.team == Team::W && !done {
                p_index = 10 * (b_row as isize + r_inc) + (b_col as isize + c_inc);
                done = true;
            } else {
                move_vector.push(10 * (b_row as isize + r_inc) + (b_col as isize + c_inc));
            }
        }
    }

    // Assuming white_i_to_p maps from position to PieceId for white pieces
    if let Some(&piece_id) = board.white_i_to_p.get(&p_index) {
        move_vector.push(piece_id as i32);
    }

    (move_vector, overlap.clone())
}

fn in_check_knight_or_pawn(board: &mut Board, pressuring: PieceId, team: Team) {
    match team {
        Team::W => {
            // If white king is in check, iterate over black pieces
            for &piece_id in &board.black_pieces {
                if let Some(moves) = board.black_available_moves.get_mut(&piece_id) {
                    if moves.contains(&board.white_indexes[&pressuring]) {
                        // Restrict black's available moves to only the pressuring move
                        *moves = vec![board.white_indexes[&pressuring]];
                    } else {
                        // Clear available moves if they don't pressure the white king
                        moves.clear();
                    }
                }
            }
        }
        Team::B => {
            // If black king is in check, iterate over white pieces
            for &piece_id in &board.white_pieces {
                if let Some(moves) = board.white_available_moves.get_mut(&piece_id) {
                    if moves.contains(&board.black_indexes[&pressuring]) {
                        // Restrict white's available moves to only the pressuring move
                        *moves = vec![board.black_indexes[&pressuring]];
                    } else {
                        // Clear available moves if they don't pressure the black king
                        moves.clear();
                    }
                }
            }
        }
        _ => {}
    }
}

fn in_check_directional(board: &mut Board, pressuring: PieceId, team: Team, direction: (i32, i32)) {
    let (king_index, pressuring_index, available_moves) = match team {
        Team::W => {
            let king_index = board.white_indexes[&PieceId::K];
            let pressuring_index = board.black_indexes[&pressuring];
            (
                &mut board.white_available_moves,
                king_index,
                pressuring_index,
            )
        }
        Team::B => {
            let king_index = board.black_indexes[&PieceId::K];
            let pressuring_index = board.white_indexes[&pressuring];
            (
                &mut board.black_available_moves,
                king_index,
                pressuring_index,
            )
        }
        _ => return,
    };

    let king_row = king_index / 10;
    let king_col = king_index % 10;
    let mut good_moves = Vec::new();
    good_moves.push(pressuring_index);

    // Determine the range to check for good moves
    let mut p_row = pressuring_index / 10 + direction.0;
    let mut p_col = pressuring_index % 10 + direction.1;

    while (0..8).contains(&p_row)
        && (0..8).contains(&p_col)
        && (p_row != king_row || p_col != king_col)
    {
        good_moves.push(10 * p_row + p_col);
        p_row += direction.0;
        p_col += direction.1;
    }

    // Update available moves based on good moves
    for (piece_id, moves) in available_moves.iter_mut() {
        if piece_id != &PieceId::K {
            moves.retain(|&m| good_moves.contains(&m));
        }
    }

    // Special handling for king's available moves
    if let Some(king_moves) = available_moves.get_mut(&PieceId::K) {
        king_moves.retain(|&m| !good_moves.contains(&m));
    }
}

pub fn ep_white(board: Board, move_piece: PieceId, indexes: usizerust) {
    let initial_coords = *board.white_indexes.get(&move_piece).unwrap_or(&0);
    let old_row = initial_coords / 10;
    let old_col = initial_coords % 10;
    let new_i = indexes;
    let new_col = new_i % 10;
    let new_row = new_i / 10;
    let captured_i = old_row * 10 + new_col as usize;
    let old_piece = *board
        .black_i_to_p
        .get(&captured_i)
        .unwrap_or(&PieceId::Error); // Assuming default

    board.black_available_moves.remove(&old_piece);
    board.black_i_to_p.remove(&captured_i);
    board.black_indexes.remove(&old_piece);
    if let Some(pos) = board.black_piece_ids.iter().position(|&id| id == old_piece) {
        board.black_piece_ids.remove(pos);
    }
    board.white_indexes.insert(move_piece, new_i as usize);
    board.white_i_to_p.remove(&initial_coords);
    board.white_i_to_p.insert(new_i as usize, move_piece);

    // Assuming `Piece` struct has a constructor or a default value
    let empty_piece = Piece {
        team: Team::N,
        kind: Kind::Empty,
        value: 0,
    };

    // Move the piece on the board
    board.full_board[new_row as usize][new_col as usize] =
        board.full_board[old_row as usize][old_col as usize].clone();
    board.full_board[old_row as usize][old_col as usize] = empty_piece.clone();
    board.full_board[old_row as usize][new_col as usize] = empty_piece;

    // Update points
    board.black_points -= 100;
    reset_board(board);
}

pub fn ep_black(board: &mut Board, move_piece: PieceId, indexes: i32) {
    let initial_coords = *board.black_indexes.get(&move_piece).unwrap_or(&0);
    let old_row = initial_coords / 10;
    let old_col = initial_coords % 10;
    let new_i = indexes;
    let new_col = new_i % 10;
    let new_row = new_i / 10;
    let captured_i = old_row * 10 + new_col;

    // Assuming Piece has a default constructor or a method to create an empty piece
    let empty_piece = Piece {
        team: Team::N,
        kind: Kind::Empty,
        value: 0,
    };

    // Remove the piece captured by en passant
    board.full_board[captured_i / 10][captured_i % 10] = empty_piece.clone();

    // Update white piece information
    if let Some(old_piece) = board.white_i_to_p.remove(&captured_i) {
        board.white_available_moves.remove(&old_piece);
        board.white_indexes.remove(&old_piece);
        if let Some(pos) = board.white_piece_ids.iter().position(|&id| id == old_piece) {
            board.white_piece_ids.remove(pos);
        }
    }

    // Update the moving black piece's position
    board.black_indexes.insert(move_piece, new_i);
    board.black_i_to_p.remove(&initial_coords);
    board.black_i_to_p.insert(new_i, move_piece);

    // Move the piece on the board and clear the old position
    board.full_board[new_row as usize][new_col as usize] =
        board.full_board[old_row as usize][old_col as usize].clone();
    board.full_board[old_row as usize][old_col as usize] = empty_piece;

    // Update points
    board.white_points -= 100;
    reset_board(board);
}
pub fn reset_board(board: &mut Board) {
    let board_blk_ids = board.black_piece_ids;
    for i in board_blk_ids{
        board.black_available_moves.insert(i, vec![]);
    }
    let board_wht_ids = board.white_piece_ids;

    for i in board_wht_ids{
        board.white_available_moves.insert(i, vec![]);
    }
}
