use crate::types::{AllMovesGenRe, AvailableMovesMap,Board, Kind, Piece, PieceId, Team};
use crate::base_functions::{contains_element, find_overlap, find_non_overlap};
pub fn w_rook_pinning(board: &Board, pinning: PieceId, overlap: &Vec<usize>) ->Vec<Vec<usize>>{
    let w_pos = board.white_indexes.get_index(pinning).unwrap();
    let w_row = w_pos / 10;
    let w_col = w_pos % 10;
    let k_pos = board.black_indexes.get_index(PieceId::K).unwrap();
    let k_row = k_pos / 10;
    let k_col = k_pos % 10;
    let mut p_indexes = 69;
    let mut move_vector: Vec<usize> = vec![w_pos];

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
                    let mut re=overlap.clone();
                    if magnitude - i == 1 {
                        re.push((w_row as i32  + j)  as usize * 10 + w_col);
                    }
                    return vec![vec![], re];
                }
                Some(piece) if piece.team == Team::B && done => {
                    return vec![vec![], overlap.clone()];
                }
                Some(piece) if piece.team == Team::B && !done => {
                    p_indexes = (w_row as i32 + j) * 10 + w_col as i32;
                    done = true;
                }
                _ => move_vector.push((w_row as i32 + j) as usize * 10 + w_col as usize),
            }
        }

        if done {
            move_vector.push(p_indexes as usize);
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
                    let mut re=overlap.clone();
                    if magnitude - i == 1 {
                        re.push((w_row as i32 * 10 + w_col as i32 + j) as usize);
                    }
                    return vec![vec![], re];
                }
                Some(piece) if piece.team == Team::B && done => {
                    return vec![vec![], overlap.clone()];
                }
                Some(piece) if piece.team == Team::B && !done => {
                    p_indexes = w_row as i32 * 10 + w_col as i32;
                    done = true;
                }
                _ => move_vector.push((w_row as i32 * 10 + w_col as i32 + j) as usize),
            }
        }
        if done {
            move_vector.push(p_indexes as usize);
        }
    }
    return vec![move_vector, overlap.clone()]
}
pub fn b_rook_pinning(board: &Board, pinning: PieceId, overlap: &Vec<usize>) ->Vec<Vec<usize>>{
    let b_pos = board.black_indexes.get_index(pinning).unwrap();
    let b_row = b_pos / 10;
    let b_col = b_pos % 10;
    let k_pos = board.white_indexes.get_index(PieceId::K).unwrap();
    let k_row = k_pos / 10;
    let k_col = k_pos % 10;
    let mut p_indexes = 69;
    let mut move_vector: Vec<usize> = vec![b_pos];

    if b_col == k_col {
        // Column alignment
        let magnitude = (k_row as i32 - b_row as i32).abs();
        let direction = if k_row > b_row { 1 } else { -1 };
        let mut done = false;

        for i in 1..=magnitude {
            let j = i * direction;
            match board
                .full_board
                .get((b_row as i32 + j) as usize)
                .and_then(|r| r.get(b_col as usize))
            {
                Some(piece) if piece.team == Team::W && done => {
                    let mut re=overlap.clone();
                    if magnitude - i == 1 {
                        re.push((b_row as i32 + j) as usize * 10 + b_col);
                    }
                    return vec![vec![], re];
                }
                Some(piece) if piece.team == Team::B  => {
                    return vec![vec![], overlap.clone()];
                }
                Some(piece) if piece.team == Team::W && !done => {
                    p_indexes = (b_row as i32 + j) * 10 + b_col as i32;
                    done = true;
                }
                _ => move_vector.push((b_row as usize + j as usize) * 10 + b_col as usize),
            }
        }

        if done {
            move_vector.push(p_indexes as usize);
        }
    } else {
        // Row alignment
        let magnitude = (k_col as i32 - b_col as i32).abs();
        let direction = if k_col > b_col { 1 } else { -1 };
        let mut done = false;

        for i in 1..=magnitude {
            let j = i * direction;
            match board
                .full_board
                .get(b_row as usize)
                .and_then(|r| r.get((b_col as i32 + j) as usize))
            {
                Some(piece) if piece.team == Team::W && done=> {
                    p_indexes = b_row as i32 * 10 + b_col as i32 + j;
                    done=true;
                }
                Some(piece) if piece.team == Team::B && done => {
                    let mut re=overlap.clone();
                    if magnitude - i == 1 {
                        re.push(b_row * 10 + (b_col as i32 + j) as usize);
                    }
                    return vec![vec![], re];
                }
                Some(piece) if piece.team == Team::B && !done => {
                    return vec![vec![], overlap.clone()];
                }
                _ => move_vector.push(b_row * 10 + (b_col as i32 + j) as usize),
            }
        }

        if done {
            move_vector.push(p_indexes as usize);
        }
    }
    return vec![move_vector, overlap.clone()]
}


pub fn w_bishop_pinning(
    board:  &Board,
    pinning: PieceId,
    overlap:  &Vec<usize>,
) -> Vec<Vec<usize>> {
    let w_pos = board.white_indexes.get_index(pinning).unwrap();
    let w_row = w_pos / 10;
    let w_col = w_pos % 10;
    let k_pos = board.black_indexes.get_index(PieceId::K).unwrap();
    let k_row = k_pos / 10;
    let k_col = k_pos % 10;

    if (w_row as isize - k_row as isize).abs() != (w_col as isize - k_col as isize).abs()
        || w_row == k_row
        || w_col == k_col
    {
        return vec![vec![], overlap.clone()];
    }

    let mut p_indexes = 69;
    let mut move_vector = vec![w_pos];
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
                return vec![vec![], overlap.clone()];
            }

            if piece.team == Team::B && done {
                let mut re=overlap.clone();
                if magnitude - i == 1 {
                    re.push((w_row as isize + r_inc) as usize + (c_inc + w_col as isize) as usize);
                }
                return vec![vec![], re];
            } else if piece.team == Team::B && !done {
                p_indexes = 10 * (w_row as isize + r_inc) as usize + (w_col as isize + c_inc) as usize;
                done = true;
            } else {
                move_vector
                    .push(10 * (w_row as isize + r_inc) as usize + (w_col as isize + c_inc) as usize);
            }
        }
    }

    if !done {
        return vec![vec![], overlap.clone()];
    }

    // Assuming black_i_to_p maps from position to PieceId for black pieces
    if let Some(piece_id) = board.black_i_to_p.get_piece(p_indexes) {
        move_vector.push(piece_id as usize);
    }

    return vec![move_vector, overlap.clone()];
}

pub fn b_bishop_pinning(
    board: &Board,
    pinning: PieceId,
    overlap: &Vec<usize>,
) -> Vec<Vec<usize>> {
    let b_pos = board.black_indexes.get_index(pinning).unwrap();
    let b_row = b_pos / 10;
    let b_col = b_pos % 10;
    let k_pos = board.white_indexes.get_index(PieceId::K).unwrap();
    let k_row = k_pos / 10;
    let k_col = k_pos % 10;

    if (b_row as isize - k_row as isize).abs() != (b_col as isize - k_col as isize).abs()
        || (b_col as isize - k_col as isize) == 0
        || (b_row as isize - k_row as isize) == 0
    {
        return vec![vec![], overlap.clone()];
    }

    let mut p_index = 69;
    let mut move_vector = vec![b_pos];
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
                let mut re=overlap.clone();
                if magnitude - i == 1 {
                    re.push(r_inc as usize + c_inc as usize);
                }
                return vec![vec![], re];
            }

            if piece.team == Team::W && done {
                return vec![vec![], overlap.clone()];
            } else if piece.team == Team::W && !done {
                p_index = 10 * (b_row as isize + r_inc) + (b_col as isize + c_inc);
                done = true;
            } else {
                move_vector.push((10 * (b_row as isize + r_inc) + (b_col as isize + c_inc)) as usize);
            }
        }
    }

    // Assuming white_i_to_p maps from position to PieceId for white pieces
    if let Some(piece_id) = board.white_i_to_p.get_piece(p_index as usize) {
        move_vector.push(piece_id as usize);
    }

    return vec![move_vector, overlap.clone()]
}

pub fn in_check_knight_or_pawn(board: &mut Board, ret:&AllMovesGenRe, pressuring: PieceId, team: Team)->AvailableMovesMap {
    let mut re=AvailableMovesMap::new();
    match team {
        Team::W => {
            // If white king is in check, iterate over black pieces
            for piece_id in board.black_piece_ids.iter() {
                let moves = ret.black_moves.get_moves(*piece_id);
                    if contains_element(&moves, board.white_indexes.get_index(pressuring).unwrap()) && *piece_id!=PieceId::K {
                        re.insert_moves(*piece_id, &vec![board.white_indexes.get_index(pressuring).unwrap()]);
                    }
                }
            return re;
        }
        Team::B => {
            // If black king is in check, iterate over white pieces
            for piece_id in board.white_piece_ids.iter() {
                let moves = ret.white_moves.get_moves(*piece_id);
                    if contains_element(&moves, board.black_indexes.get_index(pressuring).unwrap()) {
                        // Restrict white's available moves to only the pressuring move
                        re.insert_moves(*piece_id, &vec![board.black_indexes.get_index(pressuring).unwrap()])
                    }
                }
            return re; 
            }
        _ => {return re}
    }
}



pub fn in_check_directional(board: &Board, re:&AllMovesGenRe, pressuring: PieceId, team: Team, direction: Vec<i32>) -> AvailableMovesMap {
    let king_index:usize;
    let pressuring_index:usize; 
    let old_av_map:AvailableMovesMap;
    if team==Team::W {
            king_index = board.black_indexes.get_index(PieceId::K).unwrap();
            pressuring_index = board.white_indexes.get_index(pressuring).unwrap();
            old_av_map =re.white_moves.clone();
    }
    else {
            king_index = board.black_indexes.get_index(PieceId::K).unwrap();
            pressuring_index = board.white_indexes.get_index(pressuring).unwrap();
            old_av_map =re.black_moves.clone();
        }

    let king_row = king_index / 10;
    let king_col = king_index % 10;
    let mut good_moves:Vec<usize> = vec![];
    good_moves.push(pressuring_index);
    
    let mut new_advantage_map=AvailableMovesMap::new();
    // Determine the range to check for good moves
    let mut p_row = pressuring_index as i32/ 10 + direction[0];
    let mut p_col = pressuring_index as i32% 10 + direction[1];

    while 0<=p_row && p_row<=7 && 0<=p_col && 7<=p_col
        && (p_row as usize != king_row || p_col as usize != king_col)
    {
        good_moves.push(10 * p_row as usize + p_col as usize );
        p_row += direction[0];
        p_col += direction[1];
    }

    // Update available moves based on good moves
        if team==Team::W{
            for i in board.white_piece_ids.iter(){
                if *i!=PieceId::K{
                    let new=find_overlap(&old_av_map.get_moves(*i), &good_moves);
                    new_advantage_map.insert_moves(*i, &new);
                }
            }
            new_advantage_map.insert_moves(PieceId::K, &find_non_overlap(re.white_moves.get_moves(PieceId::K), good_moves.clone()));

        }
        else{
            for i in board.black_piece_ids.iter(){
                if *i!=PieceId::K{
                    let new=find_overlap(&old_av_map.get_moves(*i), &good_moves.clone());
                   new_advantage_map.insert_moves(*i, &new);
                }
            }
            new_advantage_map.insert_moves(PieceId::K, &find_non_overlap(re.black_moves.get_moves(PieceId::K), good_moves.clone()));
        }
        return new_advantage_map;
    
    
}

pub fn ep_white(mut board: Board, move_piece: PieceId, indexes: usize)->Board {
    let initial_coords = board.white_indexes.get_index(move_piece).unwrap();
    let old_row = initial_coords / 10;
    let old_col = initial_coords % 10;
    let new_i = indexes;
    let new_col = new_i % 10;
    let new_row = new_i / 10;
    let captured_i = old_row * 10 + new_col as usize;
    let old_piece = board.black_i_to_p.get_piece(new_i%10+initial_coords/10).unwrap();
    board.black_i_to_p.nullify(captured_i);
    board.black_indexes.nullify(old_piece);
    if let Some(pos) = board.black_piece_ids.iter().position(|&id| id == old_piece) {
        board.black_piece_ids.remove(pos);
    }
    board.white_indexes.change_indexes(move_piece, new_i as usize);
    board.white_i_to_p.nullify(initial_coords);
    board.white_i_to_p.insert_piece(new_i, move_piece);

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
    return board;
}

pub fn ep_black(mut board: Board, move_piece: PieceId, indexes: usize)->Board {
    let initial_coords = board.black_indexes.get_index(move_piece).unwrap();
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
    if let Some(old_piece) = board.white_i_to_p.get_piece(captured_i) {
        board.white_indexes.nullify(old_piece);
        board.white_i_to_p.nullify(captured_i);
        if let Some(pos) = board.white_piece_ids.iter().position(|&id| id == old_piece) {
            board.white_piece_ids.remove(pos);
        }
    }

    // Update the moving black piece's position
    board.black_indexes.change_indexes(move_piece, new_i);
    board.black_i_to_p.nullify(initial_coords);
    board.black_i_to_p.insert_piece(new_i, move_piece);

    // Move the piece on the board and clear the old position
    board.full_board[new_row as usize][new_col as usize] =
        board.full_board[old_row as usize][old_col as usize].clone();
    board.full_board[old_row as usize][old_col as usize] = empty_piece;

    // Update points
    board.white_points -= 100;
    return board;
}