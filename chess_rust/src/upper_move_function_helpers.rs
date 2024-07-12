use crate::types::{AllMovesGenRe, AvailableMovesMap,Board, Kind, Piece, PieceId, Team};
use crate::base_functions::{find_overlap};
pub fn w_rook_pinning(board: &Board, pinning: PieceId, overlap: &Vec<usize>) ->Vec<Vec<usize>>{
    let w_pos = board.white_indexes.get_index(pinning).unwrap();
    let w_row = w_pos / 10;
    let w_col = w_pos % 10;
    let k_pos = board.black_indexes.get_index(PieceId::K).unwrap();
    let k_row = k_pos / 10;
    let k_col = k_pos % 10;
    let mut p_indexes = 69;
    let mut move_vector: Vec<usize> = vec![w_pos];
    //here we're iterating over the vector between the king and the pinning piece. 
    //checking for the existence of 1 black piece in this vector, if there's one white piece or 2 black pieces it's not pinned. 
    //Also, if the only piece in between the pinning piece and the king is a black piece adjacent to the king, 
    //This piece needs to be removed from the king's available moves. 

    if w_col == k_col {
        // Column alignment
        let magnitude = (k_row as i32 - w_row as i32).abs();
        let direction = if k_row > w_row { 1 } else { -1 };
        let mut done = false;

        for i in 1..magnitude {
            let j = i * direction;
            match board
                .full_board
                .get((w_row as i32 + j) as usize)
                .and_then(|r| r.get(w_col as usize))
            {
                Some(piece) if piece.team == Team::W => {
                    let mut re=overlap.clone();
                    if magnitude - i == 1 && !done{
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
        if !done{
            return vec![vec![], overlap.clone()];
        }
        if done {
            move_vector.push(p_indexes as usize);
        }
         return vec![move_vector, overlap.clone()];
    } else if w_row==k_row {
        // Row alignment
        let magnitude = (k_col as i32 - w_col as i32).abs();
        let direction = if k_col > w_col { 1 } else { -1 };
        let mut done = false;

        for i in 1..magnitude {
            let j = i * direction;
            match board
                .full_board
                .get(w_row as usize)
                .and_then(|r| r.get((w_col as i32 + j) as usize))
            {
                Some(piece) if piece.team == Team::W => {
                    let mut re=overlap.clone();
                    if magnitude - i == 1 && !done{
                        re.push(((w_row as i32 * 10) + (w_col as i32 + j)) as usize);
                    }
                    return vec![vec![], re];
                }
                Some(piece) if piece.team == Team::B && done => {
                    return vec![vec![], overlap.clone()];
                }
                Some(piece) if piece.team == Team::B && !done => {
                    p_indexes = w_row as i32 * 10 + (w_col as i32+j);
                    done = true;
                }
                _ => move_vector.push((w_row as i32 * 10 + w_col as i32 + j) as usize),
            }
        }
        if done {
            move_vector.push(p_indexes as usize);
        }
        if !done{
            return vec![vec![], overlap.clone()];
        }
         return vec![move_vector, overlap.clone()];
    }
    else{
        return vec![vec![], overlap.clone()];
    }
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

        for i in 1..magnitude-1 {
            let j = i * direction;
            match board
                .full_board
                .get((b_row as i32 + j) as usize)
                .and_then(|r| r.get(b_col as usize))
            {
                Some(piece) if piece.team == Team::W && done => {
                    return vec![vec![], overlap.clone()];
                }
                Some(piece) if piece.team == Team::B  => {
                    let mut re=overlap.clone();
                    if magnitude - i == 1 && !done{
                        re.push((b_row as i32 + j) as usize * 10 + b_col);
                    }
                    return vec![vec![], re];
                }
                Some(piece) if piece.team == Team::W && !done => {
                    p_indexes = (b_row as i32 + j) * 10 + b_col as i32;
                    done = true;
                }
                _ => move_vector.push((b_row as isize + j as isize) as usize * 10  + b_col as usize),
            }
        }
        if !done{
            return vec![vec![], overlap.clone()];
        }

        if done {
            move_vector.push(p_indexes as usize);
        }
        return vec![move_vector, overlap.clone()]
    } else if b_row==k_row{
        // Row alignment
        let magnitude = (k_col as i32 - b_col as i32).abs();
        let direction = if k_col > b_col { 1 } else { -1 };
        let mut done = false;

        for i in 1..magnitude {
            let j = i * direction;
            match board
                .full_board
                .get(b_row as usize)
                .and_then(|r| r.get((b_col as i32 + j) as usize))
            {
                Some(piece) if piece.team == Team::W && !done=> {
                    p_indexes = b_row as i32 * 10 + b_col as i32 + j;
                    done=true;
                }
                Some(piece) if piece.team == Team::B => {
                    let mut re=overlap.clone();
                    if magnitude - i == 1 && !done {
                        re.push((b_row as i32 + j as i32) as usize * 10 + b_col);
                    }
                    return vec![vec![], re];
                }
                Some(piece) if piece.team == Team::W && done => {
                    return vec![vec![], overlap.clone()];
                }
                _ => move_vector.push(b_row * 10 + (b_col as i32 + j as i32) as usize),
            }
        }

        if done {
            move_vector.push(p_indexes as usize);
        }
        if !done{
            return vec![vec![], overlap.clone()];
        }
        return vec![move_vector, overlap.clone()]
    }
    else{
        return vec![vec![], overlap.clone()];
    }
    
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
    } //if the bishop is not diagonal to the black king. 

    let mut p_indexes:usize=101;
    let mut move_vector = vec![w_pos];
    let magnitude = (k_row as isize - w_row as isize).abs() as usize;
    let direction_r = if k_row > w_row { 1 } else { -1 };
    let direction_c = if k_col > w_col { 1 } else { -1 };
    let mut done = false;

    for i in 1..magnitude {
        let r_inc = (i as isize) * direction_r;
        let c_inc = (i as isize) * direction_c;       
        if let Some(piece) = board
            .full_board
            .get((w_row as isize + r_inc) as usize)
            .and_then(|row| row.get((w_col as isize + c_inc) as usize))
        {
            if piece.team == Team::W && !done{
                let mut re=overlap.clone();
                if magnitude - i == 1 && !done{
                    re.push(((w_row as isize + r_inc as isize)*10) as usize + (c_inc + w_col as isize) as usize);
                }
                return vec![vec![], re];
            }

            if piece.team == Team::B && done {
                return vec![vec![], overlap.clone()];
            } else if piece.team == Team::B && !done {
                p_indexes = 10 * (w_row as isize + r_inc as isize) as usize + (w_col as isize + c_inc as isize) as usize;
                done = true;
            } else {
                move_vector.push(10 * (w_row as isize + r_inc as isize) as usize + (w_col as isize + c_inc as isize) as usize);
            }
        }
    }

    if !done {
        return vec![vec![], overlap.clone()];
    }

    move_vector.push(p_indexes);

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

    for i in 1..magnitude {
        let r_inc = (i as isize) * direction_r;
        let c_inc = (i as isize) * direction_c;

        if let Some(piece) = board
            .full_board
            .get((b_row as isize + r_inc) as usize)
            .and_then(|row| row.get((b_col as isize + c_inc) as usize))
        {
            if piece.team == Team::B{
                let mut re=overlap.clone();
                if magnitude - i == 1 && !done{
                    re.push(((b_row as isize + r_inc)*10) as usize + (c_inc + b_col as isize) as usize);
                }
                return vec![vec![], re];
            }

            if piece.team == Team::W && done {
                return vec![vec![], overlap.clone()];
            } else if piece.team == Team::W && !done {
                p_index = 10 * (b_row as isize + r_inc) as usize + (b_col as isize + c_inc) as usize;
                done = true;
            } else {
                move_vector.push((10 * (b_row as isize + r_inc) + (b_col as isize + c_inc)) as usize);
            }
        }
    }
    if !done{
        return vec![vec![], overlap.clone()]
    }

    move_vector.push(p_index as usize);
    return vec![move_vector, overlap.clone()];
}



pub fn in_check_directional(board: &Board, re:&AvailableMovesMap, pressuring: PieceId, team: Team) -> AvailableMovesMap {
    let king_index:usize;
    let pressuring_index:usize; 
    let old_av_map=re;
    if team==Team::W {
            king_index = board.black_indexes.get_index(PieceId::K).unwrap();
            pressuring_index = board.white_indexes.get_index(pressuring).unwrap();
    }
    else {
            king_index = board.white_indexes.get_index(PieceId::K).unwrap();
            pressuring_index = board.black_indexes.get_index(pressuring).unwrap();
        }

    let king_row = king_index / 10;
    let king_col = king_index % 10;
    let mut good_moves:Vec<usize> = vec![];
    good_moves.push(pressuring_index);
    let good_moves_for_king=&old_av_map.get_moves(PieceId::K).clone();
    let mut new_av_map=AvailableMovesMap::new();
    // Good moves is the vector that the opponents pieces can move into to block the check.
    let p_row = pressuring_index as isize/10;
    let p_col = pressuring_index as isize%10;
    let r_inc=if p_row<king_row as isize{1} else if p_row>king_row as isize {-1} else {0};
    let c_inc=if p_col<king_col as isize{1} else if p_col>king_col as isize {-1} else {0};
    let magnitude=(p_row-king_row as isize).abs();
    for i in 1..magnitude{
        let good_m=((p_row +(r_inc*i)))*10+p_col+(c_inc*i);
        good_moves.push(good_m as usize);
    }

    // Update available moves based on good moves
    if team==Team::B{
        for i in board.white_piece_ids.iter(){
            if *i!=PieceId::K{
                let new=find_overlap(&old_av_map.get_moves(*i), &good_moves);
                new_av_map.insert_moves(*i, &new);
            }
        }
    }
    else{
        for i in board.black_piece_ids.iter(){
            if *i!=PieceId::K{
                //king logic is different so you want to move out of the directional check instead of into it. 
                let new=find_overlap(&old_av_map.get_moves(*i), &good_moves);
                new_av_map.insert_moves(*i, &new);
            }
        }
        
    }
    //oppossite side of king needs to be removed by check
    let oppossite_side_of_k=(king_row as isize+r_inc)*10+king_col as isize+c_inc;
    let mut bad_moves_for_king:Vec<usize>=vec![];

    if 7 >= oppossite_side_of_k / 10 && oppossite_side_of_k / 10 >= 0 && 7 >= oppossite_side_of_k % 10 && oppossite_side_of_k % 10 >= 0 {
        bad_moves_for_king.push(oppossite_side_of_k as usize);
    }

    let near_side_of_k=(king_row as isize-r_inc)*10+king_col as isize-c_inc;
    if near_side_of_k as usize!=pressuring_index{
        bad_moves_for_king.push(near_side_of_k as usize);
    }

    let mut re_king_moves:Vec<usize>=vec![];
    for v in good_moves_for_king.iter(){
        let mut in_bad_moves=false;
        for j in bad_moves_for_king.iter(){
            if v==j{
                in_bad_moves=true;
            }
        }
        if !in_bad_moves{
            re_king_moves.push(*v)
        }
    }
    new_av_map.insert_moves(PieceId::K, &re_king_moves);

    return new_av_map;
    
}