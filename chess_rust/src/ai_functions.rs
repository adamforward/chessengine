use crate::types::{Board, AvailableMovesMap, Piece,Team,Kind};
use rand::Rng;
use half::f16;



pub fn game_still_going(board: &Board, checking:bool, white_available_moves:&AvailableMovesMap, black_available_moves:&AvailableMovesMap) -> f32 {
    if board.turn % 2 == 0 {
        let mut no_moves = true; // 0 for in play, 1 for checkmate, 2 for stalemate
        for &i in board.white_piece_ids.iter() {
            let moves = white_available_moves.get_moves(i);
            if moves.len() > 0 {
                no_moves = false;
            }
        }
        if no_moves==true {
            if checking {
                return 2000.0; //stalemate
            } else {
                return 100000.0; // ai wins
            }
        }
        return 0.0;
    } else {
        let mut no_moves = true; // 0 for in play, 1 for checkmate, 2 for stalemate
        for &i in board.black_piece_ids.iter() {
            let moves = black_available_moves.get_moves(i);
            if moves.len() > 0 {
                no_moves = false;
            }
        }
        if no_moves {
            if checking {
                return 2000.0; //stalemate
            } else {
                return 100000.0; // ai wins
            } //put neural network here
        }
        return 0.0;
    }
}

pub fn board_position_advantage_eval(full_board:&Vec<Vec<Piece>>, ai_team:bool) -> f32 {
    let mut pawns: Vec<Vec<i8>> = vec![vec![0; 8]; 8];
    let mut knights: Vec<Vec<i8>> = vec![vec![0; 8]; 8];
    let mut rooks: Vec<Vec<i8>> = vec![vec![0; 8]; 8];
    let mut bishops: Vec<Vec<i8>> = vec![vec![0; 8]; 8];
    let mut queens: Vec<Vec<i8>> = vec![vec![0; 8]; 8];
    let mut kings: Vec<Vec<i8>> = vec![vec![0; 8]; 8];

    for (i, row) in full_board.iter().enumerate() {
        for (j, piece) in row.iter().enumerate() {
            match piece.team {
                Team::W | Team::B => {
                    let value = if piece.team == Team::W { 1 } else { -1 };
                    match piece.kind {
                        Kind::Pawn => pawns[i][j] = value,
                        Kind::Knight => knights[i][j] = value,
                        Kind::Rook => rooks[i][j] = value,
                        Kind::Bishop => bishops[i][j] = value,
                        Kind::Queen => queens[i][j] = value,
                        Kind::King => kings[i][j] = value,
                        _ => {}, 
                    }
                }
                _ => {},
            }
        }
    }
    
    let mut rng = rand::thread_rng();
    let random_float: f32 = rng.gen_range(0.0..=1.0);
    return random_float;
    //probability=desired possibility over total possibilities
    //nn is only trained on games with checkmate
    //so this is probability white wins
}
