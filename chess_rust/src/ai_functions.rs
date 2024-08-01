use crate::types::{Board, AvailableMovesMap, Piece, Team, Kind};
use rand::Rng;
use half::f16;

pub fn game_still_going(board: &Board, checking: bool, white_available_moves: &AvailableMovesMap, black_available_moves: &AvailableMovesMap) -> f32 {
    //neural network does not flag if the game is over or not, it just uses image recognition to calculate a winning probability
    //so this is needed for a deterministic is game still going. 
    if board.turn % 2 == 0 {
        let mut no_moves = true; // 0 for in play, 1 for checkmate, 2 for stalemate
        for &i in board.white_piece_ids.iter() {
            let moves = white_available_moves.get_moves(i);
            if moves.len() > 0 {
                no_moves = false;
            }
        }
        if no_moves {
            if checking {
                return 0.5; // stalemate
            } else {
                return 1.0; // ai wins
            }
        }
        return 0.1;
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
                return 0.5; // stalemate
            } else {
                return 1.0; // ai wins
            }
        }
        return 0.1;
    }
}

pub fn board_position_advantage_eval(full_board: &Vec<Vec<Piece>>, ai_team: bool) -> f32 {
    let mut pawns: Vec<Vec<f16>> = vec![vec![f16::from_f32(0.0); 8]; 8];
    let mut knights: Vec<Vec<f16>> = vec![vec![f16::from_f32(0.0); 8]; 8];
    let mut rooks: Vec<Vec<f16>> = vec![vec![f16::from_f32(0.0); 8]; 8];
    let mut bishops: Vec<Vec<f16>> = vec![vec![f16::from_f32(0.0); 8]; 8];
    let mut queens: Vec<Vec<f16>> = vec![vec![f16::from_f32(0.0); 8]; 8];
    let mut kings: Vec<Vec<f16>> = vec![vec![f16::from_f32(0.0); 8]; 8];

    for (i, row) in full_board.iter().enumerate() {
        for (j, piece) in row.iter().enumerate() {
            match piece.team {
                Team::W | Team::B => {
                    let value = if piece.team == Team::W { f16::from_f32(1.0) } else { f16::from_f32(-1.0) };
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
    if ai_team {
        return random_float;
    } else {
        return 1.0 - random_float;
    }
    // Probability = desired possibility over total possibilities
    // NN is only trained on games with checkmate, does not have any games with draws or win by anything other than checkmate
    // So this is probability white wins, 1 - nn outcome is probability that black wins
}
