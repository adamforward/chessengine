use crate::types::{Board, AvailableMovesMap};
use rand::Rng;



pub fn game_still_going(board: &Board, checking:bool, white_available_moves:&AvailableMovesMap, black_available_moves:&AvailableMovesMap) -> f64 {
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
pub fn board_position_advantage_eval() -> f64 {
    let mut rng = rand::thread_rng();
    rng.gen_range(-1.0..=1.0)
}
