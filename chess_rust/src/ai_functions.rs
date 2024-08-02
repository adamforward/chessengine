use crate::types::{Board, AvailableMovesMap, Piece, Team, Kind};
// use onnxruntime::environment::Environment;
// use onnxruntime::ndarray::Array;
// use onnxruntime::ndarray::ArrayD;
// use onnxruntime::GraphOptimizationLevel;
// use onnxruntime::LoggingLevel;
// use onnxruntime::session::Session;
use half::f16;
use rand::Rng;
use std::sync::Arc;

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
    // let mut pawns: Vec<Vec<f16>> = vec![vec![f16::from_f32(0.0); 8]; 8];
    // let mut knights: Vec<Vec<f16>> = vec![vec![f16::from_f32(0.0); 8]; 8];
    // let mut rooks: Vec<Vec<f16>> = vec![vec![f16::from_f32(0.0); 8]; 8];
    // let mut bishops: Vec<Vec<f16>> = vec![vec![f16::from_f32(0.0); 8]; 8];
    // let mut queens: Vec<Vec<f16>> = vec![vec![f16::from_f32(0.0); 8]; 8];
    // let mut kings: Vec<Vec<f16>> = vec![vec![f16::from_f32(0.0); 8]; 8];

    // for (i, row) in full_board.iter().enumerate() {
    //     for (j, piece) in row.iter().enumerate() {
    //         match piece.team {
    //             Team::W | Team::B => {
    //                 let value = if piece.team == Team::W { f16::from_f32(1.0) } else { f16::from_f32(-1.0) };
    //                 match piece.kind {
    //                     Kind::Pawn => pawns[i][j] = value,
    //                     Kind::Knight => knights[i][j] = value,
    //                     Kind::Rook => rooks[i][j] = value,
    //                     Kind::Bishop => bishops[i][j] = value,
    //                     Kind::Queen => queens[i][j] = value,
    //                     Kind::King => kings[i][j] = value,
    //                     _ => {}, 
    //                 }
    //             }
    //             _ => {},
    //         }
    //     }
    // }

    // // Flatten the board representation and convert it to f16
    // let mut input_data: Vec<f16> = Vec::new();
    // for i in 0..8 {
    //     for j in 0..8 {
    //         input_data.push(pawns[i][j]);
    //         input_data.push(knights[i][j]);
    //         input_data.push(rooks[i][j]);
    //         input_data.push(bishops[i][j]);
    //         input_data.push(queens[i][j]);
    //         input_data.push(kings[i][j]);
    //     }
    // }

    // // Create an environment and a session to run the model
    // let environment = Environment::builder()
    //     .with_name("test")
    //     .with_log_level(LoggingLevel::Warning)
    //     .build().unwrap();

    // let session = environment
    //     .new_session_builder()
    //     .unwrap()
    //     .with_optimization_level(GraphOptimizationLevel::Basic)
    //     .unwrap()
    //     .with_model_from_file("src/model.onnx")
    //     .unwrap();

    // // Convert the input data to an ndarray
    // let input_array = Array::from_shape_vec((1, 6, 8, 8), input_data).unwrap();
    // let input_tensor = vec![input_array.into_dyn()];

    // // Run the model
    // let outputs: Vec<Arc<ndarray::ArrayD<f16>>> = session.run(input_tensor).unwrap();

    // // Get the output
    // let result = outputs[0].as_slice().unwrap()[0].to_f32();

    // result
    return 0.4
}


