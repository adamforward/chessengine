use crate::types::{Board, AvailableMovesMap, Piece, Team, Kind, NeuralNetworkSelector, GameStateIdentifiers};
use crate::base_functions::{init_board};
use crate::upper_move_functions::{all_moves_gen, move_piece};
use crate::test_board::{print_all};
use std::time::Instant;
use half::f16;
use rand::Rng;
use std::sync::Arc;
use std::path::Path;


pub fn game_still_going(board: &Board, checking: bool, av_moves:&AvailableMovesMap) -> f32 {
    //neural network does not flag if the game is over or not, it just uses image recognition to calculate a winning probability
    //so this is needed for a deterministic is game still going. 
    let game_states=GameStateIdentifiers::new();
    if board.turn % 2 == 0{
        let mut no_moves = true; // 0 for in play, 1 for checkmate, 2 for stalemate
        for &i in board.white_piece_ids.iter() {
            let moves = av_moves.get_moves(i);
            if moves.len() > 0 {
                no_moves = false;
            }
        }
        if no_moves {
            if !checking {
                return 0.0; // stalemate
            } else {
                if board.ai_team_is_white{
                    return game_states.ai_checkmate; // ai wins
                }
                else{
                    return game_states.player_checkmate;
                }
            }
        }
        return game_states.in_play;
    } else {
        let mut no_moves = true; // 0 for in play, 1 for checkmate, 2 for stalemate
        for &i in board.black_piece_ids.iter() {
            let moves = av_moves.get_moves(i);
            if moves.len() > 0 {
                no_moves = false;
            }
        }
        if no_moves {
            if checking {
                return 0.0; // stalemate
            } else {
                if !board.ai_team_is_white{
                    return game_states.ai_checkmate; // ai wins
                }
                else{
                    return game_states.player_checkmate;
                } // player wins
            }
        }
        return game_states.in_play;
    }
}

pub fn board_position_advantage_eval(
    full_board: &Vec<Vec<Piece>>,
    ai_team: bool,
) -> f32 {
    let mut pawns: Vec<f32> = vec![0.0; 64];
    let mut knights: Vec<f32> = vec![0.0; 64];
    let mut rooks: Vec<f32> = vec![0.0; 64];
    let mut bishops: Vec<f32> = vec![0.0; 64];
    let mut queens: Vec<f32> = vec![0.0; 64];
    let mut kings: Vec<f32> = vec![0.0; 64];

    for (i, row) in full_board.iter().enumerate() {
        for (j, piece) in row.iter().enumerate() {
            let idx = i * 8 + j;
            match piece.team {
                Team::W => {
                    let value = 1.0;
                    match piece.kind {
                        Kind::Pawn => pawns[idx] = value,
                        Kind::Knight => knights[idx] = value,
                        Kind::Rook => rooks[idx] = value,
                        Kind::Bishop => bishops[idx] = value,
                        Kind::Queen => queens[idx] = value,
                        Kind::King => kings[idx] = value,
                        _ => {},
                    }
                }
                Team::B => {
                    let value = -1.0;
                    match piece.kind {
                        Kind::Pawn => pawns[idx] = value,
                        Kind::Knight => knights[idx] = value,
                        Kind::Rook => rooks[idx] = value,
                        Kind::Bishop => bishops[idx] = value,
                        Kind::Queen => queens[idx] = value,
                        Kind::King => kings[idx] = value,
                        _ => {},
                    }
                }
                _ => {},
            }
        }
    }

    // Convert the board representation into a Tensor
    // let input_data: Vec<f32> = [pawns, knights, rooks, bishops, queens, kings].concat();
    // let input_tensor = Tensor::from_slice(&input_data).view([1, 6, 8, 8]);
    // Run the model on the input tensor
    //let output = model.forward(&input_tensor);

    // Since the model uses a sigmoid function, we can directly interpret the output as the probability
    let mut rng = rand::thread_rng();

    // Generate a random f32 between -1.0 and 1.0
    let random_f32 = rng.gen_range(-1.0..1.0);
    let probability_of_white_win=random_f32;    
    // Adjust the result based on whether the AI team is white or black
    let result = if ai_team {
        probability_of_white_win as f32
    } else {
        -1.0*probability_of_white_win as f32
    };

    result
}

// pub fn get_next_move(b:Board, model:&NeuralNetworkSelector)->Board{
//     let nn=model.to_string();
//     let white:bool;
//     if b.turn%2==0{
//         white=true;
//     }
//     else{
//         white=false;
//     }
//     let moves=all_moves_gen(&b);
//     let mut probability_best_move:f32=0.0;
//     let mut best_board=init_board(true);
    
//     if white{
//         for i in b.white_piece_ids.iter(){
//             for j in moves.moves.get_moves(*i){
//                 let cloned_board=b.clone();
//                 let new_move=move_piece(cloned_board, *i, j);
//                 let start = Instant::now(); 
//                 let duration = start.elapsed();
//                 let adv=board_position_advantage_eval(&new_move.full_board, white, model);
//                 let duration_in_seconds = duration.as_secs_f64();
//                 if adv>probability_best_move{
//                     probability_best_move=adv;
//                     best_board=new_move.clone();
//                 }
//             }
//         }
//     }else{
//         for i in b.black_piece_ids.iter(){
//             for j in moves.moves.get_moves(*i){
//                 let cloned_board=b.clone();
//                 let new_move=move_piece(cloned_board, *i, j);
//                 let start = Instant::now(); 
//                 let duration = start.elapsed();
//                 let adv=board_position_advantage_eval(&new_move.full_board, white, model);
//                 let duration_in_seconds = duration.as_secs_f64();
//                 if adv>probability_best_move{
//                     probability_best_move=adv;
//                     best_board=new_move.clone();
//                 }
//             }
//         }
//     }
//     return best_board;
// }

// pub fn test_neural_networks(black:&NeuralNetworkSelector, white:&NeuralNetworkSelector)->i32{
    
//     let mut done=false; 
//     let mut game=init_board(true);
//     while done==false{
//         if game.turn%2==0{
//             println!("white {}", white.to_string());
//             game=get_next_move(game, &white);
//             print_all(game.clone());
//         }else{
//             println!("black {}", black.to_string());
//             game=get_next_move(game, &black);
//             print_all(game.clone());
//         }
//         let moves=all_moves_gen(&game);
        
//         if game_still_going(&game, moves.checking, &moves.moves)!=0.1 || game.turn>400{
//             done=true;
//         }
//     }
//     let moves2=all_moves_gen(&game);
//     if game_still_going(&game, moves2.checking, &moves2.moves)==0.0{
//         return -1;
//     }
//     if game_still_going(&game, moves2.checking, &moves2.moves)==1.0{
//         return 1;
//     }
//     return 0;
// }




