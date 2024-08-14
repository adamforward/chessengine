use crate::types::{Board, AvailableMovesMap, Piece, Team, Kind};
use crate::base_functions::{init_board};
use crate::upper_move_functions::{all_moves_gen, move_piece};
use crate::test_board::{print_all};
use std::time::Instant;
use half::f16;
use rand::Rng;
use std::sync::Arc;
use tch::{CModule, Tensor};



pub fn game_still_going(board: &Board, checking: bool, av_moves:&AvailableMovesMap) -> f32 {
    //neural network does not flag if the game is over or not, it just uses image recognition to calculate a winning probability
    //so this is needed for a deterministic is game still going. 
    if board.turn % 2 == 0 {
        let mut no_moves = true; // 0 for in play, 1 for checkmate, 2 for stalemate
        for &i in board.white_piece_ids.iter() {
            let moves = av_moves.get_moves(i);
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
            let moves = av_moves.get_moves(i);
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

pub fn board_position_advantage_eval(full_board: &Vec<Vec<Piece>>, ai_team: bool, curr_model: &str) -> f32 {
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
                        _=>{},
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
                        _=>{},
                    }
                }
                _ => {},
            }
        }
    }

    // Convert the board representation into a Tensor
    let input_data: Vec<f32> = [pawns, knights, rooks, bishops, queens, kings].concat();
    let input_tensor = Tensor::from_slice(&input_data).view([1, 6, 8, 8]);

    // Load the model with error handling
    let model_result = CModule::load(curr_model);
    let model = match model_result {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load the model: {:?}", e);
            return -1.0; // Return a default value or handle as necessary
        }
    };

    // Run the model on the input tensor
    let output = model
        .forward_ts(&[input_tensor])
        .expect("Failed to run the model")
        .softmax(-1, tch::Kind::Float);

    // Get the probability result for the ai_team
    let probability_of_white_win = f64::from(output.double_value(&[0]));

    // Adjust the result based on whether the AI team is white or black
    let result = if ai_team {
        probability_of_white_win as f32
    } else {
        1.0 - probability_of_white_win as f32
    };

    result
}

fn get_next_move(b:Board, nn:&str )->Board{
    let white:bool;
    if b.turn%2==0{
        white=true;
    }
    else{
        white=false;
    }
    let moves=all_moves_gen(&b);
    let mut probability_best_move:f32=0.0;
    let mut best_board=init_board(true);
    
    if white{
        for i in b.white_piece_ids.iter(){
            for j in moves.moves.get_moves(*i){
                print_all(b.clone());
                let cloned_board=b.clone();
                let new_move=move_piece(cloned_board, *i, j);
                let start = Instant::now(); 
                let duration = start.elapsed();
                let adv=board_position_advantage_eval(&new_move.full_board, white, nn);
                let duration_in_seconds = duration.as_secs_f64();
                println!("time taken for nn call {} {}", duration_in_seconds, nn);
                if adv>probability_best_move{
                    probability_best_move=adv;
                    best_board=new_move.clone();
                }
            }
        }
    }else{
        for i in b.black_piece_ids.iter(){
            for j in moves.moves.get_moves(*i){
                print_all(b.clone());
                let cloned_board=b.clone();
                let new_move=move_piece(cloned_board, *i, j);
                let start = Instant::now(); 
                let duration = start.elapsed();
                let adv=board_position_advantage_eval(&new_move.full_board, white, nn);
                let duration_in_seconds = duration.as_secs_f64();
                println!("time taken for nn call {} {}", duration_in_seconds, nn);
                if adv>probability_best_move{
                    probability_best_move=adv;
                    best_board=new_move.clone();
                }
            }
        }
    }
    return best_board;
}

pub fn test_neural_networks(white:&str, black:&str)->i32{
    let mut done=false; 
    let mut game=init_board(true);
    while done==false{
        if game.turn%2==0{
            game=get_next_move(game, white);
        }else{
            game=get_next_move(game, white);
        }
        let moves=all_moves_gen(&game);
        
        if game_still_going(&game, moves.checking, &moves.moves)!=0.1 || game.turn>400{
            done=true;
        }
    }
    let moves2=all_moves_gen(&game);
    if game_still_going(&game, moves2.checking, &moves2.moves)==0.0{
        return -1;
    }
    if game_still_going(&game, moves2.checking, &moves2.moves)==1.0{
        return -1;
    }
    return 0;
}


