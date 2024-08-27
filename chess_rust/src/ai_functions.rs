use crate::types::{Board, AvailableMovesMap, Piece, Team, Kind, NeuralNetworkSelector, GameStateIdentifiers};
use crate::base_functions::{init_board};
use crate::upper_move_functions::{all_moves_gen, move_piece};
use crate::test_board::{print_all};
use std::time::Instant;
use half::f16;
use rand::Rng;
use std::sync::Arc;
use tch::{nn, nn::Module, CModule, nn::OptimizerConfig, Device, Tensor};
use std::path::Path;



fn build_chess_cnn_3(vs: &nn::Path) -> nn::Sequential {
    nn::seq()
        .add(nn::conv2d(vs, 6, 32, 4, nn::ConvConfig { padding: 0, ..Default::default() }))
        .add_fn(|xs| xs.relu())
        .add(nn::conv2d(vs, 32, 32, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.avg_pool2d(&[2, 2], &[2, 2], &[0, 0], false, false, None))
        .add(nn::conv2d(vs, 32, 64, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
        .add_fn(|xs| xs.relu())
        .add(nn::conv2d(vs, 64, 64, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.max_pool2d(&[2, 2], &[2, 2], &[0, 0], &[1, 1], false))  // dilation: [1, 1], ceil_mode: false
        .add_fn(|xs| xs.view([-1, 64 * 1 * 1]))  // Flatten the tensor
        .add(nn::linear(vs, 64 * 1 * 1, 128, Default::default()))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.dropout(0.5, false))
        .add(nn::linear(vs, 128, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.dropout(0.5, false))
        .add(nn::linear(vs, 64, 1, Default::default()))
        .add_fn(|xs| xs.sigmoid())
}


fn build_chess_cnn_2(vs: &nn::Path) -> nn::Sequential {
    nn::seq()
        .add(nn::conv2d(vs, 6, 32, 4, nn::ConvConfig { padding: 0, ..Default::default() }))
        .add_fn(|xs| xs.relu())
        .add(nn::conv2d(vs, 32, 32, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.avg_pool2d(&[2, 2], &[2, 2], &[0, 0], false, false, None))  // Corresponds to Python's AvgPool2d
        .add(nn::conv2d(vs, 32, 64, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
        .add_fn(|xs| xs.relu())
        .add(nn::conv2d(vs, 64, 64, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.max_pool2d(&[2, 2], &[2, 2], &[0, 0], &[1, 1], false)) // Corresponds to Python's MaxPool2d
        .add_fn(|xs| xs.view([-1, 64 * 1 * 1]))  // Flatten the tensor
        .add(nn::linear(vs, 64 * 1 * 1, 128, Default::default()))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.dropout(0.5, false)) // Corresponds to Python's Dropout
        .add(nn::linear(vs, 128, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.dropout(0.5, false)) // Corresponds to Python's Dropout
        .add(nn::linear(vs, 64, 1, Default::default()))
        .add_fn(|xs| xs.sigmoid())  // Corresponds to Python's Sigmoid
}


fn build_chess_cnn_4(vs: &nn::Path) -> nn::Sequential {
    nn::seq()
        .add(nn::conv2d(vs, 6, 32, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
        .add_fn(|xs| xs.relu())
        .add(nn::conv2d(vs, 32, 32, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.avg_pool2d(&[2, 2], &[2, 2], &[0, 0], false, false, None)) // Updated avg_pool2d call
        .add(nn::conv2d(vs, 32, 64, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.max_pool2d(&[2, 2], &[2, 2], &[0, 0], &[1, 1], false)) // Updated max_pool2d call
        .add_fn(|xs| xs.view([-1, 64 * 2 * 2])) // Flatten the tensor
        .add(nn::linear(vs, 64 * 2 * 2, 128, Default::default()))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.dropout(0.5, false)) // Updated dropout call
        .add(nn::linear(vs, 128, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.dropout(0.5, false)) // Updated dropout call
        .add(nn::linear(vs, 64, 1, Default::default()))
        .add_fn(|xs| xs.sigmoid())
}

fn build_chess_cnn_6(vs: &nn::Path) -> nn::Sequential {
    nn::seq()
        .add(nn::conv2d(vs, 6, 32, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
        .add_fn(|xs| xs.relu())
        .add(nn::conv2d(vs, 32, 64, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
        .add_fn(|xs| xs.relu())
        .add(nn::conv2d(vs, 64, 128, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.max_pool2d(&[2, 2], &[2, 2], &[0, 0], &[1, 1], false)) // Updated max_pool2d call
        .add(nn::conv2d(vs, 128, 128, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
        .add_fn(|xs| xs.relu())
        .add(nn::conv2d(vs, 128, 256, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.max_pool2d(&[2, 2], &[2, 2], &[0, 0], &[1, 1], false)) // Updated max_pool2d call
        .add_fn(|xs| xs.view([-1, 256 * 2 * 2]))  // Flatten the tensor
        .add(nn::linear(vs, 256 * 2 * 2, 128, Default::default()))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.dropout(0.5, false)) // Updated dropout call
        .add(nn::linear(vs, 128, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.dropout(0.5, false)) // Updated dropout call
        .add(nn::linear(vs, 64, 1, Default::default()))
        .add_fn(|xs| xs.sigmoid())
}

fn build_chess_cnn_5(vs: &nn::Path) -> nn::Sequential {
    nn::seq()
        .add(nn::conv2d(vs, 6, 32, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
        .add_fn(|xs| xs.relu())
        .add(nn::conv2d(vs, 32, 32, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.avg_pool2d(&[2, 2], &[2, 2], &[0, 0], false, false, None))  // Corresponds to Python's AvgPool2d
        .add(nn::conv2d(vs, 32, 64, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
        .add_fn(|xs| xs.relu())
        .add(nn::conv2d(vs, 64, 64, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.max_pool2d(&[2, 2], &[2, 2], &[0, 0], &[1, 1], false)) // Corresponds to Python's MaxPool2d
        .add_fn(|xs| xs.view([-1, 64 * 2 * 2]))  // Flatten the tensor
        .add(nn::linear(vs, 64 * 2 * 2, 128, Default::default()))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.dropout(0.5, false)) // Corresponds to Python's Dropout
        .add(nn::linear(vs, 128, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.dropout(0.5, false)) // Corresponds to Python's Dropout
        .add(nn::linear(vs, 64, 1, Default::default()))
        .add_fn(|xs| xs.sigmoid())  // Corresponds to Python's Sigmoid
}


// fn get_nn_build(model: NeuralNetworkSelector, vs: &mut nn::VarStore) -> nn::Sequential {
//     // Build the model based on the selector
//     let sequential_model = match model {
//         NeuralNetworkSelector::Model2 => build_chess_cnn_2(&vs.root()),
//         NeuralNetworkSelector::Model3 => build_chess_cnn_3(&vs.root()),
//         NeuralNetworkSelector::Model4 => build_chess_cnn_4(&vs.root()),
//         NeuralNetworkSelector::Model5 => build_chess_cnn_5(&vs.root()),
//         NeuralNetworkSelector::Model6 => build_chess_cnn_6(&vs.root()),
//         _ => build_chess_cnn_2(&vs.root()), // Default case
//     };

//     // Load the pre-trained weights from the file path provided by NeuralNetworkSelector
//     let model_file_path = model.to_string();
//     if Path::new(&model_file_path).exists() {
//         match vs.load(&model_file_path) {
//             Ok(_) => println!("Successfully loaded pre-trained model from {}", model_file_path),
//             Err(e) => eprintln!("Failed to load the model: {:?}", e),
//         }
//     } else {
//         eprintln!("Model file does not exist at path: {}", model_file_path);
//     }

//     sequential_model
// }

fn get_nn_build(model_path: &str) -> Result<CModule, Box<dyn std::error::Error>> {
    let model = CModule::load(model_path)?;
    Ok(model)
}


pub fn game_still_going(board: &Board, checking: bool, av_moves:&AvailableMovesMap) -> f32 {
    //neural network does not flag if the game is over or not, it just uses image recognition to calculate a winning probability
    //so this is needed for a deterministic is game still going. 
    game_states=GameStateIdentifiers::new();
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
                if board.ai_team{
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
                if !board.ai_team{
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
    model_selector: &NeuralNetworkSelector,//need to just load the nn once. Will do this once I get the final nn 
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
    let input_data: Vec<f32> = [pawns, knights, rooks, bishops, queens, kings].concat();
    let input_tensor = Tensor::from_slice(&input_data).view([1, 6, 8, 8]);

    // Build and load the model based on the model selector
    let model = get_nn_build(model_selector.to_string());

    let model = match model {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error loading model: {:?}", e);
            return -1000.0; // Or handle the error appropriately
        }
    };
    // Run the model on the input tensor
    let output = model.forward(&input_tensor);

    // Since the model uses a sigmoid function, we can directly interpret the output as the probability
    let probability_of_white_win = output.double_value(&[0]);

    // Adjust the result based on whether the AI team is white or black
    let result = if ai_team {
        probability_of_white_win as f32
    } else {
        -1.0*probability_of_white_win as f32
    };

    result
}

pub fn get_next_move(b:Board, model:&NeuralNetworkSelector)->Board{
    let nn=model.to_string();
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
                let cloned_board=b.clone();
                let new_move=move_piece(cloned_board, *i, j);
                let start = Instant::now(); 
                let duration = start.elapsed();
                let adv=board_position_advantage_eval(&new_move.full_board, white, model);
                let duration_in_seconds = duration.as_secs_f64();
                if adv>probability_best_move{
                    probability_best_move=adv;
                    best_board=new_move.clone();
                }
            }
        }
    }else{
        for i in b.black_piece_ids.iter(){
            for j in moves.moves.get_moves(*i){
                let cloned_board=b.clone();
                let new_move=move_piece(cloned_board, *i, j);
                let start = Instant::now(); 
                let duration = start.elapsed();
                let adv=board_position_advantage_eval(&new_move.full_board, white, model);
                let duration_in_seconds = duration.as_secs_f64();
                if adv>probability_best_move{
                    probability_best_move=adv;
                    best_board=new_move.clone();
                }
            }
        }
    }
    return best_board;
}

pub fn test_neural_networks(black:&NeuralNetworkSelector, white:&NeuralNetworkSelector)->i32{
    
    let mut done=false; 
    let mut game=init_board(true);
    while done==false{
        if game.turn%2==0{
            println!("white {}", white.to_string());
            game=get_next_move(game, &white);
            print_all(game.clone());
        }else{
            println!("black {}", black.to_string());
            game=get_next_move(game, &black);
            print_all(game.clone());
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
        return 1;
    }
    return 0;
}




