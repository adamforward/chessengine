use std::{cell::RefCell, rc::Rc};
use crate::upper_move_functions::{all_moves_gen, move_piece};
use crate::ai_functions::{ game_still_going, board_position_advantage_eval};
use crate::types::{TreeNode, Board, AdavantageMap, PieceId, NeuralNetworkSelector, GameStateIdentifiers};
use tch::{nn, nn::VarStore, Device, Tensor};
use crate::test_board::{print_all};


pub fn generate_top_moves(num_moves: i32, parent:TreeNode)->Vec<TreeNode> {
    let gamestates:GameStateIdentifiers=GameStateIdentifiers::new();
    //this takes a game, makes all available moves then returns a vector of length num_moves for the highest win probabilities
    //board_position_advantage_eval evaluates the probability that ai wins by checkmate via convolutional nn
    //this function is set up to use the datatypes that allow for a tree structure
    let mut curr_game=parent.game.clone();
    let level=parent.level;
    let new_info=all_moves_gen(&curr_game);
    if game_still_going(&curr_game, new_info.checking, &new_info.moves)!=gamestates.in_play{
        curr_game.ai_advantage=game_still_going(&curr_game, new_info.checking, &new_info.moves); 
        return vec![]; //need deterministic instead of statistical analysis code to evaluate if game is over
    }

    let mut advantage_map:Vec<AdavantageMap>=vec![];
    if curr_game.turn%2==0{
        for &i in curr_game.white_piece_ids.iter(){
            let moves=new_info.moves.get_moves(i);
            for &j in moves.iter(){
                let param_move=curr_game.clone();
                let potential_move=move_piece(param_move, i, j);
                let advantage=board_position_advantage_eval(&potential_move.full_board, curr_game.ai_team_is_white, &NeuralNetworkSelector::Model7);
                advantage_map.push(AdavantageMap{board:potential_move, advantage});
            }
        }
    }

    else{
        for &i in curr_game.black_piece_ids.iter(){
            let moves=new_info.moves.get_moves(i);
            for &j in moves.iter(){
                let param_move=curr_game.clone();
                let potential_move=move_piece(param_move, i, j);
                let advantage=board_position_advantage_eval(&potential_move.full_board, curr_game.ai_team_is_white, &NeuralNetworkSelector::Model7);
                advantage_map.push(AdavantageMap{board:potential_move, advantage});
            }
        }
    }

    let mut re:Vec<TreeNode>=vec![];
    let ai_turn:bool;
    if ((curr_game.turn+1)%2==0 && curr_game.ai_team_is_white) || ((curr_game.turn+1)%2==1 && !curr_game.ai_team_is_white){
        ai_turn=true;
    }
    else {
        ai_turn=false
    }
    
    let immut_advantage_map=advantage_map.clone();

    for e in advantage_map.iter_mut() { // check for checkmates and assign ai advantage properties
        e.board.ai_advantage=e.advantage;

        if ai_turn && game_still_going(&curr_game, new_info.checking, &new_info.moves)==gamestates.ai_checkmate{
            e.board.ai_advantage=gamestates.ai_checkmate;//always move into checkmate
            return vec![TreeNode { children:vec![],game:e.board.clone(), level}];
            
        }

        if !ai_turn && game_still_going(&curr_game, new_info.checking, &new_info.moves)==gamestates.player_checkmate{
            e.board.ai_advantage=gamestates.player_checkmate; //always assume player moves into checkmate
            return vec![TreeNode { children:vec![],game:e.board.clone(), level}];
        }
        let mut count=0; 
        for i in immut_advantage_map.iter(){
            if ai_turn{
                if e.advantage>i.advantage{
                    count+=1;//ai wants to maximize ai win probability
                }
            }
            else{
                if e.advantage<i.advantage{
                    count+=1;//player wants to minimize it
                }
            }
        }
        if count<=num_moves{
            re.push(TreeNode { children:vec![], game:e.board.clone(), level:level+1});
        }
    }
    re
}

pub fn search(mut curr_game: TreeNode, depth: i32, width: i32, mut alpha: f32, mut beta: f32) -> f32 {      
    let game_states=GameStateIdentifiers::new();
    // Check for terminal conditions
    if curr_game.level == depth || 
       curr_game.game.ai_advantage == game_states.ai_checkmate || 
       curr_game.game.ai_advantage == game_states.stalemate || 
       curr_game.game.ai_advantage == game_states.player_checkmate {
        return curr_game.game.ai_advantage; 
    }

    let cloned_game = curr_game.clone();
    let children = generate_top_moves(width, cloned_game); // Generate the top `width` moves
    curr_game.children = children.clone();

    if curr_game.level % 2 == 1 { // Maximizing ai
        let mut max_eval = -f32::INFINITY;
        for child in children {
            let eval = search(child, depth, width, alpha, beta);
            max_eval = max_eval.max(eval);
            alpha = alpha.max(eval);
            if beta <= alpha {
                break; 
            }
        }
        return max_eval;
    } else { 
        let mut min_eval = f32::INFINITY;
        for child in children {
            let eval = search(child, depth, width, alpha, beta);
            min_eval = min_eval.min(eval);
            beta = beta.min(eval);
            if beta <= alpha {
                break; 
            }
        }
        return min_eval;
    }
}


