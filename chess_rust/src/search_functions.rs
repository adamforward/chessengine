use core::num;
use std::slice::Iter;
use std::{cell::RefCell, rc::Rc};
use crate::upper_move_functions::{all_moves_gen, move_piece};
use crate::ai_functions::{ game_still_going, board_position_advantage_eval};
use crate::types::{TreeNode, Board, AdavantageMap, PieceId, NeuralNetworkSelector, GameStateIdentifiers};
use rayon::vec;
use crate::test_board::{print_all};
use crate::base_functions::ai_turn;
//lets make this code more readable
pub fn generate_top_moves(num_moves: i32, parent:TreeNode)->Vec<TreeNode> {
    let gamestates:GameStateIdentifiers=GameStateIdentifiers::new();

    //this takes a game, makes all available moves then returns a vector of length num_moves for the highest nn outcomes.
    //board_position_advantage_eval evaluates the probability that ai wins by checkmate via convolutional nn
    //this function is set up to use the datatypes that allow for a tree structure

    let mut curr_game=parent.game.clone();
    let ai_turn=ai_turn(&curr_game);

    let new_level=parent.level +1 ;
    let new_info=all_moves_gen(&curr_game);

    if game_still_going(&curr_game, new_info.checking, &new_info.moves)!=gamestates.in_play{
        curr_game.ai_advantage=game_still_going(&curr_game, new_info.checking, &new_info.moves); 
        return vec![]; //need deterministic instead of statistical analysis code to evaluate if game is over, don't do anything if this is the case
    }

    let mut advantage_map:Vec<AdavantageMap>=vec![];
    //here, we get the data for the next round of moves so that advantage field points to the ai advantage from 1 to -1, then the board is the board class after making this move. 
    if curr_game.turn%2==0{
        for &i in curr_game.white_piece_ids.iter(){
            let moves=new_info.moves.get_moves(i);
            for &j in moves.iter(){
                let param_move=curr_game.clone();
                let potential_move=move_piece(param_move, i, j);
                let potential_new_moves=all_moves_gen(&potential_move);
                let gst_value = game_still_going(&potential_move, potential_new_moves.checking, &potential_new_moves);

                if gst_value!=gamestates.in_play{
                    advantage_map.push(AdavantageMap{board:potential_move, advantage:gst_value});
                    continue;
                    //checkmate or stalemate, that's the value we want to push. 
                }

                else{
                    let advantage=board_position_advantage_eval(&potential_move.full_board, curr_game.ai_team_is_white);
                    advantage_map.push(AdavantageMap{board:potential_move, advantage});
                }
            }
        }
    }
    //data if white turn
    else{
        for &i in curr_game.black_piece_ids.iter(){
            let moves=new_info.moves.get_moves(i);
            for &j in moves.iter(){
                let param_move=curr_game.clone();
                let potential_move=move_piece(param_move, i, j);
                let potential_new_moves=all_moves_gen(&potential_move);
                let gst_value = game_still_going(&potential_move, potential_new_moves.checking, &potential_new_moves);

                if gst_value!=gamestates.in_play{
                    advantage_map.push(AdavantageMap{board:potential_move, advantage:gst_value});
                    continue;
                }

                else{
                    let advantage=board_position_advantage_eval(&potential_move.full_board, curr_game.ai_team_is_white);
                    advantage_map.push(AdavantageMap{board:potential_move, advantage});
                }
            }
        }
    }
    //data if black turn. 
    let sorted_adv_map=sort_through_advantage_map(&advantage_map); 
    if ai_turn{
        if sorted_adv_map[0]=gamestates.ai_checkmate{
            return vec![TreeNode{level:new_level, game:sorted_adv_map[0].board.clone(), children:vec![]}];
            //if there's checkmate always move there. 
        }
        else{
            let re=vec![];
            for i in 0..num_moves{ //sort biggest to smallest 
                re.push(TreeNode{level:new_level, game:sorted_adv_map[i].board.clone(), children:vec![]});
            }
            re
        }
    }
    else{
        if sorted_adv_map[0]=gamestates.player_checkmate{
            //if there's checkmate always move there. 
            return vec![TreeNode{level:new_level, game:sorted_adv_map[sorted_adv_map.len()-1].board.clone(), children:vec![]}];
        }
        else{
            let re=vec![];
            for i in (num_moves-1)..=0{ //sort biggest to smallest
                re.push(TreeNode{level:new_level, game:sorted_adv_map[i].board.clone(), children:vec![]});
            }
            re
        }
    }
}

fn sort_through_advantage_map(advantage_map: &Vec<AdvantageMap>) -> Vec<AdvantageMap> {
    if advantage_map.len() <= 1 {
        return advantage_map.clone();
    }

    let mid = advantage_map.len() / 2;
    let left:Vec<AdvantageMap> = sort_through_advantage_map(&advantage_map[..mid].to_vec());
    let right:Vec<AdvantageMap> = sort_through_advantage_map(&advantage_map[mid..].to_vec());

    merge(&left, &right)
}

fn merge(left: &Vec<AdvantageMap>, right: &Vec<AdvantageMap>) -> Vec<AdvantageMap> {
    let mut merged:Vec<&AdvantageMap> = Vec::with_capacity(left.len() + right.len());
    let mut left_iter = left.iter();
    let mut right_iter = right.iter();

    let mut left_item = left_iter.next();
    let mut right_item = right_iter.next();

    while let (Some(l), Some(r)) = (left_item, right_item) {
        if l.advantage >= r.advantage {
            merged.push(l.clone());
            left_item = left_iter.next();
        } else {
            merged.push(r.clone());
            right_item = right_iter.next();
        }
    }

    if let Some(l) = left_item {
        merged.push(l.clone());
        merged.extend(left_iter.cloned());
    }
    if let Some(r) = right_item {
        merged.push(r.clone());
        merged.extend(right_iter.cloned());
    }

    merged
}





pub fn search(mut curr_game: TreeNode, depth: i32, width: i32, mut alpha: f32, mut beta: f32) -> f32 {      
    let game_states=GameStateIdentifiers::new();
    // Check for terminal conditions
    // if its ai turn the player will want to minimize, so we end on ai turn
    // note - at depth 0, ai just moved. Therefore ai will maximize even levels and player will maximize odd
    if (curr_game.level >= depth && ai_turn(&curr_game)) || 
       curr_game.game.ai_advantage == game_states.ai_checkmate || 
       curr_game.game.ai_advantage == game_states.stalemate || 
       curr_game.game.ai_advantage == game_states.player_checkmate {
        return curr_game.game.ai_advantage; 
    }

    let cloned_game = curr_game.clone();
    let children = generate_top_moves(width, cloned_game); // Generate the top `width` moves
    curr_game.children = children.clone();
    //level 0 nodes-player turn. 
    //so level 0 children is ai turn 

    // if curr_game.level % 2 == 1 { // Maximizing ai
    //     let mut max_eval = -f32::INFINITY;
    //     for child in children {
    //         let eval = search(child, depth, width, alpha, beta, model);
    //         max_eval = max_eval.max(eval);
    //         alpha = alpha.max(eval);
    //         if beta <= alpha {
    //             break; 
    //         }
    //     }
    //     return max_eval;
    // } else { 
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
    // }
}


