use std::vec::Vec;
use std::collections::HashMap;
use crate::upper_move_functions::{all_moves_gen, move_piece};
use crate::ai_functions::{board_position_advantage_eval, game_still_going};
use crate::types::{TreeNode, Board, AdavantageMap, PieceId};

fn generate_top_moves(mut curr_game: Board, num_moves: i32, parent:Vec<TreeNode>, level:i32)->Vec<&TreeNode> {
    let new_info=all_moves_gen(&curr_game);
    if game_still_going(&curr_game, new_info.checking, &new_info.white_moves, &new_info.black_moves)!=0.0{
        curr_game.ai_advantage=game_still_going(&curr_game, new_info.checking, &new_info.white_moves, &new_info.black_moves); 
        return vec![];
    }
    let mut advantage_map:Vec<AdavantageMap>=vec![];
    if curr_game.turn%2==0{
        for &i in curr_game.white_piece_ids.iter(){
            let moves=new_info.white_moves.get_moves(i);
            for &j in moves.iter(){
                let param_move=curr_game.clone();
                let potential_move=move_piece(param_move, i, j);
                //let advantage=board_position_advantage_eval(potential_move.full_board, potential_move.turn); call neural network
                let advantage=0.0;
                advantage_map.push(AdavantageMap{board:potential_move, advantage});
            }
        }
    }
    else{
        for &i in curr_game.black_piece_ids.iter(){
            let moves=new_info.black_moves.get_moves(i);
            for &j in moves.iter(){
                let param_move=curr_game.clone();
                let potential_move=move_piece(param_move, i, j);
                //let advantage=board_position_advantage_eval(potential_move.full_board, potential_move.turn); call neural network
                let advantage=0.0;
                advantage_map.push(AdavantageMap{board:potential_move, advantage});
            }
        }
    }
    let mut re:Vec<&TreeNode>=vec![];
    let ai_turn:bool;
    if ((curr_game.turn+1)%2==0 && curr_game.ai_team_is_white) || ((curr_game.turn+1)%2==1 && !curr_game.ai_team_is_white){
        ai_turn=true;
    }
    else {
        ai_turn=false
    }
    let immut_adv=advantage_map.clone();
    let mut done=false;
    for e in advantage_map.iter_mut() {
        if e.board.ai_team_is_white && e.board.turn==1{
            e.board.ai_advantage=-e.advantage;
        }
        else{
            e.board.ai_advantage=e.advantage;
        }
        if ai_turn && game_still_going(&curr_game, new_info.checking, &new_info.white_moves, &new_info.black_moves)==100000.0{
            e.board.ai_advantage=1000000.0;
            re.push(TreeNode::new(e.board.clone(), &parent, level));
            done=true;
        }
        if !ai_turn && game_still_going(&curr_game, new_info.checking, &new_info.white_moves, &new_info.black_moves)==-100000.0{
            e.board.ai_advantage=-1000000.0;
            re.push(TreeNode::new(e.board.clone(), &parent, level));
            done=true;
        }
        let mut loss_count=0;
        let mut count=0;
        for j in immut_adv.iter(){
            if !ai_turn && game_still_going(&curr_game, new_info.checking, &new_info.white_moves, &new_info.black_moves)==-100000.0 && loss_count<num_moves{
                loss_count+=1;
            }
            if ai_turn && game_still_going(&curr_game, new_info.checking, &new_info.white_moves, &new_info.black_moves)==100000.0 && loss_count<num_moves{
                loss_count+=1;
            }
            if e.advantage<j.advantage{
                count+=1;
            }
        }
        if count<=num_moves && loss_count!=num_moves && !done {
            re.push(TreeNode::new(e.board.clone(), &parent, level));
        }
        if loss_count==num_moves && ai_turn{
            let mut loss_re=e.board.clone();
            loss_re.ai_advantage=-1000000.0;
            re.push(TreeNode::new(e.board.clone(), &parent, level));
        }
        if loss_count==num_moves && !ai_turn{
            let mut loss_re=e.board.clone();
            loss_re.ai_advantage=-1000000.0;
            re.push(TreeNode::new(e.board.clone(), &parent, level));
        }
    }
    return re;
}

fn add_branches(board: TreeNode, num_moves:i32){
    board.children=generate_top_moves(game.board, num_moves);
    board.child
}
fn search(mut curr_game: TreeNode, depth:i32, width:i32, alpha_beta:f64, mut mini_max:f64)->f64{
    if curr_game.level==depth || curr_game.in_play==false{
        if curr_game.board.ai_advantage<mini_max{
            return curr_game.board.ai_advantage; 
        }
    }
    if curr_game.game.ai_advantage<alpha_beta{
        mini_max=-2; 
        while level>1{
            curr_game.children=vec![];
            curr_game=curr_game.parent;
            level=level-1;
        }
        curr_game.ai_advantage=-2;
        return mini_max; 
    }
    else{
        if curr_game.ai_advantage!=-2{
            add_branches(curr_game, width);
            if mini_max>curr_game.ai_advantage{
                mini_max=curr_game.ai_advantage;
            }
            for i in curr_game.children{
                let mm=search(i, depth, width, alpha_beta, mini_max);
                if mm==-2{
                    break;
                }
                if mini_max>mm{
                    mini_max=mm;
                }
            }
        }
    }
    return mini_max;
}

pub fn ai_move(mut board:Board, difficulty:i32)->Board{
    let potential_moves=generate_top_moves(board, -1);
    let mut outcome=-10000000.0;
    let mut alpha_beta=outcome;
    let winner:Board;
    for i in boards.iter{
        let tree=TreeNode{vec![], vec![], i.board, 1}; 
        let temp_search=search(tree, difficulty, difficulty/2, alpha_beta, outcome);
        if temp_search>outcome{
            outcome=temp_search;
            alpha_beta=.5-outcome; 
            winner=i;
        }
    }
    return winner; 
}
pub fn player_move(mut board:Board, start_indexes:i32, end_indexes:i32){
    if board.ai_team_is_white{
        let piece=board.black_i_to_p.get(start_indexes).unwrap_or(PieceId::Empty);
        move_piece(board, piece, end_indexes);
    }
    else{
        let piece=board.black_i_to_p.get(start_indexes);
        move_piece(board, piece, end_indexes).unwrap_or(PieceId::Empty);
    }
}