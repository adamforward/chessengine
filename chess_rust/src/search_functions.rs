use std::{cell::RefCell, rc::Rc};
use crate::upper_move_functions::{all_moves_gen, move_piece};
use crate::ai_functions::{ game_still_going, board_position_advantage_eval};
use crate::types::{TreeNode, TreeNodeRef, Board, AdavantageMap, PieceId};

pub fn generate_top_moves(num_moves: i32, parent:TreeNode)->Vec<TreeNodeRef> {
    //this takes a game, makes all available moves then returns a vector of length num_moves for the highest win probabilities
    //board_position_advantage_eval evaluates the probability that ai wins by checkmate via convolutional nn
    //this function is set up to use the datatypes that allow for a tree structure
    let mut curr_game=parent.game.clone();
    let level=parent.level;
    let parent_ref=Some(Rc::new(RefCell::new(parent)));
    let new_info=all_moves_gen(&curr_game);
    if game_still_going(&curr_game, new_info.checking, &new_info.moves)!=0.1{
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
                let advantage=board_position_advantage_eval(&potential_move.full_board, curr_game.ai_team_is_white, "");
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
                let advantage=board_position_advantage_eval(&potential_move.full_board, curr_game.ai_team_is_white, "");
                advantage_map.push(AdavantageMap{board:potential_move, advantage});
            }
        }
    }

    let mut re:Vec<TreeNodeRef>=vec![];
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

        if ai_turn && game_still_going(&curr_game, new_info.checking, &new_info.moves)==1.0{
            e.board.ai_advantage=1.0;//always move into checkmate
            return vec![Rc::new(RefCell::new(TreeNode { parent:parent_ref.clone(), children:vec![],game:e.board.clone(), level}))];
            
        }

        if !ai_turn && game_still_going(&curr_game, new_info.checking, &new_info.moves)==0.0{
            e.board.ai_advantage=0.0; //always assume player moves into checkmate
            return vec![Rc::new(RefCell::new(TreeNode { parent:parent_ref.clone(), children:vec![],game:e.board.clone(), level}))];
        }

        for i in immut_advantage_map.iter(){
            let mut count=0; 
            if ai_turn{
                if e.board.ai_advantage<i.board.ai_advantage{
                    count+=1;//ai wants to maximize ai win probability
                }
                if count<=num_moves{
                    re.push(Rc::new(RefCell::new(TreeNode { parent:parent_ref.clone(), children:vec![],game:e.board.clone(), level})));
                }
            }
            else{
                if e.board.ai_advantage>i.board.ai_advantage{
                    count+=1;//player wants to minimize it
                }
                if count<=num_moves{
                    re.push(Rc::new(RefCell::new(TreeNode { parent:parent_ref.clone(), children:vec![],game:e.board.clone(), level})));
                }
            }
        }
    }
    re
}

fn clear_children(curr_game: &TreeNodeRef) {
    let mut current = curr_game.clone();
    while current.borrow().level != 0 {
        current.borrow_mut().children = Vec::new();
        let parent = Rc::clone(&current);
        current = parent; 
    }
    current.borrow_mut().children = Vec::new(); 
}

pub fn search(curr_game: &Rc<RefCell<TreeNode>>, depth: i32, width: i32, alpha_beta: f32, mut mini_max: f32) -> f32 {
    let curr_game_borrowed = curr_game.borrow();
    
    if (curr_game_borrowed.level == depth || 
        curr_game_borrowed.game.ai_advantage == 1.0 || 
        curr_game_borrowed.game.ai_advantage == 0.0 || 
        curr_game_borrowed.game.ai_advantage == 0.5) && 
        mini_max < alpha_beta { // tree ends with checkmate or stalemate or end of search depth
            
        if curr_game_borrowed.game.ai_advantage < mini_max {
            //return the player's best move if this happens
            return curr_game_borrowed.game.ai_advantage; 
        }

    } else if (curr_game_borrowed.level == depth || 
               curr_game_borrowed.game.ai_advantage == 0.0 || 
               curr_game_borrowed.game.ai_advantage == 1.0 || 
               curr_game_borrowed.game.ai_advantage == 0.5) && 
              mini_max > alpha_beta {
                //exit tree if there is a player's final move that is better than another searched option
                clear_children(curr_game);
    } else {

        let cloned_game = curr_game.clone();
        let children = generate_top_moves(width, cloned_game.borrow().clone());
        
        curr_game.borrow_mut().children = children.clone();
        //curr_game.borrow_mut.children=children.clone();
        for child in children {
            if mini_max<alpha_beta{
                mini_max=0.0;//get out if minimax<alphabeta
                break;
            }

            let mm = search(&child, depth, width, alpha_beta, mini_max);

            if mini_max > mm {
                mini_max = mm;
            }

            
        }
    }

    if mini_max<alpha_beta{
        return 0.0;
    }
    
    mini_max
}
