use std::{cell::RefCell, rc::Rc};
use crate::upper_move_functions::{all_moves_gen, move_piece};
use crate::ai_functions::{ game_still_going};
use crate::types::{TreeNode, TreeNodeRef, Board, AdavantageMap, PieceId};

fn generate_top_moves(num_moves: i32, parent:TreeNode)->Vec<TreeNodeRef> {
    let mut curr_game=parent.game.clone();
    let level=parent.level;
    let parent_ref=Some(Rc::new(RefCell::new(parent)));
    let new_info=all_moves_gen(&curr_game);
    if game_still_going(&curr_game, new_info.checking, &new_info.moves, &new_info.moves)!=0.0{
        curr_game.ai_advantage=game_still_going(&curr_game, new_info.checking, &new_info.moves, &new_info.moves); 
        return vec![];
    }
    let mut advantage_map:Vec<AdavantageMap>=vec![];
    if curr_game.turn%2==0{
        for &i in curr_game.white_piece_ids.iter(){
            let moves=new_info.moves.get_moves(i);
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
            let moves=new_info.moves.get_moves(i);
            for &j in moves.iter(){
                let param_move=curr_game.clone();
                let potential_move=move_piece(param_move, i, j);
                //let advantage=board_position_advantage_eval(potential_move.full_board, potential_move.turn); call neural network
                let advantage=0.0;
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
    let immut_adv=advantage_map.clone();
    let mut done=false;
    for e in advantage_map.iter_mut() {
        if e.board.ai_team_is_white && e.board.turn==1{
            e.board.ai_advantage=-e.advantage;
        }
        else{
            e.board.ai_advantage=e.advantage;
        }
        if ai_turn && game_still_going(&curr_game, new_info.checking, &new_info.moves, &new_info.moves)==100000.0{
            e.board.ai_advantage=1000000.0;
            re.push(Rc::new(RefCell::new(TreeNode { parent:parent_ref.clone(), children:vec![],game:e.board.clone(), level})));
            done=true;
        }
        if !ai_turn && game_still_going(&curr_game, new_info.checking, &new_info.moves, &new_info.moves)==-100000.0{
            e.board.ai_advantage=-1000000.0;
            re.push(Rc::new(RefCell::new(TreeNode {parent:parent_ref.clone(), children:vec![],game:e.board.clone(), level})));
            done=true;
        }
        let mut loss_count=0;
        let mut count=0;
        for j in immut_adv.iter(){
            if !ai_turn && game_still_going(&curr_game, new_info.checking, &new_info.moves, &new_info.moves)==-100000.0 && loss_count<num_moves{
                loss_count+=1;
            }
            if ai_turn && game_still_going(&curr_game, new_info.checking, &new_info.moves, &new_info.moves)==100000.0 && loss_count<num_moves{
                loss_count+=1;
            }
            if e.advantage<j.advantage{
                count+=1;
            }
        }
        if count<=num_moves && loss_count!=num_moves && !done {
            re.push(Rc::new(RefCell::new(TreeNode {parent:parent_ref.clone(), children:vec![],game:e.board.clone(), level})));
        }
        if loss_count==num_moves && ai_turn{
            let mut loss_re=e.board.clone();
            loss_re.ai_advantage=-1000000.0;
            re.push(Rc::new(RefCell::new(TreeNode {parent:parent_ref.clone(), children:vec![],game:e.board.clone(), level})));
        }
        if loss_count==num_moves && !ai_turn{
            let mut loss_re=e.board.clone();
            loss_re.ai_advantage=-1000000.0;
            re.push(Rc::new(RefCell::new(TreeNode {parent:parent_ref.clone(), children:vec![],game:e.board.clone(), level})));
        }
    }
    return re;
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

fn search(curr_game: &Rc<RefCell<TreeNode>>, depth: i32, width: i32, alpha_beta: f64, mut mini_max: f64) -> f64 {
    let curr_game_borrowed = curr_game.borrow();
    
    if (curr_game_borrowed.level == depth || 
        curr_game_borrowed.game.ai_advantage == 1000000.0 || 
        curr_game_borrowed.game.ai_advantage == -1000000.0 || 
        curr_game_borrowed.game.ai_advantage == 2000.0) && 
        mini_max > alpha_beta {
        if curr_game_borrowed.game.ai_advantage < mini_max {
            return curr_game_borrowed.game.ai_advantage; 
        }
    } else if (curr_game_borrowed.level == depth || 
               curr_game_borrowed.game.ai_advantage == 1000000.0 || 
               curr_game_borrowed.game.ai_advantage == -1000000.0 || 
               curr_game_borrowed.game.ai_advantage == 2000.0) && 
              mini_max < alpha_beta {
                clear_children(curr_game);
    } else {
        let cloned_game = curr_game.clone();
        let children = generate_top_moves(width, cloned_game.borrow().clone());
        
        curr_game.borrow_mut().children = children.clone();
        
        if mini_max > curr_game_borrowed.game.ai_advantage {
            mini_max = curr_game_borrowed.game.ai_advantage;
        }

        for child in children {
            let mm = search(&child, depth, width, alpha_beta, mini_max);
            if mini_max > mm {
                mini_max = mm;
            }
        }
    }
    
    mini_max
}
