use std::vec::Vec;
use std::collections::HashMap;
use crate::upper_move_functions::{all_moves_gen, move_piece};
use crate::ai_functions::{board_position_advantage_eval, game_still_going};
use crate::types::{Board, Kind, Move, Piece, PieceId, TreeNode};

fn generate_top_moves(curr_game: &mut Board, num_moves: i32)->Vec<Board> {
    all_moves_gen(curr_game); 
    if game_still_going(curr_game)!=0{
        curr_game.ai_advantage=game_still_going(curr_game); 
    }
    let mut advantage_map:Vec<AdavantageMap>=vec![];
    let place_holder=curr_game.clone();
    if curr_game.turn%2==0{
        for &i in curr_game.white_pieces.iter_mut{
            let moves=white_available_moves.get(i).unwrap_or(vec![]);
            for &j in moves.iter_mut{
                move_piece(curr_game, i, j);
                let potential_move=curr_game.clone();
                all_moves_gen(potential_move);
                if game_still_going(potential_move)==100000{
                    let advantage=100000;
                    potential_move.in_play=false;
                    advantage_map.push(potential_move, advantage);
                    Continue;
                }
                else{
                    let advantage=board_position_advantage_eval(potential_move.full_board, potential_move.turn);
                    if game_still_going(potential_move)==2000{
                        potential_move.in_play=false;
                    }
                    advantage_map.push(potential_move, advantage);
                }
                curr_game=place_holder;
            }
    }
    }
    else{
        for i in curr_game.black_pieces.iter_mut{
            let moves=black_available_moves.get(i).unwrap_or(vec![]);
            for &j in moves.iter_mut{
                move_piece(curr_game, i, j);
                let potential_move=curr_game.clone();
                all_moves_gen(potential_move);
                if game_still_going(potential_move)==100000{
                    let advantage=100000;
                    potential_move.in_play=false;
                    advantage_map.push(potential_move, advantage);
                    Continue;
                }
                else{
                    let advantage=board_position_advantage_eval(potential_move.full_board, potential_move.turn);
                    if game_still_going(potential_move)==2000{
                        potential_move.in_play=false;
                    }
                    advantage_map.push(potential_move, advantage);
                }
                curr_game=place_holder;
            }
    }
    }
    let re:Vec<AdavantageMap>=vec![]
    for i in advantage_map.iter_mut{
        re.sort(i.advantage);
        if (ai_team_is_white && board.turn==0) || (!ai_team_is_white && board.turn==1){
            i.board.ai_advantage=-i.advantage;
        }
        else{
            i.board.ai_advantage=i.advantage;
        }
        if re.length()<num_moves || num_moves==-1{
            re.push(i);
        }
        else{
            if i.advantage<re[0].advantage{
                re.remove(0);
                re.push(i);
            }
        }

    }
    for i in re.iter_mut{
        if i+1<re[re.length()-1]{
            re.remove(i);
        }
    }
    return re;
}

fn add_branches(game:&mut TreeNode, num_moves:i32){
    let children=generate_top_moves(game, num_moves);
    for i in children.iter{
        game.children=TreeNode{Vec::New, game, i.board, game.level+1}
    }
}
fn search(curr_game:&mut TreeNode, depth:i32, width:i32, alpha_beta:f64, &mut mini_max:f64)->f64{
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

pub fn ai_move(&mut board:Board, difficulty:i32)->Board{
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
pub fn player_move(&mut board:Board, start_indexes:i32, end_indexes:i32){
    if board.ai_team_is_white{
        let piece=board.black_i_to_p.get(start_indexes).unwrap_or(PieceId::Empty);
        move_piece(board, piece, end_indexes);
    }
    else{
        let piece=board.black_i_to_p.get(start_indexes);
        move_piece(board, piece, end_indexes).unwrap_or(PieceId::Empty);
    }
}