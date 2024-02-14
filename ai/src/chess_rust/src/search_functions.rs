
pub fn generate_top_moves(curr_game: &mut Board, num_moves: i32)->vec<Board> {
    all_moves_gen(curr_game); 
    if game_still_going(curr_game)!=0{
        curr_game.ai_advantage=game_still_going(curr_game); 
    }
    let mut advantage_map:vec<AdavantageMap>=vec::New()
    let place_holder=curr_game.clone();
    if curr_game.turn%2==0{
        for &i in curr_game.white_pieces.iter_mut{
            let moves=white_available_moves.get(i);
            for &j in moves.iter_mut{
                move(curr_game, i, j);
                let potential_move=curr_game.clone();
                all_moves_gen(potential_move);
                if game_still_going(potential_move)==100000{
                    let advantage=100000;
                    potential_move.in_play=false;
                    advantage_map.push(potential_move, advantage);
                    Continue;
                }
                else{
                    let advantage=ap_function(potential_move.full_board, potential_move.turn);
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
            let moves=black_available_moves.get(i);
            for &j in moves.iter_mut{
                move(curr_game, i, j);
                let potential_move=curr_game.clone();
                all_moves_gen(potential_move);
                if game_still_going(potential_move)==100000{
                    let advantage=100000;
                    potential_move.in_play=false;
                    advantage_map.push(potential_move, advantage);
                    Continue;
                }
                else{
                    let advantage=ap_function(potential_move.full_board, potential_move.turn);
                    if game_still_going(potential_move)==2000{
                        potential_move.in_play=false;
                    }
                    advantage_map.push(potential_move, advantage);
                }
                curr_game=place_holder;
            }
    }
    }
    let re:vec<AdavantageMap>=vec::New()
    for i in advantage_map.iter_mut{
        re.sort(i.advantage)
        if (ai_team_is_white && board.turn==0) || (!ai_team_is_white && board.turn==1){
            i.board.ai_advantage=-i.advantage;
        }
        else{
            i.board.ai_advantage=i.advantage;
        }
        if re.length()<num_moves{
            re.push(i);
        }
        else{
            if i.advantage<re[re.length()-1].advantage{
                re.remove(re.length()-1);
                re.push(i);
            }
        }

    }
    return re;
}

pub fn search(curr_game:TreeNode, depth:i32, alpha_beta:f64, mini_max:f64){
    if curr_game.level==depth || curr_game.in_play==false{
        if curr_game.board.ai_advantage<mini_max{
            return curr_game.board.ai_advantage; 
        }
    }
    if curr_game.game.ai_advantage<alpha_beta{
        mini_max=alpha_beta; 
        while level>1{
            curr_game.children=vec::New();
            curr_game=curr_game.parent;
            level=level-1;
        }
        return mini_max; 
    }
    else{
        if destroy 
    }
}   