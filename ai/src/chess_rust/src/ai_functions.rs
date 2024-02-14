pub function game_still_going(board:&mut Board)->f64{
if turn%2==0{
    let mut no_moves=true// 0 for in play, 1 for checkmate, 2 for stalemate
    for &i in board.white_pieces.iter{
        if board.white_available_moves[i].length()>0{
            no_moves=false
        }
    }
    if no_moves{
        board.in_play=false;
        if{
            board.in_check_stored==false{
                return 2000;//stalemate
            }
        }
        
        else{
            return 100000// ai wins
        }//put neural network here
    }
}
else{
    let mut no_moves=true// 0 for in play, 1 for checkmate, 2 for stalemate
    for &i in board.black_pieces.iter{
        if board.black_available_moves[i].length()>0{
            no_moves=false
        }
    }
    if no_moves{
        board.in_play=false;
        if{
            board.in_check_stored==false{
                return 2000;//stalemate
            }
        }
        
        else{
            return 100000// ai wins
        }//put neural network here
    }
}
    return 0; 
}
pub fn ap_function(full_board:&vec<vec<i32>>, turn:i32) -> f64 {

}