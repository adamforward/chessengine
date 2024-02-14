pub function ai_advantage(board:Board){
if turn%2==0{
    let mut no_moves=true// 0 for in play, 1 for checkmate, 2 for stalemate
    for &i in board.white_pieces.iter{
        if board.white_available_moves[i].length()>0{
            no_moves=false
        }
    }
    for &i in board.black_pieces.iter{
        if board.black_available_moves[i].length()>0{
            no_moves=false
        }
    }
    if no_moves{
        if{
            board.in_check_stored==false{
                return 20000;//stalemate
            }
        }
        if board.ai_team_is_white && board.turn%2==0 || board.ai_team_is_black && board.turn%2==1{
            return -100000//ai loses
        }
        else{
            return 100000// ai wins
        }
        //put neural network here
    }
}
}