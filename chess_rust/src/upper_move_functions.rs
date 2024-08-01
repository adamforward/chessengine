use crate::types::{AllMovesGenRe, Team, AvailableMovesMap, Board, Kind, Piece, PieceId};
use crate::base_functions::{find_overlap, map_piece_id_to_kind, contains_element, primes, primes1, pawn_to_queen, pawn_to_knight, find_non_overlap};
use crate::base_move_functions::{generate_available_moves};
use crate::upper_move_function_helpers::{in_check_directional, b_rook_pinning, w_rook_pinning, b_bishop_pinning, w_bishop_pinning};
pub fn all_moves_gen(board: &Board)->AllMovesGenRe {
    // Generate available moves gets most of the moves right, and is a much more simple function. 
    // It handles where pieces can move, outside of castling and moving into check.
    // generate_available_moves takes indexes as a parameter and uses the full board to get this information for an individual piece. 

    // all_moves_gen returns a complete set of moves in the AllMovesGenRe structure, defined in types.rs. 
    // basically the purpose of this function is to a hashmap that returns an array of indexes for each pieceid as a key. 
    // each index represents a legal chess move for the key
    
    // In order to avoid doing every possible move then checking which ones move the team into check (this would be very inefficient),
    // this function handles different cases where the player could move into check through a number of different ways. 

    // Castling conditions for King Side Castle
    // Information that's checked here:are squares in between rook and king empty and has r or k moved 

    let mut b_ks = board.prime2 % 13 != 0 && board.prime2 % 11 != 0 && board.full_board[0][6].team == Team::N && board.full_board[0][5].team == Team::N;
    let mut w_ks = board.prime2 % 5 != 0 && board.prime2 % 3 != 0 && board.full_board[7][6].team == Team::N && board.full_board[7][5].team == Team::N;
    let white_turn = board.turn%2==0;
    let mut re_checking=false;

    // Castling conditions for Queen Side
    let mut w_qs = board.prime2 % 5 != 0 && board.prime2 % 2 != 0 && board.full_board[7][1].team == Team::N && board.full_board[7][2].team == Team::N && board.full_board[7][3].team == Team::N;
    let mut b_qs = board.prime2 % 13 != 0 && board.prime2 % 7 != 0 && board.full_board[0][1].team == Team::N&& board.full_board[0][2].team == Team::N && board.full_board[0][3].team == Team::N;
    let mut white_moves=AvailableMovesMap::new();
    let mut black_moves=AvailableMovesMap::new();

    let mut black_checking: Vec<PieceId> = Vec::new();
    let mut white_checking: Vec<PieceId> = Vec::new();
    let mut b_pinned_vectors: Vec<Vec<usize>> = Vec::new();
    let mut w_pinned_vectors: Vec<Vec<usize>> = Vec::new();
    let mut b_pinned_indexes: Vec<usize> = Vec::new();
    let mut w_pinned_indexes: Vec<usize> = Vec::new();
    
    
    let b_king_index = board.black_indexes.get_index(PieceId::K).unwrap();
    let w_king_index = board.white_indexes.get_index(PieceId::K).unwrap();
    
    
    // King moves are generated first to look for overlap
    let mut w_king_moves = generate_available_moves(&board, w_king_index / 10, w_king_index % 10);

    let mut b_king_moves = generate_available_moves(&board, b_king_index / 10, b_king_index % 10);
    
    // These represent moves that both the other team and the king generated moves have in common
    let mut over_w: Vec<usize> = vec![];
    let mut over_b: Vec<usize> = vec![];
    let mut adjacent_to_black_k:Vec<usize>= vec![];
    let mut adjacent_to_white_k:Vec<usize>= vec![];
    let mut b_rooks:Vec<usize>=vec![];
    let mut w_rooks:Vec<usize>=vec![];
    let mut b_bishops:Vec<usize>=vec![];
    let mut w_bishops:Vec<usize>=vec![];
    let mut all_moves:Vec<usize>;

    for piece_id in board.white_piece_ids.iter(){
        //generate mappings for white pieces, pieceid is a hash key for an array of indexes it can move to

        let piece_index = board.white_indexes.get_index(*piece_id).unwrap();
        let curr_row = piece_index / 10;
        let curr_col = piece_index % 10;
        if (curr_row as isize-b_king_index as isize/10).abs()<=1 && (curr_col as isize-b_king_index as isize%10).abs()<=1 && !white_turn{
            //This is used later and is for making sure you cannot move into check
            adjacent_to_black_k.push(piece_index);
        }
        

        all_moves = generate_available_moves(&board, curr_row, curr_col); 
        //generate_available_moves gives you the moves outside of castling
        //however it does not factor in the fact that you cannot move into check
        //that's all handled in this function
        white_moves.insert_moves(*piece_id, &all_moves);
        if map_piece_id_to_kind(*piece_id)!=Kind::Pawn && !white_turn{
            //pawn overlapping is more complicated since the move only shows up if there's a piece there. 
            //this is handled later in the function
            let curr_over_b=find_overlap(&all_moves, &b_king_moves);
            for i in curr_over_b.iter(){
                over_b.push(*i)
            }
        }
        //checking is only for this loop, but a value for if one team is checking other is in the returned object
        let mut checking = false;

        if contains_element(&all_moves, b_king_index){
            re_checking=true;
            checking=true;
            white_checking.push(*piece_id);
        }

        if board.full_board[curr_row][curr_col].kind==Kind::Rook || board.full_board[curr_row][curr_col].kind==Kind::Queen && checking==false && !white_turn{
            w_rooks.push(piece_index);
            //w_rooks is used at the end of the function
            let mut temp=w_rook_pinning(&board, *piece_id, &over_b);// temp is only 2 elements
            //if there is a white piece adjacent to king, being attacked by white rook, that is handled in pinning function and passed back in temp[1]
            over_b=temp[1].clone(); 
            if !temp[0].is_empty(){
                //when there is a piece pinning blacks king, the indexes for the pinned piece are returned at the end of this vector. 
                //the rest of temp[0] is the spaces between the pinning piece and black's king, including the pinning piece
                let length=temp[0].len()-1;
                let pinned_p=temp[0][length];
                temp[0].remove(length); 
                b_pinned_vectors.push(temp[0].clone());
                b_pinned_indexes.push(pinned_p);
            }
        }
        
        if board.full_board[curr_row][curr_col].kind==Kind::Bishop || board.full_board[curr_row][curr_col].kind==Kind::Queen && checking==false && !white_turn{
            //works the same as w_rook_pinning, just different logic in the function. 
            w_bishops.push(piece_index);
            let mut temp=w_bishop_pinning(&board, *piece_id, &over_b);
            over_b=temp[1].clone();

            if !temp[0].is_empty(){
                let length=temp[0].len()-1;
                let pinned_p=temp[0][length];
                temp[0].remove(length); 
                b_pinned_vectors.push(temp[0].clone());
                b_pinned_indexes.push(pinned_p);
            }
        }
        //check if any white moves are in the castling strip.
        if !white_turn{
            let castling_points=vec![7,6,5,4];
            for i in castling_points.iter(){
                for j in all_moves.iter(){
                    b_ks=b_ks && *i!=*j
                }
            }
            let castling_points=vec![0,1,2,3,4];
            for i in castling_points.iter(){
                for j in all_moves.iter(){
                    b_qs=b_qs && *i!=*j
                }
            }
        }
    };

    for i in board.black_piece_ids.iter(){
        //does the same thing as the for loop above, just different team. 
        let piece=*i;
        let piece_index = board.black_indexes.get_index(piece).unwrap();
        let curr_row=piece_index/10;
        let curr_col=piece_index%10;
        all_moves=generate_available_moves(&board, curr_row, curr_col);

        if map_piece_id_to_kind(piece)!=Kind::Pawn && white_turn{
            let curr_over_w=find_overlap(&all_moves, &w_king_moves);
            for i in curr_over_w.iter(){
                over_w.push(*i)
            }
        }

        let mut checking=false;

        if (curr_row as isize-(w_king_index as isize)/10).abs()<=1 && (curr_col as isize-w_king_index as isize%10).abs()<=1 &&white_turn{
            adjacent_to_white_k.push(piece_index);
        }

        if contains_element(&all_moves,board.white_indexes.get_index(PieceId::K).unwrap()){
            re_checking=true;
            checking=true;
            black_checking.push(piece);
        }

        if board.full_board[curr_row][curr_col].kind==Kind::Rook || board.full_board[curr_row][curr_col].kind==Kind::Queen && checking==false && white_turn{
            b_rooks.push(piece_index);
            let mut temp=b_rook_pinning(&board, piece, &over_w);
            over_w=temp[1].clone();
            if !temp[0].is_empty(){
                let length=temp[0].len()-1;
                let pinned_p=temp[0][length];
                temp[0].remove(length);
                w_pinned_vectors.push(temp[0].clone());
                w_pinned_indexes.push(pinned_p);
            }
        }


        if board.full_board[curr_row][curr_col].kind==Kind::Bishop || board.full_board[curr_row][curr_col].kind==Kind::Queen && checking==false && white_turn{
            b_bishops.push(piece_index);
            let mut temp=b_bishop_pinning(&board, piece, &over_w);
            over_w=temp[1].clone();

            if !temp[0].is_empty(){
                let length=temp[0].len()-1;
                let pinned_p=temp[0][length];
                temp[0].remove(length);
                w_pinned_vectors.push(temp[0].clone());
                w_pinned_indexes.push(pinned_p);
            }
        }

        black_moves.insert_moves(piece, &all_moves);
        let castling_points=vec![77,76,75,74];
        if white_turn{
            for i in castling_points.iter(){
                for j in all_moves.iter(){
                    w_ks=w_ks && *i!=*j
                }
            }
            let castling_points=vec![70,71,72,73,74];
            for i in castling_points.iter(){
                for j in all_moves.iter(){
                    w_qs=w_qs && *i!=*j;
                }
            }
        }
    }
    //castling move identifiers
    if w_ks==true{
        w_king_moves.push(99);
    }
    if w_qs==true{
        w_king_moves.push(100)
    }
    if b_ks==true{
        b_king_moves.push(99);
    }
    if b_qs==true{
        b_king_moves.push(100)
    }

    
    for (index, pinned_index) in w_pinned_indexes.iter().enumerate(){
        let piece=board.white_i_to_p.get_piece(*pinned_index).unwrap();
        let moves=white_moves.get_moves(piece);
        //can only move pinned piece into moves that continue to block the check. 
        white_moves.insert_moves(piece, &find_overlap(&moves, &w_pinned_vectors[index]));
    }

    
    

    for (index, pinned_index) in b_pinned_indexes.iter().enumerate(){
        let piece=board.black_i_to_p.get_piece(*pinned_index).unwrap();
        let moves=black_moves.get_moves(piece);
        black_moves.insert_moves(piece, &find_overlap(&moves, &b_pinned_vectors[index]));
    }
    //take out moves where king is moving into direct attack from other king.
    let updated_king_moves_w=find_non_overlap(over_w, w_king_moves);
    let updated_king_moves_b=find_non_overlap(over_b, b_king_moves);
    white_moves.insert_moves(PieceId::K,&updated_king_moves_w);
    black_moves.insert_moves(PieceId::K,&updated_king_moves_b);
    

    for i in black_checking{
        let k=map_piece_id_to_kind(i);
        if k==Kind::Rook || k==Kind::Bishop || k==Kind::Queen{
            white_moves=in_check_directional(&board, &white_moves, i, Team::B).clone();
            //directional checking is handled differently because you can block the check without capturing or moving king. 
        }
        else{ //knight or pawn checking, either have to take or move the king. 
            for piece_id in board.white_piece_ids.iter() {
                let pressuring_i=board.black_indexes.get_index(i).unwrap();
                if *piece_id!=PieceId::K{
                    let valid=vec![pressuring_i];
                    let moves = find_overlap(&white_moves.get_moves(*piece_id), &valid);
                    white_moves.insert_moves(*piece_id,&moves)
                }
            }   
        }
    }

    for i in white_checking{//same logic here. 
        let k=map_piece_id_to_kind(i);
        if k==Kind::Rook || k==Kind::Bishop || k==Kind::Queen{
            black_moves=in_check_directional(&board, &black_moves, i, Team::W);
        }
        else{
            for piece_id in board.black_piece_ids.iter() {
                let pressuring_i=board.white_indexes.get_index(i).unwrap();
                if *piece_id!=PieceId::K{
                    let valid=vec![pressuring_i];
                    let moves = find_overlap(&black_moves.get_moves(*piece_id), &valid);
                    black_moves.insert_moves(*piece_id,&moves);
                }
                }
            }
        }
    //need to update king moves again after the checking function for the checks below.  
    let mut updated_king_moves_black=black_moves.get_moves(PieceId::K);
    let mut updated_king_moves_white=white_moves.get_moves(PieceId::K);
        
        
    //this is for when there is a knight attacking a piece adjacent to the king (ex white knight attacking white pawn next to black king)
    let knight_moves: [(isize, isize); 8] = [
    (1, 2),
    (1, -2),
    (-1, 2),
    (-1, -2),
    (2, 1),
    (2, -1),
    (-2, 1),
    (-2, -1),
    ];

    if !white_turn{
        for i in black_moves.get_moves(PieceId::K){
            //once again, this is one of the many unessessary checks to do on both turns until debugging is done. 

            //special pawn case, moves are not generated that attack empty squares next to king in generate_available_moves for pawns. 
            if i%10<=6 && i/10<=6{
                if board.full_board[i/10+1][i%10+1].kind==Kind::Pawn && board.full_board[i/10+1][i%10+1].team==Team::W{
                    if let Some(pos) = updated_king_moves_black.iter().position(|x| *x == i) {
                        updated_king_moves_black.remove(pos);
                    }
                }
            }
            if i%10>=1 && i/10<=6{
                if board.full_board[i/10+1][i%10-1].kind==Kind::Pawn && board.full_board[i/10+1][i%10-1].team==Team::W{
                    if let Some(pos) = updated_king_moves_black.iter().position(|x| *x == i) {
                        updated_king_moves_black.remove(pos);
                    }
                }
            }

            //takes out moves where king is adjacent
            let w_king_location=board.white_indexes.get_index(PieceId::K).unwrap() as i32;
            if ((i as i32)/10-w_king_location/10).abs()<=1 && ((i as i32)%10-w_king_location%10).abs()<=1{
                if let Some(pos) = updated_king_moves_black.iter().position(|x| *x == i) {
                    updated_king_moves_black.remove(pos);
                }
            }
            }
            
            let b_king_r=board.black_indexes.get_index(PieceId::K).unwrap()/10;
            let b_king_c=board.black_indexes.get_index(PieceId::K).unwrap()%10;
            for location in adjacent_to_black_k.iter(){
            let l=*location as isize;
            let mut done=true;
            
            for j in w_bishops.iter(){
                //this is for if there's a white piece attacked by the white bishop adjacent to the black king. 
                //taking the white piece for the king would move into check.  
                let k=*j as isize;
                if k==l{
                    continue;
                }
                if (k/10 as isize - b_king_r as isize).abs() != (k%10 - b_king_c as isize).abs(){//already handled in pinning functions
                if (k/10 as isize - l/10 as isize).abs() == (k%10 as isize - l%10 as isize).abs(){
                        let direction_r:isize = if k/10 > l/10 { 1 } else { -1 };
                        let direction_c:isize= if k%10 > l%10 { 1 } else { -1 };
                        let magnitude=(k/10 as isize - l/10 as isize).abs();
                        for scalar_inc in 1..magnitude{
                            let r_inc=l/10+(scalar_inc*direction_r);
                            let c_inc=l%10+(scalar_inc*direction_c);
                            if let Some(piece) = board
                                .full_board
                                .get((r_inc) as usize)
                                .and_then(|row| row.get(c_inc as usize))
                        {
                            if piece.team!=Team::N{
                                done=false; 
                            }
                        }
                }
                if done && magnitude>0{
                    if let Some(pos) = updated_king_moves_black.iter().position(|x| *x == *location) {
                            updated_king_moves_black.remove(pos);
                    }
                }
                }
            }
            }

            for i in w_rooks.iter(){
                let k=*i as isize;
                if k==l{
                    continue;
                }
                if l/10==k/10{
                    let magnitude=(l%10-k%10).abs();
                    let direction:isize = if k%10 > l%10 { 1 } else { -1 };
                    let mut done=true;
                    for scalar_inc in 1..magnitude{
                        let c_inc=l%10+(scalar_inc*direction);
                        if let Some(piece) = board
                                .full_board
                                .get((l/10) as usize)
                                .and_then(|row| row.get((c_inc) as usize))
                        {
                            if piece.team!=Team::N{
                                done=false; 
                                break;
                            }
                        }
                    }
                    
                    if done && magnitude>0{
                        if let Some(pos) = updated_king_moves_black.iter().position(|x| *x == *location) {
                            updated_king_moves_black.remove(pos);
                        }
                    }
                }
                
                if l%10==k%10{
                    let magnitude=(l/10-k/10).abs();
                    let direction:isize = if k/10 > l/10 { 1 } else { -1 };
                    let mut done=true;
                    for scalar_inc in 1..magnitude{
                        let r_inc=l/10+(scalar_inc*direction);
                        if let Some(piece) = board
                                .full_board
                                .get(r_inc as usize)
                                .and_then(|row| row.get((l%10) as usize))
                        {
                            if piece.team!=Team::N{
                                done=false; 
                                break;
                            }
                        }
                    }
                    if done && magnitude>0{
                        if let Some(pos) = updated_king_moves_black.iter().position(|x| *x == *location) {
                            updated_king_moves_black.remove(pos);
                        }
                    }
                    }
            }

            
            for (x, y) in knight_moves.iter() {
            let move_row = l/10 + x;
            let move_col = l%10 + y;

            // Check if the move is within the bounds of the board
            if move_row >= 0 && move_row < 8 && move_col >= 0 && move_col < 8 {
                let move_row = move_row as usize;
                let move_col = move_col as usize;
                if board.full_board[move_row][move_col].kind==Kind::Knight && board.full_board[move_row][move_col].team==Team::W{
                    if let Some(pos) = updated_king_moves_black.iter().position(|x| *x == *location) {
                            updated_king_moves_black.remove(pos);
                    }
                }
            }
        }
    }
}

        let w_king_i=board.white_indexes.get_index(PieceId::K).unwrap();
        let w_king_r=w_king_i/10;
        let w_king_c=w_king_i/10;
        if white_turn{
        for i in white_moves.get_moves(PieceId::K){

        //same code but for white instead of black. 
        if i%10<=6 && i/10<=6{
            if board.full_board[i/10+1][i%10+1].kind==Kind::Pawn&& board.full_board[i/10+1][i%10+1].team==Team::B{
                if let Some(pos) = updated_king_moves_white.iter().position(|x| *x == i) {
                    updated_king_moves_white.remove(pos);
                }
            }
        }
        if i%10>=1 && i/10<=6{
            if board.full_board[i/10+1][i%10-1].kind==Kind::Pawn && board.full_board[i/10+1][i%10-1].team==Team::B{
                if let Some(pos) = updated_king_moves_white.iter().position(|x| *x == i) {
                    updated_king_moves_white.remove(pos);
                }
            }
        }
        let b_king_location=board.black_indexes.get_index(PieceId::K).unwrap() as i32;
        if ((i as i32)/10-b_king_location/10).abs()<=1 && ((i as i32)%10-b_king_location%10).abs()<=1{
            if let Some(pos) = updated_king_moves_white.iter().position(|x| *x == i) {
                updated_king_moves_white.remove(pos);
            }
        }
        }

        
        for location in adjacent_to_white_k.iter(){
        let l=*location as isize;
        
        for j in b_bishops.iter(){
            let k=*j as isize;
            if k==l{
                continue;
            }
            let mut done=false;
            if (k/10 as isize - w_king_r as isize).abs() != (k%10 - w_king_c as isize).abs(){//already handled in pinning functions
               if (k/10 as isize - l/10 as isize).abs() == (k%10 as isize - l%10 as isize).abs(){
                    let direction_r:isize = if k/10 > l/10 { 1 } else { -1 };
                    let direction_c:isize= if k%10 > l%10 { 1 } else { -1 };
                    let magnitude=(k/10 as isize - l/10 as isize).abs();
                    for scalar_inc in 1..magnitude{
                        let r_inc=l/10+(scalar_inc*direction_r);
                        let c_inc=l%10+(scalar_inc*direction_c);
                        if let Some(piece) = board
                            .full_board
                            .get((r_inc) as usize)
                            .and_then(|row| row.get(c_inc as usize))
                    {
                        if piece.team!=Team::N{
                            done=true; 
                        }
                    }
               }
               if done && magnitude>0{
                if let Some(pos) = updated_king_moves_white.iter().position(|x| *x == *location) {
                    updated_king_moves_white.remove(pos);
                }
               }
            }
        }
        }

        for i in b_rooks.iter(){
            let k=*i as isize;
            if k==l{
                continue;
            }
            if l/10==k/10{
                let magnitude=(l%10-k%10).abs();
                let direction:isize = if k%10 > l%10 { 1 } else { -1 };
                let mut done=true;

                for scalar_inc in 1..magnitude{
                    let c_inc=l%10+(scalar_inc*direction);
                    if let Some(piece) = board
                            .full_board
                            .get((l/10) as usize)
                            .and_then(|row| row.get((c_inc) as usize))
                    {
                        if piece.team!=Team::N{
                            done=false; 
                            break;
                        }
                    }
                }

                if done && magnitude>0{
                    if let Some(pos) = updated_king_moves_white.iter().position(|x| *x == *location) {
                        updated_king_moves_white.remove(pos);
                    }
                }
            }

            if l%10==k%10{
                let magnitude=(l/10-k/10).abs();
                let direction:isize = if k/10 > l/10 { 1 } else { -1 };
                let mut done=true;
                for scalar_inc in 1..magnitude{
                    let r_inc=l/10+(scalar_inc*direction);
                    if let Some(piece) = board
                            .full_board
                            .get(r_inc as usize)
                            .and_then(|row| row.get((l%10) as usize))
                    {
                        if piece.team!=Team::N{
                            done=false; 
                            break;
                        }
                    }
                }
                if done && magnitude>0{
                    if let Some(pos) = updated_king_moves_white.iter().position(|x| *x == *location) {
                        updated_king_moves_white.remove(pos);
                    }
                }
                }
        }

        
        for (x, y) in knight_moves.iter() {
        let move_row = l/10 + x;
        let move_col = l%10 + y;

        // Check if the move is within the bounds of the board
        if move_row >= 0 && move_row < 8 && move_col >= 0 && move_col < 8 {
            let move_row = move_row as usize;
            let move_col = move_col as usize;
            if board.full_board[move_row][move_col].kind==Kind::Knight && board.full_board[move_row][move_col].team==Team::B{
                if let Some(pos) = updated_king_moves_white.iter().position(|x| *x == l as usize) {
                    updated_king_moves_white.remove(pos);
                    }
                }
            }
        }
    }
}
black_moves.insert_moves(PieceId::K, &updated_king_moves_black);
white_moves.insert_moves(PieceId::K, &updated_king_moves_white);
if white_turn{
    return AllMovesGenRe {moves:white_moves, checking:re_checking};
}
else{
    return AllMovesGenRe {moves:black_moves, checking:re_checking};
}
}
//end of all_moves_gen
    



pub fn move_piece(mut board:Board, move_piece:PieceId, mut indexes:usize)->Board{
    //takes a piece id and index and updates the board struct. 
    if board.turn%2==0{
        let pawn_premotion_queen=indexes/10!=8; 
        if !pawn_premotion_queen{//pawn to knight promotions have a special identifier. 
            indexes-=80;//-80 gives you the index its moving to here. 
        }
        let initial_coords=board.white_indexes.get_index(move_piece).unwrap();
        let new_indexes=indexes;
        let old_row=initial_coords/10;
        let old_col=initial_coords%10;
        let new_row=new_indexes/10;
        let new_col=new_indexes%10;
        board.white_prime1=1;
         

        if move_piece==PieceId::K && board.prime2%5!=0{
            board.prime2=board.prime2*5;
        }
        if move_piece==PieceId::R1 && board.prime2%2!=0{
            board.prime2=board.prime2*2;
        }//for castling. 
        if move_piece==PieceId::R2 && board.prime2%3!=0{
            board.prime2=board.prime2*3;
        }

        if new_indexes==99{//ks castle
            board.full_board[7][4]=Piece{kind:Kind::Empty, team:Team::N};
            board.full_board[7][7]=Piece{kind:Kind::Empty, team:Team::N};
            board.full_board[7][6]=Piece{kind:Kind::King, team:Team::W};
            board.full_board[7][5]=Piece{kind:Kind::Rook, team:Team::W};
            board.white_indexes.change_indexes(PieceId::K, 76);
            board.white_indexes.change_indexes(PieceId::R2, 75);
            board.white_i_to_p.insert_piece(76,PieceId::K);
            board.white_i_to_p.insert_piece(75, PieceId::R2);
            board.white_i_to_p.nullify(77);
            board.white_i_to_p.nullify(74);
            board.turn+=1;
            return board;
        }

        if new_indexes==100{//qs castle 
            board.full_board[7][1]=Piece{kind:Kind::Empty, team:Team::N};
            board.full_board[7][0]=Piece{kind:Kind::Empty, team:Team::N};
            board.full_board[7][4]=Piece{kind:Kind::Empty, team:Team::N};
            board.full_board[7][2]=Piece{kind:Kind::King, team:Team::W};
            board.full_board[7][3]=Piece{kind: Kind::Rook, team:Team::W};
            board.white_indexes.change_indexes(PieceId::K, 72);
            board.white_indexes.change_indexes(PieceId::R1, 73);
            board.white_i_to_p.insert_piece(72, PieceId::K);
            board.white_i_to_p.insert_piece(73, PieceId::R1);
            board.white_i_to_p.nullify(74);
            board.white_i_to_p.nullify(70);
            board.turn+=1;
            return board;
        }

        let capturing=board.full_board[new_row][new_col].team==Team::B;//if something is captured. 
        //always need to remove old piece from mappings and piece ids
        if map_piece_id_to_kind(move_piece)==Kind::Pawn && new_col != old_col && !capturing{
            //en pessant
            let captured_i = old_row * 10 + new_col as usize;
            let old_piece = board.black_i_to_p.get_piece(captured_i).unwrap();
            board.black_i_to_p.nullify(captured_i);
            board.black_indexes.nullify(old_piece);
            if let Some(pos) = board.black_piece_ids.iter().position(|&id| id == old_piece) {
                board.black_piece_ids.remove(pos);
            }
            board.white_indexes.change_indexes(move_piece, new_indexes as usize);
            board.white_i_to_p.nullify(initial_coords);
            board.white_i_to_p.insert_piece(new_indexes, move_piece);
            board.full_board[new_row as usize][new_col as usize] =
                board.full_board[old_row as usize][old_col as usize].clone();
            board.full_board[old_row as usize][old_col as usize] = Piece{kind:Kind::Empty, team:Team::N};
            board.full_board[old_row as usize][new_col as usize] = Piece{kind:Kind::Empty, team:Team::N};
            board.black_points -= 100;
            board.turn+=1;
            return board;
        }

        if map_piece_id_to_kind(move_piece)==Kind::Pawn && (new_row as i32-old_row as i32).abs()==2{
            board.white_prime=board.white_prime*primes1(move_piece);
            board.white_prime1=primes(initial_coords%10) as i16;
        }//update en pessant fields, %primes1(piece) to see if it moved

        if capturing{
            let b_old_piece=board.black_i_to_p.get_piece(new_indexes).unwrap();
            board.black_piece_ids.remove(board.black_piece_ids.iter().position(|x| *x == b_old_piece).unwrap());
            board.black_indexes.nullify(b_old_piece);
            board.black_i_to_p.nullify(new_indexes);
        }//black piece gets captured.

        if map_piece_id_to_kind(move_piece)==Kind::Pawn && new_row ==0{
            //pawn promotion. 
            board.white_i_to_p.nullify(initial_coords);
            board.white_indexes.nullify(move_piece);
            board.white_piece_ids.remove(board.white_piece_ids.iter().position(|x| *x == move_piece).unwrap());
            if pawn_premotion_queen{
                let new_piece=pawn_to_queen(move_piece);
                board.white_i_to_p.insert_piece(new_indexes, new_piece);
                board.white_indexes.change_indexes(new_piece, new_indexes);
                board.white_piece_ids.push(new_piece);
                board.full_board[new_row][new_col]=Piece{kind:map_piece_id_to_kind(new_piece), team:Team::W};
                board.full_board[old_row][old_col]=Piece{kind:Kind::Empty, team:Team::N};
                board.white_points+=800;
            }
            else{
                let new_piece=pawn_to_knight(move_piece);
                board.white_i_to_p.insert_piece(new_indexes, new_piece);
                board.white_indexes.change_indexes(new_piece, new_indexes);
                board.white_piece_ids.push(new_piece);
                board.white_points+=200;
                board.full_board[new_row][new_col]=Piece{kind:map_piece_id_to_kind(new_piece), team:Team::W};
                board.full_board[old_row][old_col]=Piece{kind:Kind::Empty, team:Team::N};
            }
            board.turn+=1;
            board.white_i_to_p.nullify(initial_coords);
            return board;

        }
        //update indexes at the end, all the information needs to be correct after calling this function.
        board.full_board[new_row][new_col]=board.full_board[old_row][old_col].clone();
        board.full_board[old_row][old_col]=Piece{kind:Kind::Empty, team:Team::N};
        board.white_i_to_p.nullify(initial_coords);
        board.white_i_to_p.insert_piece(new_indexes, move_piece);
        board.white_indexes.change_indexes(move_piece, new_indexes);
        board.turn+=1;
    }
    else{
        let pawn_premotion_queen=indexes/10!=8; //for now, I'm just promoting to queen instead of knight.
        if !pawn_premotion_queen{
            indexes-=10;
        }
        let initial_coords=board.black_indexes.get_index(move_piece).unwrap();
        let new_indexes=indexes;
        let old_row=initial_coords/10;
        let old_col=initial_coords%10;
        let new_row=new_indexes/10;
        let new_col=new_indexes%10;
        if move_piece==PieceId::K && board.prime2%13!=0{
            board.prime2=board.prime2*13;
        }
        if move_piece==PieceId::R1 && board.prime2%7!=0{
            board.prime2=board.prime2*7;
        }
        if move_piece==PieceId::R2 && board.prime2%11!=0{
            board.prime2=board.prime2*11;
        }
        board.black_prime1=1;

        if new_indexes==99{
            board.full_board[0][4]=Piece{kind:Kind::Empty, team:Team::N};
            board.full_board[0][7]=Piece{kind:Kind::Empty, team:Team::N};
            board.full_board[0][6]=Piece{kind:Kind::King, team:Team::B};
            board.full_board[0][5]=Piece{kind:Kind::Rook, team:Team::B};
            board.black_indexes.change_indexes(PieceId::K, 6);
            board.black_indexes.change_indexes(PieceId::R2, 5);
            board.black_i_to_p.nullify(4);
            board.black_i_to_p.nullify(7);
            board.prime2=board.prime2*13;
            board.black_i_to_p.insert_piece(6, PieceId::K);
            board.black_i_to_p.insert_piece(5, PieceId::R2);
            board.turn+=1;
            return board;
        }

        if new_indexes==100{
            board.full_board[0][1]=Piece{kind:Kind::Empty, team:Team::N};
            board.full_board[0][0]=Piece{kind:Kind::Empty, team:Team::N};
            board.full_board[0][4]=Piece{kind:Kind::Empty, team:Team::N};
            board.full_board[0][2]=Piece{kind:Kind::King, team:Team::B};
            board.full_board[0][3]=Piece{kind:Kind::Rook, team:Team::B};
            board.black_indexes.change_indexes(PieceId::K, 2);
            board.black_indexes.change_indexes(PieceId::R1, 3);
            board.black_i_to_p.insert_piece(2, PieceId::K);
            board.black_i_to_p.insert_piece(3, PieceId::R1);
            board.black_i_to_p.nullify(4);
            board.black_i_to_p.nullify(0);
            board.prime2=board.prime2*13;
            board.turn+=1;
            return board;
        }

        let capturing=board.full_board[new_row][new_col].team==Team::W;
        if map_piece_id_to_kind(move_piece)==Kind::Pawn && new_col != old_col && !capturing{
            //en pessant
            let captured_i = old_row * 10 + new_col as usize;
            let old_piece = board.white_i_to_p.get_piece(captured_i).unwrap();
            board.white_i_to_p.nullify(captured_i);
            board.white_indexes.nullify(old_piece);
            if let Some(pos) = board.white_piece_ids.iter().position(|&id| id == old_piece) {
                board.white_piece_ids.remove(pos);
            }
            board.black_indexes.change_indexes(move_piece, new_indexes as usize);
            board.black_i_to_p.nullify(initial_coords);
            board.black_i_to_p.insert_piece(new_indexes, move_piece);
            board.full_board[new_row as usize][new_col as usize] =
                board.full_board[old_row as usize][old_col as usize].clone();
            board.full_board[old_row as usize][old_col as usize] = Piece{kind:Kind::Empty, team:Team::N};
            board.full_board[old_row as usize][new_col as usize] = Piece{kind:Kind::Empty, team:Team::N};
            board.white_points -= 100;
            board.turn+=1;
            return board;
        }

        if map_piece_id_to_kind(move_piece)==Kind::Pawn && (new_row as i32-old_row as i32).abs()==2{
            board.black_prime=board.black_prime*primes1(move_piece);
            board.black_prime1=primes(initial_coords%10) as i16;
        }

        if capturing{
            let w_old_piece=board.white_i_to_p.get_piece(new_indexes).unwrap();
            board.white_piece_ids.remove(board.white_piece_ids.iter().position(|x| *x == w_old_piece).unwrap());
            board.white_indexes.nullify(w_old_piece);
            board.white_i_to_p.nullify(new_indexes);
        }

        if map_piece_id_to_kind(move_piece)==Kind::Pawn && new_indexes/10 ==7{
            let pawn_premotion_queen=true;
            board.black_i_to_p.nullify(initial_coords);
            board.black_indexes.nullify(move_piece);
            board.black_piece_ids.remove(board.black_piece_ids.iter().position(|x| *x == move_piece).unwrap());
            board.black_points-=100;
            if pawn_premotion_queen{
                let new_piece=pawn_to_queen(move_piece);
                board.black_i_to_p.insert_piece(new_indexes, new_piece);
                board.black_indexes.change_indexes(new_piece, new_indexes);
                board.black_piece_ids.push(new_piece);
                board.full_board[new_row][new_col]=Piece{kind:map_piece_id_to_kind(new_piece), team:Team::B};
                board.full_board[old_row][old_col]=Piece{kind:Kind::Empty, team:Team::N};
                board.black_points+=900;
            }
            else{
                let new_piece=pawn_to_knight(move_piece);
                board.black_i_to_p.insert_piece(new_indexes, new_piece);
                board.black_indexes.change_indexes(new_piece, new_indexes);
                board.black_piece_ids.push(new_piece);
                board.black_points+=300;
                board.full_board[new_row][new_col]=Piece{kind:map_piece_id_to_kind(new_piece), team:Team::W};
                board.full_board[old_row][old_col]=Piece{kind:Kind::Empty, team:Team::N};
            }
            board.turn+=1;
            return board;
        }

        board.full_board[new_row][new_col]=board.full_board[old_row][old_col].clone();
        board.full_board[old_row][old_col]=Piece{kind:Kind::Empty, team:Team::N};
        board.black_i_to_p.nullify(initial_coords);
        board.black_i_to_p.insert_piece(new_indexes, move_piece);
        board.black_indexes.change_indexes(move_piece, new_indexes);
        board.turn+=1;
    }
    return board
}