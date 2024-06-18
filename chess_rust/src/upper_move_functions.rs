use crate::types::{AllMovesGenRe, Board, Kind, Piece, PieceId, Team};
use crate::base_functions::{find_overlap, map_piece_id_to_kind, contains_element, primes, primes1, pawn_to_queen, pawn_to_knight, find_non_overlap};
use crate::base_move_functions::{generate_available_moves};
use crate::upper_move_function_helpers::{in_check_directional, b_rook_pinning, w_rook_pinning, b_bishop_pinning, w_bishop_pinning};
pub fn all_moves_gen(board: &Board)->AllMovesGenRe {
    // Castling conditions for Black King Side
    let mut b_ks = board.prime2 % 13 != 0 && board.prime2 % 11 != 0 && board.full_board[0][6].team == Team::N && board.full_board[0][5].team == Team::N;
    let mut w_ks = board.prime2 % 5 != 0 && board.prime2 % 3 != 0 && board.full_board[7][6].team == Team::N && board.full_board[7][5].team == Team::N;

    // Castling conditions for White Queen Side
    let mut w_qs = board.prime2 % 5 != 0 && board.prime2 % 2 != 0 && board.full_board[7][1].team == Team::N && board.full_board[7][2].team == Team::N && board.full_board[7][3].team == Team::N;
    
    let mut re=AllMovesGenRe::new();

    let mut black_checking: Vec<PieceId> = Vec::new();
    let mut white_checking: Vec<PieceId> = Vec::new();
    let mut b_pinned_vectors: Vec<Vec<usize>> = Vec::new();
    let mut w_pinned_vectors: Vec<Vec<usize>> = Vec::new();
    let mut b_pinned_indexes: Vec<usize> = Vec::new();
    let mut w_pinned_indexes: Vec<usize> = Vec::new();
    // Castling conditions for Black Queen Side
    let mut b_qs = board.prime2 % 13 != 0 && board.prime2 % 7 != 0 && board.full_board[0][1].team == Team::N&& board.full_board[0][2].team == Team::N && board.full_board[0][3].team == Team::N;

    // Castling condi
    
    let b_king_index = board.black_indexes.get_index(PieceId::K).unwrap();
    let w_king_index = board.white_indexes.get_index(PieceId::K).unwrap();
    
    
    // Safely perform integer division after ensuring values are available
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

    for piece_id in board.white_piece_ids.iter() {

        let piece_index = board.white_indexes.get_index(*piece_id).unwrap();
        let curr_row = piece_index / 10;
        let curr_col = piece_index % 10;
        if (curr_row as isize-b_king_index as isize/10).abs()<=1 && (curr_col as isize-b_king_index as isize%10).abs()<=1{
            adjacent_to_black_k.push(piece_index);
        }
        

        all_moves = generate_available_moves(&board, curr_row, curr_col); 
        re.white_moves.insert_moves(*piece_id, &all_moves);
        if map_piece_id_to_kind(*piece_id)!=Kind::Pawn{
            let curr_over_b=find_overlap(&all_moves, &b_king_moves);
            for i in curr_over_b.iter(){
                over_b.push(*i)
            }
        }
        let mut checking = false;

        if contains_element(&all_moves, b_king_index){
            re.checking=true;
            checking=true;
            white_checking.push(*piece_id);
        }

        if board.full_board[curr_row][curr_col].kind==Kind::Rook || board.full_board[curr_row][curr_col].kind==Kind::Queen && checking==false{
            w_rooks.push(piece_index);
            let mut temp=w_rook_pinning(&board, *piece_id, &over_b);
            over_b=temp[1].clone();//the logic for if there's a black piece adjacent to the white king for a rook or bishop is checked here
            if !temp[0].is_empty(){
                let length=temp[0].len()-1;
                let pinned_p=temp[0][length];
                temp[0].remove(length); 
                b_pinned_vectors.push(temp[0].clone());
                b_pinned_indexes.push(pinned_p);
            }
        }
        
        if board.full_board[curr_row][curr_col].kind==Kind::Bishop || board.full_board[curr_row][curr_col].kind==Kind::Queen && checking==false{
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

    for i in board.black_piece_ids.iter(){
        let piece=*i;
        let piece_index = board.black_indexes.get_index(piece).unwrap();
        let curr_row=piece_index/10;
        let curr_col=piece_index%10;
        all_moves=generate_available_moves(&board, curr_row, curr_col);

        if map_piece_id_to_kind(piece)!=Kind::Pawn{
            let curr_over_w=find_overlap(&all_moves, &w_king_moves);
            for i in curr_over_w.iter(){
                over_w.push(*i)
            }
        }

        let mut checking=false;

        if (curr_row as isize-(w_king_index as isize)/10).abs()<=1 && (curr_col as isize-w_king_index as isize%10).abs()<=1{
            adjacent_to_white_k.push(piece_index);
        }

        if contains_element(&all_moves,board.white_indexes.get_index(PieceId::K).unwrap()){
            re.checking=true;
            checking=true;
            black_checking.push(piece);
        }

        if board.full_board[curr_row][curr_col].kind==Kind::Rook || board.full_board[curr_row][curr_col].kind==Kind::Queen && checking==false{
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


        if board.full_board[curr_row][curr_col].kind==Kind::Bishop || board.full_board[curr_row][curr_col].kind==Kind::Queen && checking==false{
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

        re.black_moves.insert_moves(piece, &all_moves);
        let castling_points=vec![77,76,75,74];
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
        let moves=re.white_moves.get_moves(piece);
        re.white_moves.insert_moves(piece, &find_overlap(&moves, &w_pinned_vectors[index]));
    }

    
    

    for (index, pinned_index) in b_pinned_indexes.iter().enumerate(){
        println!("pinned index {}", pinned_index);
        let piece=board.black_i_to_p.get_piece(*pinned_index).unwrap();
        let moves=re.black_moves.get_moves(piece);
        re.black_moves.insert_moves(piece, &find_overlap(&moves, &b_pinned_vectors[index]));
    }

    let updated_king_moves_w=find_non_overlap(over_w, w_king_moves);
    let updated_king_moves_b=find_non_overlap(over_b, b_king_moves);
    re.white_moves.insert_moves(PieceId::K,&updated_king_moves_w);
    re.black_moves.insert_moves(PieceId::K,&updated_king_moves_b);
    

    for i in black_checking{
        let k=map_piece_id_to_kind(i);
        if k==Kind::Rook || k==Kind::Bishop || k==Kind::Queen{
            re.white_moves=in_check_directional(&board, &re, i, Team::B).clone();
        }
        else{
            for piece_id in board.white_piece_ids.iter() {
                let pressuring_i=board.black_indexes.get_index(i).unwrap();
                if *piece_id!=PieceId::K{
                    let valid=vec![pressuring_i];
                    let moves = find_overlap(&re.white_moves.get_moves(*piece_id), &valid);
                    re.white_moves.insert_moves(*piece_id,&moves)
                }
            }   
        }
    }

    for i in white_checking{
        let k=map_piece_id_to_kind(i);
        if k==Kind::Rook || k==Kind::Bishop || k==Kind::Queen{
            re.black_moves=in_check_directional(&board, &re, i, Team::W);
        }
        else{
            for piece_id in board.black_piece_ids.iter() {
                let pressuring_i=board.white_indexes.get_index(i).unwrap();
                if *piece_id!=PieceId::K{
                    let valid=vec![pressuring_i];
                    let moves = find_overlap(&re.black_moves.get_moves(*piece_id), &valid);
                    re.black_moves.insert_moves(*piece_id,&moves);
                }
                }
            }
        }

    let mut updated_king_moves_black=re.black_moves.get_moves(PieceId::K);
    let mut updated_king_moves_white=re.white_moves.get_moves(PieceId::K);
        
        

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


    for i in re.black_moves.get_moves(PieceId::K){

        //special pawn case
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

        let w_king_i=board.white_indexes.get_index(PieceId::K).unwrap();
        let w_king_r=w_king_i/10;
        let w_king_c=w_king_i/10;

        for i in re.white_moves.get_moves(PieceId::K){

        //special pawn case
        if i%10<=6 && i/10<=6{
            if board.full_board[i/10+1][i%10+1].kind==Kind::Pawn&& board.full_board[i/10+1][i%10-1].team==Team::B{
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
                updated_king_moves_white.remove(updated_king_moves_white.iter().position(|x| *x == *location).unwrap());
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
                    updated_king_moves_white.remove(updated_king_moves_white.iter().position(|x| *x == *location).unwrap());
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
                    updated_king_moves_white.remove(updated_king_moves_white.iter().position(|x| *x == *location).unwrap());
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
                updated_king_moves_white.remove(updated_king_moves_white.iter().position(|x| *x == move_row+move_col).unwrap());
                }
            }
        }
    }
re.black_moves.insert_moves(PieceId::K, &updated_king_moves_black);
re.white_moves.insert_moves(PieceId::K, &updated_king_moves_white);
return re
}
    



pub fn move_piece(mut board:Board, move_piece:PieceId, indexes:usize)->Board{
    if board.turn%2==0{
        let initial_coords=board.white_indexes.get_index(move_piece).unwrap();
        let new_indexes=indexes;
        let old_row=initial_coords/10;
        let old_col=initial_coords%10;
        let new_row=new_indexes/10;
        let new_col=new_indexes%10;
        board.white_prime1=1;
        let pawn_premotion_queen=true;
        if move_piece==PieceId::K{
            board.prime2=board.prime2*13;
        }
        if move_piece==PieceId::R1{
            board.prime2=board.prime2*2;
        }
        if move_piece==PieceId::R2{
            board.prime2=board.prime2*3;
        }
        if new_indexes==99{
            board.full_board[7][4]=Piece{value:0, kind:Kind::Empty, team:Team::N};
            board.full_board[7][7]=Piece{value:0, kind:Kind::Empty, team:Team::N};
            board.full_board[7][6]=Piece{value:0, kind:Kind::King, team:Team::W};
            board.full_board[7][5]=Piece{value:500, kind:Kind::Rook, team:Team::W};
            board.white_indexes.change_indexes(PieceId::K, 76);
            board.white_indexes.change_indexes(PieceId::R2, 75);
            board.white_i_to_p.insert_piece(76,PieceId::K);
            board.white_i_to_p.insert_piece(75, PieceId::R2);
            board.white_i_to_p.nullify(77);
            board.white_i_to_p.nullify(74);
            board.prime2=board.prime2*5;
            board.turn+=1;
            return board;
        }
        if new_indexes==100{
            board.full_board[7][1]=Piece{value:0, kind:Kind::Empty, team:Team::N};
            board.full_board[7][0]=Piece{value:0, kind:Kind::Empty, team:Team::N};
            board.full_board[7][4]=Piece{value:0, kind:Kind::Empty, team:Team::N};
            board.full_board[7][2]=Piece{value:0, kind:Kind::King, team:Team::W};
            board.full_board[7][3]=Piece{value:500,kind: Kind::Rook, team:Team::W};
            board.white_indexes.change_indexes(PieceId::K, 72);
            board.white_indexes.change_indexes(PieceId::R1, 73);
            board.white_i_to_p.insert_piece(72, PieceId::K);
            board.white_i_to_p.insert_piece(73, PieceId::R1);
            board.white_i_to_p.nullify(74);
            board.white_i_to_p.nullify(70);
            board.prime2=board.prime2*5;
            board.turn+=1;
            return board;
        }
        let old_points=board.full_board[new_row][new_col].value;
        if map_piece_id_to_kind(move_piece)==Kind::Pawn && new_col != old_col && old_points==0{
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
            board.full_board[old_row as usize][old_col as usize] = Piece{value:0, kind:Kind::Empty, team:Team::N};
            board.full_board[old_row as usize][new_col as usize] = Piece{value:0, kind:Kind::Empty, team:Team::N};
            board.black_points -= 100;
            board.turn+=1;
            return board;
        }
        if map_piece_id_to_kind(move_piece)==Kind::Pawn && (new_row as i32-old_row as i32).abs()==2{
            board.white_prime=board.white_prime*primes(initial_coords%10);
            board.white_prime1=primes1(move_piece);
        }
        if old_points>0{
            let b_old_piece=board.black_i_to_p.get_piece(new_indexes).unwrap();
            board.black_piece_ids.remove(board.black_piece_ids.iter().position(|x| *x == b_old_piece).unwrap());
            board.black_indexes.nullify(b_old_piece);
            board.black_i_to_p.nullify(new_indexes);
            board.black_points-=old_points;
        }
        if map_piece_id_to_kind(move_piece)==Kind::Pawn && new_row ==0{
            board.white_i_to_p.nullify(initial_coords);
            board.white_indexes.nullify(move_piece);
            board.white_piece_ids.remove(board.white_piece_ids.iter().position(|x| *x == move_piece).unwrap());
            if pawn_premotion_queen{
                let new_piece=pawn_to_queen(move_piece);
                board.white_i_to_p.insert_piece(new_indexes, new_piece);
                board.white_indexes.change_indexes(new_piece, new_indexes);
                board.white_piece_ids.push(new_piece);
                board.full_board[new_row][new_col]=Piece{value:900, kind:map_piece_id_to_kind(new_piece), team:Team::W};
                board.full_board[old_row][old_col]=Piece{value:0, kind:Kind::Empty, team:Team::N};
                board.white_points+=800;
            }
            else{
                let new_piece=pawn_to_knight(move_piece);
                board.white_i_to_p.insert_piece(new_indexes, new_piece);
                board.white_indexes.change_indexes(new_piece, new_indexes);
                board.white_piece_ids.push(new_piece);
                board.white_points+=200;
                board.full_board[new_row][new_col]=Piece{value:300, kind:map_piece_id_to_kind(new_piece), team:Team::W};
                board.full_board[old_row][old_col]=Piece{value:0, kind:Kind::Empty, team:Team::N};
            }
            board.turn+=1;
            board.white_i_to_p.nullify(initial_coords);
            return board;
        }
        board.full_board[new_row][new_col]=board.full_board[old_row][old_col].clone();
        board.full_board[old_row][old_col]=Piece{value:0, kind:Kind::Empty, team:Team::N};
        board.white_i_to_p.nullify(initial_coords);
        board.white_i_to_p.insert_piece(new_indexes, move_piece);
        board.white_indexes.change_indexes(move_piece, new_indexes);
        board.turn+=1;
    }
    else{
        let initial_coords=board.black_indexes.get_index(move_piece).unwrap();
        let new_indexes=indexes;
        let old_row=initial_coords/10;
        let old_col=initial_coords%10;
        let new_row=new_indexes/10;
        let new_col=new_indexes%10;
        if move_piece==PieceId::K{
            board.prime2=board.prime2*13;
        }
        if move_piece==PieceId::R1{
            board.prime2=board.prime2*7;
        }
        if move_piece==PieceId::R2{
            board.prime2=board.prime2*11;
        }
        board.black_prime1=1;
        if new_indexes==99{
            board.full_board[0][4]=Piece{value:0, kind:Kind::Empty, team:Team::N};
            board.full_board[0][7]=Piece{value:0, kind:Kind::Empty, team:Team::N};
            board.full_board[0][6]=Piece{value:0, kind:Kind::King, team:Team::B};
            board.full_board[0][5]=Piece{value:500, kind:Kind::Rook, team:Team::B};
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
            board.full_board[0][1]=Piece{value:0, kind:Kind::Empty, team:Team::N};
            board.full_board[0][0]=Piece{value:0, kind:Kind::Empty, team:Team::N};
            board.full_board[0][4]=Piece{value:0, kind:Kind::Empty, team:Team::N};
            board.full_board[0][2]=Piece{value:0, kind:Kind::King, team:Team::B};
            board.full_board[0][3]=Piece{value:500, kind:Kind::Rook, team:Team::B};
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
        let old_points=board.full_board[new_row][new_col].value;
        if map_piece_id_to_kind(move_piece)==Kind::Pawn && new_col != old_col && old_points==0{
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
            board.full_board[old_row as usize][old_col as usize] = Piece{value:0, kind:Kind::Empty, team:Team::N};
            board.full_board[old_row as usize][new_col as usize] = Piece{value:0, kind:Kind::Empty, team:Team::N};
            board.white_points -= 100;
            board.turn+=1;
            return board;
        }
        if map_piece_id_to_kind(move_piece)==Kind::Pawn && (new_row as i32-old_row as i32).abs()==2{
            board.black_prime=board.black_prime*primes(initial_coords%10);
            board.black_prime1=primes1(move_piece);
        }
        if old_points>0{
            let w_old_piece=board.white_i_to_p.get_piece(new_indexes).unwrap();
            board.white_piece_ids.remove(board.white_piece_ids.iter().position(|x| *x == w_old_piece).unwrap());
            board.white_indexes.nullify(w_old_piece);
            board.white_i_to_p.nullify(new_indexes);
            board.white_points-=old_points;
        }
        if map_piece_id_to_kind(move_piece)==Kind::Pawn && new_indexes/10 ==7{
            let pawn_premotion_queen=true;
            board.black_i_to_p.nullify(initial_coords);
            board.black_indexes.nullify(move_piece);
            board.black_piece_ids.remove(board.white_piece_ids.iter().position(|x| *x == move_piece).unwrap());
            board.black_points-=100;
            if pawn_premotion_queen{
                let new_piece=pawn_to_queen(move_piece);
                board.black_i_to_p.insert_piece(new_indexes, new_piece);
                board.black_indexes.change_indexes(new_piece, new_indexes);
                board.black_piece_ids.push(new_piece);
                board.full_board[new_row][new_col]=Piece{value:900, kind:map_piece_id_to_kind(new_piece), team:Team::B};
                board.full_board[old_row][old_col]=Piece{value:0, kind:Kind::Empty, team:Team::N};
                board.black_points+=900;
            }
            else{
                let new_piece=pawn_to_knight(move_piece);
                board.black_i_to_p.insert_piece(new_indexes, new_piece);
                board.black_indexes.change_indexes(new_piece, new_indexes);
                board.black_piece_ids.push(new_piece);
                board.black_points+=300;
                board.full_board[new_row][new_col]=Piece{value:300, kind:map_piece_id_to_kind(new_piece), team:Team::W};
                board.full_board[old_row][old_col]=Piece{value:0, kind:Kind::Empty, team:Team::N};
            }
            board.turn+=1;
            return board;
        }
        board.full_board[new_row][new_col]=board.full_board[old_row][old_col].clone();
        board.full_board[old_row][old_col]=Piece{value:0, kind:Kind::Empty, team:Team::N};
        board.black_i_to_p.nullify(initial_coords);
        board.black_i_to_p.insert_piece(new_indexes, move_piece);
        board.black_indexes.change_indexes(move_piece, new_indexes);
        board.turn+=1;
    }
    return board
}