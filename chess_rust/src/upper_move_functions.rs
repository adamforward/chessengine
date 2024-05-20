use crate::types::{AllMovesGenRe, Board, Kind, Piece, PieceId, Team};
use crate::base_functions::{find_overlap, contains_element, map_piece_id_to_kind, primes, primes1, pawn_to_queen, pawn_to_knight};
use crate::base_move_functions::{generate_available_moves};
use crate::upper_move_function_helpers::{in_check_directional, b_rook_pinning, w_rook_pinning, b_bishop_pinning, w_bishop_pinning, in_check_knight_or_pawn, ep_black, ep_white};
pub fn all_moves_gen(board: &Board)->AllMovesGenRe {
    // Castling conditions for Black King Side
    let mut b_ks = board.prime2 % 13 != 0 && board.prime2 % 11 != 0 && board.full_board[0][6].team == Team::N && board.full_board[0][5].team == Team::N;
    let mut re=AllMovesGenRe::new();
    // Castling conditions for Black Queen Side
    let mut b_qs = board.prime2 % 13 != 0 && board.prime2 % 7 != 0 && board.full_board[0][1].team == Team::N&& board.full_board[0][2].team == Team::N && board.full_board[0][3].team == Team::N;

    // Castling conditions for White King Side
    let mut w_ks = board.prime2 % 5 != 0 && board.prime2 % 3 != 0 && board.full_board[7][6].team == Team::N && board.full_board[7][5].team == Team::N;

    // Castling conditions for White Queen Side
    let mut w_qs = board.prime2 % 5 != 0 && board.prime2 % 2 != 0 && board.full_board[7][1].team == Team::N && board.full_board[7][2].team == Team::N && board.full_board[7][3].team == Team::N;
    
    let black_checking: Vec<PieceId> = Vec::new();
    let mut white_checking: Vec<PieceId> = Vec::new();
    let mut b_pinned_vectors: Vec<Vec<usize>> = Vec::new();
    let mut w_pinned_vectors: Vec<Vec<usize>> = Vec::new();
    let mut b_pinned_indexes: Vec<usize> = Vec::new();
    let mut w_pinned_indexes: Vec<usize> = Vec::new();
    
    // Handle potential None values safely using `match` or `if let`
    let b_king_index = board.black_indexes.get_index(PieceId::K).unwrap();
    let w_king_index = board.white_indexes.get_index(PieceId::K).unwrap();
    
    // Safely perform integer division after ensuring values are available
    let mut w_king_moves = generate_available_moves(&board, w_king_index / 10, w_king_index % 10);
    let mut b_king_moves = generate_available_moves(&board, b_king_index / 10, b_king_index % 10);
    
    // Initialize over_w and over_b as empty vectors
    let mut over_w: Vec<usize> = vec![];
    let mut over_b: Vec<usize> = vec![];
    let mut all_moves:Vec<usize>;
    for piece_id in board.white_piece_ids.iter() {
        let piece_index = board.white_indexes.get_index(*piece_id).unwrap();
        let curr_row = piece_index / 10;
        let curr_col = piece_index % 10;
        

        // Adjusted to your available function signature and logic
        all_moves = generate_available_moves(&board, curr_row, curr_col); // Assuming this function exists and works as intended
        over_b=find_overlap(&all_moves, &w_king_moves);
        let mut checking = false;
        if contains_element(&all_moves, b_king_index){
            re.checking=true;
            checking=true;
            white_checking.push(*piece_id);
        }

        if board.full_board[curr_row][curr_col].kind==Kind::Rook || board.full_board[curr_row][curr_col].kind==Kind::Queen && checking==false{
            let mut temp=w_rook_pinning(&board, *piece_id, &over_b);
            over_b=temp[1].clone();
            if temp[0].len()>0{
            let pinned_p=temp[0][temp[0].len()-1];
            let length=temp[0].len()-1;
            temp[0].remove(length);
            if temp.len()>0{
                b_pinned_vectors.push(temp[0].clone());
                w_pinned_indexes.push(pinned_p);
            }
        }
        }
        
        re.white_moves.insert_moves(*piece_id, &all_moves);
        if board.full_board[curr_row][curr_col].kind==Kind::Bishop || board.full_board[curr_row][curr_col].kind==Kind::Queen && checking==false{
            let mut temp=w_bishop_pinning(&board, *piece_id, &over_b);
            over_b=temp[1].clone();
            if temp[0].len()>0{
            let pinned_p=temp[0][temp[0].len()-1];
            let length=temp[0].len()-1;
            temp[0].remove(length);
            if temp.len()>0{
                b_pinned_vectors.push(temp[0].clone());
                w_pinned_indexes.push(pinned_p);
            }
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
        over_w=find_overlap(&all_moves, &b_king_moves);
        let mut checking=false;

        if contains_element(&all_moves,board.white_indexes.get_index(PieceId::K).unwrap()){
            re.checking=true;
            checking=true;
            white_checking.push(piece);
        }

        if board.full_board[curr_row][curr_col].kind==Kind::Rook || board.full_board[curr_row][curr_col].kind==Kind::Queen && checking==false{
            let mut temp=b_rook_pinning(&board, piece, &over_w);
            over_w=temp[1].clone();
            let temp2=temp[0].clone();
            if temp2.len()>0{
                let pinned_p=temp2[temp.len()-1];
                temp.remove(temp2.len()-1);
                w_pinned_vectors.push(temp2);
                b_pinned_indexes.push(pinned_p);
            }
        }


        if board.full_board[curr_row][curr_col].kind==Kind::Bishop || board.full_board[curr_row][curr_col].kind==Kind::Queen && checking==false{
            let mut temp=b_bishop_pinning(&board, piece, &over_w);
            over_w=temp[1].clone();
            let temp2=temp[0].clone();
            if temp2.len()>0{
                let pinned_p=temp2[temp.len()-1];
                temp.remove(temp2.len()-1);
                w_pinned_vectors.push(temp2);
                b_pinned_indexes.push(pinned_p);
            }
        }
        re.black_moves.insert_moves(piece, &all_moves);
        let castling_points=vec![7,6,5,4];
        for i in castling_points.iter(){
            for j in all_moves.iter(){
                w_ks=w_ks && *i!=*j
            }
        }
        let castling_points=vec![0,1,2,3,4];
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
    let updated_king_moves_b=find_overlap(&over_b, &b_king_moves);
    re.black_moves.insert_moves(PieceId::K, &updated_king_moves_b);
    let updated_king_moves_w=find_overlap(&over_w, &w_king_moves);
    re.white_moves.insert_moves(PieceId::K, &updated_king_moves_w);

    if b_pinned_indexes.len()>0{
        for i in w_pinned_indexes.iter(){
            for j in w_pinned_vectors.iter(){
                let moves=re.black_moves.get_moves(board.black_i_to_p.get_piece(*i).unwrap());
                let piece=board.black_i_to_p.get_piece(*i).unwrap();
                re.black_moves.insert_moves(piece, &find_overlap(&moves, j))
            }
        }
    }
    if w_pinned_indexes.len()>0{
        for i in b_pinned_indexes.iter(){
            for j in w_pinned_vectors.iter(){
                let moves=re.white_moves.get_moves(board.black_i_to_p.get_piece(*i).unwrap());
                let piece=board.white_i_to_p.get_piece(*i).unwrap();
                re.white_moves.insert_moves(piece, &find_overlap(&moves, j))
            }
        }
    }
    if black_checking.len()>0{
        for i in black_checking{
            let k=map_piece_id_to_kind(i);
            if k==Kind::Rook || k==Kind::Bishop || k==Kind::Queen{
                let mut direction:Vec<i32>=vec![];
                direction.push(board.white_indexes.get_index(PieceId::K).unwrap() as i32/10-board.black_indexes.get_index(i).unwrap() as i32/10);
                direction.push(board.white_indexes.get_index(PieceId::K).unwrap() as i32%10-board.black_indexes.get_index(i).unwrap() as i32%10);
                in_check_directional(&board, &re, i, Team::B, direction);
            }
        }
    }
    if white_checking.len()>0{
        for i in white_checking{
            let k=map_piece_id_to_kind(i);
            if k==Kind::Rook || k==Kind::Bishop || k==Kind::Queen{
                let mut direction:Vec<i32>=vec![];
                let i_king=board.black_indexes.get_index(PieceId::K).unwrap() as i32;
                let i_p=board.black_indexes.get_index(i).unwrap() as i32;
                direction.push(i_p/10-i_king/10);
                direction.push(i_p%10-i_king%10);
                in_check_directional(&board, &re, i, Team::W, direction);
            }
        }
    }
    let mut updated_king_moves=re.black_moves.get_moves(PieceId::K).clone();

    for i in re.black_moves.get_moves(PieceId::K){

        if map_piece_id_to_kind(board.white_i_to_p.get_piece(i+11).unwrap())==Kind::Pawn || map_piece_id_to_kind(board.white_i_to_p.get_piece(i+9).unwrap())==Kind::Pawn {
            updated_king_moves.remove(updated_king_moves.iter().position(|x| *x == i).unwrap());
        }
        let w_king_location=board.white_indexes.get_index(PieceId::K).unwrap() as i32;
        if ((i as i32)/10-w_king_location/10).abs()<=1 && ((i as i32)%10-w_king_location%10).abs()<=1{
            updated_king_moves.remove(updated_king_moves.iter().position(|x| *x == i).unwrap());
        }
        let knight_moves = [
        (1, 2),
        (1, -2),
        (-1, 2),
        (-1, -2),
        (2, 1),
        (2, -1),
        (-2, 1),
        (-2, -1),
    ];
        for (i, j) in knight_moves.iter() {
        let move_row = b_king_index as i32 + i;
        let move_col = b_king_index as i32 + j;

        // Check if the move is within the bounds of the board
        if move_row >= 0 && move_row < 8 && move_col >= 0 && move_col < 8 {
            let move_row = move_row as usize;
            let move_col = move_col as usize;
            if map_piece_id_to_kind(board.white_i_to_p.get_piece(move_row+move_col).unwrap())==Kind::Knight{
                updated_king_moves.remove(updated_king_moves.iter().position(|x| *x == move_row+move_col).unwrap());
            }
        }
    }
}
    re.white_moves.insert_moves(PieceId::K, &updated_king_moves);
    let mut updated_king_moves=re.white_moves.get_moves(PieceId::K).clone();
    for i in re.white_moves.get_moves(PieceId::K){

        if map_piece_id_to_kind(board.black_i_to_p.get_piece(i+11).unwrap())==Kind::Pawn || map_piece_id_to_kind(board.black_i_to_p.get_piece(i+9).unwrap())==Kind::Pawn {
            updated_king_moves.remove(updated_king_moves.iter().position(|x| *x == i).unwrap());
        }
        let b_king_location=board.black_indexes.get_index(PieceId::K).unwrap() as i32;
        if ((i as i32)/10-b_king_location/10).abs()<=1 && ((i as i32)%10-b_king_location%10).abs()<=1{
            updated_king_moves.remove(updated_king_moves.iter().position(|x| *x == i).unwrap());
        }
        let knight_moves = [
        (1, 2),
        (1, -2),
        (-1, 2),
        (-1, -2),
        (2, 1),
        (2, -1),
        (-2, 1),
        (-2, -1),
    ];
        for (i, j) in knight_moves.iter() {
        let move_row = w_king_index as i32 + i;
        let move_col = w_king_index as i32 + j;

        // Check if the move is within the bounds of the board
        if move_row >= 0 && move_row < 8 && move_col >= 0 && move_col < 8 {
            let move_row = move_row as usize;
            let move_col = move_col as usize;
             if map_piece_id_to_kind(board.black_i_to_p.get_piece(move_row+move_col).unwrap())==Kind::Knight{
                updated_king_moves.remove(updated_king_moves.iter().position(|x| *x == move_row+move_col).unwrap());
            }
        }
    }
}
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
            board.white_i_to_p.nullify(70);
            board.prime2=board.prime2*5;
            board.turn+=1;
            return board;
        }
        if new_indexes==99{
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
            return ep_white(board, move_piece, indexes);
        }
        if map_piece_id_to_kind(move_piece)==Kind::Pawn && (new_row as i32-old_row as i32).abs()==2{
            board.white_prime=board.white_prime*primes(initial_coords%10);
            board.white_prime1=primes1(move_piece)
        }
        if old_points>0{
            let b_old_piece=board.black_i_to_p.get_piece(new_indexes).unwrap();
            board.black_piece_ids.remove(board.black_piece_ids.iter().position(|x| *x == b_old_piece).unwrap());
            board.black_indexes.nullify(b_old_piece);
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
            board.turn+=1;
            return board;
        }
        if new_indexes==99{
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
            board.prime2=board.prime2*5;
            board.turn+=1;
            return board;
        }
        let old_points=board.full_board[new_row][new_col].value;
        if map_piece_id_to_kind(move_piece)==Kind::Pawn && new_col != old_col && old_points==0{
            // ep_black(board, move_piece, indexes);
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
            board.white_i_to_p.nullify(initial_coords);
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
        board.black_indexes.change_indexes(move_piece, new_indexes);
        board.turn+=1;
    }
    return board
}