use crate::types::{Board, Kind, Move, Piece, PieceId, Team};
use crate::base_move_functions::{generate_available_moves};
use crate::upper_move_helper_functions::{in_check_directional, king_move_eliminator_white, king_move_eliminator_black, b_rook_pinning, w_rook_pinning, b_bishop_pinning, w_bishop_pinning,in_check_knight_or_pawn, ep_black, ep_white, reset_board};
use std::collections::HashMap;
use std::vec::Vec;
pub fn all_moves_gen(&mut board:Board){
    // Castling conditions for Black King Side
    let mut b_ks = board.prime2 % 13 != 0 && board.prime2 % 11 != 0 && board.full_board[0][6].team == Team::N && board.full_board[0][5].team == Team::N;

    // Castling conditions for Black Queen Side
    let mut b_qs = board.prime2 % 13 != 0 && board.prime2 % 7 != 0 && board.full_board[0][1].team == Team::N&& board.full_board[0][2].team == Team::N && board.full_board[0][3].team == Team::N;

    // Castling conditions for White King Side
    let mut w_ks = board.prime2 % 5 != 0 && board.prime2 % 3 != 0 && board.full_board[7][6].team == Team::N && board.full_board[7][5].team == Team::N;

    // Castling conditions for White Queen Side
    let mut w_qs = board.prime2 % 5 != 0 && board.prime2 % 2 != 0 && board.full_board[7][1].team == Team::N && board.full_board[7][2].team == Team::N && board.full_board[7][3].team == Team::N;

    let mut black_checking: Vec<PieceId> = Vec::new();
    let mut white_checking: Vec<PieceId> = Vec::new();
    let mut b_pinned_vectors: Vec<Vec<i32>> = Vec::new();
    let mut w_pinned_vectors: Vec<Vec<i32>> = Vec::new();
    let mut b_pinned_pieces: Vec<PieceId> = Vec::new();
    let mut w_pinned_pieces: Vec<PieceId> = Vec::new();
    
    // Handle potential None values safely using `match` or `if let`
    let b_king_index = board.black_indexes.get(&PieceId::K);
    let w_king_index = board.white_indexes.get(&PieceId::K);
    
    // Safely perform integer division after ensuring values are available
    let mut w_king_moves = generate_available_moves(board, w_king_index / 10 as usize, w_king_index % 10 as usize);
    let mut b_king_moves = generate_available_moves(board, b_king_index / 10 as usize, b_king_index % 10 as usize);
    
    // Initialize over_w and over_b as empty vectors
    let mut over_w: Vec<i32> = vec![];
    let mut over_b: Vec<i32> = vec![];
    for &piece_id in board.white_piece_ids {
        let piece_index = board.white_indexes.get(piece_id).unwrap_or(PieceId::Empty);
        let curr_row = piece_index / 10;
        let curr_col = piece_index % 10;

        // Adjusted to your available function signature and logic
        let all_moves = generate_available_moves(board, curr_row, curr_col); // Assuming this function exists and works as intended

        let mut checking = false;
        if all_moves.contains(b_king_moves){
            board.in_check_stored=true;
            checking=true;
            white_checking.push(piece);
        }

        if board.full_board[curr_row][curr_col].kind==Kind::Rook || board.full_board[curr_row][curr_col].kind==Kind::Queen && checking==false{
            let mut temp=w_rook_pinning(&board, piece);
            if temp.len()>0{
                let a=temp.clone(temp[0][temp.len()-1]);
                temp.remove(temp[0].len()-1);
                b_pinned_vectors.append(temp);
                b_pinned_pieces.append(a);
                over_b.extend(temp[1]);
            }
        }

        if board.full_board[curr_row][curr_col].kind==Kind::Bishop || board.full_board[curr_row][curr_col].kind==Kind::Queen && checking==false{
            let mut temp=b_rook_pinning(&board, piece);
            if temp.len()>0{
                let a=temp.clone(temp[0][temp.len()-1]);
                temp.remove(temp[0].len()-1);
                b_pinned_vectors.append(temp);
                b_pinned_pieces.append(a);
                over_b.extend(temp[1]);
            }
        }
        for (index, value) in b_king_moves.iter.enumerate{
            if all_moves.contains(value){
                b_king_moves.remove(index);
            }
        }
        board.white_available_moves.insert(piece, all_moves);

        b_ks=b_ks && !all_moves.contains(7) && !all_moves.contains(6) && !all_moves.contains(5) && !all_moves.contains(4);
        b_qs=b_qs && !all_moves.contains(4) && !all_moves.contains(3) && !all_moves.contains(2) && !all_moves.contains(1) && !all_moves.contains(0);
    }
    for i in board.black_piece_ids{
        let piece=i;
        let &piece_index = board.black_indexes.get(&piece_id);
        let curr_row=indexes/10;
        let curr_col=indexes%10;
        let mut all_moves=generate_available_moves(&board, curr_row, curr_col);
        let mut checking=false;

        if all_moves.contains(w_king_moves){
            board.in_check_stored=true;
            checking=true;
            black_checking.push(piece);
        }

        if board.full_board[curr_row][curr_col].kind==Kind::Rook || board.full_board[curr_row][curr_col].kind==Kind::Queen && checking==false{
            let mut temp=b_rook_pinning(&board, piece);
            if temp.len()>0{
                let a=temp.clone(temp[0][temp.len()-1]);
                temp.remove(temp[0].len()-1);
                w_pinned_vectors.append(temp);
                w_pinned_pieces.append(a);
                over_w.extend(&temp[1]);
            }
        }

        if board.full_board[curr_row][curr_col].kind==Kind::Bishop || board.full_board[curr_row][curr_col].kind==Kind::Queen && checking==false{
            let mut temp=b_bishop_pinning(&board, piece);
            if temp.len()>0{
                let a=temp.clone(temp[0][temp.len()-1]);
                temp.remove(temp[0].len()-1);
                w_pinned_vectors.append(temp);
                w_pinned_pieces.append(a);
                over_w.extend(&temp[1]);
            }
        }
        for (index, value) in w_king_moves.iter.enumerate{
            if all_moves.contains(value){
                w_king_moves.remove(index);
            }
        }
        board.black_available_moves.insert(all_moves);
        

        w_ks=w_ks && !all_moves.contains(77) && !all_moves.contains(76) && !all_moves.contains(75) && !all_moves.contains(74);
        w_qs=w_qs && !all_moves.contains(74) && !all_moves.contains(73) && !all_moves.contains(72) && !all_moves.contains(71) && !all_moves.contains(70);

    }
    if w_ks==true{
        w_king_indexes.push(99);
    }
    if w_qs==true{
        w_king_indexes.push(100)
    }
    if b_ks==true{
        w_king_indexes.push(99);
    }
    if b_qs==true{
        w_king_indexes.push(100)
    }
    for (index, &element) in over_w{
        if w_king_moves.contains(element) && board.black_i_to_p.contains(element){
            w_king_moves.remove(index);
        }
    }
    for (index, &element) in over_b{
        if b_king_moves.contains(element) && board.white_i_to_p.contains(element){
            b_king_moves.remove(index);
        }
    }
    if w_pinned_vectors.length()>0{
        for (index, &element) in w_pinned_pieces{
            let mut overlap:vec<i32>=[];
            let piece=board.white_i_to_p.get(element);
            for (index2, &element2) in w_pinned_vectors{
                if board.white_available_moves.contains(element2);
                overlap.push(element);
            }
            board.white_available_moves.insert(piece, overlap);
        }
    }
    if b_pinned_vectors.length()>0{
        for (index, &element) in b_pinned_pieces{
            let mut overlap:vec<i32>=[];
            let piece=board.black_i_to_p.get(element);
            for (index2, &element2) in b_pinned_vectors{
                if board.black_available_moves.contains(element2);
                overlap.push(element);
            }
            board.black_available_moves.insert(piece, overlap);
        }
    }
    if black_checking.length()>0{
        for &i in black_checking{
            let k=map_piece_id_to_kind(piece_id);
            if k==Kind::Rook || k==Kind::Bishop || k==Kind::Queen{
                let mut direction:vec<i32>=[];
                direction.push(board.white_indexes.get(PieceId::King)/10-board.black_indexes.get(i)/10);
                direction.push(board.white_indexes.get(PieceId::King)%10-board.black_indexes.get(i)%10);
                in_check_directional(i, Team::B, direction);
            }
        }
    }
    if white_checking.length()>0{
        for &i in white_checking{
            let k=map_piece_id_to_kind(piece_id);
            if k==Kind::Rook || k==Kind::Bishop || k==Kind::Queen{
                let mut direction:vec<i32>=[];
                direction.push(board.black_indexes.get(PieceId::King)/10-board.white_indexes.get(i)/10);
                direction.push(board.black_indexes.get(PieceId::King)%10-board.white_indexes.get(i)%10);
                in_check_directional(i, Team::W, direction);
            }
        }
    }
    king_move_eliminator_black(&mut board);
    king_move_eliminator_white(&mut board);
}

pub fn move(&mut board:Board, move_piece:PieceId, indexes:i32, pawn_premotion_queen:bool){
    board.in_check_stored=false;
    if board.turn%2==0{
        let initial_coords=board.white_indexes.get(move_piece);
        let new_indexes=indexes;
        let old_row=initial_coords/10;
        let old_col=initial_coords%10;
        let new_row=new_indexes/10;
        let new_col=new_indexes%10;
        board.white_prime1=1;
        if move_piece==PieceId::K{
            board.prime2=board.prime2*13;
        }
        if move_piece==PieceId::R1{
            board.prime2=board.prime2*2;
        }
        if move_piece==PieceId::R2{
            board.prime2=board.prime2*3;
        }
        if new_indexes=99{
            board.full_board[7][4]=Piece{0, PieceId::Empty, Team::N};
            board.full_board[7][7]=Piece{0, PieceId::Empty, Team::N};
            board.full_board[7][6]=Piece{0, PieceId::King, Team::W};
            board.full_board[7][5]=Piece{500, PieceId::Rook, Team::W};
            board.white_indexes.insert(PieceId::K, 76);
            board.white_indexes.insert(PieceId::R2, 76);
            board.prime2=board.prime2*5;
            board.turn+=1;
            reset_board(board);
            return;
        }
        if new_indexes=99{
            board.full_board[7][1]=Piece{0, PieceId::Empty, Team::N};
            board.full_board[7][0]=Piece{0, PieceId::Empty, Team::N};
            board.full_board[7][4]=Piece{0, PieceId::Empty, Team::N};
            board.full_board[7][2]=Piece{0, PieceId::King, Team::W};
            board.full_board[7][3]=Piece{500, PieceId::Rook, Team::W};
            board.white_indexes.insert(PieceId::K, 72);
            board.white_indexes.insert(PieceId::R2, 73);
            board.prime2=board.prime2*5;
            board.turn+=1;
            reset_board(board);
            return;
        }
        let old_points=board.full_board[new_row][new_col];
        if map_piece_id_to_kind(move_piece)==Kind::Pawn && new_col != old_col && old_points==0{
            ep_white(board, move_piece, indexes);
            reset_board(board);
            return;
        }
        if map_piece_id_to_kind move_piece==Kind::Pawn && (new_row-old_row).abs()==2{
            board.white_prime=board.white_prime*primes(move_col);
        }
        if old_points>0{
            let b_old_piece=board.black_i_to_p.get(new_indexes);
            board.black_available_moves.remove(b_old_piece);
            board.black_i_to_p.remove(new_indexes);
            board.black_pieces.remove(b_old_piece);
            board.black_indexes.remove(b_old_piece);
            board.black_points-=old_points;
        }
        if map_piece_id_to_kind move_piece==Kind::Pawn && row ==0{
            board.white_i_to_p.remove(initial_coords);
            board.white_indexes.remove(piece);
            board.white_pieces.remove(piece);
            board.white_available_moves.remove(piece);
            board.white_points-=100;
            if pawn_premotion_queen{
                let new_piece=pawn_to_queen(piece);
                board.white_i_to_p.push(new_indexes, new_piece);
                board.white_indexes.insert(new_piece, new_indexes);
                board.white_pieces.push(new_piece);
                board.white_available_moves.insert(new_piece, []);
                board.full_board[new_row][new_col]=Piece{900, map_piece_id_to_kind(new_piece), Team::W};
                board.full_board[new_row][new_col]=Piece{0, PieceId::empty, Team::N};
                board.white_points+=900;
            }
            else{
                let new_piece=pawn_to_knight(piece);
                board.white_i_to_p.push(new_indexes, new_piece);
                board.white_indexes.insert(new_piece, new_indexes);
                board.white_pieces.push(new_piece);
                board.white_available_moves.insert(new_piece, []);
                board.white_points+=300;
                board.full_board[new_row][new_col]=Piece{300, map_piece_id_to_kind(new_piece), Team::W};
                board.full_board[new_row][new_col]=Piece{0, PieceId::empty, Team::N};
            }
            board.turn+=1;
            return;
        }
        board.full_board[new_row][new_col]=board.full_board[old_row][old_col].clone();
        board.full_board[old_row][old_col]=Piece{0, Kind::Empty, Team::N};
        board.white_i_to_p.remove(initial_coords);
        board.white_indexes.insert(piece, new_indexes);
        board.turn+=1;
        reset_board(board);
    }
    else{
        let initial_coords=board.black_indexes.get(move_piece);
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
        if new_indexes=99{
            board.full_board[0][4]=Piece{0, Kind::Empty, Team::N};
            board.full_board[0][7]=Piece{0, Kind::Empty, Team::N};
            board.full_board[0][6]=Piece{0, Kind::King, Team::B};
            board.full_board[0][5]=Piece{500, Kind::Rook, Team::B};
            board.black_indexes.insert(PieceId::K, 76);
            board.black_indexes.insert(PieceId::R2, 76);
            board.turn+=1;
            reset_board(board);
            return;
        }
        if new_indexes=99{
            board.full_board[0][1]=Piece{0, Kind::Empty, Team::N};
            board.full_board[0][0]=Piece{0, Kind::Empty, Team::N};
            board.full_board[0][4]=Piece{0, Kind::Empty, Team::N};
            board.full_board[0][2]=Piece{0, Kind::King, Team::B};
            board.full_board[0][3]=Piece{500, Kind::Rook, Team::B};
            board.black_indexes.insert{PieceId::K, 2};
            board.black_indexes.insert{PieceId::R2, 3};
            board.prime2=board.prime2*5;
            board.turn+=1;
            reset_board(board);
            return;
        }
        let old_points=board.full_board[new_row][new_col];
        if map_piece_id_to_kind(move_piece)==Kind::Pawn && new_col != old_col && old_points==0{
            ep_black(board, move_piece, indexes);
            reset_board(board);
            return;
        }
        if map_piece_id_to_kind move_piece==Kind::Pawn && (new_row-old_row).abs()==2{
            board.black_prime=board.black_prime*primes(move_col);
        }
        if old_points>0{
            let b_old_piece=board.white_i_to_p.get(new_indexes);
            board.white_available_moves.remove(b_old_piece);
            board.white_i_to_p.remove(new_indexes);
            board.white_pieces.remove(b_old_piece);
            board.white_indexes.remove(b_old_piece);
            board.white_points-=old_points;
        }
        if map_piece_id_to_kind move_piece==Kind::Pawn && row ==7{
            board.black_i_to_p.remove(initial_coords);
            board.black_indexes.remove(piece);
            board.black_pieces.remove(piece);
            board.black_available_moves.remove(piece);
            board.black_points-=100;
            if pawn_premotion_queen{
                let new_piece=pawn_to_queen(piece);
                board.black_i_to_p.push(new_indexes, new_piece);
                board.black_indexes.insert(new_piece, new_indexes);
                board.black_pieces.push(new_piece);
                board.black_available_moves.insert(new_piece, []);
                board.full_board[new_row][new_col]=Piece{900, map_piece_id_to_kind(new_piece), Team::Black};
                board.full_board[new_row][new_col]=Piece{0, Kind::Empty, Team::None};
                board.black_points+=900;
            }
            else{
                let new_piece=pawn_to_knight(piece);
                board.black_i_to_p.push(new_indexes, new_piece);
                board.black_indexes.insert(new_piece, new_indexes);
                board.black_pieces.push(new_piece);
                board.black_available_moves.insert(new_piece, []);
                board.black_points+=300;
                board.full_board[new_row][new_col]=Piece{300, map_piece_id_to_kind(new_piece), Team::White};
                board.full_board[new_row][new_col]=Piece{0, Kind::Empty, Team::None};
            }
            board.turn+=1;
            reset_board(board);
            return;
        }
        board.full_board[new_row][new_col]=board.full_board[old_row][old_col].clone();
        board.full_board[old_row][old_col]=Piece{0, Kind::Empty, Team::None};
        board.black_i_to_p.remove(initial_coords);
        board.black_indexes.insert(piece, new_indexes);
        reset_board(board);
        board.turn+=1;
    }
}
