use crate::test_board::{split_string_to_chars, reverse_mapping_2, reverse_row_mapping, reverse_col_mapping, map_standard_format_to_kind };
use crate::types::{Board,Move, GameStateIdentifiers, Kind,PieceId,AllMovesGenRe,Team, TreeNode,MoveAdvMap,AIMoveRe};
use crate::base_functions::{contains_element, map_piece_id_to_kind, init_board};
use crate::upper_move_functions::{all_moves_gen, move_piece};
use std::{cell::RefCell, rc::Rc};
use crate::search_functions::{generate_top_moves, search};
use crate::ai_functions::{game_still_going};
use serde_json::Value;
use std::fs;
use std::collections::HashMap;


pub fn row_mapping(n: usize) -> &'static str {
    match n {
        0 => "8",
        10 => "7",
        20 => "6",
        30 => "5",
        40 => "4",
        50 => "3",
        60 => "2",
        70 => "1",
        _ => "invalid",
    }
}

pub fn col_mapping(n: usize) -> &'static str {
    match n {
        0 => "a",
        1 => "b",
        2 => "c",
        3 => "d",
        4 => "e",
        5 => "f",
        6 => "g",
        7 => "h",
        _ => "invalid",
    }
}

fn mapping_2(n: usize) -> &'static str {
    match n {
        0 => "a8",  1 => "b8",  2 => "c8",  3 => "d8",  4 => "e8",  5 => "f8",  6 => "g8",  7 => "h8",
        10 => "a7", 11 => "b7", 12 => "c7", 13 => "d7", 14 => "e7", 15 => "f7", 16 => "g7", 17 => "h7",
        20 => "a6", 21 => "b6", 22 => "c6", 23 => "d6", 24 => "e6", 25 => "f6", 26 => "g6", 27 => "h6",
        30 => "a5", 31 => "b5", 32 => "c5", 33 => "d5", 34 => "e5", 35 => "f5", 36 => "g5", 37 => "h5",
        40 => "a4", 41 => "b4", 42 => "c4", 43 => "d4", 44 => "e4", 45 => "f4", 46 => "g4", 47 => "h4",
        50 => "a3", 51 => "b3", 52 => "c3", 53 => "d3", 54 => "e3", 55 => "f3", 56 => "g3", 57 => "h3",
        60 => "a2", 61 => "b2", 62 => "c2", 63 => "d2", 64 => "e2", 65 => "f2", 66 => "g2", 67 => "h2",
        70 => "a1", 71 => "b1", 72 => "c1", 73 => "d1", 74 => "e1", 75 => "f1", 76 => "g1", 77 => "h1",
        _ => "invalid",
    }
}


fn hash_to_uci(start_index:usize, final_index:usize, white:bool, pawn:bool)->String{
    println!("start_index, final index, white, pawn {} {} {} {}", start_index, final_index, white, pawn);
    if final_index==99{
        if white{
            return "e1g1".to_string();
        }
        else{
            return "e8g8".to_string();
        }
    }
    if final_index==100{
        if white{
            return "e1c1".to_string();
        }
        else{
            return "e8c8".to_string();
        }
    }
    if pawn && white && (final_index/10==0 || final_index/10==8){
        if final_index/10==8{
            return format!("{}{}{}", mapping_2(start_index), mapping_2(final_index%10), "n");
        }
        else{
            return format!("{}{}{}", mapping_2(start_index), mapping_2(final_index), "q");
        }
    }
    if pawn && !white && (final_index/10==7 || final_index/10==8){
        if final_index/10==8{
            return format!("{}{}{}", mapping_2(start_index), mapping_2(70+final_index%10), "n");
        }
        else{
            return format!("{}{}{}", mapping_2(start_index), mapping_2(final_index), "q");
        }
    }
    return format!("{}{}", mapping_2(start_index), mapping_2(final_index));
}

fn uci_to_hash(uci:String, white:bool, ks:bool, qs:bool)->(usize, usize){
    if ks{
        if white{
            return (74, 99);
        }
        else{
            return (4, 99);
        }
    }
    if qs{
        if white{
            return (74, 100);
        }
        else{
            return (4, 100);
        }
    }
    let (initial_indexes, final_indexes) = uci.split_at(2);
    if final_indexes.len() == 2 {
        return (reverse_mapping_2(initial_indexes), reverse_mapping_2(final_indexes));
    }
    else{ //promotion
        let promo_type=&uci[4..5];
        let promo_col=&uci[2..3]; 
        if promo_type=="n"{
            return(reverse_mapping_2(initial_indexes), 80+reverse_col_mapping(promo_col));
        }
        if white{
            return(reverse_mapping_2(initial_indexes), reverse_col_mapping(promo_col));
        }
        else{
            return(reverse_mapping_2(initial_indexes), 70+reverse_col_mapping(promo_col));
        }
    }
}

fn kind_to_str_map(k: Kind) -> &'static str {
    match k {
        Kind::Knight => "N",
        Kind::Bishop => "B",
        Kind::King => "K",
        Kind::Rook => "R",
        _ => "",
    }
}

pub fn hash_to_pgn(b:&Board, piece:PieceId, index:usize, moves_info:&AllMovesGenRe)->String{
    if index==99{
        return "O-O".to_string();
    }//ks or qs castle
    if index==100{
        return "O-O-O".to_string();
    }
    let mut priority_hash=vec!["", "", "", "", "", ""];
    //this is the order that the string will be displayed in pgn
    //first priority in the string is piece identifier
    //second is original location identifier, if there are two pieces with the same potential move it'll have column identifier or row identifier if columns are the same. 
    //test_board has the mappings for these
    //third is the capturing identifier, x if capturing
    //fourth is the new idexes identifier, mappings are in other function
    //fifth is the promotion idententifier, =Q or =K if knight is getting promoted. 
    //sixth is checking identifier, + if checking. 

    let white_turn=b.turn%2==0;
    let k=map_piece_id_to_kind(piece);
    //start with first and sixth identifiers since they're easy and independent on what team is moving
    priority_hash[0]=kind_to_str_map(k);
    if moves_info.checking{
        priority_hash[5]="+";
    }
    let move_col=index%10;
    let mut move_row=index/10;

    if map_piece_id_to_kind(piece)==Kind::Pawn && move_row==7 && white_turn{
        priority_hash[4]="=Q";
    }
    if map_piece_id_to_kind(piece)==Kind::Pawn && move_row==0 && !white_turn{
        priority_hash[4]="=Q";
    }


    if white_turn{
        //pawn promotions id is the next one since we have to change the indexes if its a knight promotion
        let init_indexes=b.white_indexes.get_index(piece).unwrap();
        if index/10==8{
            move_row=7;
            priority_hash[4]="=N";//8th row is id for knight promotion
        }
        if b.full_board[move_row][move_col].team==Team::B{
            priority_hash[2]="x";
        }
        priority_hash[3]=mapping_2(move_row*10 + move_col);
        //got all but the original location identifier, this one you have to iterate through all the moves. 
        let mut pressuring_target:Vec<usize>=vec![];
        for i in b.white_piece_ids.iter(){
            if map_piece_id_to_kind(piece)!=map_piece_id_to_kind(*i){
                for j in moves_info.moves.get_moves(*i){
                    if j==move_row*10+move_col{
                        pressuring_target.push(b.white_indexes.get_index(*i).unwrap());
                    }
                }
            }
        }
        if map_piece_id_to_kind(piece)==Kind::Pawn{
            if priority_hash[2]=="x"{
                //pawns work differently where it shows this when something is captured
                priority_hash[1]=col_mapping(init_indexes%10);
            }
        }
        else if map_piece_id_to_kind(piece)!=Kind::Pawn || map_piece_id_to_kind(piece)!=Kind::King{//never a conflict with the king
            let mut col_eq=false;
            for i in pressuring_target{
                if i!=init_indexes && i%10==init_indexes%10{
                    col_eq=true;
                }
            }
            if !col_eq{
                priority_hash[1]=col_mapping(init_indexes%10);
            }
            else{
                priority_hash[1]=col_mapping(init_indexes/10);
            }
        }

    }
    else{
        let init_indexes=b.black_indexes.get_index(piece).unwrap();
        if index/10==8{
            move_row=7;
            priority_hash[4]="=N";//8th row is id for knight promotion
        }
        if b.full_board[move_row][move_col].team==Team::W{
            priority_hash[2]="x";
        }
        priority_hash[3]=mapping_2(move_row*10 + move_col);
        //got all but the original location identifier, this one you have to iterate through all the moves. 
        let mut pressuring_target:Vec<usize>=vec![];
        for i in b.black_piece_ids.iter(){
            if map_piece_id_to_kind(piece)!=map_piece_id_to_kind(*i){
                for j in moves_info.moves.get_moves(*i){
                    if j==move_row*10+move_col{
                        pressuring_target.push(b.black_indexes.get_index(*i).unwrap());
                    }
                }
            }
        }
        if map_piece_id_to_kind(piece)==Kind::Pawn{
            if priority_hash[2]=="x"{
                //pawns work differently where it shows this when something is captured
                priority_hash[1]=col_mapping(init_indexes%10);
            }
        }
        else if map_piece_id_to_kind(piece)!=Kind::Pawn || map_piece_id_to_kind(piece)!=Kind::King{//never a conflict with the king
            let mut col_eq=false;
            for i in pressuring_target{
                if i!=init_indexes && i%10==init_indexes%10{
                    col_eq=true;
                }
            }
            if !col_eq{
                priority_hash[1]=col_mapping(init_indexes%10);
            }
            else{
                priority_hash[1]=col_mapping(init_indexes/10);
            }
        }
    }
    return combine_strings(priority_hash);
}

fn combine_strings(strings: Vec<&str>) -> String {
    let mut result = String::new();
    for s in strings.iter() {
        result.push_str(s);
    }
    result
}

pub fn process_server_response(game_history: Vec<String>, ai_team_is_white:bool) -> String {
    println!("ai team {}", ai_team_is_white);
    println!("{:?}", game_history);
    if game_history.len() == 0 {
        return "e2e4".to_string();
    } else {
        let mut start_board = init_board(ai_team_is_white);
        let mut proper_order:Vec<String>=vec![];
        for i in game_history.iter().rev() {
            let mut qs=false;
            let mut ks=false; 
            proper_order.push(i.to_string());
            if start_board.turn%2==0{
                if i.to_string()=="e1g1".to_string() && start_board.prime2%5!=0{
                    ks=true;
                }
                if i.to_string()=="e1c1".to_string() && start_board.prime2%5!=0{
                    qs=true;
                }
            }
            else{
                if i.to_string()=="e8g8".to_string() && start_board.prime2%13!=0{
                    ks=true;
                }
                if i.to_string()=="e8c8".to_string() && start_board.prime2%13!=0{
                    qs=true;
                }
            }
            let (old_indexes, new_indexes) = uci_to_hash(i.to_string(), start_board.turn%2==0, ks, qs);
            let move_p: PieceId = if start_board.turn%2==0{
                start_board.white_i_to_p.get_piece(old_indexes)
            } else {
                start_board.black_i_to_p.get_piece(old_indexes)
            }
            .unwrap();
            start_board = move_piece(start_board, move_p, new_indexes);
        }
        let start_board2 = start_board.clone();
        //print_all(start_board2.clone());
        let ai_move_returned = ai_move(start_board2.clone(), &proper_order.iter().map(|s| format!("{}", s)).collect::<String>()).unwrap();
        let ai_move_returned2=ai_move_returned.clone();
        let ai_move_returned3=ai_move_returned.clone();
        let init_indexes = if start_board2.ai_team_is_white {
            start_board2.white_indexes.get_index(ai_move_returned2.m.piece).unwrap()
        } else {
            start_board2.black_indexes.get_index(ai_move_returned2.m.piece).unwrap()
        };
        println!("piece {}", ai_move_returned3.m.piece.to_string());
        let pawn= map_piece_id_to_kind(ai_move_returned3.m.piece) == Kind::Pawn;
        println!("output {}", hash_to_uci(
            init_indexes,
            ai_move_returned3.m.location,
            ai_move_returned3.b.ai_team_is_white,
            pawn,
        ));
        return hash_to_uci(
            init_indexes,
            ai_move_returned3.m.location,
            ai_move_returned3.b.ai_team_is_white,
            pawn,
        );
    }
}

fn ai_move(b:Board, uci:&str)->Option<AIMoveRe>{
    if b.turn<10{
    let b2=b.clone();
    // Load the JSON file into a string
    let data = fs::read_to_string("src/opening_book.json")
        .expect("Unable to read file");
    // Parse the JSON string into a HashMap for efficient key-value access
    let book: HashMap<String, String> = serde_json::from_str(&data)
        .expect("JSON was not well-formatted");
    if let Some(move_value) = book.get(uci) {
        let mut ks=false;
        let mut qs=false;
        //if something is returned from the opening book
        if b2.turn%2==0{
                if (move_value.to_string()=="e1g1".to_string() && b2.prime2%5==0) || move_value.to_string()=="O-O".to_string(){
                    ks=true;
                }
                if (move_value.to_string()=="e1c1".to_string() && b2.prime2%5==0) || move_value.to_string()=="O-O-O".to_string(){
                    qs=true;
                }
            }
        else{
            if (move_value.to_string()=="e8g8".to_string() && b2.prime2%13==0) || move_value.to_string()=="O-O".to_string(){
                ks=true;
            }
            if (move_value.to_string()=="e8c8".to_string() && b2.prime2%13==0)|| move_value.to_string()=="O-O-O".to_string() {
                qs=true;
            }
        }
        let param_moves=all_moves_gen(&b);
        let (start_index, end_index)=uci_to_hash(move_value.to_string(), b2.turn%2==0, ks, qs);
        let moved_board:Board;
        let the_piece:PieceId;
        let b3=b2.clone();
        let b4=b3.clone();
        if b2.turn%2==0{
            moved_board=move_piece(b4, b2.white_i_to_p.get_piece(start_index).unwrap(), end_index);
            the_piece=b3.white_i_to_p.get_piece(start_index).unwrap();
        }
        else{
            moved_board=move_piece(b4, b2.black_i_to_p.get_piece(start_index).unwrap(), end_index);
            the_piece=b3.black_i_to_p.get_piece(start_index).unwrap();
        }
        return Some(AIMoveRe{b:moved_board,m:Move{piece:the_piece, location:end_index}});
    }
    }
    let game_states=GameStateIdentifiers::new();

    let av_moves=all_moves_gen(&b);
    if game_still_going(&b, av_moves.checking, &av_moves.moves)!=game_states.in_play{
        return Some(AIMoveRe{m:Move {piece:PieceId::Error, location:10000}, b:b.clone()}) ;// end the game here.
    }
    let mut the_move:Move=Move {piece:PieceId::Error, location:10000};
    let mut re:AIMoveRe=AIMoveRe{b:b.clone(),m:the_move};
    if b.turn%2==0{
        let mut biggest_mm:f32=-f32::INFINITY;
        for i in b.white_piece_ids.iter(){
            for j in av_moves.moves.get_moves(*i).iter(){
                let searching_b=move_piece(b.clone(), *i,*j);
                let searching_treenode=TreeNode {game:searching_b.clone(), level:0, children:vec![]};
                let new_mm=search(searching_treenode, 5, 2, biggest_mm, 1.0);//will tweak depth and width params
                if new_mm>biggest_mm{
                    biggest_mm=new_mm;
                    the_move=Move {piece:*i, location:*j};
                    re= AIMoveRe{b:searching_b, m:the_move}
                }
            }
        }
        return Some(re);
    }
    else{
        let mut biggest_mm:f32=-f32::INFINITY;
        for i in b.black_piece_ids.iter(){
            for j in av_moves.moves.get_moves(*i).iter(){
                let searching_b=move_piece(b.clone(), *i,*j);
                let searching_treenode=TreeNode {game:searching_b.clone(), level:0, children:vec![]};
                let new_mm=search(searching_treenode, 5, 2, biggest_mm, 1.0);
                if biggest_mm<new_mm{
                    biggest_mm=new_mm;
                    the_move=Move {piece:*i, location:*j};
                    re= AIMoveRe{b:searching_b, m:the_move};
                }
            }
        }
        return Some(re);
    }
    }