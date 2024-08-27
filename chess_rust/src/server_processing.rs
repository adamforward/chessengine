use crate::test_board::{split_string_to_chars, reverse_mapping_2, reverse_row_mapping, reverse_col_mapping, map_standard_format_to_kind };
use crate::types::{Board,Move, GameStateIdentifiers, Kind,PieceId,AllMovesGenRe,Team, ServerProcessingRe, TreeNode,MoveAdvMap,AIMoveRe};
use crate::base_functions::{contains_element, map_piece_id_to_kind, init_board};
use crate::upper_move_functions::{all_moves_gen, move_piece};
use crate::mongo_repo::{MongoRepo, MongoBoard};
use std::{cell::RefCell, rc::Rc};
use crate::search_functions::{generate_top_moves, search};
use crate::ai_functions::{game_still_going};
use serde_json::Value;
use std::fs;
use std::collections::HashMap;
use tch::{nn, nn::VarStore, Device, Tensor};


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

pub fn pgn_to_hash(b:&Board, standard_format_move:&str, moves:&AllMovesGenRe)->Move{
    println!("{}", standard_format_move);

    if standard_format_move=="O-O-O"{ //castling
        return Move {piece:PieceId::K, location:100};
    }
    if standard_format_move=="O-O"{
        return Move {piece:PieceId::K, location:99};
    }

    let rows=vec!["8", "7", "6", "5", "4", "3", "2", "1"];
    let cols=vec!["a", "b", "c", "d", "e", "f", "g", "h"];
    let mut split_sfm=split_string_to_chars(standard_format_move);
    
    if split_sfm[split_sfm.len()-1]=="+"{ //nothing in my data model for checking
        split_sfm.remove(split_sfm.len()-1);
    }

    let move_as_str=format!("{}{}", split_sfm[split_sfm.len()-2], split_sfm[split_sfm.len()-1]);
    let loc=reverse_mapping_2(&move_as_str);

    let kind=map_standard_format_to_kind(&split_sfm[0]);
    let col=reverse_col_mapping(&split_sfm[split_sfm.len()-2]);
    let mut valid_indexes:Vec<usize>=vec![];

    if kind==Kind::King{ //simple since there's only ever 1 king
        return Move {piece:PieceId::K, location: loc};
    } 


    if b.turn%2==0{
        //101 is pawn to q, 102 is pawn to k
        if loc==101 || loc==102{

            let location=reverse_col_mapping(&split_sfm[split_sfm.len()-4]);

            for i in b.white_piece_ids.iter(){
                if contains_element(&moves.moves.get_moves(*i), location){
                    valid_indexes.push(b.white_indexes.get_index(*i).unwrap());
                }
            }

            if valid_indexes.len()==1{
                if loc==101{
                    return Move {piece: b.white_i_to_p.get_piece(valid_indexes[0]).unwrap(), location:location}
                }else{
                    return Move {piece: b.white_i_to_p.get_piece(valid_indexes[0]).unwrap(), location:80+location}
                }
            } else{
                let starting_col=reverse_col_mapping(&split_sfm[0]);
                if loc==102{
                    return Move {piece: b.white_i_to_p.get_piece(10+starting_col).unwrap(), location:80+location}
                } else{
                    return Move {piece: b.white_i_to_p.get_piece(10+starting_col).unwrap(), location:location}
                }
            }
        }

        for i in b.white_piece_ids.iter(){
            if map_piece_id_to_kind(*i)==kind{
                if kind==Kind::Pawn &&split_sfm.len()==2 && b.white_indexes.get_index(*i).unwrap()%10==col && contains_element(&moves.moves.get_moves(*i), loc){
                    return Move {piece:*i, location:loc};
                }
                if kind==Kind::Queen || kind==Kind::Knight || kind==Kind::Rook || kind==Kind::Pawn || kind==Kind::Bishop{
                    if contains_element(&moves.moves.get_moves(*i), loc){
                        valid_indexes.push(b.white_indexes.get_index(*i).unwrap());
                    }
                }
            }
        }
        if valid_indexes.len()==1{
            return Move { piece: b.white_i_to_p.get_piece(valid_indexes[0]).unwrap(), location:loc};
        }
        else{
            for i in split_sfm.iter(){
                if contains_element(&cols, i){
                    let c=reverse_col_mapping(i);
                    if c%10==valid_indexes[0]%10{
                        return Move { piece: b.white_i_to_p.get_piece(valid_indexes[0]).unwrap(), location:loc};
                    }
                    else{
                        return Move { piece: b.white_i_to_p.get_piece(valid_indexes[1]).unwrap(), location:loc};
                    }
                }
                if contains_element(&rows, i){
                    let r=reverse_row_mapping(i);
                    if r/10==valid_indexes[0]/10{
                        return Move { piece: b.white_i_to_p.get_piece(valid_indexes[0]).unwrap(), location:loc};
                    }
                    else{
                        return Move { piece: b.white_i_to_p.get_piece(valid_indexes[1]).unwrap(), location:loc};
                    }
                }
            }
        }
    }
    else{
        if loc==101 || loc==102{
            let location=reverse_col_mapping(&split_sfm[split_sfm.len()-4]) + 70;

            for i in b.black_piece_ids.iter(){
                if contains_element(&moves.moves.get_moves(*i), location){
                    valid_indexes.push(b.black_indexes.get_index(*i).unwrap());
                }
            }

            if valid_indexes.len()==1{
                if loc==101{
                    return Move {piece: b.black_i_to_p.get_piece(valid_indexes[0]).unwrap(), location:location}
                }else{
                    return Move {piece: b.black_i_to_p.get_piece(valid_indexes[0]).unwrap(), location:10+location}
                }
            }else{
                let starting_col=reverse_col_mapping(&split_sfm[0]);
                if loc==102{
                    return Move {piece: b.black_i_to_p.get_piece(60+starting_col).unwrap(), location:10+location}
                } else{
                    return Move {piece: b.black_i_to_p.get_piece(60+starting_col).unwrap(), location:location}
                }
            }
        }

        for i in b.black_piece_ids.iter(){
            if map_piece_id_to_kind(*i)==kind{
                if kind==Kind::Pawn && split_sfm.len()==2&& b.black_indexes.get_index(*i).unwrap()%10==col && contains_element(&moves.moves.get_moves(*i), loc){
                    return Move {piece:*i, location:loc};
                }
                if kind==Kind::Queen || kind==Kind::Knight || kind==Kind::Rook ||kind==Kind::Pawn || kind==Kind::Bishop{
                    if contains_element(&moves.moves.get_moves(*i), loc){
                        valid_indexes.push(b.black_indexes.get_index(*i).unwrap());
                    }
                }
            }
        }
        if valid_indexes.len()==1{
            return Move { piece: b.black_i_to_p.get_piece(valid_indexes[0]).unwrap(), location:loc};
        }
        else{
            for i in split_sfm.iter(){
                if contains_element(&cols, i){
                    let c=reverse_col_mapping(i);
                    if c%10==valid_indexes[0]%10{
                        return Move { piece: b.black_i_to_p.get_piece(valid_indexes[0]).unwrap(), location:loc};
                    }
                    else{
                        return Move { piece: b.black_i_to_p.get_piece(valid_indexes[1]).unwrap(), location:loc};
                    }
                }
                if contains_element(&rows, i){
                    let r=reverse_row_mapping(i);
                    if r/10==valid_indexes[0]/10{
                        return Move { piece: b.black_i_to_p.get_piece(valid_indexes[0]).unwrap(), location:loc};
                    }
                    else{
                        return Move { piece: b.black_i_to_p.get_piece(valid_indexes[1]).unwrap(), location:loc};
                    }
                }
            }
        }
    }
    return Move {piece:PieceId::Error, location:54321};
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

pub fn process_server_response(pgn:String, last_game:Option<MongoBoard>)->ServerProcessingRe{
    let address_pgn:&str=pgn.as_str();
    if last_game.is_none() {

        if pgn.len()==0{
            //if white start always move e4
            let initial_board=init_board(true);
            let pgn2="e4";
            let moved2=move_piece(initial_board, PieceId::P5, 44);
            return ServerProcessingRe{b:MongoBoard{board:moved2, pgn:pgn2.to_string()}, pgn:pgn2.to_string()};
        }

        else{
            //black start
            let initial_board=init_board(true);
            let moves=all_moves_gen(&initial_board);
            let hash=pgn_to_hash(&initial_board, address_pgn, &moves);
            let turn_2=move_piece(initial_board, hash.piece, hash.location);
            let moved=ai_move(turn_2.clone(), address_pgn).unwrap();
            let moved2=move_piece(moved.b, moved.m.piece, moved.m.location);
            let pgn2=hash_to_pgn(&turn_2, moved.m.piece, moved.m.location,&moves);
            let pgn3=format!("{}{}", pgn, pgn2);
            return ServerProcessingRe{b:MongoBoard{board:moved2,pgn:pgn3}, pgn:pgn2};
        }
    }
    
    else{
        //past turn 2
        let last_game=last_game.unwrap();
        let moves=all_moves_gen(&last_game.board);
        let hash=pgn_to_hash(&last_game.board,address_pgn,&moves);
        let turn_2=move_piece(last_game.board.clone(), hash.piece, hash.location);
        let ai_move_param_pgn=last_game.pgn;
        let moved=ai_move(turn_2.clone(), &ai_move_param_pgn).unwrap();
        let moved2=move_piece(moved.b.clone(), moved.m.piece, moved.m.location);
        let pgn2=hash_to_pgn(&turn_2, moved.m.piece, moved.m.location,&moves);
        let pgn3=format!("{}{}", pgn, pgn2);
        let pgn4=format!("{}{}", ai_move_param_pgn, pgn3);
        return ServerProcessingRe{b:MongoBoard{board:moved2,pgn:pgn4}, pgn:pgn2};
    }
}

fn ai_move(b:Board, pgn:&str)->Option<AIMoveRe>{
    if b.turn<10{
    let b2=b.clone();
    // Load the JSON file into a string
    let data = fs::read_to_string("opening_book.json")
        .expect("Unable to read file");
    // Parse the JSON string into a HashMap for efficient key-value access
    let book: HashMap<String, String> = serde_json::from_str(&data)
        .expect("JSON was not well-formatted");
    if let Some(move_value) = book.get(pgn) {
        //if something is returned from the opening book
        let param_moves=all_moves_gen(&b);
        let re_move:Move=pgn_to_hash(&b, move_value, &param_moves);
        let moved_board=move_piece(b2,re_move.piece, re_move.location);
        return Some(AIMoveRe{b:moved_board,m:re_move});
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
        let mut biggest_mm:f32=0.0;
        for i in b.white_piece_ids.iter(){
            for j in av_moves.moves.get_moves(*i).iter(){
                let searching_b=move_piece(b.clone(), *i,*j);
                let searching_treenode=TreeNode {game:searching_b.clone(), level:0, children:vec![]};
                let new_mm=search(searching_treenode, 5, 5, biggest_mm, 1.0);//will tweak depth and width params
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
        let mut biggest_mm:f32=0.0;
        for i in b.black_piece_ids.iter(){
            for j in av_moves.moves.get_moves(*i).iter(){
                let searching_b=move_piece(b.clone(), *i,*j);
                let searching_treenode=TreeNode {game:searching_b.clone(), level:0, children:vec![]};
                let new_mm=search(searching_treenode, 5, 5, biggest_mm, 1.0);
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