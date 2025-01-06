use crate::base_functions::{init_board, map_piece_id_to_kind, find_non_overlap, find_overlap, contains_element};
use crate::types::{Board, Piece, PieceId, Team, Kind, Move, AllMovesGenRe, TreeNode, NeuralNetworkSelector};
use crate::upper_move_functions::{all_moves_gen, move_piece};
use crate::search_functions::{search, generate_top_moves};
use crate::ai_functions::{game_still_going};
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Write;
// use rand::Rng;
use std::io;
// use crate::base_move_functions::{generate_available_moves};
pub fn reverse_mapping(b: &str) -> usize {
    match b {
        "A8" => 0,  "B8" => 1,  "C8" => 2,  "D8" => 3,  "E8" => 4,  "F8" => 5,  "G8" => 6,  "H8" => 7,
        "A7" => 10, "B7" => 11, "C7" => 12, "D7" => 13, "E7" => 14, "F7" => 15, "G7" => 16, "H7" => 17,
        "A6" => 20, "B6" => 21, "C6" => 22, "D6" => 23, "E6" => 24, "F6" => 25, "G6" => 26, "H6" => 27,
        "A5" => 30, "B5" => 31, "C5" => 32, "D5" => 33, "E5" => 34, "F5" => 35, "G5" => 36, "H5" => 37,
        "A4" => 40, "B4" => 41, "C4" => 42, "D4" => 43, "E4" => 44, "F4" => 45, "G4" => 46, "H4" => 47,
        "A3" => 50, "B3" => 51, "C3" => 52, "D3" => 53, "E3" => 54, "F3" => 55, "G3" => 56, "H3" => 57,
        "A2" => 60, "B2" => 61, "C2" => 62, "D2" => 63, "E2" => 64, "F2" => 65, "G2" => 66, "H2" => 67,
        "A1" => 70, "B1" => 71, "C1" => 72, "D1" => 73, "E1" => 74, "F1" => 75, "G1" => 76, "H1" => 77,
        _ => 100000,
    }
}

pub fn reverse_mapping_2(b: &str) -> usize {
    match b {
        "a8" => 0,  "b8" => 1,  "c8" => 2,  "d8" => 3,  "e8" => 4,  "f8" => 5,  "g8" => 6,  "h8" => 7,
        "a7" => 10, "b7" => 11, "c7" => 12, "d7" => 13, "e7" => 14, "f7" => 15, "g7" => 16, "h7" => 17,
        "a6" => 20, "b6" => 21, "c6" => 22, "d6" => 23, "e6" => 24, "f6" => 25, "g6" => 26, "h6" => 27,
        "a5" => 30, "b5" => 31, "c5" => 32, "d5" => 33, "e5" => 34, "f5" => 35, "g5" => 36, "h5" => 37,
        "a4" => 40, "b4" => 41, "c4" => 42, "d4" => 43, "e4" => 44, "f4" => 45, "g4" => 46, "h4" => 47,
        "a3" => 50, "b3" => 51, "c3" => 52, "d3" => 53, "e3" => 54, "f3" => 55, "g3" => 56, "h3" => 57,
        "a2" => 60, "b2" => 61, "c2" => 62, "d2" => 63, "e2" => 64, "f2" => 65, "g2" => 66, "h2" => 67,
        "a1" => 70, "b1" => 71, "c1" => 72, "d1" => 73, "e1" => 74, "f1" => 75, "g1" => 76, "h1" => 77,
        "O-O"=>99, "O-O-O"=>100,
        _ => 100000,
    }
}

pub fn reverse_row_mapping(b:&str)->usize{
    match b{
    "8"=>0, "7"=>10, "6"=>20, "5"=>30, "4"=>40, "3"=>50, "2"=>60, "1"=>70, _ => 100000,
    }
}

pub fn reverse_col_mapping(b:&str)->usize{
    match b{
        "a"=>0, "b"=>1, "c"=>2, "d"=>3, "e"=>4, "f"=>5, "g"=>6, "h"=>7, _ =>100000,
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
    
    if split_sfm[split_sfm.len()-1]=="+" || split_sfm[split_sfm.len()-1]=="#"{ //nothing in my data model for checking
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

pub fn map_standard_format_to_kind(input: &str)->Kind{
    match input{
        "N"=>Kind::Knight,
        "B"=>Kind::Bishop,
        "Q"=>Kind::Queen,
        "R"=>Kind::Rook,
        "K"=>Kind::King,
        _=>Kind::Pawn,
    }
}

pub fn process_moves_string(input: &str) -> Vec<String> {
    return input
        .split_whitespace()
        .filter(|s| !s.contains('.'))
        .map(|s| s.to_string())
        .collect();
} // moving the 1. e4 e5  2. Bc4 d5  3. d4 c5  4. Nf3 exd4  5. Nxd4 Nf6 etc..
// format to my data model so I can test full games from copying and pasting 

//white moves will have an even index still

pub fn split_string_to_chars(s: &str) -> Vec<String> {
    s.chars().map(|c| c.to_string()).collect()
}



fn play_std_format_game_full(game: &str) {
    // let mut rng = rand::thread_rng();
    // let random_bool: bool = rng.gen_bool(0.5);
    let mut b = init_board(true); 

    let moves = process_moves_string(game);

    for i in moves.iter() {
        let b1 = b.clone();
        let b2 = b.clone();
        let b4 = b.clone();
        let hash_moves=all_moves_gen(&b1);
        let m = pgn_to_hash(&b1, i,&hash_moves);
        let new_location;
        if b4.turn%2==0{
            new_location = index_to_string(b4.white_indexes.get_index(m.piece).unwrap());
        }else{
            new_location = index_to_string(b4.black_indexes.get_index(m.piece).unwrap());
        }
        println!("ABOUT TO MOVE {} {}", new_location, index_to_string(m.location));
        b = move_piece(b2, m.piece, m.location);
        let b3 = b.clone();
        print_all(b3);
    }
}


pub fn index_to_string(index: usize) -> String {
    if index == 99 {
        return "KS".to_string();
    } else if index == 100 {
        return "QS".to_string();
    } else if index/10==8{
        let col_map = ["A", "B", "C", "D", "E", "F", "G", "H"];
        return format!("Knight Promotion at column {}", col_map[index % 10]);
    }else {
        let row_map = ["8", "7", "6", "5", "4", "3", "2", "1"];
        let col_map = ["A", "B", "C", "D", "E", "F", "G", "H"];

        let col = (index % 10) as usize;
        let row = (index / 10) as usize;

        if col < col_map.len() && row < row_map.len() {
            return format!("{}{}", col_map[col], row_map[row]);
        } else {
            return "Invalid index".to_string();
        }
    }
}

fn print_mappings(b: Board) {
    println!("white indexes:");
    for i in b.white_piece_ids.iter() {
        if let Some(index) = b.white_indexes.get_index(*i) {
            print!("{}:{} ", i.to_string(), index_to_string(index));
        }
    }
    println!("\n");

    println!("black indexes:");
    for i in b.black_piece_ids.iter() {
        if let Some(index) = b.black_indexes.get_index(*i) {
            print!("{}:{} ", i.to_string(), index_to_string(index));
        }
    }
    println!("\n");

    let indexes: Vec<usize> = vec![
        0, 1, 2, 3, 4, 5, 6, 7,
        10, 11, 12, 13, 14, 15, 16, 17,
        20, 21, 22, 23, 24, 25, 26, 27,
        30, 31, 32, 33, 34, 35, 36, 37,
        40, 41, 42, 43, 44, 45, 46, 47,
        50, 51, 52, 53, 54, 55, 56, 57,
        60, 61, 62, 63, 64, 65, 66, 67,
        70, 71, 72, 73, 74, 75, 76, 77,
    ];

    println!("white i to p:");
    for i in indexes.iter() {
        if let Some(piece) = b.white_i_to_p.get_piece(*i) {
            print!("{}:{} ", index_to_string(*i), piece.to_string());
        }
    }
    println!("\n");

    println!("black i to p:");
    for i in indexes.iter() {
        if let Some(piece) = b.black_i_to_p.get_piece(*i) {
            print!("{}:{} ", index_to_string(*i), piece.to_string());
        }
    }
    println!("\n");
}


fn print_full_board(b: Vec<Vec<Piece>>) {
    let letters = vec!["A", "B", "C", "D", "E", "F", "G", "H"];
    let numbers = vec!["8", "7", "6", "5", "4", "3", "2", "1"];
    print!("  ");
    for letter in &letters {
        print!(" {} ", letter);
    }
    println!();
    for (i, row) in b.iter().enumerate() {
        print!("{} ", numbers[i]);
        for piece in row {
            print!("{}{} ", piece.team.to_string(), piece.kind.to_string());
        }
        println!(); 
    }
}

fn print_piece_ids(game:Board) {
    print!("white pieces: ");
    for piece_id in &game.white_piece_ids {
        print!("{} ", piece_id.to_string());
    }
    println!(); 

    print!("black pieces: ");
    for piece_id in &game.black_piece_ids {
        print!("{} ", piece_id.to_string());
    }
    println!(); 
}

// fn print_all_pre_checked_moves(b:Board){
//     println!("white:");
//     for piece_id in &b.white_piece_ids {
//         let index=b.white_indexes.get_index(*piece_id).unwrap();
//         let row=index/10;
//         let col=index%10;
//         let re=generate_available_moves(&b, row,col);
//         for j in re{
//             print!("{}:{} ", piece_id.to_string(), index_to_string(j));
//         }
//     }
//     println!("\n");
//     println!("black:");

//     for piece_id in &b.black_piece_ids {
//         let index=b.black_indexes.get_index(*piece_id).unwrap();
//         let row=index/10;
//         let col=index%10;
//         let re=generate_available_moves(&b, row,col);
//         for j in re{
//             print!("{}:{} ", piece_id.to_string(), index_to_string(j));
//         }
//     }

//     println!("\n");

// }

fn print_all_checked_moves(b:Board){
    let moves=all_moves_gen(&b);
    print!("{}{}", "checking:", moves.checking);
    println!("\n");
    if b.turn%2==0{
    print!("white moves: ");
    
    for i in b.white_piece_ids.iter(){
        print!("{} at {}: ", map_piece_id_to_kind(*i).to_string(), index_to_string(b.white_indexes.get_index(*i).unwrap()));
        let m=moves.moves.get_moves(*i);
        for j in m.iter(){
            print!("{} ", index_to_string(*j));
        }
        println!("\n");
    }
    println!("\n");
    }else{
    print!("black moves: ");
    for i in b.black_piece_ids.iter(){
        print!("{} at {}: ", map_piece_id_to_kind(*i).to_string(), index_to_string(b.black_indexes.get_index(*i).unwrap()));
        let m=moves.moves.get_moves(*i);
        for j in m.iter(){
            print!("{} ", index_to_string(*j));
        }
        println!("\n");
    }
}
}



pub fn print_all(game:Board){
    print!("turn:");
    println!("{}", game.turn);
    println!("ai advantage {}", game.ai_advantage);
    let game2=game.clone();
    println!("Black Prime {}", game.black_prime);
    println!("Black Prime1 {}", game.black_prime1);
    println!("White Prime {}", game.white_prime);
    println!("White Prime1 {}", game.white_prime1);
    let game3=game.clone();
    let game4=game.clone();
    let game5=game.clone();
    print_full_board(game.full_board);
    print_piece_ids(game2);
    print_mappings(game3);
    print_all_checked_moves(game5);
    //print_moves_log(&game.moves_log);
    base_error_check(game4);
}
fn print_moves_log(moves_log: &Vec<Move>) {
    for (index, mv) in moves_log.iter().enumerate() {
        println!("Move {}: {}", index + 1, mv.to_string());
    }
}
fn base_error_check(b:Board){
    let mut white_king_found=false;
    let mut black_king_found=false;
    let mut white_indexes=vec![];
    let mut black_indexes=vec![];
    for i in b.white_piece_ids.iter(){
        
        if *i==PieceId::K{
            white_king_found=true;
        }
        let index=b.white_indexes.get_index(*i).unwrap();
        white_indexes.push(index);
        if b.black_i_to_p.get_piece(index)!=None{
            println!("Error, black and white piece occupying the same spot");
            println!("at {}", index);
        }
        let row=index/10;
        let col=index%10; 
        if b.full_board[row][col].kind!=map_piece_id_to_kind(*i) && b.full_board[row][col].team!=Team::W{
            println!("Error, piece not found on fullboard at {}", index); 
        }
        if b.white_i_to_p.get_piece(index).unwrap_or(PieceId::Error)!=*i{
            println!("Error, piece not found at in white i to p {}", index);
        }
    }
    for i in b.black_piece_ids.iter(){
        if *i==PieceId::K{
            black_king_found=true;
        }
        let index=b.black_indexes.get_index(*i).unwrap();
        black_indexes.push(index);
        if b.white_i_to_p.get_piece(index)!=None{
            println!("Error, black and white piece occupying the same spot");
            println!("at {}", index);
        }
        let row=index/10;
        let col=index%10; 
        if b.full_board[row][col].kind!=map_piece_id_to_kind(*i) && b.full_board[row][col].team!=Team::B{
            println!("Error, piece not found on fullboard at {}", index); 
        }
        if b.black_i_to_p.get_piece(index).unwrap_or(PieceId::Error)!=*i{
            println!("Error, piece not found at in black i to p {}", index);
        }
    }
    if !black_king_found{
        println!("Error black king not found")
    }
    if !white_king_found{
        println!("Error white king not found")
    }
    let all_indexes: Vec<usize> = vec![
        0, 1, 2, 3, 4, 5, 6, 7,
        10, 11, 12, 13, 14, 15, 16, 17,
        20, 21, 22, 23, 24, 25, 26, 27,
        30, 31, 32, 33, 34, 35, 36, 37,
        40, 41, 42, 43, 44, 45, 46, 47,
        50, 51, 52, 53, 54, 55, 56, 57,
        60, 61, 62, 63, 64, 65, 66, 67,
        70, 71, 72, 73, 74, 75, 76, 77,
    ];
    let no_black_ind=find_non_overlap(all_indexes.clone(), black_indexes); 
    let no_white_ind=find_non_overlap(all_indexes, white_indexes);
    let p1=no_black_ind.clone();
    let p2=no_white_ind.clone();
    let no_ind=find_overlap(&p1,&p2);
    for i in no_black_ind.iter(){
        if b.black_i_to_p.get_piece(*i).unwrap_or(PieceId::Error)!=PieceId::Error{
            println!("Error, black i to p should be empty at {}", *i);
        }
    }
    for i in no_white_ind.iter(){
        if b.white_i_to_p.get_piece(*i).unwrap_or(PieceId::Error)!=PieceId::Error{
            println!("Error, white i to p should be empty at {}", *i);
        }
    }

    for i in no_ind{
        let row=i/10;
        let col=i%10;
        if b.full_board[row][col].team!=Team::N && (b.full_board[row][col].team!=Team::B ||b.full_board[row][col].team!=Team::W) && b.full_board[row][col].kind!=Kind::Empty{
            println!("Error, fullboard should be empty at {}", i);
        }
    }
    


}
pub fn easy_move(b:Board, piece:&str, location:&str)->Board{
    println!("About to move piece, location{}:{}", piece, location);
    let p_indexes=reverse_mapping(piece);
    let loc_indexes=reverse_mapping(location);
    if b.turn%2==0{
        let p_id=b.white_i_to_p.get_piece(p_indexes);
        return move_piece(b, p_id.unwrap(), loc_indexes);
    }
    else{
        let p_id=b.black_i_to_p.get_piece(p_indexes);
        return move_piece(b, p_id.unwrap(), loc_indexes);
    }
}

pub fn pause() {
    let mut input_string = String::new(); 
    println!("Press Enter to continue...");
    let _ = io::stdin().read_line(&mut input_string);
} 

pub fn test_b(){
    println!("New Game");
    play_std_format_game_full("1. e4 c5 2. Nf3 Nc6 3. Bb5 d6 4. O-O Bd7 5. Re1 Nf6 6. c3 a6 7. Ba4 c4 8. d4
cxd3 9. Bg5 e6 10. Qxd3 Be7 11. Bxf6 gxf6 12. Bxc6 Bxc6 13. c4 O-O 14. Nc3 Kh8
15. Rad1 Rg8 16. Qe3 Qf8 17. Nd4 Rc8 18. f4 Bd7 19. b3 Bd8 20. Nf3 b5 21. Qa7
Bc7 22. Qxa6 bxc4 23. b4 Qg7 24. g3 d5 25. exd5 Bxf4 26. Kf2 f5 27. gxf4 Qxc3
28. Qd6 Ba4 29. Rd4 Rg7 30. dxe6 Bc6 31. Ng5 Rxg5 32. Qe5+ Rg7 33. Rd8+ Rxd8 34.
Qxc3 f6 35. e7 Ra8 36. Qxf6 Be4 37. Rg1 Rxa2+ 38. Ke1");
    println!("New Game");
    // play_std_format_game_full("1. f4 e5  2. d4 f5  3. fxe5 f4  4. g3 f3  5. Nxf3 g5  6. Nxg5 Qxg5  7. e4 Qxc1  8. Qxc1 d5  9. exd6 Bxd6  10. Bc4 Bb4+  11. c3 Nc6  12. O-O Ke7  13. h4 Bf5  14. Rxf5 Nh6  15. Qxh6 Rhc8  16. Be6 Nxd4  17. cxb4 Nf3+  18. Rxf3 Rd8  19. b5 c5  20. bxc6 b5  21. c7 Rab8  22. cxb8=Q Rxb8  23. b4 Rg8  24. Qg7+ Kxe6  25. Qxg8+ Ke5  26. Qe8+ Kd6  27. Rd3+ Kc7  28. Qd8+ Kc6  29. Rc3+ Kb7  30. Qc7+ Ka8  31. Qc8");
    println!("New Game");
    play_std_format_game_full("1. f4 g5  2. Nf3 Bh6  3. Nxg5 Nf6  4. Nxf7 Kxf7  5. e4 Nxe4  6. Bd3 Ng3  7. Kf2 Bxf4  8. hxg3 Bxg3+  9. Kxg3 Rg8+  10. Kf2 Qf8  11. Ke3 e5  12. Bg6+ Rxg6  13. g4 Rg5  14. Rf1+ Kg7  15. d4 exd4+  16. Kxd4 Qxf1  17. Qf3 Qd1+  18. Bd2 Qxc2  19. Nc3 Qb1  20. Rxb1 Rc5  21. b4 Na6  22. Nd5 Rc4+  23. Kxc4 c5  24. Ne7 d5+  25. Kb5 h5  26. g5 h4  27. g6 h3  28. Qf7+ Kh8  29. Qxd5 h2  30. g7+ Kxg7  31. Nxc8 h1=Q  32. Qf5 Qh5  33. Qxh5 Kg8  34. bxc5 Nc7+  35. Kb4 a5+  36. Kc4 Rb8  37. c6 bxc6  38. Rf1 Rb4+  39. Kc5 Ne6+  40. Kxc6 Nc7  41. Qf7+ Kh8  42. Rh1+ Rh4  43. Rxh4");
    println!("New Game");
    play_std_format_game_full("1. e4 e5  2. Bc4 Qg5  3. Be6 fxe6  4. Qg4 Qxg4  5. f4 Qxg2  6. Ne2 Qxh1+  7. Kf2 Nf6  8. Nd4 Ng4+  9. Kg3 Qg1+  10. Kh4 Nxh2  11. f5 exf5  12. exf5 Qg6  13. fxg6 Bc5  14. c4 Bxd4  15. d3 e4  16. c5 exd3  17. Be3 d2  18. Bxd4 d1=Q  19. Kh3 Rf8  20. gxh7 g5  21. h8=Q Rxh8+  22. Kg3 Nf3  23. Kg4 Rh4+  24. Kf5 Nxd4+  25. Kf6 g4  26. Kg7 Rh7+  27. Kxh7 Nf5  28. Kg6 d5  29. c6 bxc6  30. b4 Qd4  31. b5 cxb5  32. Nc3 Qc5  33. Nxd5 Qc1  34. Rxc1 c5  35. Nc7+ Ke7  36. Nxa8 Be6  37. Kg5 Bg8  38. Kxf5 Be6+  39. Kf4 Bxa2  40. Re1+ Be6  41. Rxe6+ Kf8  42. Nc7 c4  43. Re8+ Kg7  44. Ne6+ Kh7  45. Rf8 g3  46. Kf5 Kh6  47. Kf6 Nc6  48. Rh8");
    println!("New Game");
    play_std_format_game_full("1. d4 e5  2. d5 e4  3. f4 exf3  4. d6 Qe7  5. g4 fxe2  6. Qxe2 Qxe2+  7. Kxe2 f5  8. gxf5 g5  9. f6 Ne7  10. f7+ Kd8  11. Nf3 g4  12. Ng5 Nd5  13. dxc7+ Ke7  14. Ne4 Ke6  15. Nc5+ Kxf7  16. Bg5 Kg6  17. Bf6 Kh6  18. h4 gxh3  19. Rxh3+ Kg6  20. Bxh8 Bh6  21. Rxh6+ Kxh6  22. c4 Nf4+  23. Kf3 Ng6  24. Bf6 Nf8  25. Be7 Ne6  26. Nxd7 Nf4  27. Bg5+ Kg6  28. Bxf4 Kf5  29. Bd6 Nc6  30. b4 Nxb4  31. c5 b5  32. cxb6 axb6  33. Nxb6 Be6  34. Nd2 Kg5  35. Bc5 Rf8+  36. Kg3 Rxf1  37. Kh2 Kg4  38. Nb3 Kf3  39. Nd4+ Kf2  40. Nxe6+ Kf3  41. Kh3 Rh1+");
    println!("New Game");
    play_std_format_game_full("1. f4 g5  2. f5 e5  3. fxe6 f5  4. e7 Bg7  5. exd8=N Kxd8  6. g4 fxg4  7. Nf3 gxf3  8. e4 f2+  9. Ke2 g4  10. Qe1 fxe1=N+"); 
}
pub fn old_test_b(){
    let game=init_board(true);
    let game2=game.clone(); 
    print_all(game);
    pause();
    let game3=move_piece(game2, PieceId::P5, reverse_mapping("E4"));
    let game4=game3.clone();
    print_all(game3);
    pause();
    let game5=easy_move(game4.clone(), "E7", "E5");
    let game6=game5.clone();
    print_all(game5);
    pause();
    let game7=easy_move(game6.clone(), "F1", "B5");
    let game8=game7.clone();
    print_all(game7);
    pause();
    let game9=easy_move(game8.clone(), "F8", "B4"); 
    let game10=game9.clone();
    print_all(game9);
    pause();
    let game11=easy_move(game10.clone(), "G1", "F3");
    let game12=game11.clone();
    print_all(game11);
    pause();
    let game13=easy_move(game12.clone(), "C7", "C6");
    let game14=game13.clone();
    print_all(game13);
    pause();
    let game15=move_piece(game14.clone(), PieceId::K, 99);
    let game16=game15.clone();
    print_all(game15);
    pause();
    let game17=easy_move(game16.clone(), "F7", "F5");
    let game18=game17.clone();
    print_all(game17);
    pause();
    let game19=easy_move(game18.clone(), "E4", "F5"); 
    let game20=game19.clone();
    print_all(game19);
    pause();
    let game21=easy_move(game20.clone(), "G7", "G5");
    let game22 = game21.clone();
    print_all(game21);
    pause();
    let game23 = easy_move(game22.clone(), "F5", "G6");
    let game24 = game23.clone();
    print_all(game23);
    pause();
    let game25 = easy_move(game24.clone(), "H7", "G6");
    let game26 = game25.clone();
    print_all(game25);
    pause();
    let game27 = easy_move(game26.clone(), "F3", "E5");
    let game28 = game27.clone();
    print_all(game27);
    pause();
    let game29 = easy_move(game28.clone(), "H8", "H2");
    let game30 = game29.clone();
    print_all(game29);
    pause();
    let game31 = easy_move(game30.clone(), "D1", "F3");
    let game32 = game31.clone();
    print_all(game31);
    pause();
    let game33 = easy_move(game32.clone(), "H2", "H1");
    let game34 = game33.clone();
    print_all(game33);
    pause();
    let game35 = easy_move(game34.clone(), "G1", "H1");
    let game36 = game35.clone();
    print_all(game35);
    pause();
    let game37 = easy_move(game36.clone(), "C6", "B5");
    let game38 = game37.clone();
    print_all(game37);
    pause();
    let game39 = easy_move(game38.clone(), "F3", "F8");
    let game40 = game39.clone();
    print_all(game39);
    pause();
    let game41 = easy_move(game40.clone(), "E8", "F8");
    let game42 = game41.clone();
    print_all(game41);
    pause();
    let game43 = easy_move(game42.clone(), "E5", "G6");
    let game44 = game43.clone();
    print_all(game43);
    pause();
    let game45 = easy_move(game44.clone(), "F8", "E8");
    let game46 = game45.clone();
    print_all(game45);
    pause();
    let game47 = easy_move(game46.clone(), "A2", "A4");
    let game48 = game47.clone();
    print_all(game47);
    pause();
    let game49 = easy_move(game48.clone(), "D7", "D5");
    let game50 = game49.clone();
    print_all(game49);
    pause();
    let game51 = easy_move(game50.clone(), "A4", "B5");
    let game52 = game51.clone();
    print_all(game51);
    pause();
    let game53 = easy_move(game52.clone(), "D5", "D4");
    let game54 = game53.clone();
    print_all(game53);
    pause();
    let game55 = easy_move(game54.clone(), "C2", "C4");
    let game56 = game55.clone();
    print_all(game55);
    pause();
    let game57 = easy_move(game56.clone(), "D4", "C3");
    let game58 = game57.clone();
    print_all(game57);
    pause();
    let game59 = easy_move(game58.clone(), "B5", "B6");
    let game60 = game59.clone();
    print_all(game59);
    pause();
    let game61 = easy_move(game60.clone(), "C3", "B2");
    let game62 = game61.clone();
    print_all(game61);
    pause();
    let game63 = easy_move(game62.clone(), "B6", "A7");
    let game64 = game63.clone();
    print_all(game63);
    pause();
    let game65 = easy_move(game64.clone(), "B2", "C1");
    let game66 = game65.clone();
    print_all(game65);
    pause();
    let game67 = easy_move(game66.clone(), "A7", "B8");
    let game68 = game67.clone();
    print_all(game67);
    pause();
    let game69 = easy_move(game68.clone(), "C8", "E6");
    let game70 = game69.clone();
    print_all(game69);
    pause();
    let game71 = easy_move(game70.clone(), "F1", "E1");
    let game72 = game71.clone();
    print_all(game71);
    pause();
    let game73 = easy_move(game72.clone(), "C1", "C2");
    let game74 = game73.clone();
    print_all(game73);
    pause();
    let game75 = easy_move(game74.clone(), "E1", "E6");
    let game76 = game75.clone();
    print_all(game75);
    pause();
    let game77 = easy_move(game76.clone(), "G8", "E7");
    let game78 = game77.clone();
    print_all(game77);
    pause();
    let game79 = easy_move(game78.clone(), "B4", "E7");
    let game80 = game79.clone();
    print_all(game79);
    pause();
    let game81 = easy_move(game80.clone(), "B8", "A8");
    let game82 = game81.clone();
    print_all(game81);
    pause();
    let game83 = easy_move(game82.clone(), "E7", "D6");
    let game84 = game83.clone();
    print_all(game83);
    pause();
    let game85 = easy_move(game84.clone(), "G6", "H8");
    let game86 = game85.clone();
    print_all(game85);
    pause();
    let game87 = easy_move(game86.clone(), "C2", "C1");
    print_all(game87);
}

fn map_my_data_model_to_fen(b: &Vec<Vec<Piece>>) -> String {
    let mut fen = String::new(); // Initialize as an empty String
    let mut count = 0;

    for i in b.iter() {
        if count != 0 {
            fen.push('/'); // Append '/' directly
        }
        count += 1;

        let mut empty_count = 0;
        for j in i.iter() {
            if j.kind == Kind::Empty {
                empty_count += 1;
            } else {
                if empty_count != 0 {
                    fen.push_str(&empty_count.to_string()); // Append the number of empty squares
                    empty_count = 0;
                }
                match j.team {
                    Team::W => {
                        let piece_char = match j.kind {
                            Kind::Pawn => "P",
                            Kind::Queen => "Q",
                            Kind::Rook => "R",
                            Kind::Knight => "N",
                            Kind::Bishop => "B",
                            _ => "K", // King
                        };
                        fen.push_str(piece_char); // Append the piece character
                    }
                    Team::B => {
                        let piece_char = match j.kind {
                            Kind::Pawn => "p",
                            Kind::Queen => "q",
                            Kind::Rook => "r",
                            Kind::Knight => "n",
                            Kind::Bishop => "b",
                            _ => "k", // King
                        };
                        fen.push_str(piece_char); // Append the piece character
                    }
                    _=>{}
                }
            }
        }

        // If there are empty squares at the end of the row, add them to the FEN string
        if empty_count != 0 {
            fen.push_str(&empty_count.to_string());
        }
    }

    fen // Return the final FEN string
}

fn get_std_format(index:usize)->String{
    if index==99{
        return "0-0".to_string();
    }
    if index==100{
        return "0-0-0".to_string();
    }
    let row_map = ["8", "7", "6", "5", "4", "3", "2", "1"];
    let col_map = ["a", "b", "c", "d", "e", "f", "g", "h"];
    return format!("{}{}", col_map[index%10], row_map[index/10]);
}

// pub fn write_to_moves_validator(file_path: &str) {
//     let white = NeuralNetworkSelector::Model5;
//     let black = NeuralNetworkSelector::Model3;

//     // Open the file in append mode
//     let mut file = OpenOptions::new()
//         .create(true)
//         .append(true)
//         .open(file_path)
//         .expect("Unable to open file");

//     for i in 0..=100 {
//         let mut game = init_board(true);
//         let mut done = false;
//         while !done {
//             if game.turn % 2 == 0 {
//                 game = get_next_move(game, &white);
//             } else {
//                 game = get_next_move(game, &black);
//             }

//             let fen: &str = &map_my_data_model_to_fen(&game.full_board);
//             let moves = all_moves_gen(&game).moves;
//             let mut standard_format_moves: Vec<String> = vec![];

//             if game.turn % 2 == 0 {
//                 for j in game.white_piece_ids.iter() {
//                     for k in moves.get_moves(*j).iter() {
//                         if k/10==8{
//                             continue;
//                         }
//                         let std_f_move = format!(
//                             "{}{}",
//                             &get_std_format(game.white_indexes.get_index(*j).unwrap()),
//                             &get_std_format(*k)
//                         );
//                         standard_format_moves.push(std_f_move);
//                     }
//                 }
//             } else {
//                 for j in game.black_piece_ids.iter() {
//                     for k in moves.get_moves(*j).iter() {
//                         if k/10==8{
//                             continue;
//                         }//don't need the knight promotion here
//                         let std_f_move = format!(
//                             "{}{}",
//                             &get_std_format(game.black_indexes.get_index(*j).unwrap()),
//                             &get_std_format(*k)
//                         );
//                         standard_format_moves.push(std_f_move);
//                     }
//                 }
//             }

//             // Write FEN and moves to file
//             let turn:&str;
//             if game.turn%2==0{
//                 turn="white"
//             }else{
//                 turn="black"
//             }
//             if let Err(e) = writeln!(
//                 file,
//                 "Turn: {} FEN: {}\nMoves: {:?}\n",
//                 turn,
//                 fen,
//                 standard_format_moves.join(", ")
//             ) {
//                 eprintln!("Could not write to file: {}", e);
//             }

//             if game_still_going(&game, all_moves_gen(&game).checking, &moves) != 0.1 || game.turn > 150 {
//                 done = true;
//             }
//         }
//     }
// }

// pub fn test_search(){
//     let b=init_board(true);
//     let searching_treenode=TreeNode{game:b.clone(), level:0, children:vec![]};
//     let biggest_mm=100.0;
//     let new_mm=search(searching_treenode, 5, 5, biggest_mm, -1.0);//will tweak depth and width params
// }