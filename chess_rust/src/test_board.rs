use crate::base_functions::{init_board};
use crate::types::{Board, Piece, PieceId};
use crate::upper_move_functions::{all_moves_gen, move_piece};
// use crate::base_move_functions::{generate_available_moves};
fn reverse_mapping(b: &str) -> usize {
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
fn index_to_string(index: usize) -> String {
    if index == 99 {
        return "KS".to_string();
    } else if index == 100 {
        return "QS".to_string();
    } else {
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
    print!("white: ");
    for i in b.white_piece_ids.iter(){
        print!("{}: ", i.to_string());
        let m=moves.white_moves.get_moves(*i);
        for j in m.iter(){
            print!("{} ", index_to_string(*j));
        }
        println!("\n");
    }
    println!("\n");
    print!("black: ");
    for i in b.black_piece_ids.iter(){
        print!("{}: ", i.to_string());
        let m=moves.black_moves.get_moves(*i);
        for j in m.iter(){
            print!("{} ", index_to_string(*j));
        }
        println!("\n");
    }
}

pub fn print_all(game:Board){
    let game2=game.clone();
    let game3=game.clone();
    // let game4=game.clone();
    let game5=game.clone();
    print_full_board(game.full_board);
    print_piece_ids(game2);
    print_mappings(game3);
    // print_all_pre_checked_moves(game4);
    print_all_checked_moves(game5);
}



pub fn test_b(){
    let game=init_board(true);
    let game2=game.clone(); 
    print_all(game);
    let game3=move_piece(game2, PieceId::P5, reverse_mapping("E4"));
    print_all(game3);
}
