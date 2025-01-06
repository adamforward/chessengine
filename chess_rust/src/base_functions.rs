use crate::types::{ Board, IToPMap, IndexMap, Kind, Piece, PieceId, Team, GameStateIdentifiers};
use std::collections::HashSet;
use std::hash::Hash;

pub fn map_piece_id_to_kind(piece: PieceId) -> Kind {
    //Kind is identifier for type of piece, PieceId is for individual pieces
    //Every pawn has same kind, but different ID
    match piece {
        PieceId::P1
        | PieceId::P2
        | PieceId::P3
        | PieceId::P4
        | PieceId::P5
        | PieceId::P6
        | PieceId::P7
        | PieceId::P8 => Kind::Pawn,

        PieceId::K1
        | PieceId::K2
        | PieceId::K3
        | PieceId::K4
        | PieceId::K5
        | PieceId::K6
        | PieceId::K7
        | PieceId::K8
        | PieceId::K9
        | PieceId::K10 => Kind::Knight,

        PieceId::B1 | PieceId::B2 => Kind::Bishop,

        PieceId::K => Kind::King,

        PieceId::Q
        | PieceId::Q1
        | PieceId::Q2
        | PieceId::Q3
        | PieceId::Q4
        | PieceId::Q5
        | PieceId::Q6
        | PieceId::Q7
        | PieceId::Q8 => Kind::Queen,

        PieceId::R1 | PieceId::R2 => Kind::Rook,

        _ => Kind::Empty,
    }
}

pub fn init_board(ai_team: bool) -> Board {
    let game_states=GameStateIdentifiers::new();
    //this is for generating a new game, so a board where everything is in starting position
    //I describe all of the different fields in types.rust
    let moves_log = Vec::new();
    let mut full_board = Vec::new();
    let w_piece_ids = vec![
        PieceId::P1,
        PieceId::P2,
        PieceId::P3,
        PieceId::P4,
        PieceId::P5,
        PieceId::P6,
        PieceId::P7,
        PieceId::P8,
        PieceId::K1,
        PieceId::K2,
        PieceId::B1,
        PieceId::B2,
        PieceId::R1,
        PieceId::R2,
        PieceId::K,
        PieceId::Q,
    ];
    let b_piece_ids = vec![
        PieceId::P1,
        PieceId::P2,
        PieceId::P3,
        PieceId::P4,
        PieceId::P5,
        PieceId::P6,
        PieceId::P7,
        PieceId::P8,
        PieceId::K1,
        PieceId::K2,
        PieceId::B1,
        PieceId::B2,
        PieceId::R1,
        PieceId::R2,
        PieceId::K,
        PieceId::Q,
    ];
    let row1 = vec![
        Piece {
            team: Team::B,
            kind: Kind::Rook,
        },
        Piece {
            team: Team::B,
            kind: Kind::Knight,
        },
        Piece {
            team: Team::B,
            kind: Kind::Bishop,
        },
        Piece {
            team: Team::B,
            kind: Kind::Queen,
        },
        Piece {
            team: Team::B,
            kind: Kind::King,
        },
        Piece {
            team: Team::B,
            kind: Kind::Bishop,
        },
        Piece {
            team: Team::B,
            kind: Kind::Knight,
        },
        Piece {
            team: Team::B,
            kind: Kind::Rook,
        },
    ];

    let row2 = vec![
        Piece {
            team: Team::B,
            kind: Kind::Pawn,
        },
        Piece {
            team: Team::B,
            kind: Kind::Pawn,
        },
        Piece {
            team: Team::B,
            kind: Kind::Pawn,
        },
        Piece {
            team: Team::B,
            kind: Kind::Pawn,
        },
        Piece {
            team: Team::B,
            kind: Kind::Pawn,
        },
        Piece {
            team: Team::B,
            kind: Kind::Pawn,
        },
        Piece {
            team: Team::B,
            kind: Kind::Pawn,
        },
        Piece {
            team: Team::B,
            kind: Kind::Pawn,
        },
    ];

    let row3 = vec![
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
    ];

    let row4 = vec![
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
    ];

    let row5 = vec![
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
    ];
    let row6 = vec![
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
        },
    ];
    let row7 = vec![
        Piece {
            team: Team::W,
            kind: Kind::Pawn,
        },
        Piece {
            team: Team::W,
            kind: Kind::Pawn,
        },
        Piece {
            team: Team::W,
            kind: Kind::Pawn,
        },
        Piece {
            team: Team::W,
            kind: Kind::Pawn,
        },
        Piece {
            team: Team::W,
            kind: Kind::Pawn,
        },
        Piece {
            team: Team::W,
            kind: Kind::Pawn,
        },
        Piece {
            team: Team::W,
            kind: Kind::Pawn,
        },
        Piece {
            team: Team::W,
            kind: Kind::Pawn,
        },
    ];
    let row8 = vec![
        Piece {
            team: Team::W,
            kind: Kind::Rook,
        },
        Piece {
            team: Team::W,
            kind: Kind::Knight,
        },
        Piece {
            team: Team::W,
            kind: Kind::Bishop,
        },
        Piece {
            team: Team::W,
            kind: Kind::Queen,
        },
        Piece {
            team: Team::W,
            kind: Kind::King,
        },
        Piece {
            team: Team::W,
            kind: Kind::Bishop,
        },
        Piece {
            team: Team::W,
            kind: Kind::Knight,
        },
        Piece {
            team: Team::W,
            kind: Kind::Rook,
        },
    ];
    full_board.push(row1);
    full_board.push(row2);
    full_board.push(row3);
    full_board.push(row4);
    full_board.push(row5);
    full_board.push(row6);
    full_board.push(row7);
    full_board.push(row8);

    let mut black_indexes = IndexMap::new();
    black_indexes.change_indexes(PieceId::R1, 0);
    black_indexes.change_indexes(PieceId::R2, 7);
    black_indexes.change_indexes(PieceId::B1, 2);
    black_indexes.change_indexes(PieceId::B2, 5);
    black_indexes.change_indexes(PieceId::K1, 1);
    black_indexes.change_indexes(PieceId::K2, 6);
    black_indexes.change_indexes(PieceId::K, 4);
    black_indexes.change_indexes(PieceId::Q, 3);
    black_indexes.change_indexes(PieceId::P1, 10);
    black_indexes.change_indexes(PieceId::P2, 11);
    black_indexes.change_indexes(PieceId::P3, 12);
    black_indexes.change_indexes(PieceId::P4, 13);
    black_indexes.change_indexes(PieceId::P5, 14);
    black_indexes.change_indexes(PieceId::P6, 15);
    black_indexes.change_indexes(PieceId::P7, 16);
    black_indexes.change_indexes(PieceId::P8, 17);

    let mut black_i_to_p = IToPMap::new();
    black_i_to_p.insert_piece(0, PieceId::R1);
    black_i_to_p.insert_piece(7, PieceId::R2);
    black_i_to_p.insert_piece(2, PieceId::B1);
    black_i_to_p.insert_piece(5, PieceId::B2);
    black_i_to_p.insert_piece(1, PieceId::K1);
    black_i_to_p.insert_piece(6, PieceId::K2);
    black_i_to_p.insert_piece(4, PieceId::K);
    black_i_to_p.insert_piece(3, PieceId::Q);
    black_i_to_p.insert_piece(10, PieceId::P1);
    black_i_to_p.insert_piece(11, PieceId::P2);
    black_i_to_p.insert_piece(12, PieceId::P3);
    black_i_to_p.insert_piece(13, PieceId::P4);
    black_i_to_p.insert_piece(14, PieceId::P5);
    black_i_to_p.insert_piece(15, PieceId::P6);
    black_i_to_p.insert_piece(16, PieceId::P7);
    black_i_to_p.insert_piece(17, PieceId::P8);

    let mut white_indexes = IndexMap::new();
    white_indexes.change_indexes(PieceId::R1, 70);
    white_indexes.change_indexes(PieceId::R2, 77);
    white_indexes.change_indexes(PieceId::B1, 72);
    white_indexes.change_indexes(PieceId::B2, 75);
    white_indexes.change_indexes(PieceId::K1, 71);
    white_indexes.change_indexes(PieceId::K2, 76);
    white_indexes.change_indexes(PieceId::K, 74);
    white_indexes.change_indexes(PieceId::Q, 73);
    white_indexes.change_indexes(PieceId::P1, 60);
    white_indexes.change_indexes(PieceId::P2, 61);
    white_indexes.change_indexes(PieceId::P3, 62);
    white_indexes.change_indexes(PieceId::P4, 63);
    white_indexes.change_indexes(PieceId::P5, 64);
    white_indexes.change_indexes(PieceId::P6, 65);
    white_indexes.change_indexes(PieceId::P7, 66);
    white_indexes.change_indexes(PieceId::P8, 67);

    let mut white_i_to_p = IToPMap::new();
    white_i_to_p.insert_piece(70, PieceId::R1);
    white_i_to_p.insert_piece(77, PieceId::R2);
    white_i_to_p.insert_piece(72, PieceId::B1);
    white_i_to_p.insert_piece(75, PieceId::B2);
    white_i_to_p.insert_piece(71, PieceId::K1);
    white_i_to_p.insert_piece(76, PieceId::K2);
    white_i_to_p.insert_piece(74, PieceId::K);
    white_i_to_p.insert_piece(73, PieceId::Q);
    white_i_to_p.insert_piece(60, PieceId::P1);
    white_i_to_p.insert_piece(61, PieceId::P2);
    white_i_to_p.insert_piece(62, PieceId::P3);
    white_i_to_p.insert_piece(63, PieceId::P4);
    white_i_to_p.insert_piece(64, PieceId::P5);
    white_i_to_p.insert_piece(65, PieceId::P6);
    white_i_to_p.insert_piece(66, PieceId::P7);
    white_i_to_p.insert_piece(67, PieceId::P8);

    return Board {
        moves_log,
        ai_team_is_white: ai_team,
        full_board,
        white_i_to_p,
        black_i_to_p,
        white_indexes,
        black_indexes,
        white_points: 3800,
        black_points: 3800,
        white_piece_ids: w_piece_ids,
        black_piece_ids: b_piece_ids,
        white_prime: 1,
        black_prime: 1,
        white_prime1: 1,
        black_prime1: 1,
        prime2: 1,
        ai_advantage: game_states.new_game,
        turn: 0,
    };
}

//each prime returned here is an identifier for did the pawn of whatever team skip ahead last turn 
//I use prime numbers to efficiently store information on what has or has not happened in the chess game. 
pub fn primes(col: usize) -> i32 {
    let primes = vec![2, 3, 5, 7, 11, 13, 17, 19];
    return primes[col]; // Dereference `col` here
} 

pub fn primes1(p: PieceId) -> i32 { 
    //while primes stores whether or not a pawn jumped ahead last turn, 
    //primes1 is for if the pawn jumped 2 squares at all, so the column doesn't matter.
    //this is because if a pawn wants to en pessant capture another piece, it can only do this if 
    //the pawn jumped ahead at the start. 
    match p {
        PieceId::P1 => 2,
        PieceId::P2 => 3,
        PieceId::P3 => 5,
        PieceId::P4 => 7,
        PieceId::P5 => 11,
        PieceId::P6 => 13,
        PieceId::P7 => 17,
        PieceId::P8 => 19,
        _ => 1,
    }
}

pub fn contains_element<T: PartialEq>(v: &Vec<T>, element: T) -> bool {
    for i in v.iter() {
        if *i == element {
            return true;
        }
    }
    false
}

//I only support pawn to queen and pawn to knight promotions. 
pub fn pawn_to_queen(p: PieceId) -> PieceId {
    match p {
        PieceId::P1 => PieceId::Q1,
        PieceId::P2 => PieceId::Q2,
        PieceId::P3 => PieceId::Q3,
        PieceId::P4 => PieceId::Q4,
        PieceId::P5 => PieceId::Q5,
        PieceId::P6 => PieceId::Q6,
        PieceId::P7 => PieceId::Q7,
        _ => PieceId::Q8,
    }
}
pub fn pawn_to_knight(p: PieceId) -> PieceId {
    match p {
        PieceId::P1 => PieceId::K3,
        PieceId::P2 => PieceId::K4,
        PieceId::P3 => PieceId::K5,
        PieceId::P4 => PieceId::K6,
        PieceId::P5 => PieceId::K7,
        PieceId::P6 => PieceId::K8,
        PieceId::P7 => PieceId::K9,
        _ => PieceId::K10,
    }
}

pub fn find_overlap<T: PartialEq + Eq + Hash + Clone>(v1: &[T], v2: &[T]) -> Vec<T> {
    let set1: HashSet<&T> = v1.iter().collect();
    let set2: HashSet<&T> = v2.iter().collect();
    set1.intersection(&set2).cloned().cloned().collect()
}

pub fn find_non_overlap<T: PartialEq + Eq + Hash + Clone>(v1: Vec<T>, v2: Vec<T>) -> Vec<T> {
    let set1: HashSet<T> = v1.into_iter().collect();
    let set2: HashSet<T> = v2.into_iter().collect();
    
    set1.symmetric_difference(&set2).cloned().collect()
}


fn safe_remove<T: PartialEq>(vec: &mut Vec<T>, element: T) {
    if let Some(pos) = vec.iter().position(|x| *x == element) {
        vec.remove(pos);
    }
}

pub fn ai_turn(b:&Board)->bool{
    return b.ai_team_is_white == (b.turn%2==0);
}