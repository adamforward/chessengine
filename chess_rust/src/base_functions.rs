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

// pub fn get_sygyzy_enum(b:&Board>)->Sygyzy{
//     let sum=b.white_piece_ids.len()+b.black_piece_ids.len();
//     if sum>5{
//         return Sygyzy::Na;
//     }
//     else{
//     let mut white_pawns = 0;
//     let mut black_pawns = 0;
//     let mut white_knights = 0;
//     let mut black_knights = 0;
//     let mut white_rooks = 0;
//     let mut black_rooks = 0;
//     let mut white_bishops = 0;
//     let mut black_bishops = 0;
//     let mut white_queens = 0;
//     let mut black_queens = 0;

//     for i in white_piece_ids.iter() {
//         match map_piece_id_to_kind(*i) {
//             Kind::Pawn => white_pawns += 1,
//             Kind::Rook => white_rooks += 1,
//             Kind::Knight => white_knights += 1,
//             Kind::Bishop => white_bishops += 1,
//             Kind::Queen => white_queens += 1,
//             Kind::King | Kind::Empty => {}, // No need to count kings or empty spaces
//         }
//     }

//     for i in black_piece_ids.iter() {
//         match map_piece_id_to_kind(*i) {
//             Kind::Pawn => black_pawns += 1,
//             Kind::Rook => black_rooks += 1,
//             Kind::Knight => black_knights += 1,
//             Kind::Bishop => black_bishops += 1,
//             Kind::Queen => black_queens += 1,
//             Kind::King | Kind::Empty => {}, // No need to count kings or empty spaces
//         }
//     }
//     match (
//             white_pawns, black_pawns,
//             white_knights, black_knights,
//             white_rooks, black_rooks,
//             white_bishops, black_bishops,
//             white_queens, black_queens,
//         ) {
//             (0, 0, 1, 0, 0, 0, 3, 0, 0, 0) => Syzygy::KBBBvK,
//             (0, 0, 1, 0, 0, 0, 2, 0, 0, 0) => Syzygy::KBBNvK,
//             (1, 0, 1, 0, 0, 0, 2, 0, 0, 0) => Syzygy::KBBPvK,
//             (0, 0, 0, 0, 0, 0, 2, 0, 0, 0) => Syzygy::KBBvK,
//             (0, 0, 0, 0, 0, 0, 2, 1, 0, 0) => Syzygy::KBBvKB,
//             (0, 0, 0, 1, 0, 0, 2, 0, 0, 0) => Syzygy::KBBvKN,
//             (0, 0, 1, 0, 0, 0, 2, 0, 0, 0) => Syzygy::KBBvKP,
//             (0, 0, 0, 0, 0, 1, 2, 0, 0, 0) => Syzygy::KBBvKQ,
//             (0, 0, 0, 0, 1, 0, 2, 0, 0, 0) => Syzygy::KBBvKR,
//             (0, 0, 2, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KBNNvK,
//             (1, 0, 1, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KBNPvK,
//             (0, 0, 1, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KBNvK,
//             (0, 0, 1, 0, 0, 0, 1, 1, 0, 0) => Syzygy::KBNvKB,
//             (0, 0, 1, 1, 0, 0, 1, 0, 0, 0) => Syzygy::KBNvKN,
//             (1, 0, 1, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KBNvKP,
//             (0, 0, 1, 0, 0, 1, 1, 0, 0, 0) => Syzygy::KBNvKQ,
//             (0, 0, 1, 0, 1, 0, 1, 0, 0, 0) => Syzygy::KBNvKR,
//             (2, 0, 0, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KBPPvK,
//             (1, 0, 0, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KBPvK,
//             (1, 0, 0, 0, 0, 0, 1, 1, 0, 0) => Syzygy::KBPvKB,
//             (1, 0, 1, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KBPvKN,
//             (2, 0, 0, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KBPvKP,
//             (1, 0, 0, 0, 0, 1, 1, 0, 0, 0) => Syzygy::KBPvKQ,
//             (1, 0, 0, 0, 1, 0, 1, 0, 0, 0) => Syzygy::KBPvKR,
//             (0, 0, 0, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KBvK,
//             (0, 0, 0, 0, 0, 0, 1, 1, 0, 0) => Syzygy::KBvKB,
//             (0, 0, 1, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KBvKN,
//             (1, 0, 0, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KBvKP,
//             (0, 0, 3, 0, 0, 0, 0, 0, 0, 0) => Syzygy::KNNNvK,
//             (1, 0, 2, 0, 0, 0, 0, 0, 0, 0) => Syzygy::KNNPvK,
//             (0, 0, 2, 0, 0, 0, 0, 0, 0, 0) => Syzygy::KNNvK,
//             (0, 0, 2, 0, 0, 0, 0, 1, 0, 0) => Syzygy::KNNvKB,
//             (0, 0, 2, 1, 0, 0, 0, 0, 0, 0) => Syzygy::KNNvKN,
//             (1, 0, 2, 0, 0, 0, 0, 0, 0, 0) => Syzygy::KNNvKP,
//             (0, 0, 2, 0, 0, 1, 0, 0, 0, 0) => Syzygy::KNNvKQ,
//             (0, 0, 2, 0, 1, 0, 0, 0, 0, 0) => Syzygy::KNNvKR,
//             (2, 0, 1, 0, 0, 0, 0, 0, 0, 0) => Syzygy::KNPPvK,
//             (1, 0, 1, 0, 0, 0, 0, 0, 0, 0) => Syzygy::KNPvK,
//             (1, 0, 1, 0, 0, 0, 0, 1, 0, 0) => Syzygy::KNPvKB,
//             (1, 0, 2, 0, 0, 0, 0, 0, 0, 0) => Syzygy::KNPvKN,
//             (2, 0, 1, 0, 0, 0, 0, 0, 0, 0) => Syzygy::KNPvKP,
//             (1, 0, 1, 0, 0, 1, 0, 0, 0, 0) => Syzygy::KNPvKQ,
//             (1, 0, 1, 0, 1, 0, 0, 0, 0, 0) => Syzygy::KNPvKR,
//             (0, 0, 1, 0, 0, 0, 0, 0, 0, 0) => Syzygy::KNvK,
//             (0, 0, 1, 0, 0, 0, 0, 1, 0, 0) => Syzygy::KNvKB,
//             (0, 0, 1, 1, 0, 0, 0, 0, 0, 0) => Syzygy::KNvKN,
//             (1, 0, 1, 0, 0, 0, 0, 0, 0, 0) => Syzygy::KNvKP,
//             (1, 0, 0, 0, 0, 0, 0, 0, 0, 0) => Syzygy::KPvK,
//             (1, 0, 0, 0, 0, 0, 0, 1, 0, 0) => Syzygy::KPvKB,
//             (1, 0, 1, 0, 0, 0, 0, 0, 0, 0) => Syzygy::KPvKN,
//             (2, 0, 0, 0, 0, 0, 0, 0, 0, 0) => Syzygy::KPvKP,
//             (0, 0, 0, 0, 0, 0, 2, 0, 0, 0) => Syzygy::KQBBvK,
//             (0, 0, 1, 0, 0, 0, 2, 0, 0, 0) => Syzygy::KQBNvK,
//             (1, 0, 1, 0, 0, 0, 2, 0, 0, 0) => Syzygy::KQBPvK,
//             (0, 0, 0, 0, 0, 0, 2, 0, 0, 0) => Syzygy::KQBvK,
//             (0, 0, 0, 0, 0, 1, 2, 1, 0, 0) => Syzygy::KQBvKB,
//             (0, 0, 0, 1, 0, 0, 2, 0, 0, 0) => Syzygy::KQBvKN,
//             (0, 0, 1, 0, 0, 0, 2, 0, 0, 0) => Syzygy::KQBvKP,
//             (0, 0, 0, 0, 0, 1, 2, 0, 0, 0) => Syzygy::KQBvKQ,
//             (0, 0, 0, 0, 1, 0, 2, 0, 0, 0) => Syzygy::KQBvKR,
//             (0, 0, 2, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KQNNvK,
//             (1, 0, 1, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KQNPvK,
//             (0, 0, 1, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KQNvK,
//             (0, 0, 1, 0, 0, 0, 1, 1, 0, 0) => Syzygy::KQNvKB,
//             (0, 0, 1, 1, 0, 0, 1, 0, 0, 0) => Syzygy::KQNvKN,
//             (1, 0, 1, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KQNvKP,
//             (0, 0, 1, 0, 0, 1, 1, 0, 0, 0) => Syzygy::KQNvKQ,
//             (0, 0, 1, 0, 1, 0, 1, 0, 0, 0) => Syzygy::KQNvKR,
//             (2, 0, 0, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KQPPvK,
//             (1, 0, 0, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KQPvK,
//             (1, 0, 0, 0, 0, 0, 1, 1, 0, 0) => Syzygy::KQPvKB,
//             (1, 0, 1, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KQPvKN,
//             (2, 0, 0, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KQPvKP,
//             (1, 0, 0, 0, 0, 1, 1, 0, 0, 0) => Syzygy::KQPvKQ,
//             (1, 0, 0, 0, 1, 0, 1, 0, 0, 0) => Syzygy::KQPvKR,
//             (0, 0, 0, 0, 0, 0, 2, 0, 0, 0) => Syzygy::KQQvK,
//             (0, 0, 0, 0, 0, 1, 2, 1, 0, 0) => Syzygy::KQQvKB,
//             (0, 0, 0, 1, 0, 0, 2, 0, 0, 0) => Syzygy::KQQvKN,
//             (0, 0, 1, 0, 0, 0, 2, 0, 0, 0) => Syzygy::KQQvKP,
//             (0, 0, 0, 0, 0, 1, 2, 0, 0, 0) => Syzygy::KQQvKQ,
//             (0, 0, 0, 0, 1, 0, 2, 0, 0, 0) => Syzygy::KQQvKR,
//             (0, 0, 1, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KQRBvK,
//             (1, 0, 1, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KQRNvK,
//             (1, 0, 1, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KQRPvK,
//             (0, 0, 0, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KQRvK,
//             (0, 0, 0, 0, 0, 1, 1, 1, 0, 0) => Syzygy::KQRvKB,
//             (0, 0, 0, 1, 0, 0, 1, 0, 0, 0) => Syzygy::KQRvKN,
//             (0, 0, 1, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KQRvKP,
//             (0, 0, 0, 0, 0, 1, 1, 0, 0, 0) => Syzygy::KQRvKQ,
//             (0, 0, 0, 0, 1, 0, 1, 0, 0, 0) => Syzygy::KQRvKR,
//             (0, 0, 0, 0, 0, 0, 2, 0, 0, 0) => Syzygy::KRBBvK,
//             (0, 0, 1, 0, 0, 0, 2, 0, 0, 0) => Syzygy::KRBNvK,
//             (1, 0, 1, 0, 0, 0, 2, 0, 0, 0) => Syzygy::KRBPvK,
//             (0, 0, 0, 0, 0, 0, 2, 0, 0, 0) => Syzygy::KRBvK,
//             (0, 0, 0, 0, 0, 1, 2, 1, 0, 0) => Syzygy::KRBvKB,
//             (0, 0, 0, 1, 0, 0, 2, 0, 0, 0) => Syzygy::KRBvKN,
//             (0, 0, 1, 0, 0, 0, 2, 0, 0, 0) => Syzygy::KRBvKP,
//             (0, 0, 0, 0, 0, 1, 2, 0, 0, 0) => Syzygy::KRBvKQ,
//             (0, 0, 0, 0, 1, 0, 2, 0, 0, 0) => Syzygy::KRBvKR,
//             (0, 0, 2, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KRNNvK,
//             (1, 0, 1, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KRNPvK,
//             (0, 0, 1, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KRNvK,
//             (0, 0, 1, 0, 0, 0, 1, 1, 0, 0) => Syzygy::KRNvKB,
//             (0, 0, 1, 1, 0, 0, 1, 0, 0, 0) => Syzygy::KRNvKN,
//             (1, 0, 1, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KRNvKP,
//             (0, 0, 1, 0, 0, 1, 1, 0, 0, 0) => Syzygy::KRNvKQ,
//             (0, 0, 1, 0, 1, 0, 1, 0, 0, 0) => Syzygy::KRNvKR,
//             (2, 0, 0, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KRPPvK,
//             (1, 0, 0, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KRPvK,
//             (1, 0, 0, 0, 0, 0, 1, 1, 0, 0) => Syzygy::KRPvKB,
//             (1, 0, 1, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KRPvKN,
//             (2, 0, 0, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KRPvKP,
//             (1, 0, 0, 0, 0, 1, 1, 0, 0, 0) => Syzygy::KRPvKQ,
//             (1, 0, 0, 0, 1, 0, 1, 0, 0, 0) => Syzygy::KRPvKR,
//             (0, 0, 0, 0, 0, 0, 3, 0, 0, 0) => Syzygy::KRRRvK,
//             (0, 0, 0, 0, 0, 0, 2, 0, 0, 0) => Syzygy::KRRvK,
//             (0, 0, 0, 0, 0, 1, 2, 1, 0, 0) => Syzygy::KRRvKB,
//             (0, 0, 0, 1, 0, 0, 2, 0, 0, 0) => Syzygy::KRRvKN,
//             (0, 0, 1, 0, 0, 0, 2, 0, 0, 0) => Syzygy::KRRvKP,
//             (0, 0, 0, 0, 0, 1, 2, 0, 0, 0) => Syzygy::KRRvKQ,
//             (0, 0, 0, 0, 1, 0, 2, 0, 0, 0) => Syzygy::KRRvKR,
//             (0, 0, 0, 1, 0, 0, 0, 3, 0, 0) => Syzygy::KBBBvK, // Mirror of KBBBvK
//             (0, 0, 0, 1, 0, 0, 0, 2, 0, 0) => Syzygy::KBBNvK, // Mirror of KBBNvK
//             (0, 1, 0, 1, 0, 0, 0, 2, 0, 0) => Syzygy::KBBPvK, // Mirror of KBBPvK
//             (0, 0, 0, 0, 0, 0, 0, 2, 0, 0) => Syzygy::KBBvK, // Mirror of KBBvK
//             (0, 0, 0, 0, 0, 1, 0, 2, 0, 0) => Syzygy::KBBvKB, // Mirror of KBBvKB
//             (0, 0, 1, 0, 0, 0, 0, 2, 0, 0) => Syzygy::KBBvKN, // Mirror of KBBvKN
//             (0, 1, 0, 0, 0, 0, 0, 2, 0, 0) => Syzygy::KBBvKP, // Mirror of KBBvKP
//             (0, 0, 0, 0, 1, 0, 0, 2, 0, 0) => Syzygy::KBBvKQ, // Mirror of KBBvKQ
//             (0, 0, 0, 0, 0, 1, 0, 2, 0, 0) => Syzygy::KBBvKR, // Mirror of KBBvKR
//             (0, 0, 0, 2, 0, 0, 0, 1, 0, 0) => Syzygy::KBNNvK, // Mirror of KBNNvK
//             (0, 1, 0, 1, 0, 0, 0, 1, 0, 0) => Syzygy::KBNPvK, // Mirror of KBNPvK
//             (0, 0, 0, 1, 0, 0, 0, 1, 0, 0) => Syzygy::KBNvK, // Mirror of KBNvK
//             (0, 0, 0, 1, 1, 0, 0, 1, 0, 0) => Syzygy::KBNvKB, // Mirror of KBNvKB
//             (0, 1, 1, 0, 0, 0, 0, 1, 0, 0) => Syzygy::KBNvKN, // Mirror of KBNvKN
//             (0, 1, 0, 1, 0, 0, 0, 1, 0, 0) => Syzygy::KBNvKP, // Mirror of KBNvKP
//             (0, 0, 1, 0, 1, 0, 0, 1, 0, 0) => Syzygy::KBNvKQ, // Mirror of KBNvKQ
//             (0, 0, 1, 0, 0, 1, 0, 1, 0, 0) => Syzygy::KBNvKR, // Mirror of KBNvKR
//             (0, 1, 0, 0, 0, 0, 0, 1, 0, 0) => Syzygy::KBPPvK, // Mirror of KBPPvK
//             (0, 1, 0, 0, 0, 0, 0, 1, 0, 0) => Syzygy::KBPvK, // Mirror of KBPvK
//             (0, 1, 0, 0, 1, 0, 0, 1, 0, 0) => Syzygy::KBPvKB, // Mirror of KBPvKB
//             (1, 0, 0, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KBPvKN, // Mirror of KBPvKN
//             (1, 0, 0, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KBPvKP, // Mirror of KBPvKP
//             (0, 1, 0, 0, 1, 0, 0, 1, 0, 0) => Syzygy::KBPvKQ, // Mirror of KBPvKQ
//             (0, 1, 0, 0, 1, 0, 0, 1, 0, 0) => Syzygy::KBPvKR, // Mirror of KBPvKR
//             (0, 0, 0, 0, 1, 0, 0, 1, 0, 0) => Syzygy::KBvK, // Mirror of KBvK
//             (0, 0, 0, 0, 1, 1, 0, 1, 0, 0) => Syzygy::KBvKB, // Mirror of KBvKB
//             (1, 0, 0, 0, 1, 0, 0, 1, 0, 0) => Syzygy::KBvKN, // Mirror of KBvKN
//             (0, 1, 0, 0, 1, 0, 0, 1, 0, 0) => Syzygy::KBvKP, // Mirror of KBvKP
//             (0, 0, 0, 3, 0, 0, 0, 0, 0, 0) => Syzygy::KNNNvK, // Mirror of KNNNvK
//             (0, 1, 0, 2, 0, 0, 0, 0, 0, 0) => Syzygy::KNNPvK, // Mirror of KNNPvK
//             (0, 0, 0, 2, 0, 0, 0, 0, 0, 0) => Syzygy::KNNvK, // Mirror of KNNvK
//             (0, 0, 0, 2, 0, 0, 1, 0, 0, 0) => Syzygy::KNNvKB, // Mirror of KNNvKB
//             (0, 1, 1, 1, 0, 0, 0, 0, 0, 0) => Syzygy::KNNvKN, // Mirror of KNNvKN
//             (0, 1, 0, 2, 0, 0, 0, 0, 0, 0) => Syzygy::KNNvKP, // Mirror of KNNvKP
//             (0, 0, 1, 1, 1, 0, 0, 0, 0, 0) => Syzygy::KNNvKQ, // Mirror of KNNvKQ
//             (0, 0, 1, 1, 0, 1, 0, 0, 0, 0) => Syzygy::KNNvKR, // Mirror of KNNvKR
//             (0, 1, 0, 1, 0, 0, 0, 0, 0, 0) => Syzygy::KNPPvK, // Mirror of KNPPvK
//             (0, 1, 0, 1, 0, 0, 0, 0, 0, 0) => Syzygy::KNPvK, // Mirror of KNPvK
//             (0, 1, 0, 1, 1, 0, 0, 0, 0, 0) => Syzygy::KNPvKB, // Mirror of KNPvKB
//             (1, 0, 1, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KNPvKN, // Mirror of KNPvKN
//             (1, 0, 1, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KNPvKP, // Mirror of KNPvKP
//             (0, 1, 0, 1, 1, 0, 0, 0, 0, 0) => Syzygy::KNPvKQ, // Mirror of KNPvKQ
//             (0, 1, 0, 1, 1, 0, 0, 0, 0, 0) => Syzygy::KNPvKR, // Mirror of KNPvKR
//             (0, 0, 0, 1, 0, 0, 0, 0, 0, 0) => Syzygy::KNvK, // Mirror of KNvK
//             (0, 0, 0, 1, 1, 0, 0, 0, 0, 0) => Syzygy::KNvKB, // Mirror of KNvKB
//             (1, 1, 1, 1, 0, 0, 0, 0, 0, 0) => Syzygy::KNvKN, // Mirror of KNvKN
//             (1, 0, 1, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KNvKP, // Mirror of KNvKP
//             (1, 0, 0, 0, 0, 0, 1, 0, 0, 0) => Syzygy::KPvK, // Mirror of KPvK
//             (1, 0, 0, 0, 1, 0, 1, 0, 0, 0) => Syzygy::KPvKB, // Mirror of KPvKB
//             (1, 1, 1, 0, 1, 0, 1, 0, 0, 0) => Syzygy::KPvKN, // Mirror of KPvKN
//             (2, 0, 1, 0, 1, 0, 1, 0, 0, 0) => Syzygy::KPvKP, // Mirror of KPvKP
//             (0, 0, 0, 2, 0, 0, 0, 1, 0, 0) => Syzygy::KQBBvK, // Mirror of KQBBvK
//             (0, 1, 0, 1, 0, 0, 0, 2, 0, 0) => Syzygy::KQBNvK, // Mirror of KQBNvK
//             (1, 0, 0, 1, 1, 0, 1, 2, 0, 0) => Syzygy::KQBPvK, // Mirror of KQBPvK
//             (0, 0, 0, 2, 0, 0, 0, 2, 0, 0) => Syzygy::KQBvK, // Mirror of KQBvK
//             (0, 0, 0, 2, 0, 1, 0, 2, 0, 0) => Syzygy::KQBvKB, // Mirror of KQBvKB
//             (0, 1, 1, 1, 0, 0, 0, 2, 0, 0) => Syzygy::KQBvKN, // Mirror of KQBvKN
//             (0, 1, 0, 1, 1, 0, 0, 2, 0, 0) => Syzygy::KQBvKP, // Mirror of KQBvKP
//             (0, 0, 1, 0, 1, 0, 1, 2, 0, 0) => Syzygy::KQBvKQ, // Mirror of KQBvKQ
//             (0, 0, 1, 0, 0, 1, 1, 2, 0, 0) => Syzygy::KQBvKR, // Mirror of KQBvKR
//             (0, 1, 0, 0, 0, 0, 0, 2, 0, 0) => Syzygy::KQNNvK, // Mirror of KQNNvK
//             (1, 0, 1, 0, 1, 0, 1, 1, 0, 0) => Syzygy::KQNPvK, // Mirror of KQNPvK
//             (0, 0, 1, 0, 1, 0, 1, 1, 0, 0) => Syzygy::KQNvK, // Mirror of KQNvK
//             (0, 0, 1, 1, 1, 0, 1, 1, 0, 0) => Syzygy::KQNvKB, // Mirror of KQNvKB
//             (0, 1, 1, 1, 1, 0, 1, 1, 0, 0) => Syzygy::KQNvKN, // Mirror of KQNvKN
//             (0, 1, 0, 1, 1, 0, 1, 1, 0, 0) => Syzygy::KQNvKP, // Mirror of KQNvKP
//             (0, 0, 1, 1, 1, 0, 1, 1, 0, 0) => Syzygy::KQNvKQ, // Mirror of KQNvKQ
//             (0, 0, 1, 1, 0, 1, 1, 1, 0, 0) => Syzygy::KQNvKR, // Mirror of KQNvKR
//             (0, 1, 0, 0, 0, 0, 1, 1, 0, 0) => Syzygy::KQPPvK, // Mirror of KQPPvK
//             (1, 0, 0, 0, 0, 0, 1, 1, 0, 0) => Syzygy::KQPvK, // Mirror of KQPvK
//             (1, 0, 0, 0, 1, 0, 1, 1, 0, 0) => Syzygy::KQPvKB, // Mirror of KQPvKB
//             (1, 1, 1, 0, 1, 0, 1, 1, 0, 0) => Syzygy::KQPvKN, // Mirror of KQPvKN
//             (2, 0, 1, 0, 1, 0, 1, 1, 0, 0) => Syzygy::KQPvKP, // Mirror of KQPvKP
//             (1, 0, 1, 0, 1, 0, 1, 1, 0, 0) => Syzygy::KQPvKQ, // Mirror of KQPvKQ
//             (1, 0, 1, 0, 1, 0, 1, 1, 0, 0) => Syzygy::KQPvKR, // Mirror of KQPvKR
//             (0, 0, 0, 2, 0, 0, 0, 2, 0, 0) => Syzygy::KQQvK, // Mirror of KQQvK
//             (0, 0, 0, 2, 0, 1, 0, 2, 0, 0) => Syzygy::KQQvKB, // Mirror of KQQvKB
//             (0, 1, 1, 1, 0, 0, 0, 2, 0, 0) => Syzygy::KQQvKN, // Mirror of KQQvKN
//             (0, 1, 0, 1, 1, 0, 0, 2, 0, 0) => Syzygy::KQQvKP, // Mirror of KQQvKP
//             (0, 0, 1, 0, 1, 0, 1, 2, 0, 0) => Syzygy::KQQvKQ, // Mirror of KQQvKQ
//             (0, 0, 1, 0, 0, 1, 1, 2, 0, 0) => Syzygy::KQQvKR, // Mirror of KQQvKR
//             (0, 0, 1, 0, 1, 0, 1, 0, 0, 0) => Syzygy::KQRBvK, // Mirror of KQRBvK
//             (1, 0, 1, 0, 1, 0, 1, 0, 0, 0) => Syzygy::KQRNvK, // Mirror of KQRNvK
//             (1, 0, 1, 0, 1, 0, 1, 0, 0, 0) => Syzygy::KQRPvK, // Mirror of KQRPvK
//             (0, 0, 0, 0, 1, 0, 1, 0, 0, 0) => Syzygy::KQRvK, // Mirror of KQRvK
//             (0, 0, 0, 0, 1, 1, 1, 1, 0, 0) => Syzygy::KQRvKB, // Mirror of KQRvKB
//             (0, 1, 1, 1, 0, 0, 1, 0, 0, 0) => Syzygy::KQRvKN, // Mirror of KQRvKN
//             (0, 1, 0, 1, 1, 0, 1, 0, 0, 0) => Syzygy::KQRvKP, // Mirror of KQRvKP
//             (0, 0, 1, 1, 1, 0, 1, 1, 0, 0) => Syzygy::KQRvKQ, // Mirror of KQRvKQ
//             (0, 0, 1, 1, 0, 1, 1, 1, 0, 0) => Syzygy::KQRvKR, // Mirror of KQRvKR
//             (0, 0, 0, 2, 0, 0, 0, 2, 0, 0) => Syzygy::KRBBvK, // Mirror of KRBBvK
//             (0, 1, 0, 1, 0, 0, 0, 2, 0, 0) => Syzygy::KRBNvK, // Mirror of KRBNvK
//             (1, 0, 1, 0, 1, 0, 1, 2, 0, 0) => Syzygy::KRBPvK, // Mirror of KRBPvK
//             (0, 0, 0, 2, 0, 0, 0, 2, 0, 0) => Syzygy::KRBvK, // Mirror of KRBvK
//             (0, 0, 0, 2, 0, 1, 0, 2, 0, 0) => Syzygy::KRBvKB, // Mirror of KRBvKB
//             (0, 1, 1, 1, 0, 0, 0, 2, 0, 0) => Syzygy::KRBvKN, // Mirror of KRBvKN
//             (0, 1, 0, 1, 1, 0, 0, 2, 0, 0) => Syzygy::KRBvKP, // Mirror of KRBvKP
//             (0, 0, 1, 0, 1, 0, 1, 2, 0, 0) => Syzygy::KRBvKQ, // Mirror of KRBvKQ
//             (0, 0, 1, 0, 0, 1, 1, 2, 0, 0) => Syzygy::KRBvKR, // Mirror of KRBvKR
//             (0, 1, 0, 0, 0, 0, 0, 2, 0, 0) => Syzygy::KRNNvK, // Mirror of KRNNvK
//             (1, 0, 1, 0, 1, 0, 1, 1, 0, 0) => Syzygy::KRNPvK, // Mirror of KRNPvK
//             (0, 0, 1, 0, 1, 0, 1, 1, 0, 0) => Syzygy::KRNvK, // Mirror of KRNvK
//             (0, 0, 1, 1, 1, 0, 1, 1, 0, 0) => Syzygy::KRNvKB, // Mirror of KRNvKB
//             (0, 1, 1, 1, 1, 0, 1, 1, 0, 0) => Syzygy::KRNvKN, // Mirror of KRNvKN
//             (0, 1, 0, 1, 1, 0, 1, 1, 0, 0) => Syzygy::KRNvKP, // Mirror of KRNvKP
//             (0, 0, 1, 1, 1, 0, 1, 1, 0, 0) => Syzygy::KRNvKQ, // Mirror of KRNvKQ
//             (0, 0, 1, 1, 0, 1, 1, 1, 0, 0) => Syzygy::KRNvKR, // Mirror of KRNvKR
//             (0, 1, 0, 0, 0, 0, 1, 1, 0, 0) => Syzygy::KRPPvK, // Mirror of KRPPvK
//             (1, 0, 0, 0, 0, 0, 1, 1, 0, 0) => Syzygy::KRPvK, // Mirror of KRPvK
//             (1, 0, 0, 0, 1, 0, 1, 1, 0, 0) => Syzygy::KRPvKB, // Mirror of KRPvKB
//             (1, 1, 1, 0, 1, 0, 1, 1, 0, 0) => Syzygy::KRPvKN, // Mirror of KRPvKN
//             (2, 0, 1, 0, 1, 0, 1, 1, 0, 0) => Syzygy::KRPvKP, // Mirror of KRPvKP
//             (1, 0, 1, 0, 1, 0, 1, 1, 0, 0) => Syzygy::KRPvKQ, // Mirror of KRPvKQ
//             (1, 0, 1, 0, 1, 0, 1, 1, 0, 0) => Syzygy::KRPvKR, // Mirror of KRPvKR
//             (0, 0, 0, 3, 0, 0, 0, 0, 0, 0) => Syzygy::KRRRvK, // Mirror of KRRRvK
//             (0, 0, 0, 2, 0, 0, 0, 0, 0, 0) => Syzygy::KRRvK, // Mirror of KRRvK
//             (0, 0, 0, 2, 0, 1, 0, 2, 0, 0) => Syzygy::KRRvKB, // Mirror of KRRvKB
//             (0, 1, 1, 1, 0, 0, 0, 2, 0, 0) => Syzygy::KRRvKN, // Mirror of KRRvKN
//             (0, 1, 0, 1, 1, 0, 0, 2, 0, 0) => Syzygy::KRRvKP, // Mirror of KRRvKP
//             (0, 0, 1, 0, 1, 0, 1, 2, 0, 0) => Syzygy::KRRvKQ, // Mirror of KRRvKQ
//             (0, 0, 1, 0, 0, 1, 1, 2, 0, 0) => Syzygy::KRRvKR, // Mirror of KRRvKR
//             _ => Syzygy::Na,
//         }

//     }
// }