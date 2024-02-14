use crate::types::{Board, Kind, Move, Piece, PieceId, Team};
use std::collections::HashMap;
pub fn map_piece_id_to_kind(piece: PieceId) -> Kind {
    match piece {
        PieceId::P1 | PieceId::P2 | PieceId::P3 | PieceId::P4 |
        PieceId::P5 | PieceId::P6 | PieceId::P7 | PieceId::P8 => Kind::Pawn,
        
        PieceId::K1 | PieceId::K2 | PieceId::K3 | PieceId::K4 |
        PieceId::K5 | PieceId::K6 | PieceId::K7 | PieceId::K8 |
        PieceId::K9 | PieceId::K10 => Kind::Knight,
        
        PieceId::B1 | PieceId::B2 => Kind::Bishop,
        
        PieceId::K => Kind::King,
        
        PieceId::Q | PieceId::Q1 | PieceId::Q2 | PieceId::Q3 |
        PieceId::Q4 | PieceId::Q5 | PieceId::Q6 | PieceId::Q7 => Kind::Queen,
        
        PieceId::R1 | PieceId::R2 => Kind::Rook,
        
        _ => Kind::Empty,
    }
    
}

pub fn init_board(ai_team: bool) -> Board {
    let mut moves_log = Vec::new();
    let mut full_board = Vec::new();
    let mut w_piece_ids = vec![
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
    let mut b_piece_ids = vec![
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
            value: 500,
        },
        Piece {
            team: Team::B,
            kind: Kind::Knight,
            value: 300,
        },
        Piece {
            team: Team::B,
            kind: Kind::Bishop,
            value: 300,
        },
        Piece {
            team: Team::B,
            kind: Kind::King,
            value: 0,
        },
        Piece {
            team: Team::B,
            kind: Kind::Queen,
            value: 900,
        },
        Piece {
            team: Team::B,
            kind: Kind::Bishop,
            value: 300,
        },
        Piece {
            team: Team::B,
            kind: Kind::Knight,
            value: 300,
        },
        Piece {
            team: Team::B,
            kind: Kind::Rook,
            value: 500,
        },
    ];

    let row2 = vec![
        Piece {
            team: Team::B,
            kind: Kind::Pawn,
            value: 100,
        },
        Piece {
            team: Team::B,
            kind: Kind::Pawn,
            value: 100,
        },
        Piece {
            team: Team::B,
            kind: Kind::Pawn,
            value: 100,
        },
        Piece {
            team: Team::B,
            kind: Kind::Pawn,
            value: 100,
        },
        Piece {
            team: Team::B,
            kind: Kind::Pawn,
            value: 100,
        },
        Piece {
            team: Team::B,
            kind: Kind::Pawn,
            value: 100,
        },
        Piece {
            team: Team::B,
            kind: Kind::Pawn,
            value: 100,
        },
        Piece {
            team: Team::B,
            kind: Kind::Pawn,
            value: 100,
        },
    ];

    let row3 = vec![
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
    ];

    let row4 = vec![
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
    ];

    let row5 = vec![
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
    ];
    let row6 = vec![
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
        Piece {
            team: Team::N,
            kind: Kind::Empty,
            value: 0,
        },
    ];
    let row7 = vec![
        Piece {
            team: Team::W,
            kind: Kind::Pawn,
            value: 100,
        },
        Piece {
            team: Team::W,
            kind: Kind::Pawn,
            value: 100,
        },
        Piece {
            team: Team::W,
            kind: Kind::Pawn,
            value: 100,
        },
        Piece {
            team: Team::W,
            kind: Kind::Pawn,
            value: 100,
        },
        Piece {
            team: Team::W,
            kind: Kind::Pawn,
            value: 100,
        },
        Piece {
            team: Team::W,
            kind: Kind::Pawn,
            value: 100,
        },
        Piece {
            team: Team::W,
            kind: Kind::Pawn,
            value: 100,
        },
        Piece {
            team: Team::W,
            kind: Kind::Pawn,
            value: 100,
        },
    ];
    let row8 = vec![
        Piece {
            team: Team::W,
            kind: Kind::Rook,
            value: 500,
        },
        Piece {
            team: Team::W,
            kind: Kind::Knight,
            value: 300,
        },
        Piece {
            team: Team::W,
            kind: Kind::Bishop,
            value: 300,
        },
        Piece {
            team: Team::W,
            kind: Kind::King,
            value: 0,
        },
        Piece {
            team: Team::W,
            kind: Kind::Queen,
            value: 900,
        },
        Piece {
            team: Team::W,
            kind: Kind::Bishop,
            value: 300,
        },
        Piece {
            team: Team::W,
            kind: Kind::Knight,
            value: 300,
        },
        Piece {
            team: Team::W,
            kind: Kind::Rook,
            value: 500,
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

    let mut black_indexes = HashMap::new();
    black_indexes.insert(PieceId::R1, 0);
    black_indexes.insert(PieceId::R2, 7);
    black_indexes.insert(PieceId::B1, 2);
    black_indexes.insert(PieceId::B2, 5);
    black_indexes.insert(PieceId::K1, 1);
    black_indexes.insert(PieceId::K2, 6);
    black_indexes.insert(PieceId::K, 3);
    black_indexes.insert(PieceId::Q, 4);
    black_indexes.insert(PieceId::P1, 10);
    black_indexes.insert(PieceId::P2, 11);
    black_indexes.insert(PieceId::P3, 12);
    black_indexes.insert(PieceId::P4, 13);
    black_indexes.insert(PieceId::P5, 14);
    black_indexes.insert(PieceId::P6, 15);
    black_indexes.insert(PieceId::P7, 16);
    black_indexes.insert(PieceId::P8, 17);

    let mut black_i_to_p = HashMap::new();
    black_i_to_p.insert(0, PieceId::R1);
    black_i_to_p.insert(7, PieceId::R2);
    black_i_to_p.insert(2, PieceId::B1);
    black_i_to_p.insert(5, PieceId::B2);
    black_i_to_p.insert(1, PieceId::K1);
    black_i_to_p.insert(6, PieceId::K2);
    black_i_to_p.insert(3, PieceId::K);
    black_i_to_p.insert(4, PieceId::Q);
    black_i_to_p.insert(10, PieceId::P1);
    black_i_to_p.insert(11, PieceId::P2);
    black_i_to_p.insert(12, PieceId::P3);
    black_i_to_p.insert(13, PieceId::P4);
    black_i_to_p.insert(14, PieceId::P5);
    black_i_to_p.insert(15, PieceId::P6);
    black_i_to_p.insert(16, PieceId::P7);
    black_i_to_p.insert(17, PieceId::P8);

    let mut white_indexes = HashMap::new();
    white_indexes.insert(PieceId::R1, 70);
    white_indexes.insert(PieceId::R2, 77);
    white_indexes.insert(PieceId::B1, 72);
    white_indexes.insert(PieceId::B2, 75);
    white_indexes.insert(PieceId::K1, 71);
    white_indexes.insert(PieceId::K2, 76);
    white_indexes.insert(PieceId::K, 73);
    white_indexes.insert(PieceId::Q, 74);
    white_indexes.insert(PieceId::P1, 60);
    white_indexes.insert(PieceId::P2, 61);
    white_indexes.insert(PieceId::P3, 62);
    white_indexes.insert(PieceId::P4, 63);
    white_indexes.insert(PieceId::P5, 64);
    white_indexes.insert(PieceId::P6, 65);
    white_indexes.insert(PieceId::P7, 66);
    white_indexes.insert(PieceId::P8, 67);

    let mut white_i_to_p = HashMap::new();
    white_i_to_p.insert(0, PieceId::R1);
    white_i_to_p.insert(7, PieceId::R2);
    white_i_to_p.insert(2, PieceId::B1);
    white_i_to_p.insert(5, PieceId::B2);
    white_i_to_p.insert(1, PieceId::K1);
    white_i_to_p.insert(6, PieceId::K2);
    white_i_to_p.insert(3, PieceId::K);
    white_i_to_p.insert(4, PieceId::Q);
    white_i_to_p.insert(10, PieceId::P1);
    white_i_to_p.insert(11, PieceId::P2);
    white_i_to_p.insert(12, PieceId::P3);
    white_i_to_p.insert(13, PieceId::P4);
    white_i_to_p.insert(14, PieceId::P5);
    white_i_to_p.insert(15, PieceId::P6);
    white_i_to_p.insert(16, PieceId::P7);
    white_i_to_p.insert(17, PieceId::P8);

    let mut white_available_moves = HashMap::new();
    let mut black_available_moves = HashMap::new();

    return Board {
        moves_log,
        in_check_stored: false,
        ai_team_is_white: ai_team,
        in_play: true,
        full_board,
        white_available_moves,
        black_available_moves,
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
        ai_advantage: 0,
        turn: 0,
    };
}
fn primes(col:i32) -> i32{
    let primes=Vec::vec![2,3,5,7,11,13,17,19];
    return primes[col];
}
fn primes1(p:PieceId) -> i32{
        match p{
            PieceId::P1 => 2,
            PieceId::P2 => 3,
            PieceId::P3 => 5,
            PieceId::P4 => 7,
            PieceId::P5 => 11,
            PieceId::P6 => 13,
            PieceId::P7 => 17,
            PieceId::P8 => 19,
            _ => Kind::1,
        }
        return p(PieceId)
}
fn pawn_to_queen(p:PieceId)-> i32{
        match p {
            PieceId::P1 => Kind::Q1,
            PieceId::P2 => Kind::Q2,
            PieceId::P3 => Kind::Q3,
            PieceId::P4 => Kind::Q4,
            PieceId::P5 => Kind::Q5,
            PieceId::P6 => Kind::Q6,
            PieceId::P7 => Kind::Q7,
            PieceId::P8 => Kind::Q8,
            _ => Kind::Empty,
        }
}
fn pawn_to_knight(p:PieceId)-> i32{
    match p {
        PieceId::P1 => Kind::K3,
        PieceId::P2 => Kind::K4,
        PieceId::P3 => Kind::K5,
        PieceId::P4 => Kind::K6,
        PieceId::P5 => Kind::K7,
        PieceId::P6 => Kind::K8,
        PieceId::P7 => Kind::K9,
        PieceId::P8 => Kind::K10,
        _ => Kind::Empty,
    }
    return p(PieceId)
}
fn mapping(n: i32) -> String {
    let mut n = n.abs(); // Take the absolute value if n is negative

    if n == 99 {
        return String::from("KS");
    } else if n == 100 {
        return String::from("QS");
    } else {
        let mut ret = String::new();
        let row_map = vec!["8", "7", "6", "5", "4", "3", "2", "1"];
        let col_map = vec!["A", "B", "C", "D", "E", "F", "G", "H"];

        let col = n % 10;
        let row = n / 10;

        let a = col_map[col as usize];
        let b = row_map[row as usize];

        ret.push_str(a);
        ret.push_str(b);

        return ret;
    }
}


