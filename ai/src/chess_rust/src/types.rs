use std::collections::HashMap;

#[derive(Eq, Hash, PartialEq, Debug, Clone, Copy)]
pub enum Kind {
    Pawn,
    Rook,
    Knight,
    King,
    Queen,
    Bishop,
    Empty,
}

#[derive(Eq, Hash, PartialEq, Debug, Clone, Copy)]
pub enum PieceId {
    P1,
    P2,
    P3,
    P4,
    P5,
    P6,
    P7,
    P8,
    K1,
    K2,
    K3,
    K4,
    K5,
    K6,
    K7,
    K8,
    K9,
    K10,
    B1,
    B2,
    R1,
    R2,
    K,
    Q,
    Q1,
    Q2,
    Q3,
    Q4,
    Q5,
    Q6,
    Q7,
}

pub enum Team {
    W,
    B,
    N,
}

pub struct Piece {
    pub team: Team,
    pub kind: Kind,
    pub value: i32,
}
pub struct Move {
    pub piece: PieceId,
    pub location: i32,
}

pub struct Board {
    pub moves_log: Vec<Move>,
    pub in_check_stored: bool,
    pub ai_team_is_white: bool,
    pub in_play: bool,
    pub full_board: Vec<Vec<Piece>>,
    pub turn: i32,
    pub black_indexes: HashMap<PieceId, i32>,
    pub white_indexes: HashMap<PieceId, i32>,
    pub black_i_to_p: HashMap<i32, PieceId>,
    pub white_i_to_p: HashMap<i32, PieceId>,
    pub black_points: i32,
    pub white_points: i32,
    pub white_piece_ids: Vec<PieceId>,
    pub black_piece_ids: Vec<PieceId>,
    pub white_available_moves: HashMap<PieceId, Vec<i32>>,
    pub black_available_moves: HashMap<PieceId, Vec<i32>>,
    pub white_prime: i32,
    pub black_prime: i32,
    pub white_prime1: i32,
    pub black_prime1: i32,
    pub prime2: i32,
    pub ai_advantage: i64,
}
