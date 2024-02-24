use std::collections::HashMap;
use std::vec::Vec;

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
    Q8,
    Q9,
}

#[derive(Eq, Hash, PartialEq, Debug, Clone, Copy)]
pub enum Team {
    W,
    B,
    N,
}

#[derive(Eq, Hash, PartialEq, Debug, Clone, Copy)]
pub enum GameState {
    InPlay,
    AiWin,
    PlayerWin,
    Stalemate,
}

#[derive(Eq, Hash, PartialEq, Debug, Clone, Copy)]
pub struct Piece {
    pub team: Team,
    pub kind: Kind,
    pub value: i32,
}

#[derive(Eq, Hash, PartialEq, Debug, Clone, Copy)]
pub struct Move {
    pub piece: PieceId,
    pub location: i32,
}

#[derive(Eq, Hash, PartialEq, Debug, Clone, Copy)]
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
    pub ai_advantage: f64,
}

#[derive(Eq, Hash, PartialEq, Debug, Clone, Copy)]
pub struct TreeNode {
    pub children: Vec<TreeNode>,
    pub parent: Vec<TreeNode>,
    pub board: Board,
    pub level: i32,
}

#[derive(Eq, Hash, PartialEq, Debug, Clone, Copy)]
pub struct AdavantageMap {
    pub advantage: f64,
    pub board: Board,
}
//all of these types are for sending to front end.

#[derive(Eq, Hash, PartialEq, Debug, Clone, Copy)]
pub struct LocationToMove {
    pub location: i32,
    pub moves: Vec<i32>,
}

#[derive(Eq, Hash, PartialEq, Debug, Clone, Copy)]
pub struct RenderPiece {
    pub kind: Kind,
    pub team: Team,
}

#[derive(Eq, Hash, PartialEq, Debug, Clone, Copy)]
pub struct APIResponse {
    pub boards_searched: String,
    pub player_moves: Vec<LocationToMove>,
    pub full_board: Vec<Vec<RenderPiece>>,
    pub game_state: GameState,
}
