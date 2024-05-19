use std::{cell::RefCell, rc::Rc};


#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    Pawn,
    Rook,
    Knight,
    King,
    Queen,
    Bishop,
    Empty,
}
#[derive(Clone, Debug, Copy, PartialEq)]
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
    Error,
}
#[derive(Clone, Debug, PartialEq, Copy)]
pub enum Team {
    W,
    B,
    N,
}
#[derive(Clone, Debug)]
pub enum GameState {
    InPlay,
    AiWin,
    PlayerWin,
    Stalemate,
}
#[derive(Clone, Debug)]
pub struct Piece {
    pub team: Team,
    pub kind: Kind,
    pub value: i32,
}
#[derive(Clone, Debug)]
pub struct Move {
    pub piece: PieceId,
    pub location: i32,
}
#[derive(Clone, Debug)]
pub struct IndexMap {
    hash: Vec<Option<usize>>,
}
impl IndexMap {
    pub fn new() -> Self {
        let num_variants = 39;
        IndexMap {
            hash: vec![None; num_variants],
        }
    }
    pub fn change_indexes(&mut self, piece_id: PieceId, index: usize) {
        self.hash[piece_id as usize] = Some(index);
    }
    pub fn get_index(&self, piece_id: PieceId) -> Option<usize> {
        self.hash[piece_id as usize]
    }
    pub fn nullify(&mut self, piece_id: PieceId) {
        self.hash[piece_id as usize] = None;
    }
}
#[derive(Clone, Debug)]
pub struct AvailableMovesMap {
    pub hash: Vec<Vec<usize>>,
}
impl AvailableMovesMap {
    pub fn new() -> Self {
        let num_variants = 39; 
        AvailableMovesMap {
            hash: vec![vec![]; num_variants],
        }
    }

    pub fn insert_moves(&mut self, piece_id: PieceId, moves: &Vec<usize>) {
        self.hash[piece_id as usize] = moves.clone();
    }

    pub fn get_moves(&self, piece: PieceId) -> Vec<usize> {
        let index = piece as usize;
        if index < self.hash.len() {
            self.hash[index].clone() 
        } else {
            vec![] 
        }
    }
    pub fn reset(&mut self) {
        let num_variants = 39;
        self.hash = vec![vec![]; num_variants];
    }
}
#[derive(Clone, Debug)]
pub struct IToPMap {
    pub hash: Vec<Option<PieceId>>,
}

#[derive(Clone, Debug)]
pub struct TreeNode{
    pub level:i32,
    pub game:Board,
    pub parent:Option<TreeNodeRef>,
    pub children:Vec<TreeNodeRef>,
}

pub type TreeNodeRef = Rc<RefCell<TreeNode>>; //RefCell<T> and Cell<T> is a type that allows for interior mutability,

impl IToPMap {
    pub fn new() -> Self {
        let num_variants = 77;
        IToPMap {
            hash: vec![None; num_variants], 
        }
    }
    pub fn insert_piece(&mut self, index: usize, piece_id: PieceId) {
        self.hash[index] = Some(piece_id); 
    }
    pub fn get_piece(&self, index: usize) -> Option<PieceId> {
        self.hash[index]
    }

    pub fn nullify(&mut self, index: usize) {
        self.hash[index] = None;
    }
}

#[derive(Clone, Debug)]
pub struct Board {
    pub moves_log: Vec<Move>,
    pub ai_team_is_white: bool,
    pub full_board: Vec<Vec<Piece>>,
    pub turn: i32,
    pub black_indexes: IndexMap,
    pub white_indexes: IndexMap,
    pub black_i_to_p: IToPMap,
    pub white_i_to_p: IToPMap,
    pub black_points: i32,
    pub white_points: i32,
    pub white_piece_ids: Vec<PieceId>,
    pub black_piece_ids: Vec<PieceId>,
    pub white_prime: i32,
    pub black_prime: i32,
    pub white_prime1: i32,
    pub black_prime1: i32,
    pub prime2: i32,
    pub ai_advantage: f64,
}

pub struct AllMovesGenRe {
    pub black_moves:AvailableMovesMap,
    pub white_moves:AvailableMovesMap,
    pub checking:bool
}
impl AllMovesGenRe {
    pub fn new()->AllMovesGenRe{
        AllMovesGenRe{
            black_moves:AvailableMovesMap::new(),
            white_moves:AvailableMovesMap::new(),
            checking:false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AdavantageMap {
    pub advantage: f64,
    pub board: Board,
}


pub struct LocationToMove {
    pub location: i32,
    pub moves: Vec<i32>,
}

pub struct RenderPiece {
    pub kind: Kind,
    pub team: Team,
}

pub struct APIResponse {
    pub boards_searched: String,
    pub player_moves: Vec<LocationToMove>,
    pub full_board: Vec<Vec<RenderPiece>>,
    pub game_state: GameState,
}
