use std::{cell::RefCell, rc::Rc};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Kind {
    Pawn,
    Rook,
    Knight,
    King,
    Queen,
    Bishop,
    Empty,
}

impl Kind { //this is just for testing. 
    pub fn to_string(&self) -> &str {
        match self {
            Kind::Pawn => "P",
            Kind::Rook => "R",
            Kind::Knight => "k",
            Kind::King => "K",
            Kind::Queen => "Q",
            Kind::Bishop => "B",
            Kind::Empty => "E",
        }
    }
}


#[derive(Clone, Debug, Copy, PartialEq, Serialize, Deserialize)]
pub enum PieceId { // unique identifiers for every possible piece on the board
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
    Error,
}

impl PieceId { //just for testing
    pub fn to_string(&self) -> &str {
        match self {
            PieceId::P1 => "P1",
            PieceId::P2 => "P2",
            PieceId::P3 => "P3",
            PieceId::P4 => "P4",
            PieceId::P5 => "P5",
            PieceId::P6 => "P6",
            PieceId::P7 => "P7",
            PieceId::P8 => "P8",
            PieceId::K1 => "K1",
            PieceId::K2 => "K2",
            PieceId::K3 => "K3",
            PieceId::K4 => "K4",
            PieceId::K5 => "K5",
            PieceId::K6 => "K6",
            PieceId::K7 => "K7",
            PieceId::K8 => "K8",
            PieceId::K9 => "K9",
            PieceId::K10 => "K10",
            PieceId::B1 => "B1",
            PieceId::B2 => "B2",
            PieceId::R1 => "R1",
            PieceId::R2 => "R2",
            PieceId::K => "K",
            PieceId::Q => "Q",
            PieceId::Q1 => "Q1",
            PieceId::Q2 => "Q2",
            PieceId::Q3 => "Q3",
            PieceId::Q4 => "Q4",
            PieceId::Q5 => "Q5",
            PieceId::Q6 => "Q6",
            PieceId::Q7 => "Q7",
            PieceId::Q8 => "Q8",
            PieceId::Error => "Error",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Copy, Serialize, Deserialize)]
pub enum Team { //white, black or empty square
    W,
    B,
    N,
}

impl Team {
    pub fn to_string(&self) -> &str { //once again just for testing
        match self {
            Team::W => "W",
            Team::B => "B",
            Team::N => "N",
        }
    }
}

#[derive(Clone, Debug)]
pub enum GameState { //will be used for sending info to front end. 
    InPlay,
    AiWin,
    PlayerWin,
    Stalemate,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Piece {//this is what is represented on the full_board field
    pub team: Team,
    pub kind: Kind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Move {//parameters for the move_piece function. 
    pub piece: PieceId,
    pub location: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexMap {//this stores every unique PieceId to its location on the board. 
    //for every index, row is stored in the 10s place and column in stored in the 1s place
    //indexes can range from 0 (A8) to 77 (H1), /10 for row %10 for column
    //NOTE: pieceids that are captured or do not exist yet (pawn promotions) should NEVER be looked up here
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
pub struct AvailableMovesMap {//this maps the PieceId to where it can move to
    //key is pieceid and value is vector of places the unique piece can move
    //unlike the other maps, this one has to be calculated from scratch each turn instead of just changing one or two key-value pairs each turn.
    //Due to this, the mappings for piece moves are not included in the board class 
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
}


#[derive(Clone, Debug)]
pub struct TreeNode{
    pub level:i32,
    pub game:Board,
    pub parent:Option<TreeNodeRef>,
    pub children:Vec<TreeNodeRef>,
}

pub type TreeNodeRef = Rc<RefCell<TreeNode>>; //RefCell<T> and Cell<T> is a type that allows for interior mutability,

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IToPMap {
    pub hash: Vec<Option<PieceId>>,
}
impl IToPMap { //index maps to what PieceId is located on that index
    //once again, the get_piece method should NEVER be called on an empty index
    pub fn new() -> Self {
        let num_variants = 78;
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

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Board {
    pub moves_log: Vec<Move>, //moves_log is just for testing and maybe recording games in the future
    //moves_log is not actually used by the engine

    pub ai_team_is_white: bool,
    pub full_board: Vec<Vec<Piece>>, // represents current state of chess board
    pub turn: i32,//%2==0 means its whites turn, else ->blacks turn 

    pub black_indexes: IndexMap,
    pub white_indexes: IndexMap, //description of these maps are in definitions
    pub black_i_to_p: IToPMap,
    pub white_i_to_p: IToPMap,

    //these points fields are not currently used anywhere by engine, but the whole idea 
    //is pawn=1, knight=3..., may be important later or I may take it out. 
    pub black_points: i32, 
    pub white_points: i32,

    //the piece ids represent which unique pieces are currently on the board for each team
    //these vectors are iterated over to both generate moves and search for best move
    //there should NEVER be a piece in here that is not on the board
    pub white_piece_ids: Vec<PieceId>,
    pub black_piece_ids: Vec<PieceId>,

    //origanally, I was using arrays of booleans to store information on if these events happened:
    //pawn skipping last turn
    //pawn skipping at any turn 
    //knight, king or rook is moved
    //This information was moved over to the prime numbers
    //multiply by a prime when one of these events happens, do % on the stored value for that prime # when checking to see if it happened or not

    //white_prime is multiplied by a prime number that is hashed from the pieceId of the pawn that moved when a white pawn skips forward 2
    //Primes1 is the hashing function for this
    //if a white pawn wants to en pessant, it needed to have skipped forward 2 earlier in the game
    //white_prime stores that info for white and black_prime1 for black
    pub white_prime: i32,
    pub black_prime: i32,

    //the vals white_prime1 and black_prime1 represent which pawn for their teams skipped forward last turn. 
    //after black moves, white prime is reset to 1 and vice versa. 
    //This is for knowing if you can en pessant, since the opposite pawn would have had to move last turn to do this
    pub white_prime1: i16,
    pub black_prime1: i16,
    //prime2 is for castling. you cannot castle if you have moved the king, or the respective side's rook
    //2,3,5,7,11,13 are hardcoded to represent this info
    pub prime2: i32,
    //ai_advantage is a field that works with the searching algorithm to find best move. It is generated by the neural network. 
    pub ai_advantage: f32,
}

pub struct AllMovesGenRe {
    pub moves:AvailableMovesMap,
    pub checking:bool
}

#[derive(Clone, Debug)]
pub struct AdavantageMap {
    pub advantage: f32,
    pub board: Board,
}

#[derive(Clone, Debug)]
pub struct MoveAdvMap{
    pub advantage:f32,
    pub m:Move,
}

pub struct LocationToMove {
    pub location: i32,
    pub moves: Vec<i32>,
}

pub struct RenderPiece {
    pub kind: Kind,
    pub team: Team,
}

pub struct ServerProcessingRe {
    pub b:Board, 
    pub pgn:String,
}

pub struct AIMoveRe{
    pub m:Move,
    pub b:Board,
}
