mod base_functions;
mod types; // Only if `types.rs` is a separate file // Only if `base_functions.rs` is a separate file

use crate::base_functions::init_board;
// use crate::types::{Board, Piece, Kind, PieceId, Team}; // Import everything you need from types
fn main() {
    let board = init_board(true);
}
