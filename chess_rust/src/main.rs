mod ai_functions;
mod base_functions;
mod base_move_functions;
mod search_functions;
mod types;
mod upper_move_function_helpers;
mod upper_move_functions;
mod test_board;
mod server_processing;
mod mongo_repo;

use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use mongodb::{bson::doc, options::ClientOptions, Client};
use std::sync::Arc;
use tokio::sync::Mutex;
use serde::{Deserialize, Serialize};
//use crate::mongo_repo::{AppState, get_last_game_state, insert_game_state, delete_game_state};
use crate::server_processing::{process_server_response};
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Move{
    pgn:String,
    user_id:String
}

struct AppState {
    counter: Mutex<usize>,
}

async fn handle_move(
    move_data: web::Json<Move>,
    data: web::Data<Arc<Mutex<AppState>>>,
) -> impl Responder {
    let moves = vec![
        "e4", "Nf6", "e5", "Nd5", "d4", "d6", "Nf3", "Bg4", "Bc4", "e6", 
        "O-O", "Nb6", "Be2", "Be7", "h3", "Bh5", "Bf4", "Nc6", "c3", "O-O", 
        "Nbd2", "d5", "b4", "a5", "a3", "Qd7", "Qc2", "Bg6", "Bd3", "Rfc8", 
        "Rfb1", "Bf8", "h4", "Ne7", "g3", "Qa4", "Ne1", "Qxc2", "Bxc2", 
        "Bxc2", "Nxc2", "Na4", "Rb3", "b6", "Kf1", "c5", "bxc5", "bxc5", 
        "dxc5", "Rxc5", "Nb1", "Rac8", "Be3", "Rc4", "Bd4", "Nc6", "Rb5", 
        "Nxd4", "Nxd4", "Nxc3", "Nxc3", "Rxd4", "Ne2", "Ra4", "Ke1", "Rxa3", 
        "Rab1", "Bb4+", "Kf1", "Rd3"
    ];

    let state = data.lock().await;

    let mut counter = state.counter.lock().await;
    let index = *counter;

    if index >= moves.len() {
        return HttpResponse::BadRequest().json("Index out of bounds");
    }

    let result_move = moves.get(index).unwrap_or(&"").clone();

    // Increment the counter for the next request
    *counter += 2;

    // Return the move at the current index
    HttpResponse::Ok().json(result_move)
}// async fn handle_move(
//     move_data: web::Json<Move>,
//     data: web::Data<Arc<Mutex<AppState>>>,
// ) -> impl Responder {
//     let pgn = move_data.pgn.clone();
//     let user_id = move_data.user_id.clone();
//     let collection = &data.lock().await.mongo_collection;

//     let last_game = get_last_game_state(collection, &user_id).await;

//     let result = process_server_response(pgn.clone(), last_game);

//     if result.pgn.is_empty() {
//         match delete_game_state(collection, &user_id).await {
//             Ok(_) => HttpResponse::Ok().json(result.pgn),
//             Err(e) => {
//                 eprintln!("Failed to delete board state: {}", e);
//                 HttpResponse::InternalServerError().json("Failed to delete board state")
//             }
//         }
//     } else {
//         match insert_game_state(collection, &user_id, &result.b).await {
//             Ok(_) => HttpResponse::Ok().json(result.pgn),
//             Err(e) => {
//                 eprintln!("Failed to store board state: {}", e);
//                 HttpResponse::InternalServerError().json("Failed to store board state")
//             }
//         }
//     }
// }

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let state = Arc::new(Mutex::new(AppState {
        counter: Mutex::new(0),
    }));

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(state.clone()))
            .route("/handle_move", web::post().to(handle_move))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
// async fn main() -> std::io::Result<()> {
//     // MongoDB setup
//     let client_options = ClientOptions::parse("mongodb://localhost:27017").await.unwrap();
//     let client = Client::with_options(client_options).unwrap();
//     let database = client.database("chess_db");
//     let collection = database.collection::<bson::Document>("moves");

//     // let app_state = Arc::new(Mutex::new(AppState {
//     //     mongo_collection: collection,
//     // }));

//     HttpServer::new(move || {
//         App::new()
//             .app_data(web::Data::new(app_state.clone()))
//             .route("/api/move", web::post().to(handle_move))
//     })
//     .bind("127.0.0.1:8080")?
//     .run()
//     .await
// }