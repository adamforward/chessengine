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
use crate::mongo_repo::{AppState, get_last_game_state, insert_game_state, delete_game_state};
use crate::server_processing::{process_server_response};
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Move{
    pgn:String,
    user_id:String
}


async fn handle_move(
    move_data: web::Json<Move>,
    data: web::Data<Arc<Mutex<AppState>>>,
) -> impl Responder {
    let pgn = move_data.pgn.clone();
    let user_id = move_data.user_id.clone();
    let collection = &data.lock().await.mongo_collection;

    let last_game = get_last_game_state(collection, &user_id).await;

    let result = process_server_response(pgn.clone(), last_game);

    if result.pgn.is_empty() {
        match delete_game_state(collection, &user_id).await {
            Ok(_) => HttpResponse::Ok().json(result.pgn),
            Err(e) => {
                eprintln!("Failed to delete board state: {}", e);
                HttpResponse::InternalServerError().json("Failed to delete board state")
            }
        }
    } else {
        match insert_game_state(collection, &user_id, &result.b).await {
            Ok(_) => HttpResponse::Ok().json(result.pgn),
            Err(e) => {
                eprintln!("Failed to store board state: {}", e);
                HttpResponse::InternalServerError().json("Failed to store board state")
            }
        }
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // MongoDB setup
    let client_options = ClientOptions::parse("mongodb://localhost:27017").await.unwrap();
    let client = Client::with_options(client_options).unwrap();
    let database = client.database("chess_db");
    let collection = database.collection::<bson::Document>("moves");

    let app_state = Arc::new(Mutex::new(AppState {
        mongo_collection: collection,
    }));

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(app_state.clone()))
            .route("/api/move", web::post().to(handle_move))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}