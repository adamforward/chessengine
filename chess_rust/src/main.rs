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
use crate::ai_functions::{test_neural_networks};
use crate::mongo_repo::{AppState, get_last_game_state, insert_game_state, delete_game_state};
use crate::server_processing::{process_server_response};
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Move{
    pgn:String,
    user_id:String
}

// struct AppState {
//     counter: Mutex<usize>,
// }

// async fn handle_move(
//     move_data: web::Json<Move>,
//     data: web::Data<Arc<Mutex<AppState>>>,
// ) -> impl Responder {
//     let moves = vec![
//         "e4", "Nf6", "e5", "Nd5", "d4", "d6", "Nf3", "Bg4", "Bc4", "e6", 
//         "O-O", "Nb6", "Be2", "Be7", "h3", "Bh5", "Bf4", "Nc6", "c3", "O-O", 
//         "Nbd2", "d5", "b4", "a5", "a3", "Qd7", "Qc2", "Bg6", "Bd3", "Rfc8", 
//         "Rfb1", "Bf8", "h4", "Ne7", "g3", "Qa4", "Ne1", "Qxc2", "Bxc2", 
//         "Bxc2", "Nxc2", "Na4", "Rb3", "b6", "Kf1", "c5", "bxc5", "bxc5", 
//         "dxc5", "Rxc5", "Nb1", "Rac8", "Be3", "Rc4", "Bd4", "Nc6", "Rb5", 
//         "Nxd4", "Nxd4", "Nxc3", "Nxc3", "Rxd4", "Ne2", "Ra4", "Ke1", "Rxa3", 
//         "Rab1", "Bb4+", "Kf1", "Rd3"
//     ];

//     let state = data.lock().await;

//     let mut counter = state.counter.lock().await;
//     let index = *counter;

//     if index >= moves.len() {
//         return HttpResponse::BadRequest().json("Index out of bounds");
//     }

//     let result_move = moves.get(index).unwrap_or(&"").clone();

//     // Increment the counter for the next request
//     *counter += 2;

//     // Return the move at the current index
//     HttpResponse::Ok().json(result_move)
// }
// async fn handle_move(
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

fn main() {
    println!("Current directory: {:?}", std::env::current_dir());

    let mut chess_cnn_checkpoint_epoch_15_score = 0;
    let mut chess_cnn_model_final_3x3x3x3_f16_score = 0;
    let mut chess_cnn_model_final_f16_3_convs_score = 0;
    let mut chess_cnn_model_final_f16_score = 0;
    let mut chess_cnn_model_final_3xpool2xconv_f32_score = 0;

    // Define the absolute paths to your models
    let chess_cnn_checkpoint_epoch_15 = "/Users/adamforward/Desktop/chess/chess_rust/src/chess_cnn_checkpoint_epoch_15.pt";
    let chess_cnn_model_final_3x3x3x3_f16 = "/Users/adamforward/Desktop/chess/chess_rust/src/chess_cnn_model_final_3x3x3x3_f16.pt";
    let chess_cnn_model_final_f16_3_convs = "/Users/adamforward/Desktop/chess/chess_rust/src/chess_cnn_model_final_3xpool2xconv_f16.pt";
    let chess_cnn_model_final_f16 = "/Users/adamforward/Desktop/chess/chess_rust/src/chess_cnn_model_final_f16_3_convs.pt";
    let chess_cnn_model_final_3xpool2xconv_f32 = "/Users/adamforward/Desktop/chess/chess_rust/src/chess_cnn_model_final_f16.pt";

    for _ in 0..100 {
        // Test chess_cnn_checkpoint_epoch_15 against chess_cnn_model_final_3x3x3x3_f16
        let game1 = test_neural_networks(chess_cnn_checkpoint_epoch_15, chess_cnn_model_final_3x3x3x3_f16);
        chess_cnn_checkpoint_epoch_15_score += game1;
        chess_cnn_model_final_3x3x3x3_f16_score -= game1;
        let game2 = test_neural_networks(chess_cnn_model_final_3x3x3x3_f16, chess_cnn_checkpoint_epoch_15);
        chess_cnn_checkpoint_epoch_15_score -= game2;
        chess_cnn_model_final_3x3x3x3_f16_score += game2;

        // Test chess_cnn_checkpoint_epoch_15 against chess_cnn_model_final_f16_3_convs
        let game3 = test_neural_networks(chess_cnn_checkpoint_epoch_15, chess_cnn_model_final_f16_3_convs);
        chess_cnn_checkpoint_epoch_15_score += game3;
        chess_cnn_model_final_f16_3_convs_score -= game3;
        let game4 = test_neural_networks(chess_cnn_model_final_f16_3_convs, chess_cnn_checkpoint_epoch_15);
        chess_cnn_checkpoint_epoch_15_score -= game4;
        chess_cnn_model_final_f16_3_convs_score += game4;

        // Test chess_cnn_checkpoint_epoch_15 against chess_cnn_model_final_f16
        let game5 = test_neural_networks(chess_cnn_checkpoint_epoch_15, chess_cnn_model_final_f16);
        chess_cnn_checkpoint_epoch_15_score += game5;
        chess_cnn_model_final_f16_score -= game5;
        let game6 = test_neural_networks(chess_cnn_model_final_f16, chess_cnn_checkpoint_epoch_15);
        chess_cnn_checkpoint_epoch_15_score -= game6;
        chess_cnn_model_final_f16_score += game6;

        // Test chess_cnn_checkpoint_epoch_15 against chess_cnn_model_final_3xpool2xconv_f32
        let game7 = test_neural_networks(chess_cnn_checkpoint_epoch_15, chess_cnn_model_final_3xpool2xconv_f32);
        chess_cnn_checkpoint_epoch_15_score += game7;
        chess_cnn_model_final_3xpool2xconv_f32_score -= game7;
        let game8 = test_neural_networks(chess_cnn_model_final_3xpool2xconv_f32, chess_cnn_checkpoint_epoch_15);
        chess_cnn_checkpoint_epoch_15_score -= game8;
        chess_cnn_model_final_3xpool2xconv_f32_score += game8;

        // Test chess_cnn_model_final_3x3x3x3_f16 against chess_cnn_model_final_f16_3_convs
        let game9 = test_neural_networks(chess_cnn_model_final_3x3x3x3_f16, chess_cnn_model_final_f16_3_convs);
        chess_cnn_model_final_3x3x3x3_f16_score += game9;
        chess_cnn_model_final_f16_3_convs_score -= game9;
        let game10 = test_neural_networks(chess_cnn_model_final_f16_3_convs, chess_cnn_model_final_3x3x3x3_f16);
        chess_cnn_model_final_3x3x3x3_f16_score -= game10;
        chess_cnn_model_final_f16_3_convs_score += game10;

        // Test chess_cnn_model_final_3x3x3x3_f16 against chess_cnn_model_final_f16
        let game11 = test_neural_networks(chess_cnn_model_final_3x3x3x3_f16, chess_cnn_model_final_f16);
        chess_cnn_model_final_3x3x3x3_f16_score += game11;
        chess_cnn_model_final_f16_score -= game11;
        let game12 = test_neural_networks(chess_cnn_model_final_f16, chess_cnn_model_final_3x3x3x3_f16);
        chess_cnn_model_final_3x3x3x3_f16_score -= game12;
        chess_cnn_model_final_f16_score += game12;

        // Test chess_cnn_model_final_3x3x3x3_f16 against chess_cnn_model_final_3xpool2xconv_f32
        let game13 = test_neural_networks(chess_cnn_model_final_3x3x3x3_f16, chess_cnn_model_final_3xpool2xconv_f32);
        chess_cnn_model_final_3x3x3x3_f16_score += game13;
        chess_cnn_model_final_3xpool2xconv_f32_score -= game13;
        let game14 = test_neural_networks(chess_cnn_model_final_3xpool2xconv_f32, chess_cnn_model_final_3x3x3x3_f16);
        chess_cnn_model_final_3x3x3x3_f16_score -= game14;
        chess_cnn_model_final_3xpool2xconv_f32_score += game14;

        // Test chess_cnn_model_final_f16_3_convs against chess_cnn_model_final_f16
        let game15 = test_neural_networks(chess_cnn_model_final_f16_3_convs, chess_cnn_model_final_f16);
        chess_cnn_model_final_f16_3_convs_score += game15;
        chess_cnn_model_final_f16_score -= game15;
        let game16 = test_neural_networks(chess_cnn_model_final_f16, chess_cnn_model_final_f16_3_convs);
        chess_cnn_model_final_f16_3_convs_score -= game16;
        chess_cnn_model_final_f16_score += game16;

        // Test chess_cnn_model_final_f16_3_convs against chess_cnn_model_final_3xpool2xconv_f32
        let game17 = test_neural_networks(chess_cnn_model_final_f16_3_convs, chess_cnn_model_final_3xpool2xconv_f32);
        chess_cnn_model_final_f16_3_convs_score += game17;
        chess_cnn_model_final_3xpool2xconv_f32_score -= game17;
        let game18 = test_neural_networks(chess_cnn_model_final_3xpool2xconv_f32, chess_cnn_model_final_f16_3_convs);
        chess_cnn_model_final_f16_3_convs_score -= game18;
        chess_cnn_model_final_3xpool2xconv_f32_score += game18;

        // Test chess_cnn_model_final_f16 against chess_cnn_model_final_3xpool2xconv_f32
        let game19 = test_neural_networks(chess_cnn_model_final_f16, chess_cnn_model_final_3xpool2xconv_f32);
        chess_cnn_model_final_f16_score += game19;
        chess_cnn_model_final_3xpool2xconv_f32_score -= game19;
        let game20 = test_neural_networks(chess_cnn_model_final_3xpool2xconv_f32, chess_cnn_model_final_f16);
        chess_cnn_model_final_f16_score -= game20;
        chess_cnn_model_final_3xpool2xconv_f32_score += game20;
    }
    println!("chess_cnn_model_final_f16 score {}", chess_cnn_model_final_f16_score);
    println!("chess_cnn_checkpoint_epoch_15_score {}",chess_cnn_checkpoint_epoch_15_score);
    println!("chess_cnn_model_final_3x3x3x3_f16_score {}", chess_cnn_model_final_3x3x3x3_f16_score);
    println!("chess_cnn_model_final_f16_3_convs_score {}", chess_cnn_model_final_f16_3_convs_score);
    println!("chess_cnn_model_final_3xpool2xconv_f32_score {}", chess_cnn_model_final_3xpool2xconv_f32_score);
}

// #[actix_web::main]
// async fn main() -> std::io::Result<()> {
//     let state = Arc::new(Mutex::new(AppState {
//         counter: Mutex::new(0),
//     }));

//     HttpServer::new(move || {
//         App::new()
//             .app_data(web::Data::new(state.clone()))
//             .route("/handle_move", web::post().to(handle_move))
//     })
//     .bind("127.0.0.1:8080")?
//     .run()
//     .await
//}
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