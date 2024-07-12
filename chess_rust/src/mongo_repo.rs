use mongodb::{bson::{doc, Document}, options::ClientOptions, Client, Collection};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::types::{Board};

pub struct MongoRepo {
    col: Collection<UserGame>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct UserGame {
    pub user_id: String,
    pub board: Board,
}

pub struct AppState {
    pub mongo_collection: Collection<bson::Document>,
}
impl MongoRepo {
    pub async fn init() -> Self {
        let client_options = ClientOptions::parse("mongodb://localhost:27017").await.unwrap();
        let client = Client::with_options(client_options).unwrap();
        let db = client.database("chess_game");
        let col = db.collection::<UserGame>("games");

        MongoRepo { col }
    }

    pub async fn create_game(&self, user_game: UserGame) -> mongodb::error::Result<()> {
        self.col.insert_one(user_game, None).await.map(|_| ())
    }

    pub async fn get_game(&self, user_id: &str) -> mongodb::error::Result<Option<UserGame>> {
        self.col.find_one(doc! { "user_id": user_id }, None).await
    }

    pub async fn delete_game(&self, user_id: &str) -> mongodb::error::Result<()> {
        self.col.delete_one(doc! { "user_id": user_id }, None).await.map(|_| ())
    }
}

pub async fn get_last_game_state(collection: &Collection<Document>, user_id: &str) -> Option<Board> {
    let filter = doc! { "user_id": user_id };
    match collection.find_one(filter, None).await {
        Ok(Some(doc)) => {
            if let Ok(board) = bson::from_document::<Board>(doc) {
                Some(board)
            } else {
                None
            }
        }
        _ => None,
    }
}

pub async fn insert_game_state(
    collection: &Collection<Document>,
    user_id: &str,
    board: &Board,
) -> Result<(), mongodb::error::Error> {
    let mut board_doc = bson::to_document(board).unwrap();
    board_doc.insert("user_id", user_id);

    match collection.insert_one(board_doc, None).await {
        Ok(_) => Ok(()),
        Err(e) => Err(e),
    }
}

pub async fn delete_game_state(
    collection: &Collection<Document>,
    user_id: &str,
) -> Result<(), mongodb::error::Error> {
    let filter = doc! { "user_id": user_id };

    match collection.delete_one(filter, None).await {
        Ok(_) => Ok(()),
        Err(e) => Err(e),
    }
}