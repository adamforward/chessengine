AF Chess Engine
AF Chess Engine is a chess-playing application powered by a Rust backend and a neural network model. The project is deployed at afchessengine.com, where you can experience the chess engine in action.

Project Overview
This repository contains the code for the backend of the AF Chess Engine. The backend is implemented in Rust and uses a neural network (nn7.pt) along with an opening book (opening_book.json) to enhance its chess-playing capabilities.

Note: This repository does not include the large files necessary to run the backend. These files are essential for the engine's functionality.

nn7.pt: The neural network model used for move evaluation.
opening_book.json: The precomputed opening book for optimized gameplay.
If you need access to these files, please email adamforward19@gmail.com for more information.

Deployment Details
Website: https://afchessengine.com
Backend: Rust-based API hosted at https://api.afchessengine.com.
The backend and frontend communicate seamlessly to provide a smooth and interactive chess-playing experience.

Getting Started
Prerequisites
To run the backend locally, ensure you have the following installed:

Rust (latest stable version recommended)
Docker (for containerized deployment)
Required large files (nn7.pt and opening_book.json)
Running Locally
Clone the repository:

bash
Copy code
git clone https://github.com/your-repo-name.git
cd chess_rust
Obtain the required large files (nn7.pt and opening_book.json) by emailing adamforward19@gmail.com. Place these files in the appropriate directory.

Build and run the backend:

bash
Copy code
cargo build --release
cargo run
The server will be accessible at http://127.0.0.1:3000. You can use curl or any HTTP client to test the endpoints.

Using Docker
Build the Docker image:

bash
Copy code
docker build -t chess_rust_backend .
Run the container:

bash
Copy code
docker run -d -p 3000:3000 --name chess_rust_backend chess_rust_backend
The server will now be available at http://127.0.0.1:3000.

API Endpoints
/move (POST)
Description: Processes a move request and returns the next best move.
Request Body:
json
{
  "player_team_is_white": false,
  "uci": [] (array of strings)
}
Response:
json
Copy code
"e2e4"
About
For more details about the AF Chess Engine, visit the "About" section of the website.
