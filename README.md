Chess Game with AI
This is a project to create a chess game with an AI opponent. The game has been implemented using the object-oriented programming paradigm in Python. The AI opponent uses the minimax algorithm along with alpha-beta pruning to make its moves.

Requirements
Python 3.x
numpy
copy
Getting Started
Clone the repository to your local machine.

shell
Copy code
$ git clone https://github.com/<username>/chess-game-with-ai.git
Move into the project directory

shell
Copy code
$ cd chess-game-with-ai
Running the Game
The game can be run by executing the main.py file.

css
Copy code
$ python main.py
Game Implementation
The game has been implemented using the following classes:

board class to represent the chess board.
piece class to represent individual chess pieces.
treeNode class to represent the nodes of the game tree for the AI to use in its decision making.
The main.py file initializes the game and contains the main game loop where moves are made by the player and AI opponent.

The AI's move is determined by calling the search function in the main.py file, which uses the minimax algorithm along with alpha-beta pruning to make its move.

Contributing
We welcome contributions to this project. To contribute, please follow these steps:

Fork the repository.
Clone the repository to your local machine.
Create a new branch with a descriptive name for your changes.
Make the changes and commit them to your branch.
Push the branch to your fork.
Submit a pull request.
License
This project is licensed under the MIT License. See LICENSE for details.



