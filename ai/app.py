from flask import Flask, request, jsonify
import chess
app = Flask(__name__)


@app.route('/start', methods=['POST'])
def startGame():
    data = request.get_json()
    team = data.get('team')
    difficulty = data.get('difficulty')


@app.route('/move', methods=['POST'])
def response():
