# importing this data from chesstree.net, using the master's opening tree.
class openingMoveTree:
    # I'm having it set up so certain boards correspond to certain moves, and it doesn't matter how the game got there.
    def __init__(self, wPieces, wIndexes, bPieces, bIndexes, turn):
        self.wPieces = []
        self.wIndexes = {}
        self.bPieces = []
        self.bIndexes = {}
        for i in wPieces:
            self.wPieces.append(i)
            self.wIndexes[i] = wIndexes[i]
        for i in bPieces:
            self.bPieces.append(i)
            self.bIndexes[i] = bIndexes[i]

    def deepCompare(self, other) -> bool:
        return self.wPieces == other.wPieces and self.wIndexes == other.wIndexes and self.bPieces == other.bPieces and self.bIndexes == other.bIndexes


class reference:
    def __init__(self) -> None:
        self.Board = []
        self.move = []
        # I'm using the code to play against myself and produce opening tree nodes, then the corresponding index is the AI's move
        turn0 = openingMoveTree(['r1', 'r2', 'b1', 'b2', 'k2', 'K', 'q', 'k1', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'], ['r1', 'r2', 'b1', 'b2', 'k2', 'K', 'q', 'k1', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'], {
                                'r1': 70, 'r2': 77, 'b1': 72, 'b2': 75, 'k1': 71, 'k2': 76, 'K': 74, 'q': 73, 'p1': 60, 'p2': 61, 'p3': 62, 'p4': 63, 'p5': 64, 'p6': 65, 'p7': 66, 'p8': 67}, {'r1': 0, 'r2': 7, 'b1': 2, 'b2': 5, 'k1': 1, 'k2': 6, 'K': 4, 'q': 3, 'p1': 10, 'p2': 11, 'p3': 12, 'p4': 13, 'p5': 14, 'p6': 15, 'p7': 16, 'p8': 17})
        self.Board.append(turn0)
        self.move.append(["p5", 44])
        whiteOpening1 = (['r1', 'r2', 'b1', 'b2', 'k2', 'K', 'q', 'k1', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'], ['r1', 'r2', 'b1', 'b2', 'k2', 'K', 'q', 'k1', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'], {
                         'r1': 70, 'r2': 77, 'b1': 72, 'b2': 75, 'k1': 71, 'k2': 76, 'K': 74, 'q': 73, 'p1': 60, 'p2': 61, 'p3': 62, 'p4': 63, 'p5': 44, 'p6': 65, 'p7': 66, 'p8': 67}, {'r1': 0, 'r2': 7, 'b1': 2, 'b2': 5, 'k1': 1, 'k2': 6, 'K': 4, 'q': 3, 'p1': 10, 'p2': 11, 'p3': 12, 'p4': 13, 'p5': 14, 'p6': 15, 'p7': 16, 'p8': 17})
        self.Board.append(whiteOpening1)
        self.move.append(["p3", 32])
        b2 = openingMoveTree(['r1', 'r2', 'b1', 'b2', 'k2', 'K', 'q', 'k1', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'], ['r1', 'r2', 'b1', 'b2', 'k2', 'K', 'q', 'k1', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'], {
                             'r1': 70, 'r2': 77, 'b1': 72, 'b2': 75, 'k1': 71, 'k2': 76, 'K': 74, 'q': 73, 'p1': 60, 'p2': 61, 'p3': 62, 'p4': 63, 'p5': 44, 'p6': 65, 'p7': 66, 'p8': 67}, {'r1': 0, 'r2': 7, 'b1': 2, 'b2': 5, 'k1': 1, 'k2': 6, 'K': 4, 'q': 3, 'p1': 10, 'p2': 11, 'p3': 32, 'p4': 13, 'p5': 14, 'p6': 15, 'p7': 16, 'p8': 17})
        self.Board.append(b2)
        self.move.append(["k2", 55])
        w2 = openingMoveTree(['r1', 'r2', 'b1', 'b2', 'k2', 'K', 'q', 'k1', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'], ['r1', 'r2', 'b1', 'b2', 'k2', 'K', 'q', 'k1', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'], {
                             'r1': 70, 'r2': 77, 'b1': 72, 'b2': 75, 'k1': 71, 'k2': 55, 'K': 74, 'q': 73, 'p1': 60, 'p2': 61, 'p3': 62, 'p4': 63, 'p5': 44, 'p6': 65, 'p7': 66, 'p8': 67}, {'r1': 0, 'r2': 7, 'b1': 2, 'b2': 5, 'k1': 1, 'k2': 6, 'K': 4, 'q': 3, 'p1': 10, 'p2': 11, 'p3': 32, 'p4': 13, 'p5': 14, 'p6': 15, 'p7': 16, 'p8': 17})
        self.Board.append(w2)
        self.move.append(["p4", 23])
        b3 = openingMoveTree(['r1', 'r2', 'b1', 'b2', 'k2', 'K', 'q', 'k1', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'], ['r1', 'r2', 'b1', 'b2', 'k2', 'K', 'q', 'k1', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'], {
                             'r1': 70, 'r2': 77, 'b1': 72, 'b2': 75, 'k1': 71, 'k2': 55, 'K': 74, 'q': 73, 'p1': 60, 'p2': 61, 'p3': 62, 'p4': 63, 'p5': 44, 'p6': 65, 'p7': 66, 'p8': 67}, {'r1': 0, 'r2': 7, 'b1': 2, 'b2': 5, 'k1': 1, 'k2': 6, 'K': 4, 'q': 3, 'p1': 10, 'p2': 11, 'p3': 32, 'p4': 23, 'p5': 14, 'p6': 15, 'p7': 16, 'p8': 17})
        self.Board.append(b3)
        self.move.append(["p4", 43])
        w3 = openingMoveTree(['r1', 'r2', 'b1', 'b2', 'k2', 'K', 'q', 'k1', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'], ['r1', 'r2', 'b1', 'b2', 'k2', 'K', 'q', 'k1', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'], {
                             'r1': 70, 'r2': 77, 'b1': 72, 'b2': 75, 'k1': 71, 'k2': 55, 'K': 74, 'q': 73, 'p1': 60, 'p2': 61, 'p3': 62, 'p4': 43, 'p5': 44, 'p6': 65, 'p7': 66, 'p8': 67}, {'r1': 0, 'r2': 7, 'b1': 2, 'b2': 5, 'k1': 1, 'k2': 6, 'K': 4, 'q': 3, 'p1': 10, 'p2': 11, 'p3': 32, 'p4': 23, 'p5': 14, 'p6': 15, 'p7': 16, 'p8': 17})
        self.Board.append(w3)
        self.move.append(["p3", 43])
        b4 = openingMoveTree(['r1', 'r2', 'b1', 'b2', 'k2', 'K', 'q', 'k1', 'p1', 'p2', 'p3', 'p5', 'p6', 'p7', 'p8'], ['r1', 'r2', 'b1', 'b2', 'k2', 'K', 'q', 'k1', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'], {
                             'r1': 70, 'r2': 77, 'b1': 72, 'b2': 75, 'k1': 71, 'k2': 55, 'K': 74, 'q': 73, 'p1': 60, 'p2': 61, 'p3': 62, 'p5': 44, 'p6': 65, 'p7': 66, 'p8': 67}, {'r1': 0, 'r2': 7, 'b1': 2, 'b2': 5, 'k1': 1, 'k2': 6, 'K': 4, 'q': 3, 'p1': 10, 'p2': 11, 'p3': 43, 'p4': 23, 'p5': 14, 'p6': 15, 'p7': 16, 'p8': 17})
        self.Board.append(b4)
        self.move.append(["k2", 43])


def openingMoveGenerator(game: openingMoveTree):
    moves = reference()
    for i in range(0, len(moves.Board), 1):
        if game.deepCompare(moves.Board[i]) == True:
            return moves[i]
    return []
