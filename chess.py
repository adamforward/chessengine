import copy
# from chess import genOpeningTree
# opening tree has data from


def mapping(n: int) -> str:  # maps indexes into the standard form, makes it a little easier to compare my programs generated outputs to a real board
    if n < 0:
        n = n*-1
    elif n == 99:
        return "KS"
    elif n == 100:
        return "QS"
    else:
        ret = ""
        rowMap = {0: "8", 1: "7", 2: "6", 3: "5",
                  4: "4", 5: "3", 6: "2", 7: "1"}
        colMap = {0: "A", 1: "B", 2: "C", 3: "D",
                  4: "E", 5: "F", 6: "G", 7: "H"}
        col = n % 10
        row = n//10
        a = colMap[col]
        b = rowMap[row]
        ret += a
        ret += b
        return ret


# need to construct two opening trees, 1 for white and one for black

class primeMethods:
    def primes(col: int) -> int:
        primes = (2, 3, 5, 7, 11, 13, 17, 19)
        return primes[col]

    def primes1(piece: str) -> int:
        primes1 = {"p1": 2, "p2": 3, "p3": 5, "p4": 7,
                   "p5": 11, "p6": 13, "p7": 17, "p8": 19}
        return primes1[piece]

    def queenMapping(pawn: str) -> str:
        newPMap = {"p1": "q1", "p2": "q2", "p3": "q3", "p4": "q4",
                   "p5": "q5", "p6": "q6", "p7": "q7", "p8": "q8"}
        return newPMap[pawn]


class piece:
    def __init__(self, val: int, kind, team):
        self.val = val
        self.kind = kind
        self.team = team

    def copy(self):
        return piece(self.val, self.kind, self.team)
# this tuple represents data for the pawns that skipped ahead two based on columns from the start (needed for en pessant captures).
# since the data in the board class is copied each time a simiulated move is made in the searching algorithm, I decided to cut this info down from
# 2 vectors (len()==8) in the class to two integers using this data.
# Using a


class board:
    def __init__(self):
        self.movesLog = []
        self.inCheckStored = False
        self.AIteamIsWhite = True
        self.inPlay = True
        self.fullBoard = [[piece(500, 'r', 'b'), piece(300, 'k', 'b'), piece(300, 'b', 'b'), piece(900, 'q', 'b'), piece(0, 'K', 'b'), piece(300, 'b', 'b'),
                           piece(300, 'k', 'b'), piece(500, 'r', 'b')], [piece(100, 'p', 'b'), piece(100, 'p', 'b'), piece(100, 'p', 'b'), piece(100, 'p', 'b'), piece(100, 'p', 'b'),
                                                                         piece(100, 'p', 'b'), piece(100, 'p', 'b'), piece(100, 'p', 'b')], [piece(0, 'n', 'n'), piece(0, 'n', 'n'), piece(0, 'n', 'n'),
                                                                                                                                             piece(0, 'n', 'n'), piece(0, 'n', 'n'), piece(0, 'n', 'n'), piece(0, 'n', 'n'), piece(0, 'n', 'n')],
                          [piece(0, 'n', 'n'), piece(0, 'n', 'n'), piece(0, 'n', 'n'), piece(0, 'n', 'n'), piece(
                              0, 'n', 'n'), piece(0, 'n', 'n'), piece(0, 'n', 'n'), piece(0, 'n', 'n')],
                          [piece(0, 'n', 'n'), piece(0, 'n', 'n'), piece(0, 'n', 'n'), piece(0, 'n', 'n'), piece(
                              0, 'n', 'n'), piece(0, 'n', 'n'), piece(0, 'n', 'n'), piece(0, 'n', 'n'),],
                          [piece(0, 'n', 'n'), piece(0, 'n', 'n'), piece(0, 'n', 'n'), piece(0, 'n', 'n'), piece(
                              0, 'n', 'n'), piece(0, 'n', 'n'), piece(0, 'n', 'n'), piece(0, 'n', 'n')],
                          [piece(100, 'p', 'w'), piece(100, 'p', 'w'), piece(100, 'p', 'w'), piece(100, 'p', 'w'), piece(100, 'p', 'w'),
                           piece(100, 'p', 'w'), piece(100, 'p', 'w'), piece(100, 'p', 'w')],
                          [piece(500, 'r', 'w'), piece(300, 'k', 'w'), piece(300, 'b', 'w'), piece(900, 'q', 'w'), piece(0, 'K', 'w'), piece(300, 'b', 'w'),
                           piece(300, 'k', 'w'), piece(500, 'r', 'w')]]  # this is used to generate available moves and keep track of everything that's written.
        self.turn = 0  # every time turn=1
        self.blackIndexes = {"r1": 0, "r2": 7, "b1": 2, "b2": 5, "k1": 1, "k2": 6, "K": 4, "q": 3,
                             "p1": 10, "p2": 11, "p3": 12, "p4": 13, "p5": 14, "p6": 15, "p7": 16, "p8": 17}
        self.whiteIndexes = {"r1": 70, "r2": 77, "b1": 72, "b2": 75, "k1": 71, "k2": 76, "K": 74, "q": 73,
                             "p1": 60, "p2": 61, "p3": 62, "p4": 63, "p5": 64, "p6": 65, "p7": 66, "p8": 67}
        self.blackIToP = {0: "r1", 7: "r2", 2: "b1", 5: "b2", 1: "k1", 6: "k2", 4: "K", 3: "q",
                          10: "p1", 11: "p2", 12: "p3", 13: "p4", 14: "p5", 15: "p6", 16: "p7", 17: "p8"}  # important in cutting down on runtime in many methods used below.
        self.whiteIToP = {70: "r1", 77: "r2", 72: "b1", 75: "b2", 71: "k1", 76: "k2", 74: "K", 73: "q",
                          60: "p1", 61: "p2", 62: "p3", 63: "p4", 64: "p5", 65: "p6", 66: "p7", 67: "p8"}  # index to pieces
        self.blackPoints = 3800
        self.whitePoints = 3800
        self.whitePieces = ["r1", "r2", "b1", "b2", "k2", "K", "q",
                            "k1", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]
        self.blackPieces = ["r1", "r2", "b1", "b2", "k2", "K", "q",
                            "k1", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]
        self.whiteaVailableMoves = {"r1": [], "r2": [], "b1": [], "b2": [], "k2": [], "K": [], "q": [
        ], "k1": [], "p1": [], "p2": [], "p3": [], "p4": [], "p5": [], "p6": [], "p7": [], "p8": []}
        self.blackAvailableMoves = {"r1": [], "r2": [], "b1": [], "b2": [], "k2": [], "K": [], "q": [
        ], "k1": [], "p1": [], "p2": [], "p3": [], "p4": [], "p5": [], "p6": [], "p7": [], "p8": []}
        self.whitePrime = 1
        self.blackPrime = 1
        self.whitePrime1 = 1
        self.blackPrime1 = 1
        # self.lastTurnSkipW=False
        # self.lastTurnSkipB=False#for en pessant captures
        # self.wHasMovedKing=False
        # self.wHasMovedR1=False
        # self.wHasMovedR2=False
        # self.bHasMovedKing=False
        # self.bHasMovedR1=False
        # self.bHasMovedR2=False
        # this field represents all of the commented out lines of code above.
        self.prime2 = 1
        # if prime2%2==0, white has moved R1, %3==0 is white R2, %5==0 is white king, next 3 prime #s correspond to black
        self.AIAdvantage = 0

    # changing this function up, dividing it into different sections for pieces

    def generateAvailableMoves(self, row: int, col: int):
        # returns an integer, /10=row, %10=col
        if self.fullBoard[row][col].kind == 'p':
            # only pawns vary in the indexes they can move to
            if self.fullBoard[row][col].team == 'w':
                return self.generatePawnMovesw(row, col)
            else:
                return self.generatePawnMovesb(row, col)
        elif self.fullBoard[row][col].kind == 'k':
            return self.knightMoves(row, col)
        elif self.fullBoard[row][col].kind == 'r':
            return self.rookMoves(row, col)
        elif self.fullBoard[row][col].kind == 'b':
            return self.bishopMoves(row, col)
        elif self.fullBoard[row][col].kind == 'K':
            return self.kingMoves(row, col)
        elif self.fullBoard[row][col].kind == 'q':
            re = self.rookMoves(row, col)
            re.extend(self.bishopMoves(row, col))
            return re
        else:  # gen available moves looks good
            return None

    def generatePawnMovesw(self, row, col):  # white starts at row 6
        re = []
        if self.fullBoard[row-1][col].team == 'n':
            # print("CHECK")
         # boundary conditions never met, because in the move function if it goes to the end it becomes queen
            re.append(10*(row-1)+col)  # move forward 1
            # skipping first
            if row == 6 and self.fullBoard[row-2][col].team == 'n':
                re.append(40+col)
        if 1 <= col:
            if self.fullBoard[row-1][col-1].team == 'b' and 0 <= col and col <= 7:
                re.append(10*(row-1)+col-1)
        if col <= 6:
            if self.fullBoard[row-1][col+1].team == 'b':
                re.append(10*(row-1)+col+1)
        if row == 3 and col >= 1:
            # avoid out of bounds error, no -1th index of fullboard
            if self.fullBoard[3][col-1].team == 'b' and self.fullBoard[3][col-1].kind == 'p' and self.fullBoard[2][col-1].team == 'n':
                if self.blackPrime % primeMethods.primes(col-1) == 0 and self.whitePrime1 % primeMethods.primes1(self.whiteIToP[row*10+col]) == 0:
                    ep = 20+col-1  # check right and left if it's not on the edge
                    re.append(ep)
        if row == 3 and col <= 6:
            # en pessant, same idea but col index is 6
            if self.fullBoard[3][col+1].team == 'b' and self.fullBoard[3][col+1].kind == 'p':
                if self.blackPrime % primeMethods.primes(col+1) == 0 and self.whitePrime1 % primeMethods.primes1(self.whiteIToP[row*10+col]) == 0:
                    ep = 20+col+1
                    # check right and left if it's not on the edge
                    re.append(ep)
        return re

    # black starts at row 1, index goes up. only difference is the reference point of 3 for en pessant changes to 5, row+=1 instead of -=1
    def generatePawnMovesb(self, row, col):
        re = []
        if self.fullBoard[row+1][col].team == 'n':
            # print("CHECK")
            # boundary conditions never met, because in the move function if it goes to the end it becomes queen
            re.append(10*row+10+col)  # move forward 1
            # skipping first
            if row == 1 and self.fullBoard[row+2][col].team == 'n':
                re.append(30+col)
        if 1 <= col:
            if self.fullBoard[row+1][col-1].team == 'w' and 0 <= col and col <= 7:
                if self.fullBoard[row+1][col-1].kind == 'K':
                    re.append(10*row+10+col-1)
        if col <= 6:
            if self.fullBoard[row+1][col+1].team == 'w':
                re.append(10*(row+1)+col+1)
        if row == 4 and col >= 1:
            # en pessant, avoid out of bounds exception
            if self.fullBoard[4][col-1].team == 'w' and self.fullBoard[4][col-1].kind == 'p' and self.fullBoard[5][col-1].team == 'n':
                if self.whitePrime % primeMethods.primes(col-1) == 0 and self.blackPrime1 % primeMethods.primes1(self.blackIToP[row*10+col]) == 0:
                    r = 50+col-1
                    re.append(r)
        if row == 4 and col <= 6:
            # en pessant, same idea but col index is 6
            if self.fullBoard[4][col+1].team == 'w' and self.fullBoard[4][col+1].kind == 'p' and self.fullBoard[5][col+1].team == 'n':
                # en pessant, same idea but col index is 6
                if self.whitePrime % primeMethods.primes(col+1) == 0 and self.blackPrime1 % primeMethods.primes1(self.blackIToP[row*10+col]) == 0:
                    r = 50+col+1
                    re.append(r)
        return re

    def knightMoves(self, row: int, col: int):  # debugged
        re = []
        team = self.fullBoard[row][col].team
        for i in (1, -1):
            for j in (2, -2):
                moveRow = row+i
                moveCol = col+j
                if moveRow < 0 or moveRow >= 8 or moveCol < 0 or moveCol >= 7:
                    continue
                if self.fullBoard[moveRow][moveCol].team != team:
                    re.append(moveRow*10+moveCol)
        for i in (1, -1):
            for j in (2, -2):
                moveCol = col+i
                moveRow = row+j
                if moveRow < 0 or moveRow >= len(self.fullBoard) or moveCol < 0 or moveCol >= len(self.fullBoard[moveRow]):
                    continue
                if self.fullBoard[moveRow][moveCol].team != team:
                    re.append(moveRow*10+moveCol)
        return re

    def bishopMoves(self, row: int, col: int):
        re = []
        team = self.fullBoard[row][col].team
        for i in (1, -1):

            for j in (1, -1):
                moveRow = row+i
                moveCol = col+j
                while 0 <= moveRow <= 7 and 0 <= moveCol <= 7 and self.fullBoard[moveRow][moveCol].team != team:
                    re.append(moveRow*10+moveCol)
                    if self.fullBoard[moveRow][moveCol].team != 'n':
                        break
                    moveRow += i
                    moveCol += j
        return re

    # same idea as bishop, but only one changes at a time.

    def rookMoves(self, row: int, col: int):
        re = []
        team = self.fullBoard[row][col].team  # debugged
        for i in (1, -1):
            moveRow = row + i
            while 0 <= moveRow <= 7 and self.fullBoard[moveRow][col].team != team:
                re.append(moveRow * 10 + col)
                if self.fullBoard[moveRow][col].team != "n":
                    break
                moveRow += i
            moveCol = col + i
            while 0 <= moveCol <= 7 and self.fullBoard[row][moveCol].team != team:
                re.append(row * 10 + moveCol)
                if self.fullBoard[row][moveCol].team != "n":
                    break
                moveCol += i
        return re

    def kingMoves(self, row: int, col: int):  # debugged
        team = self.fullBoard[row][col].team
        re = []
        moveRow = 0
        moveCol = 0
        for i in (1, -1):
            for j in (1, -1):
                moveRow = row+i  # diagonals
                moveCol = col+j
                if 0 <= moveRow <= 7 and self.fullBoard[moveRow][moveCol].team != team and 0 <= moveCol <= 7:
                    re.append(10*moveRow+moveCol)
        for i in (1, -1):
            moveCol = col+i
            if 0 <= moveCol <= 7 and self.fullBoard[row][moveCol].team != team:
                re.append(10*row+moveCol)
        for i in (1, -1):
            moveRow = row+i  # vertical
            if 0 <= moveRow <= 7 and self.fullBoard[moveRow][col].team != team:
                re.append(10*moveRow+col)
        return re

    # only call this function after all moves have been generated and checked. going to search through a tree of ints for advantage parameter.
    def AIAdvantageEval(self):
        self.allMovesGen()
        if self.turn <= 20:
            self.earlyGameAIEval()
        elif self.turn > 20 and len(self.blackPieces) > 5 and len(self.whitePieces > 5):
            self.midGameAIEval()
        else:
            self.lateGameAIEval()

    def kingAIAdvantage(self) -> int:
        re = 0
        kingVals = [[5, 4, 2, 2, 1, 1, 2, 2, 5],
                    [4, 3, 1, 1, 1, 1, 1, 3, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 2],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [2, 1, 1, 1, 1, 1, 1, 1, 2],
                    [4, 3, 1, 1, 1, 1, 1, 3, 4],
                    [5, 4, 2, 2, 1, 1, 2, 2, 5]]
        re += kingVals[self.whiteIndexes["K"]//10][self.whiteIndexes["K"] % 10]
        re -= kingVals[self.blackIndexes["K"]//10][self.blackIndexes["K"] % 10]
        if self.AIteamIsWhite == True:
            return re
        else:
            return -re

    # seperating out early, mid and late game functions to make things more readable and organized, easy for debugging
    def earlyGameAIEval(self):
        whiteAdvantage = 0
        blackAdvantage = 0
        whiteAdvantage += self.kingAIAdvantage()
        noMovesW = True
        noMovesB = True
        for i in self.blackPieces:
            ind = i
            currIndex = self.blackIndexes[ind]
            currRow = currIndex//10
            currCol = currIndex % 10
            if currIndex == 33 or currIndex == 34 or currIndex == 44 or currIndex == 43:  # favor moves from the middle
                blackAdvantage += 2*len(self.blackAvailableMoves[ind])
            if self.blackAvailableMoves[i] != None:
                if len(self.blackAvailableMoves[ind]) > 0:
                    # more moves means more piece development
                    blackAdvantage += len(self.blackAvailableMoves[ind])
                    noMovesB = False
                    for j in self.blackAvailableMoves[ind]:
                        if j == 99 or j == 100:
                            continue
                        moveIndexes = j
                        moveRow = moveIndexes//10
                        moveCol = moveIndexes % 10
                        if moveIndexes == 33 or moveIndexes == 34 or moveIndexes == 44 or moveIndexes == 43:
                            # moves to the middle
                            blackAdvantage += 3  # once again, I will look at these weights after playing against it
                        if self.turn % 2 == 1:
                            if self.fullBoard[moveRow][moveCol].team == 'w' and self.fullBoard[moveRow][moveCol].val > self.fullBoard[currRow][currCol].val:
                                blackAdvantage += (
                                    self.fullBoard[moveRow][moveCol].val-self.fullBoard[currRow][currCol].val)//2
                            # if b pawn is attacking w queen and its black's turn, advantage is 850 points
        if noMovesB == True and self.turn % 2 == 1:  # CheckMate, cannot stalemate within first 32 turns
            whiteAdvantage = 1000000
            blackAdvantage = -1000000
            self.inPlay = False

        for i in self.whitePieces:
            ind = i
            currIndex = self.whiteIndexes[i]
            currRow = currIndex//10
            currCol = currIndex % 10
            if currIndex == 33 or currIndex == 34 or currIndex == 44 or currIndex == 43:  # favor moves from the middle
                whiteAdvantage += 2*len(self.whiteaVailableMoves[ind])
            if self.whiteaVailableMoves[ind] != None:
                # more moves means more piece development
                whiteAdvantage += len(self.whiteaVailableMoves[ind])
                noMovesW = False
                for j in self.whiteaVailableMoves[ind]:
                    if j == 99 or j == 100:
                        continue
                    moveIndexes = j
                    moveRow = moveIndexes//10
                    moveCol = moveIndexes % 10
                    if moveIndexes == 33 or moveIndexes == 34 or moveIndexes == 44 or moveIndexes == 43:
                        # moves to the middle
                        whiteAdvantage += 3  # once again, I will look at these weights after playing against it

                    if self.turn % 2 == 1 and self.fullBoard[moveRow][moveCol].team == 'b' and self.fullBoard[moveRow][moveCol].val > self.fullBoard[currRow][currCol].val:
                        whiteAdvantage += (self.fullBoard[moveIndexes//10][moveIndexes %
                                           10].val-self.fullBoard[currRow][currCol].val)//2
        if noMovesW == True and self.turn % 2 == 0:  # CheckMate, cannot stalemate within first 20 turns
            blackAdvantage = 1000000
            whiteAdvantage = -1000000
            self.inPlay = False
        if self.AIteamIsWhite == True:
            self.AIAdvantage = self.whitePoints-self.blackPoints + \
                whiteAdvantage-blackAdvantage  # white team for AI
        else:
            self.AIAdvantage = self.blackPoints-self.whitePoints + \
                blackAdvantage-whiteAdvantage  # black team for AI

    # if turn is greater than 32 and if both teams have at least 5 pieces
    def midGameAIEval(self):
        blackAdvantage = 0
        whiteAdvantage = 0
        noMovesBlack = False
        noMovesWhite = False
        if self.turn % 2 == 0 and self.inCheckStored == True:  # late and middle game.
            blackAdvantage += 50
        if self.turn % 2 == 1 and self.inCheckStored == True:
            whiteAdvantage += 50

        for i in self.blackPieces:
            currIndex = self.blackIndexes[self.blackPieces[i]]
            currRow = currIndex//10
            currCol = currIndex % 10
            # favor moves from the middle heavier in midgame
            if currIndex == 33 or currIndex == 34 or currIndex == 44 or currIndex == 43:
                blackAdvantage += 3 * \
                    len(self.blackAvailableMoves[self.blackPieces[i]])
            if self.fullBoard[currRow][currCol].kind == 'p':
                if currRow == 5:
                    blackAdvantage += 50
                if currRow == 6:
                    blackAdvantage += 100
            if len(self.blackAvailableMoves[self.blackPieces[i]]) != 0:
                noMovesBlack = False
            moveIndexes = self.blackAvailableMoves[self.blackPieces[i]][j]
            moveRow = moveIndexes//10
            moveCol = moveIndexes % 10
            if moveRow == 5:
                blackAdvantage += 1
            if moveRow == 6:
                blackAdvantage += 3
            if moveRow == 7:
                blackAdvantage += 5  # favor moves to other teams side slightly
            if moveIndexes == 33 or moveIndexes == 34 or moveIndexes == 43 or moveIndexes == 44:
                blackAdvantage += 5
            if self.turn % 2 == 1 and self.fullBoard[moveRow][moveCol].team == 'w' and self.fullBoard[moveRow][moveCol].val > self.fullBoard[currRow][currCol]:
                blackAdvantage += (self.fullBoard[moveIndexes//10][moveIndexes %
                                   10].val-self.fullBoard[currRow][currCol].val)//2

        for i in self.whitePieces:
            temp = i
            currIndex = self.whiteIndexes[temp]
            currRow = currIndex//10
            currCol = currIndex % 10
            # favor moves from the middle heavier in midgame
            if currIndex == 33 or currIndex == 34 or currIndex == 44 or currIndex == 43:
                whiteAdvantage += 3*len(self.whiteaVailableMoves[temp])
            if self.fullBoard[currRow][currCol].kind == 'p':
                if currRow == 2:
                    whiteAdvantage += 50
                if currRow == 1:
                    whiteAdvantage += 100
                if len(self.whiteaVailableMoves[self.whitePieces[i]]) != 0:
                    noMovesWhite = False
            for j in self.whiteaVailableMoves[self.whitePieces[i]]:
                moveIndexes = self.whiteaVailableMoves[self.whitePieces[i]][j]
                moveRow = moveIndexes//10
                moveCol = moveIndexes % 10
                if moveRow == 5:
                    whiteAdvantage += 1
                if moveRow == 6:
                    whiteAdvantage += 3
                if moveRow == 7:
                    whiteAdvantage += 5  # favor moves to other teams side slightly
                if moveIndexes == 33 or moveIndexes == 34 or moveIndexes == 43 or moveIndexes == 44:
                    whiteAdvantage += 5
                if self.turn % 2 == 1 and self.fullBoard[moveRow][moveCol].team == 'b' and self.fullBoard[moveRow][moveCol].val > self.fullBoard[currRow][currCol]:
                    whiteAdvantage += (self.fullBoard[moveIndexes//10][moveIndexes %
                                       10].val-self.fullBoard[currRow][currCol].val)//2

        if noMovesWhite == True and self.turn % 2 == 1:  # CheckMate, cannot stalemate within first 32 turns
            if self.inCheckStored == True:
                blackAdvantage = 1000000
                whiteAdvantage = -1000000
                self.inPlay = False
            else:
                self.advantage = 0
                self.inPlay = False
        if noMovesBlack == True and self.turn % 2 == 0:
            if self.inCheckStored == True:
                blackAdvantage = -1000000
                whiteAdvantage = 1000000
                self.inPlay = False
            else:
                blackAdvantage = 0
                whiteAdvantage = 0
                self.whitePoints = 0
                self.blackPoints = 0
                self.inPlay = False
        if self.AIteamIsWhite == True:
            self.AIAdvantage = self.whitePoints-self.blackPoints + \
                whiteAdvantage-blackAdvantage  # white team for AI
        else:
            self.AIAdvantage = self.blackPoints-self.whitePoints + \
                blackAdvantage-whiteAdvantage  # black team for AI

    # I probably want to create some algorithm for comparing search depth to pieces left, less pieces left means deeper search
    def lateGameAIEval(self):
        if len(self.whitePieces) + len(self.blackPieces) == 2:  # 2 kings left
            self.inPlay = True
            self.AIAdvantage = 0
            return
        whiteAdvantage = 0
        blackAdvantage = 0
        noMovesW = True
        noMovesB = True
        if self.blackPoints < self.whitePoints:
            # incentivise trades if you're up in the late game
            whiteAdvantage += 3*(len(self.blackPieces)+len(self.whitePieces))
        if self.blackPoints > self.whitePoints:
            blackAdvantage += 3*(len(self.blackPieces)+len(self.whitePieces))
        if self.turn % 2 == 0 and self.inCheckStored == True:  # late and middle game.
            blackAdvantage += 50
        if self.turn % 2 == 1 and self.inCheckStored == True:
            whiteAdvantage += 50
        # If last piece is the black king, push it to the edge for checkmate
        if len(self.blackPieces) == 1:
            bKingRow = self.blackIndexes//10
            bKingCol = self.blackIndexes % 10
            if bKingRow == 2 or bKingRow == 5:
                # has to be weighted heavily, this is to pushes the king to the edge for checkmate
                whiteAdvantage += 100
            if bKingCol == 2 or bKingCol == 5:
                whiteAdvantage += 100
            if bKingRow == 1 or bKingRow == 6:
                whiteAdvantage += 150
            if bKingCol == 1 or bKingCol == 6:
                whiteAdvantage += 150
            if bKingRow == 0 or bKingRow == 7:
                whiteAdvantage += 200
            if bKingCol == 0 or bKingCol == 7:
                whiteAdvantage += 200
        if len(self.whitePieces) == 1:  # same for white
            wKingRow = self.blackIndexes//10
            wKingCol = self.blackIndexes % 10
            if wKingRow == 2 or wKingRow == 5:
                # has to be weighted heavily, this is to pushes the king to the edge for checkmate
                blackAdvantage += 100
            if wKingCol == 2 or wKingCol == 5:
                blackAdvantage += 100
            if wKingRow == 1 or wKingRow == 6:
                blackAdvantage += 150
            if wKingCol == 1 or wKingCol == 6:
                blackAdvantage += 150
            if wKingRow == 0 or wKingRow == 7:
                blackAdvantage += 200
            if wKingCol == 0 or wKingCol == 7:
                blackAdvantage += 200

        for i in self.blackPieces:
            currIndexes = self.blackIndexes[self.blackPieces[i]]
            currRow = currIndexes//10
            currCol = currIndexes % 10
            if self.fullBoard[currRow][currCol].kind == 'p':
                if currRow > 4:
                    blackAdvantage += currRow*30  # this way, higher rows get more points for pawns
            if len(self.blackAvailableMoves[self.blackPieces[i]]) != 0:
                noMovesB = False
            # no middle control weight for late game
            for j in self.blackAvailableMoves[self.blackPieces[i]]:
                moveIndexes = self.blackAvailableMoves[self.blackPieces[i]][j]
                moveRow = moveIndexes//10
                moveCol = moveIndexes % 10
                if self.turn % 2 == 1 and self.fullBoard[moveRow][moveCol].team == 'b' and self.fullBoard[moveRow][moveCol].val > self.fullBoard[currRow][currCol]:
                    whiteAdvantage += (self.fullBoard[moveRow][moveCol].val -
                                       self.fullBoard[currRow][currCol].val)//2

        for i in self.whitePieces:
            currIndexes = self.whiteIndexes[self.whitePieces[i]]
            currRow = currIndexes//10
            currCol = currIndexes % 10
            if self.fullBoard[currRow][currCol].kind == 'p':
                if currRow < 3:
                    whiteAdvantage += (7-currRow)*30
            if len(self.whiteaVailableMoves[self.blackPieces[i]]) != 0:
                noMovesW = False
            for j in self.whiteaVailableMoves[self.whitePieces[i]]:
                moveIndexes = self.whiteaVailableMoves[self.whitePieces[i]][j]
                moveRow = moveIndexes//10
                moveCol = moveIndexes % 10
                if self.turn % 2 == 1 and self.fullBoard[moveRow][moveCol].team == 'w' and self.fullBoard[moveRow][moveCol].val > self.fullBoard[currRow][currCol]:
                    whiteAdvantage += (self.fullBoard[moveIndexes//10][moveIndexes %
                                       10].val-self.fullBoard[currRow][currCol].val)//2
        if noMovesW == True and self.turn % 2 == 1:  # CheckMate
            if self.inCheckStored == True:
                blackAdvantage = 1000000
                whiteAdvantage = -1000000
                self.inPlay = False
            else:
                self.advantage = 0
                self.inPlay = False
        if noMovesB == True and self.turn % 2 == 0:
            if self.inCheckStored == True:
                blackAdvantage = -1000000
                whiteAdvantage = 1000000
                self.inPlay = False
            else:
                blackAdvantage = 0
                whiteAdvantage = 0
                self.whitePoints = 0
                self.blackPoints = 0
                self.inPlay = False
        if self.AIteamIsWhite == True:
            self.AIAdvantage = self.whitePoints-self.blackPoints + \
                whiteAdvantage-blackAdvantage  # white team for AI
        else:
            self.AIAdvantage = self.blackPoints-self.whitePoints + \
                blackAdvantage-whiteAdvantage  # black team for AI

    def allMovesGen(self):  # only call the move function after this is called.
        bKS = self.prime2 % 13 != 0 and self.prime2 % 11 != 0 and self.fullBoard[
            0][6].team == 'n' and self.fullBoard[0][5].team == 'n'
        bQS = self.prime2 % 13 != 0 and self.prime2 % 7 != 0 and self.fullBoard[0][
            1].team == 'n' and self.fullBoard[0][2].team == 'n' and self.fullBoard[0][3].team == 'n'
        wKS = self.prime2 % 5 != 0 and self.prime2 % 3 != 0 and self.fullBoard[
            7][6].team == 'n' and self.fullBoard[7][5].team == 'n'
        wQS = self.prime2 % 5 != 0 and self.prime2 % 2 != 0 and self.fullBoard[7][
            1].team == 'n' and self.fullBoard[7][2].team == 'n' and self.fullBoard[7][3].team == 'n'
        # the lines of code check to see if the squares in between K and R are empty and if you've moved those pieces.
        # self.canKSCastle() cutting this to improve efficiency. Represented by (9,9)
        # self.canQSCastle() is checked here, represented by (10,10)
        blackChecking = []
        whiteChecking = []
        bPinnedVectors = []
        wPinnedVectors = []
        bPinnedPieces = []
        wPinnedPieces = []
        wKingMoves = self.generateAvailableMoves(
            self.whiteIndexes["K"]//10, self.whiteIndexes["K"] % 10)
        bKingMoves = self.generateAvailableMoves(
            self.blackIndexes["K"]//10, self.blackIndexes["K"] % 10)  # will need to remove some of these
        overW = []
        overB = []

        for i in self.whitePieces:
            if i == "K":
                continue
            tempPiece = i
            currRow = self.whiteIndexes[tempPiece]//10
            currCol = self.whiteIndexes[tempPiece] % 10
            allMoves = self.generateAvailableMoves(
                currRow, currCol)  # array of places you can move to
            self.whiteaVailableMoves[tempPiece] = allMoves
            checking = False
            if allMoves != None:
                # in check condition
                if self.blackIndexes["K"] in self.whiteaVailableMoves[tempPiece]:
                    # only look to see if black's in check if it's blacks turn, cannot move into check
                    self.inCheckStored = True
                    whiteChecking.append(tempPiece)
                    checking = True
            if self.fullBoard[currRow][currCol].kind == 'r' or self.fullBoard[currRow][currCol].kind == 'q' and checking == False:
                if self.whiteIndexes[tempPiece]//10 == self.blackIndexes["K"]//10 and self.whiteIndexes[tempPiece] % 10 == self.blackIndexes["K"] % 10:
                    temp = self.wRookPinning(tempPiece)
                    if temp != []:
                        a = temp[len(temp)-1]
                        temp.pop(len(temp)-1)
                        bPinnedVectors.append(temp)
                        bPinnedPieces.append(a)
            if (self.fullBoard[currRow][currCol].kind == 'q' or self.fullBoard[currRow][currCol].kind == 'b') and checking == False:
                if abs(self.blackIndexes["K"] % 10-self.whiteIndexes[tempPiece] % 10) ==\
                        abs(self.blackIndexes["K"]//10-self.whiteIndexes[tempPiece]//10):  # if they're on same diagonal
                    temp = self.wBishopPinning(tempPiece)
                    if temp != []:
                        a = temp[len(temp)-1]
                        temp.pop(len(temp)-1)
                        bPinnedVectors.append(temp)
                        bPinnedPieces.append(a)
            if allMoves != None and bKingMoves != None:
                for j in bKingMoves:
                    if j in allMoves:
                        overB.append(j)
            # looking to modify this function, only need to check whether or not q, r, b are pressuring king
            if bKS == True and allMoves != None:  # can QS castle
                if 76 in allMoves or 75 in allMoves or 74 in allMoves or 77 in allMoves:
                    bKS = False
            if bKS == True and allMoves != None:  # can KS castle
                if 74 in allMoves or 73 in allMoves or 72 in allMoves or 71 in allMoves or 70 in allMoves:
                    bQS = False

        for i in self.blackPieces:
            if i == "K":
                continue
            tempPiece = i
            # print(tempPiece)
            currRow = self.blackIndexes[tempPiece]//10
            currCol = self.blackIndexes[tempPiece] % 10
            allMoves = self.generateAvailableMoves(currRow, currCol)
            self.blackAvailableMoves[tempPiece] = allMoves
            checking = False
            if allMoves != None:
                # in check condition
                if self.whiteIndexes["K"] in self.blackAvailableMoves[tempPiece]:

                    blackChecking.append(tempPiece)
                    checking = True
            if (self.fullBoard[currRow][currCol].kind == 'r' or self.fullBoard[currRow][currCol].kind == 'q') and checking == False and self.inPlay == True:
                if self.blackIndexes[tempPiece]//10 == self.whiteIndexes["K"]//10 or self.blackIndexes[tempPiece] % 10 == self.whiteIndexes["K"] % 10:
                    temp = self.bRookPinning(tempPiece)
                    if temp != []:
                        a = temp[len(temp)-1]
                        temp.pop(len(temp)-1)
                        wPinnedVectors.append(temp)
                        wPinnedPieces.append(a)
            if (self.fullBoard[currRow][currCol].kind == 'q' or self.fullBoard[currRow][currCol].kind == 'b') and checking == False and self.inPlay == True:
                if abs(self.whiteIndexes["K"] % 10-self.blackIndexes[tempPiece] % 10) ==\
                        abs(self.whiteIndexes["K"]//10-self.blackIndexes[tempPiece]//10):  # if they're on same diagonal
                    temp = self.bBishopPinning(tempPiece)
                    if temp != []:
                        a = temp[len(temp)-1]
                        temp.pop(len(temp)-1)
                        wPinnedVectors.append(temp)
                        wPinnedPieces.append(a)
            if allMoves != None and wKingMoves != None:
                for j in wKingMoves:
                    if j in allMoves:
                        overW.append(j)
            if wKS == True and allMoves != None:  # can QS castle
                if 76 in allMoves or 75 in allMoves or 74 in allMoves or 77 in allMoves:
                    wKS = False
            if wQS == True and allMoves != None:
                if 44 in allMoves or 73 in allMoves or 72 in allMoves or 71 in allMoves or 70 in allMoves:
                    wQS = False
        if bKingMoves != None:
            for i in overB:
                if i in bKingMoves:
                    bKingMoves.remove(i)
        if wKingMoves != None:
            for i in overW:
                if i in wKingMoves:
                    wKingMoves.remove(i)
        if self.whiteIndexes["K"]//10 > 1:
            # pawns don't generate this move unless there is already something there
            if self.whiteIndexes["K"] % 10 > 0:
                if self.fullBoard[self.whiteIndexes["K"]//10-2][self.whiteIndexes["K"] % 10-1].kind == 'p'\
                        and self.fullBoard[self.whiteIndexes["K"]//10-2][self.whiteIndexes["K"] % 10-1].team == 'b':
                    if self.whiteIndexes["K"]-21 in wKingMoves:
                        wKingMoves.remove(self.whiteIndexes["K"]-21)
            if self.whiteIndexes["K"] % 10 < 7:
                if self.fullBoard[self.whiteIndexes["K"]//10-2][self.whiteIndexes["K"] % 10+1].kind == 'p'\
                        and self.fullBoard[self.whiteIndexes["K"]//10-2][self.whiteIndexes["K"] % 10+1].team == 'b':
                    if self.whiteIndexes["K"]-19 in wKingMoves:
                        wKingMoves.remove(self.whiteIndexes["K"]-19)
        if self.blackIndexes["K"]//10 < 6:
            if self.blackIndexes["K"] % 10 > 0:
                if self.fullBoard[self.blackIndexes["K"]//10+2][self.blackIndexes["K"] % 10-1].kind == 'p'\
                        and self.fullBoard[self.blackIndexes["K"]//10+2][self.blackIndexes["K"] % 10-1].team == 'w':
                    if self.blackIndexes["K"]+19 in bKingMoves:
                        bKingMoves.remove(self.blackIndexes["K"]+19)
            if self.blackIndexes["K"] % 10 < 7:
                if self.fullBoard[self.blackIndexes["K"]//10-2][self.blackIndexes["K"] % 10+1].kind == 'p'\
                        and self.fullBoard[self.blackIndexes["K"]//10-2][self.blackIndexes["K"] % 1+1].team == 'w':
                    if self.blackIndexes["K"]+21 in bKingMoves:
                        bKingMoves.remove(self.blackIndexes["K"]+21)
        self.blackAvailableMoves["K"] = bKingMoves
        self.whiteaVailableMoves["K"] = wKingMoves
        if bKS == True:
            self.blackAvailableMoves["K"].append(99)  # KS castle check.
        if bQS == True:
            # if condition is met, black can QS castle, cuts down on iterations.
            self.blackAvailableMoves["K"].append(100)
        if wKS == True:
            self.whiteaVailableMoves["K"].append(99)
        if wQS == True:
            self.whiteaVailableMoves["K"].append(100)
        if len(bPinnedVectors) != 0:
            for i in range(len(bPinnedPieces)):  # i should represent a vector
                overLap = []
                pinnedP = bPinnedPieces[i]
                for j in bPinnedVectors[i]:
                    if j in self.blackAvailableMoves[pinnedP]:
                        overLap.append(j)
                self.blackAvailableMoves[pinnedP] = overLap
        if len(wPinnedVectors) != 0:
            for i in range(len(wPinnedPieces)):
                overLap = []
                pinnedP = wPinnedPieces[i]
                for j in wPinnedVectors[i]:
                    if j in self.whiteaVailableMoves[pinnedP]:
                        overLap.append(j)
                self.whiteaVailableMoves[pinnedP] = overLap

        if len(whiteChecking) > 0:  # now that moves and necessary info has been generated, need to eliminate moves that put the king into check
            for i in whiteChecking:
                if i == "b" or i == "r" or i == "q":
                    direction = []
                    direction.append(
                        self.blackIndexes["K"]//10-self.whiteIndexes[i]//10)
                    direction.append(
                        self.blackIndexes["K"] % 10-self.whiteIndexes[i] % 10)
                    self.inCheck2(i, "w", direction)

                else:
                    self.inCheck1(i, "w")
        if len(blackChecking) > 0:
            for i in blackChecking:
                if i == "b" or i == "r" or i == "q":
                    direction = []
                    direction.append(
                        self.whiteIndexes["K"]//10-self.blackIndexes[i]//10)
                    direction.append(
                        self.whiteIndexes["K"] % 10-self.blackIndexes[i] % 10)
                    self.inCheck2(i, "b", direction)
                else:
                    self.inCheck1(i, "b")

    def wRookPinning(self, pinning: str):
        wRow = self.whiteIndexes[pinning] // 10
        wCol = self.whiteIndexes[pinning] % 10
        kRow = self.blackIndexes["K"] // 10
        kCol = self.blackIndexes["K"] % 10
        pIndexes = -1
        moveVector = [self.whiteIndexes[pinning]]
        if wCol == kCol:  # only called if wCol==kCol or wRow==kRow
            magnitude = kRow - wRow
            direction = magnitude // abs(magnitude)
            done = False
            for i in range(1, magnitude):
                j = i * direction
                if self.fullBoard[wRow + j][wCol].team == 'w' or (self.fullBoard[wRow + j][wCol].team == 'b' and done == True):
                    return []
                elif self.fullBoard[wRow + j][wCol].team == 'b' and done == False:
                    pIndexes = 10 * (wRow + j) + wCol
                    done = True  # Set done to True after finding a black piece
                else:
                    moveVector.append(10 * (wRow + j) + wCol)  # if team='n'
            moveVector.append(self.blackIToP[pIndexes])
            if done == False:
                return []
            return moveVector
        else:
            magnitude = kCol - wCol
            direction = magnitude // abs(magnitude)
            done = False
            for i in range(1, magnitude):
                j = i * direction
                if self.fullBoard[wRow][wCol + j].team == 'w' or (self.fullBoard[wRow][wCol + j].team == 'b' and done == True):
                    return []
                elif self.fullBoard[wRow][wCol + j].team == 'b' and done == False:
                    pIndexes = 10 * wRow + j + wCol
                    done = True  # Set done to True after finding a black piece
                else:
                    moveVector.append(10 * wRow + j + wCol)  # if team='n'
            if done == False:
                return []
            moveVector.append(self.blackIToP[pIndexes])
            print(moveVector)
            return moveVector

    def bRookPinning(self, pinning: str):
        bRow = self.blackIndexes[pinning] // 10
        bCol = self.blackIndexes[pinning] % 10
        kRow = self.whiteIndexes["K"] // 10
        kCol = self.whiteIndexes["K"] % 10
        pIndex = -1
        moveVector = [self.blackIndexes[pinning]]
        if bCol == kCol:  # only called if wCol==kCol or wRow==kRow
            magnitude = kRow - bRow
            direction = magnitude // abs(magnitude)
            done = False
            for i in range(1, magnitude):
                j = i * direction
                if self.fullBoard[bRow + j][bCol].team == 'b' or (self.fullBoard[bRow + j][bCol].team == 'w' and done == True):
                    return []
                # Corrected condition
                elif self.fullBoard[bRow + j][bCol].team == 'w' and done == False:
                    pIndex = 10 * (bRow + j) + bCol
                    done = True
                else:
                    moveVector.append(10 * (bRow + j) + bCol)  # if team='n'
            moveVector.append(self.whiteIToP[pIndex])
            if done == False:
                return []
            return moveVector
        else:
            magnitude = kCol - bCol
            direction = magnitude // abs(magnitude)
            done = False
            for i in range(1, magnitude):
                j = i * direction
                if self.fullBoard[bRow][bCol + j].team == 'b' or (self.fullBoard[bRow][bCol + j].team == 'w' and done == True):
                    return None
                # Corrected condition
                elif self.fullBoard[bRow][bCol + j].team == 'w' and done == False:
                    pIndex = 10 * bRow + j + bCol
                    done = True
                else:
                    moveVector.append(10 * bRow + j + bCol)  # if team='n'
            if done == False:
                return []
            moveVector.append(self.whiteIToP[pIndex])
            print(moveVector)
            return moveVector

    def wBishopPinning(self, pinning: str):
        wRow = self.whiteIndexes[pinning]//10
        wCol = self.whiteIndexes[pinning] % 10
        kRow = self.blackIndexes["K"]//10
        kCol = self.blackIndexes["K"] % 10
        if abs(wRow-kRow) != abs(wCol-kCol) or wRow == kRow or wCol == kCol:
            return []
        pIndexes = -1
        moveVector = [self.whiteIndexes[pinning]]
        # only called if wCol==kCol or wRow==kRow
        magnitude = kRow-wRow
        directionR = magnitude//abs(magnitude)
        m2 = kCol-wCol
        directionC = m2//abs(m2)
        done = False
        for i in range(1, abs(magnitude)):
            cInc = i*directionC
            rInc = i*directionR
            if self.fullBoard[wRow+rInc][wCol+cInc].team == 'w' or (self.fullBoard[wRow+rInc][wCol+cInc].team == 'b' and done == True):
                return []
            elif self.fullBoard[wRow+rInc][wCol+cInc].team == 'b' and done == False:
                pIndexes = 10*(wRow+rInc)+wCol+cInc
                done = True
            else:
                moveVector.append(10*(wRow+rInc)+cInc+wCol)  # if team='n'
        if done == False:
            return []
        moveVector.append(self.blackIToP[pIndexes])
        print(moveVector)
        return moveVector

    def printInfo(self):  # debbugging function only
        print("movesLog")
        print(self.movesLog)
        print("inPlay:")
        print(self.inPlay)
        print("games simulated:")
        print("w:")
        for i in self.whitePieces:
            moveIndexesw = []
            if self.whiteaVailableMoves[i] != None:
                for j in self.whiteaVailableMoves[i]:
                    moveIndexesw.append(mapping(j))
            print(i)
            print("Indexes: ", mapping(self.whiteIndexes[i]))
            print("Available Moves: ", moveIndexesw)
        print("b:")
        for i in self.blackPieces:
            moveIndexesb = []
            for j in self.blackAvailableMoves[i]:
                moveIndexesb.append(mapping(j))
            print(i)
            print("Indexes: ", mapping(self.blackIndexes[i]))
            print("Available Moves: ", moveIndexesb)
        print("AI Team: ", "w")
        print("AI Advantage: ", self.AIAdvantage)
        self.printBoard()

    def bBishopPinning(self, pinning: str):
        bRow = self.blackIndexes[pinning]//10
        bCol = self.blackIndexes[pinning] % 10
        kRow = self.whiteIndexes["K"]//10
        kCol = self.whiteIndexes["K"] % 10
        if abs(bRow-kRow) != abs(bCol-kCol) or abs(bCol-kCol) == 0 or abs(bRow-kRow) == 0:
            return []
        pIndex = -1
        moveVector = [self.blackIndexes[pinning]]
        # only called if wCol==kCol or wRow==kRow
        magnitude = kRow-bRow
        directionR = magnitude//abs(magnitude)
        m2 = kCol-bCol
        directionC = m2//abs(m2)
        done = False
        for i in range(1, abs(magnitude), 1):
            cInc = i*directionC
            rInc = i*directionR
            if self.fullBoard[bRow+rInc][bCol+cInc].team == 'b' or (self.fullBoard[bRow+rInc][bCol+cInc].team == 'w' and done == True):
                return []
            elif self.fullBoard[bRow+rInc][bCol+cInc].team == 'w' and done == False:
                pIndex = 10*(bRow+rInc)+bCol+cInc
                done = True
            else:
                moveVector.append(10*(bRow+rInc)+cInc+bCol)  # if team='n'
        moveVector.append(self.whiteIToP[pIndex])
        print(moveVector)
        return moveVector

    def inCheck1(self, pressuring: str, team: str):  # if king is in check from knight or pawn
        # team corresponds to team in check
        if team == "w":  # either has to capture the piece or move the king.
            for i in self.blackPieces:  # if white is in check
                if i == "K":
                    continue  # all moves for king are checked in gen moves function
                # clear moves here
                if self.whiteIndexes[pressuring] in self.blackAvailableMoves[i]:
                    for j in self.blackAvailableMoves[i]:
                        if j != self.whiteIndexes[pressuring]:
                            self.blackAvailableMoves[i].remove(j)
                else:
                    self.blackAvailableMoves[i] = []
        else:
            for i in self.whitePieces:  # if white is in check
                if i == "K":
                    continue  # all moves for king are checked in gen moves function
                # clear moves here
                if self.blackIndexes[pressuring] in self.whiteaVailableMoves[i]:
                    for j in self.whiteaVailableMoves[i]:
                        if j != self.blackIndexes[pressuring]:
                            self.whiteaVailableMoves[i].remove(j)
                else:
                    self.whiteaVailableMoves[i] = []

    # if getting checked by bishop, knight or rook
    def inCheck2(self, pressuring: str, team: str, direction: list[int]):
        # similar situation to pinning, but has to move in the pinning directionD
        print("in check 2 being called")
        goodMoves = []
        if team == "w":  # white checking black
            kInd = self.blackIndexes["K"]
            kingRow = kInd//10
            kingCol = kInd % 10
            pRow = self.whiteIndexes[pressuring]//10
            pCol = self.whiteIndexes[pressuring] % 10
            # you can take the piece that's pressuring the king and you can move the pieces in the same line between the long range piece and king.
            goodMoves.append(self.whiteIndexes[pressuring])

            if direction[0] > 0 and direction[1] > 0:  # king row>p row, kCol>p col
                pRow += 1
                pCol += 1
                while pRow < kingRow and pCol < kingCol:
                    goodMoves.append(10*pRow+pCol)
                    pRow += 1
                    pCol += 1

            elif direction[0] == 0 and direction[1] > 0:
                pCol += 1
                while pCol < kingCol:
                    goodMoves.append(10*pRow+pCol)
                    pCol += 1

            elif direction[0] > 0 and direction[1] == 0:
                pRow += 1
                while pRow < kingRow:
                    goodMoves.append(10*pRow+pCol)
                    pRow += 1

            elif direction[0] == 0 and direction[1] < 0:
                pCol -= 1
                while pCol > kingCol:
                    goodMoves.append(10*pRow+pCol)
                    pCol -= 1

            elif direction[0] < 0 and direction[1] < 0:
                pRow -= 1
                pCol -= 1
                while pCol > kingCol and pRow > kingRow:
                    goodMoves.append(10*pRow+pCol)
                    pRow -= 1
                    pCol -= 1

            elif direction[0] > 0 and direction[1] < 0:
                pRow += 1
                pCol -= 1
                while pCol > kingCol and pRow < kingRow:
                    goodMoves.append(10*pRow+pCol)
                    pRow += 1
                    pCol -= 1

            elif direction[0] < 0 and direction[1] > 0:
                pRow -= 1
                pCol += 1
                while pRow > kingRow and pCol < kingCol:
                    goodMoves.append(10*pRow+pCol)
                    pRow -= 1
                    pCol += 1
            else:  # direction[0]<0 and direction[1]==0
                pRow -= 1
                while pRow > kingRow:
                    goodMoves.append(10*pRow+pCol)
                    pRow -= 1

            for i in self.blackPieces:
                overLap = []
                if i == "K":
                    continue  # condition already checked.
                else:
                    for j in self.blackAvailableMoves[i]:
                        if j in goodMoves:
                            # gets rid of all the moves that do not block the check.
                            overLap.append(j)
                    self.blackAvailableMoves[i] = overLap
            goodMoves.pop(0)
            for i in self.blackAvailableMoves["K"]:
                if i in goodMoves:
                    self.blackAvailableMoves["K"].remove(i)

        else:
            kInd = self.whiteIndexes["K"]
            kingRow = kInd//10
            kingCol = kInd % 10
            pRow = self.blackIndexes[pressuring]//10
            pCol = self.blackIndexes[pressuring] % 10
            # you can take the piece that's pressuring the king and you can move the pieces in the same line between the long range piece and king.
            goodMoves.append(self.blackIndexes[pressuring])
            if direction[0] > 0 and direction[1] > 0:  # king row>p row, kCol>p col
                pRow += 1
                pCol += 1
                while pRow < kingRow and pCol < kingCol:
                    goodMoves.append(10*pRow+pCol)
                    pRow += 1
                    pCol += 1

            elif direction[0] == 0 and direction[1] > 0:
                pCol += 1
                while pCol < kingCol:
                    goodMoves.append(10*pRow+pCol)
                    pCol += 1

            elif direction[0] > 0 and direction[1] == 0:
                pRow += 1
                while pRow < kingRow:
                    goodMoves.append(10*pRow+pCol)
                    pRow += 1

            elif direction[0] == 0 and direction[1] < 0:
                pCol -= 1
                while pCol > kingCol:
                    goodMoves.append(10*pRow+pCol)
                    pCol -= 1

            elif direction[0] < 0 and direction[1] < 0:
                pRow -= 1
                pCol -= 1
                while pCol > kingCol and pRow > kingRow:
                    goodMoves.append(10*pRow+pCol)
                    pRow -= 1
                    pCol -= 1

            elif direction[0] > 0 and direction[1] < 0:
                pRow += 1
                pCol -= 1
                while pCol > kingCol and pRow < kingRow:
                    goodMoves.append(10*pRow+pCol)
                    pRow += 1
                    pCol -= 1

            elif direction[0] < 0 and direction[1] > 0:
                pRow -= 1
                pCol += 1
                while pRow > kingRow and pCol < kingCol:
                    goodMoves.append(10*pRow+pCol)
                    pRow -= 1
                    pCol += 1
            else:  # direction[0]<0 and direction[1]==0
                pRow -= 1
                while pRow > kingRow:
                    goodMoves.append(10*pRow+pCol)
                    pRow -= 1

            for i in self.whitePieces:
                overLap = []
                if i == "K":
                    continue  # condition already checked.
                else:
                    for j in self.whiteaVailableMoves[i]:
                        if j in goodMoves:
                            # gets rid of all the moves that do not block the check.
                            overLap.append(j)
                    self.whiteaVailableMoves[i] = overLap
            for i in self.whiteaVailableMoves["K"]:
                if i in goodMoves:
                    self.whiteaVailableMoves["K"].remove(i)
            goodMoves.pop(0)

    # this will only be called after the gen all moves, so you dont have to run it twice
    def move(self, movePiece, indexes):
        self.movesLog.append([movePiece, indexes])
        if self.turn % 2 == 0:  # if it's white's turn.#availableMoveNum is the index,
            initialCoords = self.whiteIndexes[movePiece]
            newIndexes = indexes
            oldRow = initialCoords//10
            oldCol = initialCoords % 10
            newRow = indexes//10
            newCol = indexes % 10
            self.whitePrime = 1
            if newIndexes == 99:  # king side castle
                self.fullBoard[7][4] = piece(0, 'n', 'n')
                self.fullBoard[7][7] = piece(0, 'n', 'n')
                self.fullBoard[7][6] = piece(0, 'K', 'w')
                self.fullBoard[7][5] = piece(500, 'r', 'w')
                self.whiteIndexes["K"] = 76
                self.whiteIndexes["r2"] = 75
                self.whiteIToP[76] = "K"
                self.whiteIToP[75] = "r2"
                self.turn += 1
                return
            elif newIndexes == 100:  # QSCastle
                self.fullBoard[7][4] = piece(0, 'n', 'n')
                self.fullBoard[7][3] = piece(500, 'r', 'w')
                self.fullBoard[7][2] = piece(0, 'K', 'w')
                self.fullBoard[7][0] = piece(0, 'n', 'n')
                self.fullBoard[7][1] = piece(0, 'n', 'n')
                self.whiteIndexes["K"] = 72
                self.whiteIndexes["r1"] = 73
                self.whiteIToP[72] = "K"
                self.whiteIToP[73] = "r2"
                self.turn += 1
                return
            else:
                # old piece refers to the one that's being captured.
                oldpoints = self.fullBoard[newRow][newCol].val
                if oldpoints > 0:  # if a black piece is captured
                    boldPiece = self.blackIToP[newIndexes]
                    self.blackAvailableMoves.pop(boldPiece)
                    self.blackIndexes.pop(boldPiece)
                    self.blackPieces.remove(boldPiece)
                    self.blackIToP.pop(newIndexes)
                    self.blackPoints -= oldpoints
                if initialCoords == 70:  # if they take rook before its moved
                    self.prime2 = self.prime2*2  # 2 corresponds to WR1
                if initialCoords == 77:
                    self.prime2 = self.prime2*3
                if initialCoords == 74:
                    self.prime2 = self.prime2*5
                if self.fullBoard[oldRow][oldCol].kind == 'p':
                    if oldCol != newCol and self.fullBoard[newRow][newCol].team == 'n':
                        self.EPWhite(movePiece, indexes)
                        return
                    if oldRow-newRow == 2:
                        a = primeMethods.primes(oldCol)
                        self.whitePrime = a
                        self.whitePrime1 = self.whitePrime1 * \
                            primeMethods.primes1(movePiece)
                    if newIndexes//10 == 0:  # pawn to queen
                        self.fullBoard[newRow][newCol] = piece(900, 'q', 'w')
                        self.whitePoints += 800
                        newPiece = primeMethods.queenMapping(movePiece)
                        self.whitePieces.append(newPiece)
                        self.whiteIndexes[newPiece] = newIndexes
                        self.whiteIToP[newIndexes] = newPiece
                        self.whiteaVailableMoves.pop(movePiece)
                        self.whitePieces.remove(movePiece)
                        self.whiteIndexes.pop(movePiece)
                        self.whiteIToP.pop(initialCoords)
                        self.turn += 1
                        # need to return here, because in this special situation, index map updating is completely different.
                        self.fullBoard[oldRow][oldCol] = piece(0, 'n', 'n')
                        return
                self.fullBoard[newRow][newCol] = piece.copy(
                    self.fullBoard[oldRow][oldCol])
                self.whiteIndexes[movePiece] = newIndexes
                self.whiteIToP.pop(initialCoords)
                # have to reset all fields to reflect information on the new board.
                self.whiteIToP[newIndexes] = movePiece
                self.fullBoard[oldRow][oldCol] = piece(0, 'n', 'n')
                self.turn += 1

        else:  # blacks turn
            self.blackPrime = 1
            initialCoords = self.blackIndexes[movePiece]
            newIndexes = indexes
            oldRow = initialCoords//10
            oldCol = initialCoords % 10
            newRow = newIndexes//10
            newCol = newIndexes % 10
            if newIndexes == 99:  # king side castle
                self.fullBoard[0][4] = piece(0, 'n', 'n')
                self.fullBoard[0][7] = piece(0, 'n', 'n')
                self.fullBoard[0][6] = piece(0, 'K', 'b')
                self.fullBoard[0][5] = piece(500, 'r', 'b')
                self.blackIndexes["K"] = 76
                self.blackIndexes["r2"] = 75
                self.blackIToP[76] = "K"
                self.blackIToP[75] = "r2"
                self.turn += 1
                return
            elif newIndexes == 100:  # QSCastle
                self.fullBoard[0][4] = piece(0, 'n', 'n')
                self.fullBoard[0][3] = piece(500, 'r', 'b')
                self.fullBoard[0][2] = piece(0, 'K', 'b')
                self.fullBoard[0][0] = piece(0, 'n', 'n')
                self.fullBoard[0][1] = piece(0, 'n', 'n')
                self.blackIndexes["K"] = 2
                self.blackIndexes["r1"] = 3
                self.blackIToP[2] = "K"
                self.blackIToP[3] = "r2"
                self.turn += 1
                return
            else:
                if movePiece == "K":
                    self.prime2 = self.prime2*13
                if movePiece == "r1":
                    self.prime2 = self.prime2*7
                if movePiece == "r2":
                    self.prime2 = self.prime2*11
                # old piece refers to the one that's being captured.
                oldpoints = self.fullBoard[newRow][newCol].val
                if oldpoints > 0:  # if a black piece is captured
                    woldPiece = self.whiteIToP[newIndexes]
                    self.whiteaVailableMoves.pop(woldPiece)
                    self.whiteIndexes.pop(woldPiece)
                    self.whitePieces.remove(woldPiece)
                    self.whiteIToP.pop(newIndexes)
                    self.whitePoints -= oldpoints

                if self.fullBoard[oldRow][oldCol].kind == 'p':  # pawn to queen
                    if oldCol != newCol and self.fullBoard[newRow][newCol].team == 'n':
                        self.EPBlack(movePiece, indexes)
                        return
                    if newRow-oldRow == 2:
                        self.blackPrime = primeMethods.primes(oldCol)
                        self.blackPrime1 = self.blackPrime1 * \
                            primeMethods.primes1(movePiece)
                    if newRow == 7:
                        self.fullBoard[newRow][newCol] = piece(900, 'q', 'w')
                        self.blackPoints += 800
                        newP = primeMethods.queenMapping(movePiece)
                        self.blackPieces.append(newP)
                        self.blackAvailableMoves.pop(movePiece)
                        self.blackPieces.remove(movePiece)
                        self.blackIndexes.pop(movePiece)
                        self.blackIToP.pop(initialCoords)
                        self.blackIndexes[newP] = newIndexes
                        self.blackIToP[newP] = newIndexes
                        self.turn += 1
                        # need to return here, because in this special situation, index map updating is completely different.
                        self.fullBoard[oldRow][oldCol] = piece(0, 'n', 'n')
                        return
            self.fullBoard[newRow][newCol] = piece.copy(
                self.fullBoard[oldRow][oldCol])
            self.blackIndexes[movePiece] = newIndexes
            # have to reset all fields to reflect information on the new board.
            self.blackIToP[newIndexes] = movePiece
            self.fullBoard[oldRow][oldCol] = piece(0, 'n', 'n')
            self.turn += 1
            self.blackIToP.pop(initialCoords)

    def EPWhite(self, movePiece, indexes):
        initialCoords = self.whiteIndexes[movePiece]
        oldRow = initialCoords//10
        oldCol = initialCoords % 10
        newI = indexes
        newCol = newI % 10
        newRow = newI//10
        capturedI = oldRow*10+newCol
        oldPiece = self.blackIToP[capturedI]
        self.blackAvailableMoves.pop(oldPiece)
        self.blackIToP.pop(capturedI)
        self.blackIndexes.pop(oldPiece)
        self.blackPieces.remove(oldPiece)
        self.whiteIndexes[movePiece] = newI
        self.whiteIToP.pop(initialCoords)
        # have to reset all fields to reflect information on the new board.
        self.whiteIToP[newI] = movePiece
        self.fullBoard[newRow][newCol] = self.fullBoard[oldRow][oldCol].copy()
        self.fullBoard[oldRow][oldCol] = piece(0, 'n', 'n')
        self.fullBoard[oldRow][newCol] = self.fullBoard[oldRow][oldCol].copy()
        self.blackPoints -= 100
        return

    def EPBlack(self, movePiece, indexes):
        initialCoords = self.blackIndexes[movePiece]
        oldRow = initialCoords//10
        oldCol = initialCoords % 10
        newI = indexes
        newCol = newI % 10
        newRow = newI//10
        capturedI = oldRow*10+newCol
        self.fullBoard[capturedI//10][capturedI % 10] = piece(0, 'n', 'n')
        oldPiece = self.whiteIToP[capturedI]
        self.whiteaVailableMoves.pop(oldPiece)
        self.whiteIToP.pop(capturedI)
        self.whiteIndexes.pop(oldPiece)
        self.whitePieces.remove(oldPiece)
        self.blackIndexes[movePiece] = newI
        self.blackIToP.pop(initialCoords)
        # have to reset all fields to reflect information on the new board.
        self.blackIToP[newI] = movePiece
        self.fullBoard[newRow][newCol] = self.fullBoard[oldRow][oldCol].copy()
        self.fullBoard[oldRow][oldCol] = piece(0, 'n', 'n')
        self.whitePoints -= 100
        return

    def printBoard(self):
        labels = "  A  B  C  D  E  F  G  H"
        for i in range(8):
            row = " ".join([self.format_cell(self.fullBoard[i][j])
                           for j in range(8)])
            print(f"{8-i} {row}")
        print(labels)

    def format_cell(self, cell):
        return f"{cell.team}{cell.kind}"

    def printMirrorBoard(self):
        for i in range(8):
            row = " ".join([self.format_cell(self.fullBoard[7-i][7-j])
                           for j in range(8)])
            print(f"{i+1} {row}")
        print("  H  G  F  E  D  C  B  A")


def generateTopMoves(currGame: board, numMoves: int):
    advantageMap = {}
    results = []
    advantageVals = []

    currGame.allMovesGen()

    pieces = currGame.whitePieces if currGame.turn % 2 == 0 else currGame.blackPieces

    for i in pieces:
        availableMoves = currGame.whiteaVailableMoves if currGame.turn % 2 == 0 else currGame.blackAvailableMoves
        if availableMoves[i] is not None:
            for j in availableMoves[i]:
                placeholder = copy.deepcopy(currGame)
                placeholder.move(i, j)

                placeholder.allMovesGen()
                placeholder.AIAdvantageEval()
                advantage = placeholder.AIAdvantage
                advantageMap[advantage] = [i, j]
                advantageVals.append(advantage)

    while len(advantageVals) <= numMoves:
        numMoves -= 1

    advantageVals.sort(reverse=True)
    for i in range(numMoves):
        advantage = advantageVals[i]
        move = advantageMap[advantage]
        placeholder = copy.deepcopy(currGame)
        placeholder.move(move[0], move[1])
        results.append(placeholder)

    return results


class treeNode:
    def __init__(self, pgame: board, pLevel: int, pParent):
        self.children = []  # list of board classes
        self.level = pLevel
        self.parent = pParent  # previous board, treeNode object
        self.game = pgame

    def incLevel(self, next: list[board]):
        for i in next:
            self.children.append(treeNode(i, self.level+1, self))


# Later, I want the depth to be predetermined by what stage of the game it is, earlier=less depth.
def search(currGame: treeNode, depth: int, alphaBeta: int):
    # search function reaches an error when there is a checkmate. It should just iterate back, I'll look into this.
    currGame.game.printInfo()
    currGame.game.AIAdvantageEval()
    destroy = False
    miniMax = 1000000
    # if game is over, do not branch further on the tree.
    if currGame.level == depth or currGame.game.inPlay == False:
        if currGame.game.AIAdvantage < miniMax:
            miniMax = currGame.game.AIAdvantage
            return miniMax
    # if too bad of an advantage is reached, exit the search function
    elif currGame.game.AIAdvantage < alphaBeta:
        destroy = True
        miniMax = -1000000000
        while currGame.level != 0:
            # go to the top of the tree, set children to none to stop searching
            currGame = currGame.parent
        currGame.children = []  # edit here
        return miniMax
    else:  # not ending, above alphaBeta
        if destroy == False and currGame.children == []:
            topMoves = generateTopMoves(currGame.game, 3)
            # 5 top moves for now, may change this based on how things run
            currGame.incLevel(topMoves)
        for i in currGame.children:
            if destroy == True:
                break
            else:
                mm = search(i, depth, alphaBeta)
                if mm < miniMax:
                    # use backtracking/recursion to generate everything. returning will jump to this statement.
                    miniMax = mm
    return miniMax  # this parameter is what the AI will base each move on


def AImove(game: board):
    # if game.turn<=10:
    # tree=openingMoveTree(game.whitePieces, game.blackPieces, game.whiteIndexes, game.blackIndexes)
    # move=openingMoveGenerator(tree)
    # if len(move)>0:
    # return move
    game.allMovesGen()
    game.AIAdvantageEval()
    if game.inPlay == False:  # prevent calling the search function if game is over
        return
    bestSearch = -100000000
    alphaBeta = -10000000
    moveIndexes = []
    if game.AIteamIsWhite == True:  # AI team taken from player input
        for i in game.whitePieces:
            if game.whiteaVailableMoves[i] != None:
                for j in game.whiteaVailableMoves[i]:
                    reference = copy.deepcopy(game)
                    reference.move(i, j)
                    currSearch = treeNode(reference, 0, None)  # Edit here
                    currScore = search(currSearch, 4, alphaBeta)
                    if bestSearch < currScore:  # eventually I want to figure out algorithms for evaluating depth and alphaBeta based on board conditions, but I need to look at runtimes first.
                        moveIndexes = [i, j]
                        bestSearch = currScore
                    alphaBeta = bestSearch//2+1
    else:
        for i in game.blackPieces:
            if game.blackAvailableMoves[i] != None:
                for j in game.blackAvailableMoves[i]:
                    reference = copy.deepcopy(game)
                    reference.move(i, j)
                    currSearch = treeNode(reference, 0, None)  # Edit here
                    currScore = search(currSearch, 4, alphaBeta)
                    if bestSearch < currScore:  # eventually I want to figure out algorithms for evaluating depth and alphaBeta based on board conditions, but I need to look at runtimes first.
                        moveIndexes = [i, j]
                        bestSearch = currScore
                    alphaBeta = bestSearch//2+1
    return moveIndexes


def reverseMapping(b: str):
    theMap = {
        "A8": 0,  "B8": 1,  "C8": 2,  "D8": 3,  "E8": 4,  "F8": 5,  "G8": 6,  "H8": 7,
        "A7": 10, "B7": 11, "C7": 12, "D7": 13, "E7": 14, "F7": 15, "G7": 16, "H7": 17,
        "A6": 20, "B6": 21, "C6": 22, "D6": 23, "E6": 24, "F6": 25, "G6": 26, "H6": 27,
        "A5": 30, "B5": 31, "C5": 32, "D5": 33, "E5": 34, "F5": 35, "G5": 36, "H5": 37,
        "A4": 40, "B4": 41, "C4": 42, "D4": 43, "E4": 44, "F4": 45, "G4": 46, "H4": 47,
        "A3": 50, "B3": 51, "C3": 52, "D3": 53, "E3": 54, "F3": 55, "G3": 56, "H3": 57,
        "A2": 60, "B2": 61, "C2": 62, "D2": 63, "E2": 64, "F2": 65, "G2": 66, "H2": 67,
        "A1": 70, "B1": 71, "C1": 72, "D1": 73, "E1": 74, "F1": 75, "G1": 76, "H1": 77,
    }
    if b in theMap:
        return theMap[b]
    else:
        return 100000


# this is the beggining of the main

test = board()
stop = True
while stop == False:
    test.allMovesGen()
    print("turn")
    print(test.turn)
    genT = generateTopMoves(test, 5)
    valid = False
    if test.turn % 2 == 0:
        print("White's turn")
        while valid == False:
            test.printBoard()
            a = input()
            b = input()
            c = reverseMapping(b)
            if a in test.whitePieces:
                if c in test.whiteaVailableMoves[a]:
                    valid = True
                else:
                    print("Please enter a valid move")
            else:
                print("Please enter a valid piece")
                continue

        test.move(a, c)
        test.AIAdvantageEval()
        test.printInfo()

        print("Press any key to continue. Enter STOP if you want to exit the game")
        cont = input()
        if cont == 'STOP':
            stop = True
        else:
            continue

    else:
        print("Black's turn")
        while valid == False:
            test.printBoard()
            a = input()
            b = input()
            c = reverseMapping(b)
            if a in test.blackPieces:
                if c in test.blackAvailableMoves[a]:
                    valid = True
                else:
                    print("Please enter a valid move")
                    continue
            else:
                print("Please enter a valid piece")
                continue
        test.move(a, c)
        test.AIAdvantageEval()
        test.printInfo()
        print("Press any key to continue. Enter STOP if you want to exit the game")
        cont = input()
        if cont == 'STOP':
            stop = True
        else:
            continue
stop2 = True

# searchFunctionTest=board()
# searchFunctionTest.allMovesGen()
# searchFunctionTest.AIAdvantageEval()
# searchFunctionTest.printInfo()
# s=treeNode(searchFunctionTest,0, None)

# minimax=search(s, 3, -5000)
# print("minimax value")
# print(minimax)
# c=input()
game2 = board()
x = False
while x == True:
    # print("openingMoveTree(",game2.whitePieces,",", game2.blackPieces, ",", game2.whiteIndexes, ",", game2.blackIndexes, ")")
    game2.allMovesGen()
    game2.printInfo()
    if (game2.turn % 2 == 0):
        print("white turn")
    else:
        print("black turn")
    print("what piece do you want to move?")
    d = False
    a = ""
    b = ""
    while d == False:
        a = input()
        print("indexes?")
        b = input()
        c = reverseMapping(b)
        if game2.turn % 2 == 0:
            if a in game2.whitePieces and c in game2.whiteaVailableMoves[a]:
                d = True
            else:
                print("something is wrong")
        else:
            if a in game2.blackPieces and c in game2.blackAvailableMoves[a]:
                d = True
            else:
                print("something is wrong")
    game2.move(a, c)
    # p5 E4
    # p3 C5
    # k2 F3
    # p4 D6
    # p4 D4
    # p5 D5


theGame = board()
theGame.inPlay = True
print("Which team would you like to play as? Enter W for white or B for black")
validPTeam = False
playerTeam = ""
while validPTeam == False:
    playerTeam = input()
    if playerTeam == "W" or playerTeam == "B":
        validPTeam = True
    else:
        print("Please enter a valid team")
if playerTeam == "W":
    theGame.AIteamIsWhite = False
else:
    theGame.AIteamIsWhite = True


if theGame.AIteamIsWhite == True:

    while theGame.inPlay == True:
        if theGame.turn % 2 == 0:
            AIMove = AImove(theGame)
            print(
                mapping(theGame.whiteIndexes[AIMove[0]]), ' to ', mapping(AIMove[1]))
            theGame.move(AIMove[0], AIMove[1])
            theGame.allMovesGen()
            theGame.AIAdvantageEval()
        else:
            playerMistake = True
            while playerMistake == True:
                playerMoveIndexes = -1
                playerValidPiece = False
                playerValidMove = False
                theGame.printMirrorBoard()
                theGame.allMovesGen()
                playerSTR = ""
                print("Enter the indexes of the piece you want to move. Example: H4")
                while playerValidPiece == False:
                    playerPiece = input()
                    playerIndexes = reverseMapping(playerPiece)
                    if playerIndexes in theGame.blackIToP:
                        playerSTR = theGame.blackIToP[playerIndexes]
                        if len(theGame.blackAvailableMoves[playerSTR]) > 0:
                            playerValidPiece = True
                            playerSTR = theGame.blackIToP[playerIndexes]
                        else:
                            print("Please enter a piece that can move.")
                    else:
                        print("Please enter valid indexes of a black piece.")
                print("Which indexes would you like to move it to?")
                while playerValidMove == False:
                    playerMove = input()
                    playerMoveIndexes = reverseMapping(
                        playerMove)  # AI team is black
                    if playerMoveIndexes in theGame.blackAvailableMoves[playerSTR]:
                        validMovePiece = True
                        break
                    else:
                        print("Please enter valid indexes to move to")
                print("Are you sure you want to make this move? Type in CHANGE if you would like to select a different move. Press enter to continue.")
                con = input()
                if con == "CHANGE":
                    continue
                else:
                    playerMistake = False
                    theGame.move(playerSTR, playerMoveIndexes)
                    theGame.allMovesGen()
                    theGame.AIAdvantageEval()
else:
    while theGame.inPlay == True:
        if theGame.turn % 2 == 1:
            AIMove = AImove(theGame)
            print(
                mapping(theGame.blackIndexes[AIMove[0]]), ' to ', mapping(AIMove[1]))
            theGame.move(AIMove[0], AIMove[1])
            theGame.allMovesGen()
            theGame.AIAdvantageEval()
        else:
            playerMistake = True
            while playerMistake == True:
                playerMoveIndexes = -1
                playerValidPiece = False
                playerValidMove = False
                theGame.allMovesGen()
                theGame.printBoard()
                playerSTR = ""
                print("Enter the indexes of the piece you want to move. Example: H4")
                while playerValidPiece == False:
                    playerPiece = input()
                    playerIndexes = reverseMapping(playerPiece)
                    if playerIndexes in theGame.whiteIToP:
                        playerSTR = theGame.whiteIToP[playerIndexes]
                        if len(theGame.whiteaVailableMoves[playerSTR]) > 0:
                            playerValidPiece = True
                            playerSTR = theGame.whiteIToP[playerIndexes]
                        else:
                            print("Please enter a piece that can move.")
                    else:
                        print("Please enter valid indexes of a white piece.")
                print("Which indexes would you like to move it to?")
                while playerValidMove == False:
                    playerMove = input()
                    playerMoveIndexes = reverseMapping(
                        playerMove)  # AI team is black
                    if playerMoveIndexes in theGame.whiteaVailableMoves[playerSTR]:
                        validMovePiece = True
                        break
                    else:
                        print("Please enter valid indexes to move to")
                print("Are you sure you want to make this move? Type in CHANGE if you would like to select a different move. Press enter to continue.")
                con = input()
                if con == "CHANGE":
                    continue
                else:
                    playerMistake = False
                    theGame.move(playerSTR, playerMoveIndexes)
                    theGame.allMovesGen()
                    theGame.AIAdvantageEval()
theGame.AIAdvantageEval()
a = []
b = a[4]
if theGame.AIAdvantage == 0:
    print("It's a tie")
elif theGame.AIAdvantage < 0:
    print("Checkmate, you win. Congradulations!!!")
else:
    print("Checkmate, you lose. Better luck next time!!!")
# both boards are printed correctly, working on getting the player to move correctly
