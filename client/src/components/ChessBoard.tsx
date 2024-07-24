import { useCallback, useMemo, useState } from "react";
import { Chess, PieceSymbol } from "chess.js";
import { Chessboard } from "react-chessboard";
import {
  Piece,
  PromotionPieceOption,
  Square,
} from "react-chessboard/dist/chessboard/types";

export default function PlayRandomMoveEngine() {
  const [game, setGame] = useState(new Chess());
  const makeAMove = useCallback(
    (
      move:
        | string
        | {
            from: string;
            to: string;
            promotion?: string | undefined;
          }
    ) => {
      // If the move is valid, update the game state with a new Chess instance
      if (move !== null) {
        game.move(move);
        const updatedGame = new Chess(game.fen()); // Create a new Chess instance with the updated FEN
        setGame(updatedGame); // Update the state with the new instance
      }
    },
    [game]
  );

  const makeRandomMove = useCallback(() => {
    const possibleMoves = game.moves();
    if (game.isGameOver() || game.isDraw() || possibleMoves.length === 0)
      return; // exit if the game is over
    const randomIndex = Math.floor(Math.random() * possibleMoves.length);
    makeAMove(possibleMoves[randomIndex]);
  }, [game]);

  const onDrop = useCallback(
    (sourceSquare: Square, targetSquare: Square, piece: Piece) => {
      console.log("we have been dropped", targetSquare);
      if (
        (targetSquare.includes("8") || targetSquare.includes("1")) &&
        piece.includes("P")
      ) {
        console.log("PROMOTIONS!");
        return false;
      }

      const move = makeAMove({
        from: sourceSquare,
        to: targetSquare,
      });

      // illegal move
      if (move === null) return false;

      setTimeout(makeRandomMove, 500);

      return true;
    },
    []
  );

  const onPromotion = useCallback(
    (
      piece?: PromotionPieceOption,
      promoteFromSquare?: Square,
      promoteToSquare?: Square
    ) => {
      if (!piece || !promoteFromSquare || !promoteToSquare) return false;

      const prom = convertToChessJs(piece);

      // it seems like making the promotion switches the turn...
      makeAMove({
        from: promoteFromSquare,
        to: promoteToSquare,
        promotion: prom,
      });
      return true;
    },
    []
  );

  return (
    <Chessboard
      id="chessboard"
      position={game.fen()}
      onPieceDrop={onDrop}
      onPromotionPieceSelect={onPromotion}
    />
  );
}

const convertToChessJs = (piece: PromotionPieceOption): PieceSymbol => {
  // export declare type (chessjs) PieceSymbol = 'p' | 'n' | 'b' | 'r' | 'q' | 'k';
  // export type (chessboard) PromotionPieceOption = "wQ" | "wR" | "wN" | "wB" | "bQ" | "bR" | "bN" | "bB";
  // this could work too:
  // return piece.toLowerCase().split("")[1] as PieceSymbol; // "wQ" -> "q"
  switch (piece) {
    case "bB":
    case "wB":
      return "b";
    case "bN":
    case "wN":
      return "n";
    case "bR":
    case "wR":
      return "r";
    case "bQ":
    case "wQ":
    default:
      return "q";
  }
};
