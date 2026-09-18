import time
import chess
from typing import Optional, List, Tuple
from model import ChessNet, load_model, evaluate_board, compute_heuristic_score, MATERIAL_WEIGHTS

class ChessEngine:
    """
    Intelligent Chess Search Engine combining Negamax with Alpha-Beta Pruning,
    Move Ordering (MVV-LVA), and Neural / Heuristic Evaluation blending.
    """
    def __init__(
        self,
        model_path: str = "chess_model.pkl",
        default_depth: int = 3,
        use_neural: bool = True,
        device: str = "cpu"
    ):
        self.default_depth = default_depth
        self.use_neural = use_neural
        self.device = device
        self.nodes_visited = 0
        
        # Load or initialize neural network model
        self.model = load_model(model_path, device=self.device) if self.use_neural else None

    def evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluates the board from the perspective of the player whose turn it is to move.
        Returns a score where positive means advantage for the active player.
        """
        # 1. Check terminal game-over states
        if board.is_checkmate():
            # Current player is checkmated (loss)
            return -10000.0
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition():
            return 0.0

        # 2. Compute classical material and positional score
        heuristic_score = compute_heuristic_score(board)

        # 3. Compute neural network score (if enabled)
        if self.use_neural and self.model is not None:
            # Neural model outputs score in [-1.0, 1.0] from White's perspective
            nn_score = evaluate_board(board, self.model, device=self.device)
            # Scale nn_score to roughly align with pawn units (-1.0 to 1.0 -> -10.0 to 10.0)
            combined_white_eval = 0.65 * heuristic_score + 0.35 * (nn_score * 8.0)
        else:
            combined_white_eval = heuristic_score

        # Convert to active player's perspective for Negamax
        perspective = 1.0 if board.turn == chess.WHITE else -1.0
        return combined_white_eval * perspective

    def score_move(self, board: chess.Board, move: chess.Move) -> int:
        """
        Assigns a heuristic priority score to a move to optimize Alpha-Beta branch ordering.
        Prioritizes captures (MVV-LVA), promotions, and checks.
        """
        score = 0

        # 1. Pawn Promotion
        if move.promotion:
            score += 900

        # 2. Captures: Most Valuable Victim - Least Valuable Attacker (MVV-LVA)
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)

            victim_val = int(MATERIAL_WEIGHTS.get(victim.piece_type, 1.0) * 100) if victim else 100
            attacker_val = int(MATERIAL_WEIGHTS.get(attacker.piece_type, 1.0) * 10) if attacker else 10
            score += 1000 + (victim_val - attacker_val)

        # 3. Checks: Prioritize moves that give check
        board.push(move)
        if board.is_check():
            score += 200
        board.pop()

        return score

    def order_moves(self, board: chess.Board, legal_moves: List[chess.Move]) -> List[chess.Move]:
        """
        Sorts legal moves in descending order of tactical promise.
        Examining strong moves first triggers immediate Beta Cutoffs in Alpha-Beta.
        """
        scored_moves = [(self.score_move(board, move), move) for move in legal_moves]
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in scored_moves]

    def quiescence_search(self, board: chess.Board, alpha: float, beta: float, max_depth: int = 4) -> float:
        """
        Quiescence Search: Continues searching tactical capture moves beyond the depth horizon
        to avoid the 'Horizon Effect' where an exchange is left unresolved.
        """
        self.nodes_visited += 1

        # Stand-pat score: evaluating without making any more moves
        stand_pat = self.evaluate_position(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        if max_depth <= 0 or board.is_game_over():
            return alpha

        # Generate only capture moves
        capture_moves = [move for move in board.legal_moves if board.is_capture(move)]
        ordered_captures = self.order_moves(board, capture_moves)

        for move in ordered_captures:
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha, max_depth - 1)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        ply: int = 0
    ) -> float:
        """
        Recursive Negamax algorithm with Alpha-Beta Pruning.
        
        alpha: best score the maximizing player can guarantee
        beta:  best score the minimizing opponent will allow
        """
        self.nodes_visited += 1

        # Check terminal conditions
        if board.is_checkmate():
            # Prefer shorter paths to checkmate: subtract ply penalty
            return -10000.0 + ply
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition():
            return 0.0

        # Horizon reached: resolve tactical captures via Quiescence Search
        if depth <= 0:
            return self.quiescence_search(board, alpha, beta)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return self.evaluate_position(board)

        # Move ordering for rapid cutoffs
        ordered_moves = self.order_moves(board, legal_moves)

        for move in ordered_moves:
            board.push(move)
            # Invert perspective and search child
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()

            # Beta cutoff: opponent had a better alternative earlier, prune branch!
            if score >= beta:
                return beta

            # Update alpha: we found a new best guaranteed score
            if score > alpha:
                alpha = score

        return alpha

    def get_best_move(
        self,
        board: chess.Board,
        depth: Optional[int] = None
    ) -> Tuple[Optional[chess.Move], float]:
        """
        Finds the optimal legal move for the current position using Alpha-Beta search.
        Returns: (best_move, evaluation_score)
        """
        search_depth = depth if depth is not None else self.default_depth
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None, 0.0

        # If only one legal move exists, play it immediately without wasting compute
        if len(legal_moves) == 1:
            return legal_moves[0], self.evaluate_position(board)

        self.nodes_visited = 0
        start_time = time.time()

        best_move = None
        best_score = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        ordered_moves = self.order_moves(board, legal_moves)

        for move in ordered_moves:
            board.push(move)
            score = -self.negamax(board, search_depth - 1, -beta, -alpha, ply=1)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

        elapsed = time.time() - start_time
        nps = int(self.nodes_visited / elapsed) if elapsed > 0 else self.nodes_visited
        print(f"[Engine] Depth: {search_depth} | Best Move: {best_move} | Eval: {best_score:.2f} | Nodes: {self.nodes_visited} | Time: {elapsed:.2f}s ({nps} nps)")

        return best_move, best_score


if __name__ == "__main__":
    # Quick self-test of the search engine
    print("--- Testing engine.py ---")
    test_board = chess.Board()
    engine = ChessEngine(default_depth=3, use_neural=False)
    
    move, score = engine.get_best_move(test_board, depth=3)
    print(f"Engine selected root move: {move} with score: {score:.2f}")

    # Tactical test: Scholar's mate threat test
    # e4 e5, Qh5 Nc6, Bc4 Nf6?? (White can play Qxf7#)
    mate_board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    print(f"\nTactical Mate-in-1 Test Board:\n{mate_board}")
    mate_move, mate_score = engine.get_best_move(mate_board, depth=2)
    print(f"Engine move in mate position: {mate_move} (Expected: Qxf7# / h5f7)")
    assert str(mate_move) == "h5f7", f"Engine failed to find Qxf7# mate! Selected: {mate_move}"
    print("Test passed successfully! Engine found Scholar's Mate.")
