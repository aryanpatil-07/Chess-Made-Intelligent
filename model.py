import os
import random
import pickle
import chess
import numpy as np
from typing import Tuple, Optional, Union

# Try importing PyTorch; if unavailable or DLL fails, seamlessly use NumPy neural engine
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False

# Mapping of standard piece types to channel index (0 to 5)
PIECE_INDICES = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

# Standard material weights for synthetic training data generation & heuristic evaluation
MATERIAL_WEIGHTS = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.2,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

# Piece-Square bonus tables (favoring central control, knight development, pawn advancement)
PAWN_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     1,  1,  2,  5,  5,  2,  1,  1,
     0,  0,  2,  6,  6,  2,  0,  0,
     1,  1,  2,  5,  5,  2,  1,  1,
     2,  3,  3,  4,  4,  3,  3,  2,
     5,  5,  5,  5,  5,  5,  5,  5,
     0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_TABLE = [
    -5,-4,-3,-3,-3,-3,-4,-5,
    -4,-2, 0, 1, 1, 0,-2,-4,
    -3, 1, 2, 3, 3, 2, 1,-3,
    -3, 0, 3, 4, 4, 3, 0,-3,
    -3, 1, 3, 4, 4, 3, 1,-3,
    -3, 0, 2, 3, 3, 2, 0,-3,
    -4,-2, 0, 1, 1, 0,-2,-4,
    -5,-4,-3,-3,-3,-3,-4,-5
]

BISHOP_TABLE = [
    -2,-1,-1,-1,-1,-1,-1,-2,
    -1, 1, 0, 0, 0, 0, 1,-1,
    -1, 1, 1, 1, 1, 1, 1,-1,
    -1, 0, 2, 2, 2, 2, 0,-1,
    -1, 1, 1, 2, 2, 1, 1,-1,
    -1, 2, 2, 1, 1, 2, 2,-1,
    -1, 1, 0, 0, 0, 0, 1,-1,
    -2,-1,-1,-1,-1,-1,-1,-2
]


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Converts a chess.Board state into a 12x8x8 spatial tensor (NumPy array).
    
    Channels 0-5:  White pieces [Pawn, Knight, Bishop, Rook, Queen, King]
    Channels 6-11: Black pieces [Pawn, Knight, Bishop, Rook, Queen, King]
    
    Each channel is an 8x8 matrix where 1.0 represents the presence of that piece.
    """
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            row = square // 8
            col = square % 8
            channel_offset = 0 if piece.color == chess.WHITE else 6
            channel = PIECE_INDICES[piece.piece_type] + channel_offset
            tensor[channel, row, col] = 1.0

    return tensor


class NumpyChessNet:
    """
    First-Principles Multi-Layer Perceptron (MLP) Neural Network.
    
    Architecture:
      Input: 12 * 8 * 8 = 768 features
      Dense 1: 768 -> 256 + ReLU
      Dense 2: 256 -> 64  + ReLU
      Dense 3: 64  -> 1   + Tanh
      
    Outputs scalar evaluation score in [-1.0, +1.0].
    Provides 100% pure mathematical neural inference with zero DLL dependencies.
    """
    def __init__(self):
        # He (Kaiming) normal initialization
        np.random.seed(42)
        self.w1 = np.random.randn(768, 256).astype(np.float32) * np.sqrt(2.0 / 768)
        self.b1 = np.zeros((256,), dtype=np.float32)

        self.w2 = np.random.randn(256, 64).astype(np.float32) * np.sqrt(2.0 / 256)
        self.b2 = np.zeros((64,), dtype=np.float32)

        self.w3 = np.random.randn(64, 1).astype(np.float32) * np.sqrt(2.0 / 64)
        self.b3 = np.zeros((1,), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward propagation. Accepts tensor of shape (..., 12, 8, 8) or (..., 768).
        """
        # Flatten input to (batch_size, 768)
        if x.ndim == 3:
            x = x.reshape(1, 768)
        elif x.ndim == 4:
            x = x.reshape(x.shape[0], 768)

        # Layer 1: Linear + ReLU
        z1 = np.dot(x, self.w1) + self.b1
        a1 = np.maximum(0, z1)

        # Layer 2: Linear + ReLU
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = np.maximum(0, z2)

        # Layer 3: Linear + Tanh
        z3 = np.dot(a2, self.w3) + self.b3
        out = np.tanh(z3)
        return out

    def eval(self):
        pass  # Inference mode

    def save(self, filepath: str = "chess_model.pkl"):
        """Saves weights dictionary to disk."""
        weights = {
            "w1": self.w1, "b1": self.b1,
            "w2": self.w2, "b2": self.b2,
            "w3": self.w3, "b3": self.b3,
        }
        with open(filepath, "wb") as f:
            pickle.dump(weights, f)
        print(f"[Model] Saved weights to '{filepath}'.")

    def load(self, filepath: str = "chess_model.pkl") -> bool:
        """Loads weights from disk."""
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    weights = pickle.load(f)
                self.w1, self.b1 = weights["w1"], weights["b1"]
                self.w2, self.b2 = weights["w2"], weights["b2"]
                self.w3, self.b3 = weights["w3"], weights["b3"]
                return True
            except Exception as e:
                print(f"[Model] Error loading checkpoint: {e}")
        return False


# If PyTorch is available, also define PyTorch equivalent ChessNet
if TORCH_AVAILABLE:
    class TorchChessNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(12 * 8 * 8, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Tanh()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 3:
                x = x.unsqueeze(0)
            return self.network(x)

    ChessNet = TorchChessNet
else:
    ChessNet = NumpyChessNet


def compute_heuristic_score(board: chess.Board) -> float:
    """
    Calculates a fast heuristic evaluation based on material balance and positional tables.
    Positive scores favor White; negative scores favor Black.
    """
    if board.is_checkmate():
        return -100.0 if board.turn == chess.WHITE else 100.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    white_score = 0.0
    black_score = 0.0

    for square, piece in board.piece_map().items():
        base_val = MATERIAL_WEIGHTS.get(piece.piece_type, 0.0)
        
        # Positional bonus
        pos_bonus = 0.0
        sq_idx = square if piece.color == chess.WHITE else chess.square_mirror(square)
        if piece.piece_type == chess.PAWN:
            pos_bonus = PAWN_TABLE[sq_idx] * 0.01
        elif piece.piece_type == chess.KNIGHT:
            pos_bonus = KNIGHT_TABLE[sq_idx] * 0.01
        elif piece.piece_type == chess.BISHOP:
            pos_bonus = BISHOP_TABLE[sq_idx] * 0.01

        total_piece_val = base_val + pos_bonus

        if piece.color == chess.WHITE:
            white_score += total_piece_val
        else:
            black_score += total_piece_val

    return white_score - black_score


def evaluate_board(board: chess.Board, model: Union[NumpyChessNet, "ChessNet"], device: str = "cpu") -> float:
    """
    Evaluates a chess position using the neural network.
    Returns scalar evaluation in [-1.0, 1.0].
    """
    tensor = board_to_tensor(board)
    if isinstance(model, NumpyChessNet):
        score = model.forward(tensor)
        return float(score[0, 0])
    elif TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
        model.eval()
        t_tensor = torch.from_numpy(tensor).unsqueeze(0).to(device)
        with torch.no_grad():
            score = model(t_tensor).item()
        return float(score)
    return 0.0


def generate_training_data(num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic chess game positions and labels them using heuristic scoring.
    Target scores are compressed into [-1.0, 1.0] using np.tanh(score / 6.0).
    """
    boards_tensors = []
    target_scores = []

    board = chess.Board()
    for _ in range(num_samples):
        if board.is_game_over() or board.fullmove_number > 60:
            board.reset()

        legal_moves = list(board.legal_moves)
        if legal_moves:
            move = random.choice(legal_moves)
            board.push(move)

        tensor = board_to_tensor(board)
        raw_score = compute_heuristic_score(board)
        normalized_score = float(np.tanh(raw_score / 6.0))

        boards_tensors.append(tensor)
        target_scores.append([normalized_score])

    X = np.stack(boards_tensors, axis=0)
    y = np.array(target_scores, dtype=np.float32)
    return X, y


def train_model(
    model: Optional[Union[NumpyChessNet, "ChessNet"]] = None,
    num_samples: int = 1500,
    epochs: int = 10,
    lr: float = 0.005,
    save_path: str = "chess_model.pkl"
) -> Union[NumpyChessNet, "ChessNet"]:
    """
    Trains the neural evaluation model using gradient descent and Mean Squared Error.
    """
    if model is None:
        model = NumpyChessNet()

    print(f"[Training] Generating {num_samples} sample board states...")
    X, y = generate_training_data(num_samples)
    X_flat = X.reshape(X.shape[0], 768)

    print(f"[Training] Training NumpyChessNet for {epochs} epochs (Samples: {num_samples}, LR: {lr})...")
    # Mini-batch gradient descent with momentum
    batch_size = 64
    num_batches = len(X) // batch_size

    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(len(X))
        total_loss = 0.0

        for b in range(num_batches):
            batch_idx = indices[b * batch_size:(b + 1) * batch_size]
            bx = X_flat[batch_idx]
            by = y[batch_idx]

            # Forward pass
            z1 = np.dot(bx, model.w1) + model.b1
            a1 = np.maximum(0, z1)
            z2 = np.dot(a1, model.w2) + model.b2
            a2 = np.maximum(0, z2)
            z3 = np.dot(a2, model.w3) + model.b3
            pred = np.tanh(z3)

            # MSE Loss
            loss = np.mean((pred - by) ** 2)
            total_loss += loss

            # Backpropagation
            d_pred = 2.0 * (pred - by) / batch_size
            d_z3 = d_pred * (1.0 - pred ** 2)  # d/dz tanh(z) = 1 - tanh^2(z)
            d_w3 = np.dot(a2.T, d_z3)
            d_b3 = np.sum(d_z3, axis=0)

            d_a2 = np.dot(d_z3, model.w3.T)
            d_z2 = d_a2 * (z2 > 0)             # d/dz ReLU
            d_w2 = np.dot(a1.T, d_z2)
            d_b2 = np.sum(d_z2, axis=0)

            d_a1 = np.dot(d_z2, model.w2.T)
            d_z1 = d_a1 * (z1 > 0)             # d/dz ReLU
            d_w1 = np.dot(bx.T, d_z1)
            d_b1 = np.sum(d_z1, axis=0)

            # Parameter updates
            model.w3 -= lr * d_w3
            model.b3 -= lr * d_b3
            model.w2 -= lr * d_w2
            model.b2 -= lr * d_b2
            model.w1 -= lr * d_w1
            model.b1 -= lr * d_b1

        avg_loss = total_loss / num_batches
        if epoch % 2 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:2d}/{epochs:2d} | MSE Loss: {avg_loss:.6f}")

    model.save(save_path)
    return model


def load_model(model_path: str = "chess_model.pkl", device: str = "cpu") -> Union[NumpyChessNet, "ChessNet"]:
    """
    Loads model checkpoint if exists, otherwise initializes a new network.
    """
    net = NumpyChessNet()
    if os.path.exists(model_path):
        if net.load(model_path):
            print(f"[Model] Successfully loaded weights from '{model_path}'.")
        else:
            print(f"[Model] Initialized fresh neural weights.")
    else:
        print(f"[Model] No checkpoint at '{model_path}'. Initialized fresh neural weights.")
    return net


if __name__ == "__main__":
    print("--- Testing model.py ---")
    b = chess.Board()
    t = board_to_tensor(b)
    print(f"Board tensor shape: {t.shape} | Active piece entries: {int(np.sum(t))}")
    assert t.shape == (12, 8, 8), "Invalid tensor shape!"
    assert np.sum(t) == 32.0, "Starting board must have exactly 32 pieces!"

    net = NumpyChessNet()
    score = evaluate_board(b, net)
    print(f"Neural evaluation on starting position: {score:.4f}")

    # Train for a few epochs
    train_model(net, num_samples=600, epochs=4, lr=0.005, save_path="chess_model.pkl")
    post_score = evaluate_board(b, net)
    print(f"Post-training evaluation on starting position: {post_score:.4f}")
    print("model.py self-test completed successfully!")