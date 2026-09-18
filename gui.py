import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Set
import chess

from engine import ChessEngine

# High-resolution Unicode Chess Glyphs
UNICODE_PIECES = {
    # White pieces
    chess.Piece(chess.PAWN, chess.WHITE): "♙",
    chess.Piece(chess.KNIGHT, chess.WHITE): "♘",
    chess.Piece(chess.BISHOP, chess.WHITE): "♗",
    chess.Piece(chess.ROOK, chess.WHITE): "♖",
    chess.Piece(chess.QUEEN, chess.WHITE): "♕",
    chess.Piece(chess.KING, chess.WHITE): "♔",
    # Black pieces
    chess.Piece(chess.PAWN, chess.BLACK): "♟",
    chess.Piece(chess.KNIGHT, chess.BLACK): "♞",
    chess.Piece(chess.BISHOP, chess.BLACK): "♝",
    chess.Piece(chess.ROOK, chess.BLACK): "♜",
    chess.Piece(chess.QUEEN, chess.BLACK): "♛",
    chess.Piece(chess.KING, chess.BLACK): "♚",
}

# Aesthetic Color Palette
COLOR_LIGHT_SQUARE = "#F0D9B5"     # Warm cream
COLOR_DARK_SQUARE = "#B58863"      # Rich hazelnut walnut
COLOR_SELECTED = "#F6D04D"         # Amber highlight
COLOR_LAST_MOVE = "#D4E157"        # Pale olive yellow
COLOR_CHECK = "#EF5350"            # Soft crimson warning
COLOR_HINT_DOT = "#558B2F"         # Olive green dot for empty targets
COLOR_HINT_CAPTURE = "#E53935"     # Scarlet ring for captures
BG_DARK = "#262421"                # Chess.com style dark sidebar
BG_PANEL = "#312E2B"
FG_TEXT = "#FFFFFF"
FG_MUTED = "#BABABA"
ACCENT_GREEN = "#81B64C"


class ChessGUI:
    """
    Modern, responsive Tkinter Graphical User Interface for Chess AI.
    Runs AI calculation on a background thread to prevent interface lag.
    """
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Chess-Made-Intelligent | Human vs AI")
        self.root.geometry("900x640")
        self.root.resizable(False, False)
        self.root.configure(bg=BG_DARK)

        # Core Game State
        self.board = chess.Board()
        self.engine = ChessEngine(default_depth=3, use_neural=True)
        self.human_color = chess.WHITE
        self.selected_square: Optional[chess.Square] = None
        self.legal_destinations: Set[chess.Square] = set()
        self.last_move: Optional[chess.Move] = None
        self.is_ai_thinking = False

        # Display constants
        self.square_size = 72
        self.board_size = self.square_size * 8

        self._init_ui()
        self._draw_board()

    def _init_ui(self):
        """Constructs layout: Board on left, controls & move ledger on right."""
        main_frame = tk.Frame(self.root, bg=BG_DARK)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 1. Left Frame: Chessboard Canvas
        canvas_container = tk.Frame(main_frame, bg="#1E1C1A", padx=4, pady=4, relief=tk.RAISED, bd=2)
        canvas_container.pack(side=tk.LEFT, padx=(0, 20))

        self.canvas = tk.Canvas(
            canvas_container,
            width=self.board_size,
            height=self.board_size,
            highlightthickness=0,
            bg=COLOR_DARK_SQUARE
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # 2. Right Frame: Control Panel & Status
        sidebar = tk.Frame(main_frame, bg=BG_PANEL, width=280, padx=16, pady=16, relief=tk.GROOVE, bd=1)
        sidebar.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Title
        title_label = tk.Label(
            sidebar,
            text="CHESS A.I.",
            font=("Helvetica", 18, "bold"),
            bg=BG_PANEL,
            fg=ACCENT_GREEN
        )
        title_label.pack(anchor=tk.W, pady=(0, 4))

        subtitle = tk.Label(
            sidebar,
            text="Deep Search & Neural Evaluation",
            font=("Helvetica", 9),
            bg=BG_PANEL,
            fg=FG_MUTED
        )
        subtitle.pack(anchor=tk.W, pady=(0, 14))

        # Status Badge
        self.status_var = tk.StringVar(value="Your turn (White)")
        self.status_label = tk.Label(
            sidebar,
            textvariable=self.status_var,
            font=("Helvetica", 12, "bold"),
            bg="#211F1D",
            fg=FG_TEXT,
            padx=12,
            pady=8,
            relief=tk.FLAT
        )
        self.status_label.pack(fill=tk.X, pady=(0, 12))

        # Evaluation indicator
        self.eval_var = tk.StringVar(value="Evaluation: 0.00")
        self.eval_label = tk.Label(
            sidebar,
            textvariable=self.eval_var,
            font=("Helvetica", 10),
            bg=BG_PANEL,
            fg=FG_MUTED
        )
        self.eval_label.pack(anchor=tk.W, pady=(0, 10))

        # Difficulty Dropdown
        diff_frame = tk.Frame(sidebar, bg=BG_PANEL)
        diff_frame.pack(fill=tk.X, pady=(0, 12))
        
        diff_lbl = tk.Label(diff_frame, text="AI Search Depth:", font=("Helvetica", 10, "bold"), bg=BG_PANEL, fg=FG_TEXT)
        diff_lbl.pack(side=tk.LEFT)

        self.depth_var = tk.IntVar(value=3)
        self.depth_menu = ttk.Combobox(
            diff_frame,
            textvariable=self.depth_var,
            values=[1, 2, 3, 4],
            width=5,
            state="readonly"
        )
        self.depth_menu.pack(side=tk.RIGHT)
        self.depth_menu.bind("<<ComboboxSelected>>", self._on_depth_change)

        # Move History Box
        history_lbl = tk.Label(sidebar, text="Move History:", font=("Helvetica", 10, "bold"), bg=BG_PANEL, fg=FG_TEXT)
        history_lbl.pack(anchor=tk.W, pady=(0, 4))

        history_frame = tk.Frame(sidebar, bg="#211F1D")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 14))

        self.history_listbox = tk.Listbox(
            history_frame,
            bg="#211F1D",
            fg=FG_TEXT,
            font=("Courier", 10),
            highlightthickness=0,
            relief=tk.FLAT,
            selectbackground=ACCENT_GREEN
        )
        history_scroll = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_listbox.yview)
        self.history_listbox.configure(yscrollcommand=history_scroll.set)

        self.history_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Action Buttons Frame
        btn_frame = tk.Frame(sidebar, bg=BG_PANEL)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)

        btn_style = {
            "font": ("Helvetica", 10, "bold"),
            "bg": "#4A4642",
            "fg": FG_TEXT,
            "activebackground": "#5D5853",
            "activeforeground": FG_TEXT,
            "relief": tk.FLAT,
            "pady": 6
        }

        self.btn_new = tk.Button(btn_frame, text="New Game", command=self._reset_game, **btn_style)
        self.btn_new.pack(fill=tk.X, pady=(0, 6))

        self.btn_undo = tk.Button(btn_frame, text="Undo Move", command=self._undo_move, **btn_style)
        self.btn_undo.pack(fill=tk.X, pady=(0, 6))

        self.btn_flip = tk.Button(btn_frame, text="Flip Colors (Play Black)", command=self._flip_colors, **btn_style)
        self.btn_flip.pack(fill=tk.X)

    def _square_to_coords(self, square: chess.Square):
        """Converts chess square index (0-63) to GUI canvas (x1, y1, x2, y2)."""
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        # If playing as White, Rank 8 is at top (row 0), Rank 1 at bottom (row 7)
        # If playing as Black, board is inverted
        if self.human_color == chess.WHITE:
            col = file
            row = 7 - rank
        else:
            col = 7 - file
            row = rank

        x1 = col * self.square_size
        y1 = row * self.square_size
        x2 = x1 + self.square_size
        y2 = y1 + self.square_size
        return x1, y1, x2, y2

    def _coords_to_square(self, x: int, y: int) -> Optional[chess.Square]:
        """Converts pixel click (x, y) into a chess square index."""
        col = x // self.square_size
        row = y // self.square_size

        if not (0 <= col < 8 and 0 <= row < 8):
            return None

        if self.human_color == chess.WHITE:
            file = col
            rank = 7 - row
        else:
            file = 7 - col
            rank = row

        return chess.square(file, rank)

    def _draw_board(self):
        """Redraws the entire board, highlights, pieces, and move indicators."""
        self.canvas.delete("all")

        # 1. Draw 64 squares
        for rank in range(8):
            for file in range(8):
                sq = chess.square(file, rank)
                x1, y1, x2, y2 = self._square_to_coords(sq)
                is_light = (file + rank) % 2 != 0
                color = COLOR_LIGHT_SQUARE if is_light else COLOR_DARK_SQUARE

                # Highlight last move squares
                if self.last_move and (sq == self.last_move.from_square or sq == self.last_move.to_square):
                    color = COLOR_LAST_MOVE

                # Highlight King in check
                piece = self.board.piece_at(sq)
                if piece and piece.piece_type == chess.KING and piece.color == self.board.turn and self.board.is_check():
                    color = COLOR_CHECK

                # Highlight selected square
                if sq == self.selected_square:
                    color = COLOR_SELECTED

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="", tags="square")

        # 2. Draw Rank and File Labels on borders
        for i in range(8):
            f_letter = chr(ord('a') + (i if self.human_color == chess.WHITE else 7 - i))
            r_number = str(8 - i if self.human_color == chess.WHITE else i + 1)
            
            # File label on bottom right of rank 1
            self.canvas.create_text(
                i * self.square_size + 10,
                self.board_size - 10,
                text=f_letter,
                font=("Helvetica", 9, "bold"),
                fill="#7D7565"
            )
            # Rank label on top left of file a
            self.canvas.create_text(
                self.board_size - 10,
                i * self.square_size + 10,
                text=r_number,
                font=("Helvetica", 9, "bold"),
                fill="#7D7565"
            )

        # 3. Draw Legal Move Destination Markers
        for dest in self.legal_destinations:
            x1, y1, x2, y2 = self._square_to_coords(dest)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            target_piece = self.board.piece_at(dest)

            if target_piece:
                # Enemy piece: draw capture ring
                r = self.square_size // 2 - 4
                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=COLOR_HINT_CAPTURE, width=4)
            else:
                # Empty square: draw small circular dot
                r = 10
                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=COLOR_HINT_DOT, outline="")

        # 4. Draw Pieces
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece:
                x1, y1, x2, y2 = self._square_to_coords(sq)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                glyph = UNICODE_PIECES.get(piece, "")
                
                # Piece shadow for high contrast & 3D look
                self.canvas.create_text(
                    cx + 1, cy + 2,
                    text=glyph,
                    font=("Segoe UI Symbol", 40),
                    fill="#1A1A1A"
                )
                
                # Actual piece color
                piece_color = "#FFFFFF" if piece.color == chess.WHITE else "#1E1E1E"
                self.canvas.create_text(
                    cx, cy,
                    text=glyph,
                    font=("Segoe UI Symbol", 40),
                    fill=piece_color
                )

    def _on_canvas_click(self, event):
        """Handles human user interaction with mouse clicks."""
        if self.is_ai_thinking or self.board.is_game_over():
            return

        # It must be the human's turn
        if self.board.turn != self.human_color:
            return

        clicked_sq = self._coords_to_square(event.x, event.y)
        if clicked_sq is None:
            return

        # Case 1: Selecting a destination square for a previously selected piece
        if self.selected_square is not None and clicked_sq in self.legal_destinations:
            move = chess.Move(self.selected_square, clicked_sq)
            
            # Handle Pawn Promotion (auto-promote to Queen)
            selected_piece = self.board.piece_at(self.selected_square)
            if selected_piece and selected_piece.piece_type == chess.PAWN:
                target_rank = chess.square_rank(clicked_sq)
                if (selected_piece.color == chess.WHITE and target_rank == 7) or \
                   (selected_piece.color == chess.BLACK and target_rank == 0):
                    move.promotion = chess.QUEEN

            if move in self.board.legal_moves:
                self._make_move(move)
                self.selected_square = None
                self.legal_destinations.clear()
                self._draw_board()

                # Trigger AI reply
                if not self.board.is_game_over():
                    self._start_ai_turn()
                return

        # Case 2: Selecting a friendly piece to move
        piece = self.board.piece_at(clicked_sq)
        if piece and piece.color == self.human_color:
            self.selected_square = clicked_sq
            # Calculate all legal destination squares for this piece
            self.legal_destinations = {
                m.to_square for m in self.board.legal_moves if m.from_square == clicked_sq
            }
        else:
            self.selected_square = None
            self.legal_destinations.clear()

        self._draw_board()

    def _make_move(self, move: chess.Move):
        """Applies a move to the board and appends it to the move history listbox."""
        san = self.board.san(move)
        move_num = self.board.fullmove_number

        if self.board.turn == chess.WHITE:
            self.history_listbox.insert(tk.END, f"{move_num}. {san}")
        else:
            # Append Black move to the last line
            last_idx = self.history_listbox.size() - 1
            if last_idx >= 0:
                prev_text = self.history_listbox.get(last_idx)
                self.history_listbox.delete(last_idx)
                self.history_listbox.insert(tk.END, f"{prev_text}   {san}")
            else:
                self.history_listbox.insert(tk.END, f"{move_num}... {san}")

        self.history_listbox.yview(tk.END)
        self.board.push(move)
        self.last_move = move
        self._update_status()

    def _start_ai_turn(self):
        """Spawns a daemon thread for AI search to prevent GUI lockup."""
        self.is_ai_thinking = True
        self.status_var.set("AI is calculating...")
        self.status_label.configure(bg="#D97706")  # Amber warning color
        
        # Threaded search execution
        thread = threading.Thread(target=self._run_ai_search, daemon=True)
        thread.start()

    def _run_ai_search(self):
        """Worker thread function executing Alpha-Beta search."""
        depth = self.depth_var.get()
        best_move, score = self.engine.get_best_move(self.board, depth=depth)
        
        # Dispatch move back onto Tkinter main thread
        self.root.after(0, self._finish_ai_turn, best_move, score)

    def _finish_ai_turn(self, move: Optional[chess.Move], score: float):
        """Executes on the Tkinter main thread once AI calculation completes."""
        self.is_ai_thinking = False

        if move and move in self.board.legal_moves:
            self._make_move(move)

        eval_sign = "+" if score > 0 else ""
        self.eval_var.set(f"Evaluation: {eval_sign}{score:.2f}")

        self._draw_board()
        self._update_status()

    def _update_status(self):
        """Updates the game status indicator, check status, and game-over modals."""
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            self.status_var.set(f"Checkmate! {winner} wins!")
            self.status_label.configure(bg="#DC2626")  # Red
            messagebox.showinfo("Game Over", f"Checkmate! {winner} wins!")
        elif self.board.is_stalemate():
            self.status_var.set("Draw: Stalemate")
            self.status_label.configure(bg="#4B5563")
            messagebox.showinfo("Game Over", "Game drawn by stalemate.")
        elif self.board.is_insufficient_material():
            self.status_var.set("Draw: Insufficient Material")
            self.status_label.configure(bg="#4B5563")
        elif self.board.can_claim_threefold_repetition():
            self.status_var.set("Draw: Threefold Repetition")
            self.status_label.configure(bg="#4B5563")
        elif self.board.is_check():
            turn_name = "White" if self.board.turn == chess.WHITE else "Black"
            self.status_var.set(f"Check! ({turn_name}'s turn)")
            self.status_label.configure(bg="#DC2626")
        else:
            if self.board.turn == self.human_color:
                color_name = "White" if self.human_color == chess.WHITE else "Black"
                self.status_var.set(f"Your turn ({color_name})")
                self.status_label.configure(bg="#211F1D")
            else:
                self.status_var.set("AI thinking...")
                self.status_label.configure(bg="#D97706")

    def _reset_game(self):
        """Restarts the game from the initial position."""
        if self.is_ai_thinking:
            return
        self.board.reset()
        self.selected_square = None
        self.legal_destinations.clear()
        self.last_move = None
        self.history_listbox.delete(0, tk.END)
        self.eval_var.set("Evaluation: 0.00")
        self._update_status()
        self._draw_board()

        if self.human_color == chess.BLACK:
            self._start_ai_turn()

    def _undo_move(self):
        """Reverts the last two half-moves (one AI move and one human move)."""
        if self.is_ai_thinking:
            return

        # Undo twice to return to human's turn
        if len(self.board.move_stack) >= 2:
            self.board.pop()
            self.board.pop()
            # Remove last history entry
            if self.history_listbox.size() > 0:
                self.history_listbox.delete(self.history_listbox.size() - 1)
        elif len(self.board.move_stack) == 1:
            self.board.pop()
            if self.history_listbox.size() > 0:
                self.history_listbox.delete(self.history_listbox.size() - 1)

        self.last_move = self.board.peek() if self.board.move_stack else None
        self.selected_square = None
        self.legal_destinations.clear()
        self._update_status()
        self._draw_board()

    def _flip_colors(self):
        """Swaps human player color between White and Black."""
        if self.is_ai_thinking:
            return
        self.human_color = chess.BLACK if self.human_color == chess.WHITE else chess.WHITE
        new_name = "Black" if self.human_color == chess.BLACK else "White"
        self.btn_flip.configure(text=f"Flip Colors (Play {'White' if self.human_color == chess.BLACK else 'Black'})")
        self._reset_game()

    def _on_depth_change(self, event):
        """Updates AI search depth on dropdown selection."""
        new_depth = self.depth_var.get()
        self.engine.default_depth = new_depth
        print(f"[GUI] Search depth changed to: {new_depth}")


def main():
    root = tk.Tk()
    app = ChessGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
