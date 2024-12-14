import io
import torch
import torch.nn as nn
import numpy as np
import chess.pgn
from chess import Board
import torch.nn.functional as F

PIECE_CHARS = "♔♕♖♗♘♙⭘♟♞♝♜♛♚"

def encode_board(board: Board) -> np.array:
    # String-encode the board.
    # If board.turn = 1 then it is now white's turn which means this is a potential move
    # being contemplated by black, and therefore we reverse the char order to rotate the board
    # for black's perspective
    # If board.turn = 0 then it is now black's turn which means this is a potential move
    # being contemplated by white, and therefore we leave the char order as white's perspective.
    # Also reverse PIECE_CHARS indexing order if black's turn to reflect "my" and "opponent" pieces.
    step = 1 - 2 * board.turn
    unicode = board.unicode().replace(' ','').replace('\n','')[::step]
    return np.array([PIECE_CHARS[::step].index(c) for c in unicode], dtype=int).reshape(8,8)

class HalfKPInputLayer(nn.Module):
    """
    HalfKP input layer, which encodes the 10 non-king pieces on 64 squares for each side's king position.
    """
    def __init__(self):
        super().__init__()
        self.input_weights = nn.Parameter(torch.randint(low=-32768, high=32767, size=(64, 640 + 1, 256), dtype=torch.int16)) # 64x(64x10+1)x256
        self.bias = nn.Parameter(torch.zeros(256, dtype=torch.int16))

    def forward(self, piece_positions, king_positions):
        """ 
        Input: 
        - piece_positions: (batch_size, 64, 10) boolean tensor where each entry indicates presence of a piece.
        - king_positions: (batch_size, 2) integers representing the indices of white and black kings on the board.
        """
        batch_size = piece_positions.shape[0]
        outputs = torch.zeros((batch_size, 256), dtype=torch.int16, device=piece_positions.device)
        for i in range(batch_size):
            wk_pos, bk_pos = king_positions[i]
            white_half = torch.sum(piece_positions[i] @ self.input_weights[wk_pos, :-1, :], dim=0)
            black_half = torch.sum(piece_positions[i] @ self.input_weights[bk_pos, :-1, :], dim=0)
            white_half += self.input_weights[wk_pos, -1, :]  # Add the constant bias term
            black_half += self.input_weights[bk_pos, -1, :]  # Add the constant bias term
            outputs[i] = white_half + black_half + self.bias
        return outputs


class TransformLayer(nn.Module):
    """
    The Transform Layer which forms a 512-element vector of 8-bit ints from the HalfKP input.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.clamp(x, min=0, max=127).to(torch.int8)  # Clip 16-bit int to 8-bit range
        side_to_move = x
        other_side = x
        return torch.cat([side_to_move, other_side], dim=1)  # Create a 512-element vector

class FullyConnectedLayer(nn.Module):
    """
    Fully connected layer which takes in 8-bit inputs, applies weights, and outputs 8-bit values.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.randint(low=-128, high=127, size=(out_dim, in_dim), dtype=torch.int8))
        self.bias = nn.Parameter(torch.zeros(out_dim, dtype=torch.int32))

    def forward(self, x):
        x = x @ self.weights.T.to(x.dtype)  # Multiply inputs with weights
        x = x + self.bias  # Add bias
        x = torch.div(x, 64, rounding_mode='floor')  # Divide by 64
        x = torch.clamp(x, min=0, max=127).to(torch.int8)  # Clip to 8-bit range
        return x

class OutputLayer(nn.Module):
    """
    Final layer which produces a scalar evaluation.
    """
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randint(low=-128, high=127, size=(32,), dtype=torch.int8))
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.int32))

    def forward(self, x):
        x = torch.sum(x * self.weights.to(x.dtype))  # Inner product
        x = x + self.bias  # Add bias
        return x // 16  # Divide by FV_SCALE (16) to produce the final evaluation


class NNUE(nn.Module):
    """
    Full NNUE network consisting of the HalfKP input layer, transform layer, two hidden layers, and the output layer.
    """
    def __init__(self):
        super().__init__()
        self.input_layer = HalfKPInputLayer()
        self.transform_layer = TransformLayer()
        self.hidden_layer_1 = FullyConnectedLayer(512, 32)
        self.hidden_layer_2 = FullyConnectedLayer(32, 32)
        self.output_layer = OutputLayer()

    def forward(self, piece_positions, king_positions):
        x = self.input_layer(piece_positions, king_positions)  # HalfKP input layer
        x = self.transform_layer(x)  # Transform to 512-element vector
        x = self.hidden_layer_1(x)  # First hidden layer
        x = self.hidden_layer_2(x)  # Second hidden layer
        x = self.output_layer(x)  # Output layer producing final evaluation
        return x

    def score(self, pgn, move):
        '''
        pgn: string e.g. "1.e4 a6 2.Bc4 "
        move: string e.g. "a5 "
        '''
        # init a game and board
        game = chess.pgn.read_game(io.StringIO(pgn))
        board = Board()
        # catch board up on game to present
        for past_move in list(game.mainline_moves()):
            board.push(past_move)
        # push the move to score
        board.push_san(move)
        # convert to tensor, unsqueezing a dummy batch dimension
        board_tensor = torch.tensor(encode_board(board)).unsqueeze(0)
        return self.forward(board_tensor).item()