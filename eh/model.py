import io
import torch
import torch.nn as nn
import numpy as np
import chess.pgn
from chess import Board
import torch.nn.functional as F

FEATURES_DIM = 64 * (64 * 10 + 1)

EXPECTED_OUTPUT_DIM = 2 * FEATURES_DIM
print("features dim:", FEATURES_DIM)
print("expected output dim:", EXPECTED_OUTPUT_DIM)

PIECE_TYPES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
COLORS = [chess.BLACK, chess.WHITE]

def encode_board(board: Board) -> np.array:
    opps = np.zeros(FEATURES_DIM)
    mys = np.zeros(FEATURES_DIM)

    if board.turn:
        my_color = chess.WHITE
        opp_color = chess.BLACK 
    else:
        my_color = chess.BLACK  
        opp_color = chess.WHITE

    my_king, opp_king = board.king(my_color), board.king(opp_color)
    print("my_king", my_king, "opp_king", opp_king)

    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(i, j))
            if piece and piece.piece_type != chess.KING:
                piece_index = PIECE_TYPES.index(piece.piece_type)
                print("piece index:", piece_index, "piece:", piece)
                if piece.color == opp_color:
                    opps[(opp_king * (64 * 10 + 1)) + (piece_index * 64) + (i * 8) + j] = 1.0
                else:
                    mys[(my_king * (64 * 10 + 1)) + (piece_index * 64) + (i * 8) + j] = 1.0
    return np.concatenate((opps, mys))

class HalfKPInputLayer(nn.Module):
    """
    HalfKP input layer, which encodes the 10 non-king pieces on 64 squares for each side's king position.
    """
    def __init__(self):
        super().__init__()
        
        self.my_pieces = nn.Sequential(
            nn.Linear(64 * (64 * 10 + 1), 256, dtype=torch.float32),
            nn.ReLU(),
        )
        
        self.opp_pieces = nn.Sequential(
            nn.Linear(64 * (64 * 10 + 1), 256, dtype=torch.float32),
            nn.ReLU(),
        )

        # self.input_weights = nn.Parameter(torch.randint(low=-32768, high=32767, size=(64, 640 + 1, 256), dtype=torch.int16)) # 64x(64x10+1)x256
        # self.bias = nn.Parameter(torch.zeros(256, dtype=torch.int16))

    def forward(self, input):
        print("input model forward", input.shape, input.dtype)

        my_pos = self.my_pieces(input[:FEATURES_DIM])
        opp_pos = self.opp_pieces(input[FEATURES_DIM:])

        return torch.cat((my_pos, opp_pos))

class Model(nn.Module):
    def __init__(self, nlayers, device='cpu'):
        super().__init__()

        self.input_layer = HalfKPInputLayer()
        self.transform = nn.Sequential(
            nn.Linear(512, 32, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(32, 32,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(32, 1,dtype=torch.float32),
        )

    def forward(self, inputs):
        print("main forward", inputs.shape, inputs.dtype)
        inputs = inputs.to(torch.float32)
        inputs = self.input_layer(inputs)
        print("second input:", input.shape, input.dtype)
        return self.transform(inputs)

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
        # board_tensor = torch.tensor(encode_board(board)).unsqueeze(0)
        # input = encode_board(board)
        # print("raw input:", input.shape, input.dtype)
        # input = torch.tensor(input, dtype=torch.float32).unsqueeze(0)
        # print("first input:", input.shape, input.dtype)
        # return self.forward(input).item()
        board_tensor = torch.tensor(encode_board(board)).unsqueeze(0)
        return self.forward(board_tensor).item()
