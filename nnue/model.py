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

FEATURES_DIM = 64 * (64 * 10 + 1)

EXPECTED_OUTPUT_DIM = 2 * FEATURES_DIM
print("features dim:", FEATURES_DIM)
print("expected output dim:", EXPECTED_OUTPUT_DIM)

PIECE_TYPES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
COLORS = [chess.BLACK, chess.WHITE]

# def encode_board(board: Board) -> np.array:
#     opps = np.zeros(FEATURES_DIM)
#     mys = np.zeros(FEATURES_DIM)

#     if board.turn:
#         my_color = chess.WHITE
#         opp_color = chess.BLACK 
#     else:
#         my_color = chess.BLACK  
#         opp_color = chess.WHITE

#     my_king, opp_king = board.king(my_color), board.king(opp_color)
#     print("my_king", my_king, "opp_king", opp_king)

#     for i in range(8):
#         for j in range(8):
#             piece = board.piece_at(chess.square(i, j))
#             if piece and piece.piece_type != chess.KING:
#                 piece_index = PIECE_TYPES.index(piece.piece_type)
#                 print("piece index:", piece_index, "piece:", piece)
#                 if piece.color == opp_color:
#                     opps[(opp_king * (64 * 10 + 1)) + (piece_index * 64) + (i * 8) + j] = 1.0
#                 else:
#                     mys[(my_king * (64 * 10 + 1)) + (piece_index * 64) + (i * 8) + j] = 1.0
#     return np.concatenate((opps, mys))

FEATURES_DIM = 64 * (64 * 10 + 1)

EXPECTED_OUTPUT_DIM = 2 * FEATURES_DIM
print("features dim:", FEATURES_DIM)
print("expected output dim:", EXPECTED_OUTPUT_DIM)

OPP_PIECE_TYPES = [11, 10, 9, 8, 7]
MY_PIECE_TYPES = [1, 2, 3, 4, 5]

def encode_dataset(d, device='cpu'):
    my_king, opp_king = None, None

    for i in range(8):
        for j in range(8):
            piece = d[i][j]
            if piece == 0:
                my_king = (i, j)
            elif piece == 12:
                opp_king = (i, j)

    opps = torch.zeros(FEATURES_DIM, device=device)
    mys = torch.zeros(FEATURES_DIM, device=device)

    for i in range(8):
        for j in range(8):
            piece = d[i][j]

            if piece != 6 and piece != 12 and piece != 0:
                if piece in OPP_PIECE_TYPES:
                    piece_index = OPP_PIECE_TYPES.index(piece)
                    opps[(opp_king[0]*8+opp_king[1]) * (64 * 10 + 1) + (piece_index * 64) + (i * 8) + j] = 1
                elif piece in MY_PIECE_TYPES:
                    piece_index = MY_PIECE_TYPES.index(piece)
                    mys[(my_king[0]*8+my_king[1]) * (64 * 10 + 1) + (piece_index * 64) + (i * 8) + j] = 1
                    
    return torch.concatenate((opps, mys))

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
        print("halfkp input", input)

        # Split input into two halves at FEATURES_DIM
        my_input = input[:, :FEATURES_DIM]  # First 41024 features
        opp_input = input[:, FEATURES_DIM:]  # Last 41024 features

        print("halfkp my_input", my_input.shape, my_input.dtype)
        print("halfkp opp_input", opp_input.shape, opp_input.dtype)

        my_pos = self.my_pieces(my_input)
        opp_pos = self.opp_pieces(opp_input)

        print("halfkp my_pos", my_pos.shape, my_pos.dtype)
        print("halfkp opp_pos", opp_pos.shape, opp_pos.dtype)

        return torch.cat((my_pos, opp_pos), dim=1)

class Model(nn.Module):
    def __init__(self, nlayers, device='cpu'):
        super().__init__()

        self.device = device

        self.input_layer = HalfKPInputLayer()
        self.transform = nn.Sequential(
            nn.Linear(512, 32, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(32, 32,dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(32, 1,dtype=torch.float32),
        )

    def forward(self, inputs: torch.Tensor):
        # print("main forward", inputs.shape, inputs.dtype)
        inputs = inputs.to(torch.float32)
        print("main forward", inputs)

        inputs = torch.stack([encode_dataset(board, self.device) for board in inputs])

        # inputs = torch.tensor([encode_dataset(board) for board in inputs])

        inputs = self.input_layer(inputs)
        print("second input:", inputs.shape, inputs.dtype)
        output =  self.transform(inputs)
        print("output", output.shape, output.dtype)
        print("raw output", output)
        return output.flatten()

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
        # input = torch.tensor(input, dtype=torch.float32)
        # print("first input:", input.shape, input.dtype)
        # return self.forward(input).item()
        board_tensor = torch.tensor(encode_board(board)).unsqueeze(0)
        return self.forward(board_tensor).item()
