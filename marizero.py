import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np
from collections import deque
from board import Board
from mcts import PolicyPi
from const import Stone, N

C = 8
H = 64
LEARNING_RATE = 1e-3
L2_CONST = 1e-4
GAMMA = 0.99

class Net(nn.Module):
    """ network for both policy p and value function 
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(C, H, (3,3,), 1, 1)
        self.bn1 = nn.BatchNorm2d(H)
        self.conv2 = nn.Conv2d(H, H, (3,3,), 1, 1)
        self.bn2 = nn.BatchNorm2d(H)
        self.conv3 = nn.Conv2d(H, H, (3,3,), 1, 1)
        self.bn3 = nn.BatchNorm2d(H)
        self.conv4 = nn.Conv2d(H, H, (3,3,), 1, 1)
        self.bn4 = nn.BatchNorm2d(H)

        self.p_conv = nn.Conv2d(H, 2, (1,1,), 1, 0)
        self.p_bn = nn.BatchNorm2d(2)
        self.p_fc1 = nn.Linear(2*N*N, N*N)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.v_conv = nn.Conv2d(H, 1, (1,1,), 1, 0)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(N*N, 256)
        self.v_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))

        p_x = self.p_conv(x)
        p_x = F.relu(self.p_bn(p_x))
        p_x = p_x.view(-1, 2*N*N)
        p_x = self.p_fc1(p_x)
        p_x = self.logsoftmax(p_x) 

        v_x = self.v_conv(x)
        v_x = F.relu(self.v_bn(v_x))
        v_x = v_x.view(-1, N*N)
        v_x = self.v_fc1(v_x)
        v_x = F.relu(v_x)
        v_x = self.v_fc2(v_x)
        v_x = F.tanh(v_x)
        return p_x, v_x


def legal_mask(S): return S[:,3,:,:].flatten().data.numpy()


class MariZero(object):
    """ MariZero: MariAI without any human domain knowledge.
    MariAI: AI based on MCTS play rollouts
    Feel free to meet them all at http://sofimarie.com

    fn and var names based on the following notaion summarized:
    a_t = argmax_a (Q(s,a) + u(s,a)) 
    u(s,a) = c_puct * P(s,a) * sqrt(Sigma_b N(s,b)) / (1 + N(s,a))
    N(s,a) = Sigma_i^n 1(s,a,i)
    Q(s,a) = W(s,a) / N(s,a), where W(s,a) := W(s,a) + v
    (P(s,-), v) = f_theta(s)
    pi(a|s) = N(s,a)^(1/tau) / Sigma_b N(s,b)^(1/tau)
    """
    def __init__(self, game=None):
        self.game = game
        self.data = deque(maxlen=10000)
        self.model = Net()
        self.pi = PolicyPi()
        self.optim = optim.Adam(self.model.parameters(), 
                                weight_decay=L2_CONST, lr=LEARNING_RATE)

    def save_model(self, name):
        torch.save(self.model.state_dict(), f'./model/{name}.pt')

    def load_model(self, name):
        model = Net()
        model.load_state_dict(torch.load(f'./model/{name}.pt'))
        model.eval()
        return model

    def read_state(self, board):
        """ board -> S
        Defines the input layer S: read the current state from the board given.
        return state S as a tensor of 10 channels
        0 -> color turn to play 
        1 -> BLACK stones
        2 -> WHITE stones
        3 -> EMPTY space of legal-move 
        4-6 -> enemy's stones captured (one-hot)
        7-9 -> my stones captured (one-hot)
        """
        if board.turn == Stone.BLACK:
            (cap_self, cap_enemy) = (board.scoreB, board.scoreW)
        else:
            (cap_self, cap_enemy) = (board.scoreW, board.scoreB)
        S = torch.zeros(1, C, N, N)
        S[:,0,:,:] = board.turn
        for x in range(N):
            for y in range(N):
                if board.get_stone(x, y) == Stone.BLACK:
                    S[:,1,x,y] = 1
                elif board.get_stone(x, y) == Stone.WHITE:
                    S[:,2,x,y] = 1
                else:
                    if not board.validate_move(x, y): S[:,3,x,y] = 1
        if cap_enemy:
            b2, b1, b0 = [ int(i) for i in f'{cap_enemy-1:03b}' ]
            S[:,4,:,:] = b2
            S[:,5,:,:] = b1
            S[:,6,:,:] = b0
        if cap_self:
            b2, b1, b0 = [ int(i) for i in f'{cap_self-1:03b}' ]
            S[:,7,:,:] = b2
            S[:,8,:,:] = b1
            S[:,9,:,:] = b0
        return S

    def f_theta(self, states, gpu=False):
        """ [ state S ] -> [ (P(s,-), v) ]
        get policy-value function from network, (p,v) = f_theta(s)
        p := P(s,-), where P(s,a) := Pr(a|s)
        """
        batch_logP, batch_v = self.model(states)
        batch_P = np.exp(batch_logP.data.numpy())
        batch_v = batch_v.data.numpy()
        batch_P, batch_v

    def policy_p(self, board):
        """ board -> P(s,-), v
        Used in MCTS when expanding tree.
        Illegal moves are filtered here.
        """
        S = self.read_state(board)
        logP, v = self.model(S)
        logP += 1e-10
        P = np.exp(logP.flatten().data.numpy()) * legal_mask(S)
        Psum = P.sum()
        assert Psum != 0
        P /= Psum
        return P, v
        
    def augment_data(self, data_set):
        """ augment data set by flipping and rotating
        by definition x8 data can be produced
        """
        equi = []
        for data in data_set:
            rot  = [ np.rot90(data, i) for i in range(4) ] 
            flip = [ np.fliplr(r) for r in rot ]
            equi.extend(rot + flip)
        self.data.extend(equi)

    def next_move(self, board):
        """ interface responsible for answering game.py module
        """
        S = self.read_state(board)
        # TODO

    def self_play(self):
        board = Board()

        while True:
            pass
        

    def train(self):
        """
        loss := (z-v)^2 - pi'log(p) + c||theta||

        """
        self.model.train()
        while True:
            pass


    def update_parameters(self, stack, is_winner):
        pass
        # TODO



if __name__ == '__main__':
    mario = MariZero()

