import math
import numpy as np
import torch
from copy import deepcopy
from const import N
import marizero as mario

C_PUCT = math.sqrt(2)
N_SEARCH = 20

class Node(object):
    """ definition of node used in Monte-Carlo search for policy pi
    """
    def __init__(self, prev, P):
        self.prev = prev
        self.next = {}
        self.N = 0
        self.Q = 0
        self.u = 0
        self.P = P

    def is_root(self):
        return self.prev is None

    def is_leaf(self):
        return len(self.next) == 0

    def Q_plus_u(self):
        """ Upper-confidence bound on Q-value
        update u(s,a) for this node and return U(s,a) := Q(s,a) + u(s,a)
        """
        self.u = C_PUCT * self.P * math.sqrt(self.prev.N) / (1 + self.N)
        return self.Q + self.u


def valid_move_mask(S): 
    """ returns a flattened array of valid move mask-layer
    """
    return S[:,3,:,:].flatten().data.numpy()

def xy(move): return move // N, move % N

def softmax(x):
    p = np.exp(x-np.max(x))
    p /= p.sum(axis=0)
    return p

class TT(object):
    """
    To decide the next move -> to find a =~ pi
    pi(a|s) = N(s,a)^(1/tau) / Sigma_b N(s,b)^(1/tau)
    a_t = argmax_a (Q(s,a) + u(s,a)) 

    u(s,a) = c_puct * P(s,a) * sqrt(Sigma_b N(s,b)) / (1 + N(s,a))
    N(s,a) = Sigma_i^n 1(s,a,i)
    Q(s,a) = W(s,a) / N(s,a), where W(s,a) := W(s,a) + v
    (P(s,-), v) = f_theta(s)

    """
    def __init__(self, net):
        self.root = Node(None, 1.)
        self.net = net

    def reset_tree(self):
        self.update_root(-1)

    def update_root(self, move):
        if move in self.root.next:
            self.root = self.root.next[move]
            self.root.prev = None
        else:
            self.root = Node(None, 1.)

    def select(self, board):
        node = self.root
        while True:
            if node.is_leaf(): return node
            move, node = max(node.next.items(), key=lambda x: x[1].Q_plus_u())
            assert board.is_illegal_move(*xy(move)) == 0
            board.make_move(*xy(move))

    def expand(self, node, P):
        for move, prob in P.items():
            if prob <= 0: continue
            node.next[move] = Node(node, prob)

    def backup(self, node, v):
        node.N += 1
        node.Q += 1.*(v - node.Q) / node.N
        if node.prev: self.backup(node.prev, -v)

    def search(self, board):
        """ single search without any MC rollouts
        process (select -> expand and evaluate -> backup) 1x
        """
        leaf = self.select(board)
        winner = board.check_game_end()
        if winner:
            v = winner == board.whose_turn() and 1. or -1.
        else:
            P, v = self.fn_policy_value(board, 10)
            self.expand(leaf, P)
        self.backup(leaf, -v)

    def fn_policy_value(self, board, cut=None):
        """ board -> ( P(s,-), v )
        (p,v) = f_theta(s)
        get policy p and value fn v from network feed-forward.
        p := P(s,-), where P(s,a) = Pr(a|s)
        used when expanding tree and illegal moves are filtered here.
        """
        S = mario.read_state(board)
        S = torch.FloatTensor(S)
        logP, v = self.net(S)
        logP += 1e-10
        P = np.exp(logP.flatten().data.numpy()) * valid_move_mask(S)
        Psum = P.sum()
        assert Psum != 0
        P /= Psum
        moves = cut and sorted(range(len(P)), key=lambda x: P[x])[-cut:] \
                     or range(len(P))
        P = cut and [ P[move] for move in moves ] or P
        P = dict(zip(moves, P))
        return P, v

    def fn_pi(self, board, num_search=N_SEARCH):
        """ board -> pi(a|s) look-up table
        get policy pi as defined in the zero paper
        pi(a|s) = N(s,a)^(1/tau) / Sigma_b N(s,b)^(1/tau)
        tau: temperature controling the degree of exploration 
        simply the normalized visit count when tau=1.
        the smaller tau, the more relying on the visit count.

        """
        for _ in range(num_search):
            self.search(deepcopy(board))
        tau = board.moves < 8 and 1 or 1e-3
        moves, visits = zip(*[ (move, node.N) 
                               for move, node in self.root.next.items() ])
        visits = softmax(1./tau * np.log(np.array(visits)+1) + 1e-10)
        return dict(zip(moves, visits))


