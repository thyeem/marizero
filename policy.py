import math
import numpy as np
import torch
from copy import deepcopy
from const import N
import marizero as mario

C_PUCT = 0.5
N_SEARCH = 20

def xy(move): return move // N, move % N

def softmax(x):
    p = np.exp(x-np.max(x))
    p /= p.sum(axis=0)
    return p

class Node(object):
    """ definition of node used in Monte-Carlo search for policy pi
    """
    def __init__(self, move, prev=None):
        self.move = move
        self.is_expanded = False
        self.prev = prev
        self.next = {}
        self.next_P = np.zeros([361], dtype=np.float32)
        self.next_N = np.zeros([361], dtype=np.float32)
        self.next_W = np.zeros([361], dtype=np.float32)

    @property
    def N(self):
        return self.prev.next_N[self.move]

    @N.setter
    def N(self, value):
        self.prev.next_N[self.move] = value

    @property
    def W(self):
        return self.prev.next_W[self.move]

    @N.setter
    def W(self, value):
        self.prev.next_W[self.move] = value

    def next_Q(self):
        return self.next_W / (self.next_N+1)

    def next_u(self):
        return C_PUCT * math.sqrt(self.N) * (self.next_P / (self.next_N+1))

    def best_next(self):
        """ Upper-confidence bound on Q-value
        return argmax_a [ U(s,a) := Q(s,a) + u(s,a) ]
        """
        return np.argmax(self.next_Q() + self.next_u())

    def print_tree(self, node, indent=2, cutoff=None):
        """ recursively dumps node-tree
        usage: node.print_tree(node, cutoff=5)
        """
        x, y = xy(self.move) 
        N_ = not self.prev and self.next_N.sum() or self.N
        P_ = not self.prev and -1 or self.prev.next_P[self.move]
        Q_ = not self.prev and -1 or self.W / self.N
        u_ = not self.prev and -1 \
                            or C_PUCT * math.sqrt(self.prev.N) * P_ / (N_+1)
        U_ = Q_ + u_
        print(f'{" "*indent} ({x:2d},{y:2d})  N {N_:6d}  U {U_:6.4f}  '
              f'Q {Q_:6.4f}  u {u_:6.4f}  P {P_:6.4f}')
        if node.is_expanded:
            args = np.argsort(self.next_Q() + self.next_u())
            args = cutoff and args[-cutoff:][::-1]
            children = [ node.next[arg] for arg in args ]
            for child in children:
                self.print_tree(child, indent+2, cutoff=cutoff)



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
            board.make_move(*xy(move), True)

    def expand(self, node, P):
        for move in range(len(P)):
            if P[move] < 1e-10: continue
            node.next[move] = Node(node, P[move])

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
            v = -1.
        else:
            P, v = self.fn_policy_value(board)
            self.expand(leaf, P)
        self.backup(leaf, -v)

    def fn_pi(self, board, num_search=N_SEARCH):
        """ board -> ( [move], [prob] ) as two-cols form of pi(a|s) 
        get policy pi as defined in the zero paper
        pi(a|s) = N(s,a)^(1/tau) / Sigma_b N(s,b)^(1/tau)
        tau: temperature controling the degree of exploration 
        simply the normalized visit count when tau=1.
        the smaller tau, the more relying on the visit count.

        """
        for _ in range(num_search):
            self.search(deepcopy(board))
        tau = board.moves < 5 and 1 or 1e-3
        moves, visits = zip(*[ (move, node.N) 
                               for move, node in self.root.next.items() ])
        probs = softmax(1./tau * np.log(np.array(visits)+1) + 1e-10)
        return (moves, probs)

    def fn_policy_value(self, board):
        """ board -> ( P(s,-), v )
        (p,v) = f_theta(s)
        get policy p and value fn v from network feed-forward.
        p := P(s,-), where P(s,a) = Pr(a|s)
        """
        S = mario.read_state(board)
        S = torch.FloatTensor(S)
        logP, v = self.net(S)
        v = v.flatten().item()
        logP += 1e-10
        P = np.exp(logP.flatten().data.numpy())
        P /= P.sum()
        invalid = [ move for move in range(len(P)) 
                    if board.is_illegal_move(*xy(move)) ] 
        P[invalid] = -1
        return P, v



