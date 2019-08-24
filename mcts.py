import math
import numpy as np
from copy import deepcopy
from const import N

C_PUCT = math.sqrt(2)
N_SEARCH = 1600

class Node(object):
    """ definition of node used in Monte-Carlo search for pi policy
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


class TT(object):
    """
    To decide the next move -> to find a =~ pi
    pi(a|s) = N(s,a)^(1/tau) / Sigma_b N(s,b)^(1/tau)
    a_t = argmax_a (Q(s,a) + u(s,a)) 

    u(s,a) = c_puct * P(s,a) * sqrt(Sigma_b N(s,b)) / (1 + N(s,a))
    N(s,a) = Sigma_i^n 1(s,a,i)
    Q(s,a) = W(s,a) / N(s,a), where W(s,a) := W(s,a) + v
    (P(s,-), v) = f_theta(s)

    fn_policy_value(self.net, board) ->
    returns tuple( P(s,a), v ) as a result of self.net feed-forward.

    """
    def __init__(self, net, fn_policy_value):
        self.root = Node(None, 1.)
        self.net = net
        self.fn_policy_value = fn_policy_value

    def reset_tree(self):
        self.update_root(-1)

    def select(self):
        node = self.root
        while True:
            if node.is_leaf(): return node
            move, node = max(node.next.items(), key=lambda x: x[1].Q_plus_u())

    def backup(self, node):
        pass


    def search(self):
        """ single search without any MC rollouts
        process (select -> expand and evaluate -> backup) 1x
        """
        #TODO
        self.select()
        self.backup()

    def update_root(self, move):
        if move in self.root.next:
            self.root = self.root.next[move]
            self.root.prev = None
        else:
            self.root = Node(None, 1.)


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
        sumN = sum( node.N ** (1/tau) for _, node in self.root.next.items() )
        pi = [ ( move, node.N ** (1/tau) / sumN, ) 
               for move, node in self.root.next.items() ]
        return dict(pi)


