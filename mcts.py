import math
import numpy as np
from copy import deepcopy

C_PUCT = math.sqrt(2)
N_SEARCH = 1600

class Node(object):
    """ definition of node used in Monte-Carlo search for pi policy
    To decide the next move -> to find a =~ pi
    a_t = argmax_a (Q(s,a) + u(s,a)) 
    u(s,a) = c_puct * P(s,a) * sqrt(Sigma_b N(s,b)) / (1 + N(s,a))
    N(s,a) = Sigma_i^n 1(s,a,i)
    Q(s,a) = W(s,a) / N(s,a), where W(s,a) := W(s,a) + v
    (P(s,-), v) = f_theta(s)
    pi(a|s) = N(s,a)^(1/tau) / Sigma_b N(s,b)^(1/tau)

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
        """ update u(s,a) for this node and return Q(s,a) + u(s,a)
        """
        self.u = C_PUCT * self.P * math.sqrt(self.prev.N) / (1 + self.N)
        return self.Q + self.u


class PolicyPi(object):

    def __init__(self, policy_p):
        self.root = Node(None, 1.)
        self.policy_p = policy_p

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

    def fn_pi(self, board, tau=1):
        """ get pi as defined in the paper
        pi(a|s) = N(s,a)^(1/tau) / Sigma_b N(s,b)^(1/tau)
        """
        for _ in range(N_SEARCH):
            self.search(deepcopy(board))
        
        sumN = sum( node.N ** (1/tau) for _, node in self.root.next.items() )
        pi = [ ( move, node.N ** (1/tau) / sumN, ) 
               for move, node in self.root.next.items() ]
        return dict(pi)

    def get_move(self, board, tau=1):

        pi = self.fn_pi(board, tau)

