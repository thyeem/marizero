import numpy as np
import torch
import sys

N=2

def augment_data(data):
    res = []
    for S, pi, z in data:
        _R = [ np.rot90(S, i).copy() for i in range(4) ] 
        _F = [ np.fliplr(r).copy() for r in _R ]
        _S = _R + _F
        pi = pi.reshape((N,N))
        _R = [ np.rot90(pi, i).copy() for i in range(4) ] 
        _F = [ np.fliplr(r).copy() for r in _R ]
        _pi =[ x.flatten() for x in _R + _F ]
        _z = np.repeat(z, len(_S))
        res.extend(zip(_S, _pi, _z))
    return res


q = np.array([[1,2],[3,4]])
q = np.random.randn(1, 2, 2)

print(q)
r = np.array([range(4)])
z = np.array(range(4))

w = list(zip(q, r, z))
res = augment_data(w)




S, pi, z = zip(*w)
print(S)
S, pi, z = zip(*res)
print(S)


S = torch.stack([ torch.FloatTensor(x) for x in S ])
pi = torch.stack([ torch.FloatTensor(x) for x in pi ])
z = torch.cat([ torch.FloatTensor([x]) for x in z ], dim=0)
print(S)
