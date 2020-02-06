
import GMatElastoPlasticQPot.Cartesian2d as GMat
import numpy as np

def EQ(a, b):
  assert np.abs(a-b) < 1.e-12

K = 12.3
G = 45.6

gamma = 0.02
epsm = 0.12

Eps = np.array(
    [[epsm, gamma],
     [gamma, epsm]])

# Elastic

mat = GMat.Elastic(K, G)

Sig = mat.Stress(Eps)

EQ(Sig[0,0], K * epsm)
EQ(Sig[1,1], K * epsm)
EQ(Sig[0,1], G * gamma)
EQ(Sig[1,0], G * gamma)

# Cusp

mat = GMat.Cusp(K, G, [0.01, 0.03, 0.10])

Sig = mat.Stress(Eps)

EQ(Sig[0,0], K * epsm)
EQ(Sig[1,1], K * epsm)
EQ(Sig[0,1], G * 0.0)
EQ(Sig[1,0], G * 0.0)

EQ(mat.epsp(Eps), 0.02)

EQ(mat.find(Eps), 1)

# Smooth

mat = GMat.Smooth(K, G, [0.01, 0.03, 0.10])

Sig = mat.Stress(Eps)

EQ(Sig[0,0], K * epsm)
EQ(Sig[1,1], K * epsm)
EQ(Sig[0,1], G * 0.0)
EQ(Sig[1,0], G * 0.0)

EQ(mat.epsp(Eps), 0.02)

EQ(mat.find(Eps), 1)

# Matrix

nelem = 3
nip = 2
mat = GMat.Matrix(nelem, nip)

# row 0: elastic
I = np.zeros([nelem, nip], dtype='int')
I[0,:] = 1
mat.setElastic(I, K, G)

# row 1: cups
I = np.zeros([nelem, nip], dtype='int')
I[1,:] = 1
mat.setCusp(I, K, G, [0.01, 0.03, 0.10])

# row 2: smooth
I = np.zeros([nelem, nip], dtype='int')
I[2,:] = 1
mat.setSmooth(I, K, G, [0.01, 0.03, 0.10])

eps = np.zeros((nelem, nip, 2, 2))
for i in range(2):
    for j in range(2):
        eps[:, :, i, j] = Eps[i, j]

sig = mat.Stress(eps)
epsp = mat.Epsp(eps)

for q in range(nip):

    EQ(sig[0,q,0,0], K * epsm)
    EQ(sig[0,q,1,1], K * epsm)
    EQ(sig[0,q,0,1], G * gamma)
    EQ(sig[0,q,0,1], G * gamma)
    EQ(sig[1,q,0,0], K * epsm)
    EQ(sig[1,q,1,1], K * epsm)
    EQ(sig[1,q,0,1], G * 0.0)
    EQ(sig[1,q,0,1], G * 0.0)
    EQ(sig[2,q,0,0], K * epsm)
    EQ(sig[2,q,1,1], K * epsm)
    EQ(sig[2,q,0,1], G * 0.0)
    EQ(sig[2,q,0,1], G * 0.0)

    EQ(epsp[0,q], 0.0)
    EQ(epsp[1,q], gamma)
    EQ(epsp[2,q], gamma)


print('All checks passed')
