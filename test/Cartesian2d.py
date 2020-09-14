
import GMatElastoPlasticQPot.Cartesian2d as GMat
import numpy as np

def CLOSE(a, b):
  assert np.abs(a-b) < 1.e-12

K = 12.3
G = 45.6

gamma = 0.02
epsm = 0.12

Eps = np.array(
    [[epsm, gamma],
     [gamma, epsm]])

# Elastic - stress

mat = GMat.Elastic(K, G)
mat.setStrain(Eps)
Sig = mat.Stress()

CLOSE(Sig[0,0], K * epsm)
CLOSE(Sig[1,1], K * epsm)
CLOSE(Sig[0,1], G * gamma)
CLOSE(Sig[1,0], G * gamma)

# Cusp - stress

mat = GMat.Cusp(K, G, [0.01, 0.03, 0.10])
mat.setStrain(Eps)
Sig = mat.Stress()

CLOSE(Sig[0,0], K * epsm)
CLOSE(Sig[1,1], K * epsm)
CLOSE(Sig[0,1], G * 0.0)
CLOSE(Sig[1,0], G * 0.0)
CLOSE(mat.epsp(Eps), 0.02)
CLOSE(mat.find(Eps), 1)

# Smooth - stress

mat = GMat.Smooth(K, G, [0.01, 0.03, 0.10])
mat.setStrain(Eps)
Sig = mat.Stress()

CLOSE(Sig[0,0], K * epsm)
CLOSE(Sig[1,1], K * epsm)
CLOSE(Sig[0,1], G * 0.0)
CLOSE(Sig[1,0], G * 0.0)
CLOSE(mat.epsp(Eps), 0.02)
CLOSE(mat.find(Eps), 1)

# Matrix

nelem = 3
nip = 2
mat = GMat.Matrix(nelem, nip)

I = np.zeros([nelem, nip], dtype='int')
I[0,:] = 1
mat.setElastic(I, K, G)

I = np.zeros([nelem, nip], dtype='int')
I[1,:] = 1
mat.setCusp(I, K, G, [0.01, 0.03, 0.10])

I = np.zeros([nelem, nip], dtype='int')
I[2,:] = 1
mat.setSmooth(I, K, G, [0.01, 0.03, 0.10])

eps = np.zeros((nelem, nip, 2, 2))
for i in range(2):
    for j in range(2):
        eps[:, :, i, j] = Eps[i, j]

mat.setStrain(eps)
sig = mat.Stress()
epsp = mat.Epsp()

for q in range(nip):

    CLOSE(sig[0,q,0,0], K * epsm)
    CLOSE(sig[0,q,1,1], K * epsm)
    CLOSE(sig[0,q,0,1], G * gamma)
    CLOSE(sig[0,q,0,1], G * gamma)
    CLOSE(epsp[0,q], 0.0)

    CLOSE(sig[1,q,0,0], K * epsm)
    CLOSE(sig[1,q,1,1], K * epsm)
    CLOSE(sig[1,q,0,1], G * 0.0)
    CLOSE(sig[1,q,0,1], G * 0.0)
    CLOSE(epsp[1,q], gamma)

    CLOSE(sig[2,q,0,0], K * epsm)
    CLOSE(sig[2,q,1,1], K * epsm)
    CLOSE(sig[2,q,0,1], G * 0.0)
    CLOSE(sig[2,q,0,1], G * 0.0)
    CLOSE(epsp[2,q], gamma)

print('All checks passed')
