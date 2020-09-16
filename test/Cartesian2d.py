
import GMatElastoPlasticQPot.Cartesian2d as GMat
import numpy as np

def ISCLOSE(a, b):
  assert np.abs(a-b) < 1.e-12

def A4_ddot_B2(A, B):
    return np.einsum('ijkl,lk->ij', A, B)

# Id - Tensor2

A = np.random.random([2, 2])
I = GMat.I2()
Id = GMat.I4d()
Is = GMat.I4s()
A = A4_ddot_B2(Is, A)
assert np.allclose(A4_ddot_B2(Id, A), A - GMat.Hydrostatic(A) * I)

# Deviatoric - Tensor2

A = np.random.random([2, 2])
B = np.array(A, copy=True)
tr = B[0, 0] + B[1, 1]
B[0, 0] -= 0.5 * tr
B[1, 1] -= 0.5 * tr
assert np.allclose(GMat.Deviatoric(A), B)

# Material points

K = 12.3
G = 45.6

gamma = 0.02
epsm = 0.12

Eps = np.array(
    [[epsm, gamma],
     [gamma, epsm]])

ISCLOSE(float(GMat.Epsd(Eps)), gamma)

# Elastic - stress

mat = GMat.Elastic(K, G)
mat.setStrain(Eps)
Sig = mat.Stress()

ISCLOSE(Sig[0,0], K * epsm)
ISCLOSE(Sig[1,1], K * epsm)
ISCLOSE(Sig[0,1], G * gamma)
ISCLOSE(Sig[1,0], G * gamma)

# Cusp - stress

mat = GMat.Cusp(K, G, [0.01, 0.03, 0.10])
mat.setStrain(Eps)
Sig = mat.Stress()

ISCLOSE(Sig[0,0], K * epsm)
ISCLOSE(Sig[1,1], K * epsm)
ISCLOSE(Sig[0,1], G * 0.0)
ISCLOSE(Sig[1,0], G * 0.0)
ISCLOSE(mat.epsp(), 0.02)
ISCLOSE(mat.currentIndex(), 1)

# Smooth - stress

mat = GMat.Smooth(K, G, [0.01, 0.03, 0.10])
mat.setStrain(Eps)
Sig = mat.Stress()

ISCLOSE(Sig[0,0], K * epsm)
ISCLOSE(Sig[1,1], K * epsm)
ISCLOSE(Sig[0,1], G * 0.0)
ISCLOSE(Sig[1,0], G * 0.0)
ISCLOSE(mat.epsp(), 0.02)
ISCLOSE(mat.currentIndex(), 1)

# Array2d

nelem = 3
nip = 2
mat = GMat.Array2d([nelem, nip])

I = np.zeros([nelem, nip], dtype='int')
I[0,:] = 1
mat.setElastic(I, K, G)

I = np.zeros([nelem, nip], dtype='int')
I[1,:] = 1
mat.setCusp(I, K, G, 0.01 + 0.02 * np.arange(100))

I = np.zeros([nelem, nip], dtype='int')
I[2,:] = 1
mat.setSmooth(I, K, G, 0.01 + 0.02 * np.arange(100))

eps = np.zeros((nelem, nip, 2, 2))

for e in range(nelem):
    for q in range(nip):
        fac = float((e + 1) * nip + (q + 1))
        eps[e, q, :, :] = fac * Eps

mat.setStrain(eps)
sig = mat.Stress()
epsp = mat.Epsp()

for q in range(nip):

    e = 0
    fac = float((e + 1) * nip + (q + 1))
    ISCLOSE(sig[e, q, 0, 0], fac * K * epsm)
    ISCLOSE(sig[e, q, 1, 1], fac * K * epsm)
    ISCLOSE(sig[e, q, 0, 1], fac * G * gamma)
    ISCLOSE(sig[e, q, 1, 0], fac * G * gamma)
    ISCLOSE(epsp[e, q], 0.0)

    e = 1
    fac = float((e + 1) * nip + (q + 1))
    ISCLOSE(sig[e, q, 0, 0], fac * K * epsm)
    ISCLOSE(sig[e, q, 1, 1], fac * K * epsm)
    ISCLOSE(sig[e, q, 0, 1], 0.0)
    ISCLOSE(sig[e, q, 1, 0], 0.0)
    ISCLOSE(epsp[e, q], fac * gamma)

    e = 2
    fac = float((e + 1) * nip + (q + 1))
    ISCLOSE(sig[e, q, 0, 0], fac * K * epsm)
    ISCLOSE(sig[e, q, 1, 1], fac * K * epsm)
    ISCLOSE(sig[e, q, 0, 1], 0.0)
    ISCLOSE(sig[e, q, 1, 0], 0.0)
    ISCLOSE(epsp[e, q], fac * gamma)

print('All checks passed')
