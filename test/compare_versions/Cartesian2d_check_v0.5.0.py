import h5py
import numpy as np
import GMatElastoPlasticQPot.Cartesian2d as GMat

def A4_ddot_B2(A, B):
    return np.einsum('...ijkl,...lk->...ij', A, B)

with h5py.File('Cartesian2d_random.hdf5', 'r') as data:

    shape = data['/shape'][...]

    mat = GMat.Matrix(shape[0], shape[1])

    I = data['/cusp/I'][...]
    idx = data['/cusp/idx'][...]
    K = data['/cusp/K'][...]
    G = data['/cusp/G'][...]
    epsy = data['/cusp/epsy'][...]

    mat.setCusp(I, idx, K, G, epsy)

    I = data['/elastic/I'][...]
    idx = data['/elastic/idx'][...]
    K = data['/elastic/K'][...]
    G = data['/elastic/G'][...]

    mat.setElastic(I, idx, K, G)

    for i in range(20):

        GradU = data['/random/{0:d}/GradU'.format(i)][...]

        Eps = np.einsum('...ijkl,...lk->...ij', mat.I4s(), GradU)
        idx = mat.Find(Eps)

        assert np.allclose(mat.Stress(Eps), data['/random/{0:d}/Stress'.format(i)][...])
        assert np.allclose(mat.Find(Eps), data['/random/{0:d}/CurrentIndex'.format(i)][...])
        assert np.allclose(mat.Epsy(idx), data['/random/{0:d}/CurrentYieldLeft'.format(i)][...])
        assert np.allclose(mat.Epsy(idx + 1), data['/random/{0:d}/CurrentYieldRight'.format(i)][...])

