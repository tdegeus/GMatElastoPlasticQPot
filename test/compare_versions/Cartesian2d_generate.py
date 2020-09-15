import h5py
import numpy as np
import GMatElastoPlasticQPot.Cartesian2d as GMat

def A4_ddot_B2(A, B):
    return

with h5py.File('Cartesian2d_random.hdf5', 'w') as data:

    nelem = 500
    nip = 4
    nplas = int(nelem / 2)
    nelas = nelem - nplas

    shape = np.array([nelem, nip], dtype='int')

    data['/shape'] = shape

    mat = GMat.Array2d(shape)

    I = np.zeros((nelem, nip), dtype='int')
    idx = np.zeros((nelem, nip), dtype='int')

    I[:nplas, :] = 1
    idx[:nplas, :] = np.arange(nplas * nip).reshape(-1, nip)
    epsy = np.cumsum(np.random.random([nplas * nip, 500]), 1)
    K = np.ones(nplas * nip)
    G = np.ones(nplas * nip)

    data['/cusp/I'] = I
    data['/cusp/idx'] = idx
    data['/cusp/K'] = K
    data['/cusp/G'] = G
    data['/cusp/epsy'] = epsy

    mat.setCusp(I, idx, K, G, epsy)

    I = np.zeros((nelem, nip), dtype='int')
    idx = np.zeros((nelem, nip), dtype='int')

    I[nplas:, :] = 1
    idx[nplas:, :] = np.arange(nelas * nip).reshape(-1, nip)
    K = np.ones(nelas * nip)
    G = np.ones(nelas * nip)

    data['/elastic/I'] = I
    data['/elastic/idx'] = idx
    data['/elastic/K'] = K
    data['/elastic/G'] = G

    mat.setElastic(I, idx, K, G)

    for i in range(20):

        GradU = 200 * np.random.random([nelem, nip, 2, 2])

        data['/random/{0:d}/GradU'.format(i)] = GradU

        Eps = np.einsum('...ijkl,...lk->...ij', mat.I4s(), GradU)
        mat.setStrain(Eps)

        data['/random/{0:d}/Stress'.format(i)] = mat.Stress()
        data['/random/{0:d}/CurrentIndex'.format(i)] = mat.CurrentIndex()
        data['/random/{0:d}/CurrentYieldLeft'.format(i)] = mat.CurrentYieldLeft()
        data['/random/{0:d}/CurrentYieldRight'.format(i)] = mat.CurrentYieldRight()

