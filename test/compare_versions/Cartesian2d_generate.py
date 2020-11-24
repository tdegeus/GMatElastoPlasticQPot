import h5py
import numpy as np
import GMatElastoPlasticQPot.Cartesian2d as GMat

with h5py.File('Cartesian2d_random.hdf5', 'w') as data:

    nelem = 1000
    nip = 4
    iden = 3.0 * np.random.random([nelem, nip])
    iden = np.where(iden < 1.0, 0.0, iden)
    iden = np.where((iden >= 1.0) * (iden < 2.0), 1.0, iden)
    iden = np.where(iden >= 2.0, 2.0, iden)
    iden = iden.astype(np.int)

    shape = np.array([nelem, nip], np.int)

    data['/shape'] = shape

    mat = GMat.Array2d(shape)

    I = np.where(iden == 0, 1, 0).astype(np.int)
    n = np.sum(I)
    idx = np.zeros(I.size, np.int)
    idx[np.argwhere(I.ravel() == 1).ravel()] = np.arange(n)
    idx = idx.reshape(I.shape)
    epsy = np.cumsum(np.random.random([n, 500]), 1)
    K = np.ones(n)
    G = np.ones(n)

    data['/cusp/I'] = I
    data['/cusp/idx'] = idx
    data['/cusp/K'] = K
    data['/cusp/G'] = G
    data['/cusp/epsy'] = epsy

    mat.setCusp(I, idx, K, G, epsy)

    I = np.where(iden == 1, 1, 0).astype(np.int)
    n = np.sum(I)
    idx = np.zeros(I.size, np.int)
    idx[np.argwhere(I.ravel() == 1).ravel()] = np.arange(n)
    idx = idx.reshape(I.shape)
    epsy = np.cumsum(np.random.random([n, 500]), 1)
    K = np.ones(n)
    G = np.ones(n)

    data['/smooth/I'] = I
    data['/smooth/idx'] = idx
    data['/smooth/K'] = K
    data['/smooth/G'] = G
    data['/smooth/epsy'] = epsy

    mat.setSmooth(I, idx, K, G, epsy)

    I = np.where(iden == 2, 1, 0).astype(np.int)
    n = np.sum(I)
    idx = np.zeros(I.size, np.int)
    idx[np.argwhere(I.ravel() == 1).ravel()] = np.arange(n)
    idx = idx.reshape(I.shape)
    K = np.ones(n)
    G = np.ones(n)

    data['/elastic/I'] = I
    data['/elastic/idx'] = idx
    data['/elastic/K'] = K
    data['/elastic/G'] = G

    mat.setElastic(I, idx, K, G)
    mat.check()

    for i in range(20):

        GradU = 200 * np.random.random([nelem, nip, 2, 2])

        data['/random/{0:d}/GradU'.format(i)] = GradU

        Eps = np.einsum('...ijkl,...lk->...ij', mat.I4s(), GradU)
        mat.setStrain(Eps)

        data['/random/{0:d}/Stress'.format(i)] = mat.Stress()
        data['/random/{0:d}/CurrentIndex'.format(i)] = mat.CurrentIndex()
        data['/random/{0:d}/CurrentYieldLeft'.format(i)] = mat.CurrentYieldLeft()
        data['/random/{0:d}/CurrentYieldRight'.format(i)] = mat.CurrentYieldRight()

