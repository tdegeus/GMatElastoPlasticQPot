import GMatElastoPlasticQPot.Cartesian2d as GMat
import GMatTensor.Cartesian2d as tensor
import h5py
import numpy as np

with h5py.File("Cartesian2d_random.hdf5", "w") as data:

    shape = [1000, 4]
    iden = 3 * np.random.random(shape)
    iden = np.where(iden < 1, 0, iden)
    iden = np.where((iden >= 1) * (iden < 2), 1, iden)
    iden = np.where(iden >= 2, 2, iden)
    iden = iden.astype(int)

    data["/shape"] = shape

    Sig = np.empty(shape + [2, 2])
    C = np.empty(shape + [2, 2, 2, 2])
    index = np.zeros(shape, dtype=np.uint64)
    epsy_left = -np.inf * np.ones(shape)
    epsy_right = np.inf * np.ones(shape)

    mat = {}

    # cusp

    I = np.where(iden == 0, 1, 0).astype(int)
    n = np.sum(I)
    idx = np.zeros(I.size, int)
    idx[np.argwhere(I.ravel() == 1).ravel()] = np.arange(n)
    idx = idx.reshape(I.shape)

    K = np.random.random(n)
    G = np.random.random(n)
    epsy = np.cumsum(np.random.random([n, 500]), 1)
    epsy[:, 0] = -epsy[:, 1]
    mat["Cusp1d"] = {"mat": GMat.Cusp1d(K, G, epsy), "is": iden == 0}

    data["/cusp/I"] = I
    data["/cusp/idx"] = idx
    data["/cusp/K"] = K
    data["/cusp/G"] = G
    data["/cusp/epsy"] = epsy
    data["/iden"] = iden

    # smooth

    I = np.where(iden == 1, 1, 0).astype(int)
    n = np.sum(I)
    idx = np.zeros(I.size, int)
    idx[np.argwhere(I.ravel() == 1).ravel()] = np.arange(n)
    idx = idx.reshape(I.shape)

    K = np.random.random(n)
    G = np.random.random(n)
    epsy = np.cumsum(np.random.random([n, 500]), 1)
    epsy[:, 0] = -epsy[:, 1]
    mat["Smooth1d"] = {"mat": GMat.Smooth1d(K, G, epsy), "is": iden == 1}

    data["/smooth/I"] = I
    data["/smooth/idx"] = idx
    data["/smooth/K"] = K
    data["/smooth/G"] = G
    data["/smooth/epsy"] = epsy

    # elastic

    I = np.where(iden == 2, 1, 0).astype(int)
    n = np.sum(I)
    idx = np.zeros(I.size, int)
    idx[np.argwhere(I.ravel() == 1).ravel()] = np.arange(n)
    idx = idx.reshape(I.shape)

    K = np.random.random(n)
    G = np.random.random(n)
    mat["Elastic1d"] = {"mat": GMat.Elastic1d(K, G), "is": iden == 2}

    data["/elastic/I"] = I
    data["/elastic/idx"] = idx
    data["/elastic/K"] = K
    data["/elastic/G"] = G
    data["/elastic/epsy"] = epsy

    for m in mat:
        mat[m]["is_tensor2"] = np.zeros(shape + [2, 2], bool)
        mat[m]["is_tensor4"] = np.zeros(shape + [2, 2, 2, 2], bool)
        mat[m]["is_tensor2"] += (mat[m]["is"]).reshape(shape + [1, 1])
        mat[m]["is_tensor4"] += (mat[m]["is"]).reshape(shape + [1, 1, 1, 1])

    I4s = tensor.Array2d(shape).I4s

    for i in range(20):

        GradU = 200 * np.random.random(shape + [2, 2])

        data[f"/random/{i:d}/GradU"] = GradU
        Eps = tensor.A4_ddot_B2(I4s, GradU)

        for m in mat:
            mat[m]["mat"].Eps = Eps[mat[m]["is_tensor2"]].reshape(-1, 2, 2)
            Sig[mat[m]["is_tensor2"]] = mat[m]["mat"].Sig.reshape(-1)
            C[mat[m]["is_tensor4"]] = mat[m]["mat"].C.reshape(-1)
            if m == "Elastic1d":
                continue
            index[mat[m]["is"]] = mat[m]["mat"].i
            epsy_left[mat[m]["is"]] = mat[m]["mat"].epsy_left
            epsy_right[mat[m]["is"]] = mat[m]["mat"].epsy_right

        data[f"/random/{i:d}/Stress"] = Sig
        data[f"/random/{i:d}/Tangent"] = C
        data[f"/random/{i:d}/CurrentIndex"] = index
        data[f"/random/{i:d}/CurrentYieldLeft"] = epsy_left
        data[f"/random/{i:d}/CurrentYieldRight"] = epsy_right
