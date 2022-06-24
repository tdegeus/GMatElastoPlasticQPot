import unittest

import GMatElastoPlasticQPot.Cartesian2d as GMat
import GMatTensor.Cartesian2d as tensor
import h5py
import numpy as np


class Test(unittest.TestCase):
    def test_main(self):

        with h5py.File("Cartesian2d_random.hdf5", "r") as data:

            shape = list(data["/shape"][...])
            I4s = tensor.Array2d(shape).I4s
            mat = {
                "Cusp1d": {
                    "mat": GMat.Cusp1d(
                        data["/cusp/K"][...], data["/cusp/G"][...], data["/cusp/epsy"][...]
                    ),
                    "is": data["/iden"][...] == 0,
                },
                "Smooth1d": {
                    "mat": GMat.Smooth1d(
                        data["/smooth/K"][...], data["/smooth/G"][...], data["/smooth/epsy"][...]
                    ),
                    "is": data["/iden"][...] == 1,
                },
                "Elastic1d": {
                    "mat": GMat.Elastic1d(data["/elastic/K"][...], data["/elastic/G"][...]),
                    "is": data["/iden"][...] == 2,
                },
            }

            for m in mat:
                mat[m]["is_tensor2"] = np.zeros(shape + [2, 2], bool)
                mat[m]["is_tensor4"] = np.zeros(shape + [2, 2, 2, 2], bool)
                mat[m]["is_tensor2"] += (mat[m]["is"]).reshape(shape + [1, 1])
                mat[m]["is_tensor4"] += (mat[m]["is"]).reshape(shape + [1, 1, 1, 1])

            Sig = np.empty(shape + [2, 2])
            C = np.empty(shape + [2, 2, 2, 2])
            index = np.zeros(shape, dtype=np.uint64)
            epsy_left = -np.inf * np.ones(shape)
            epsy_right = np.inf * np.ones(shape)

            for i in range(20):

                GradU = data[f"/random/{i:d}/GradU"][...]

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

                self.assertTrue(np.allclose(Sig, data[f"/random/{i:d}/Stress"][...]))
                self.assertTrue(np.allclose(C, data[f"/random/{i:d}/Tangent"][...]))
                self.assertTrue(
                    np.allclose(
                        epsy_left,
                        data[f"/random/{i:d}/CurrentYieldLeft"][...],
                    )
                )
                self.assertTrue(
                    np.allclose(
                        epsy_right,
                        data[f"/random/{i:d}/CurrentYieldRight"][...],
                    )
                )
                self.assertTrue(np.all(index == data[f"/random/{i:d}/CurrentIndex"][...]))


if __name__ == "__main__":

    unittest.main()
