#include <GMatElastoPlasticQPot/Cartesian2d.h>
#include <xtensor/xrandom.hpp>

namespace GMat = GMatElastoPlasticQPot::Cartesian2d;

int main()
{

    xt::xtensor<double, 2> A = xt::random::randn<double>({2, 2});
    A(0, 0) = 1.0;
    A(1, 1) = 1.0;

    xt::xtensor<double, 4> M = xt::zeros<double>({800, 1000, 2, 2});
    for (size_t i = 0; i < M.shape(0); ++i) {
        for (size_t j = 0; j < M.shape(1); ++j) {
            xt::view(M, i, j, xt::all(), xt::all()) = static_cast<double>(i * M.shape(1) + j) * A;
        }
    }

    auto e = GMat::Epsd(M);
    GMat::epsd(M, e);

    return 0;
}
