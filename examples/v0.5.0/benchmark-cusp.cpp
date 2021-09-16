#include <GMatElastoPlasticQPot/Cartesian2d.h>

namespace GMat = GMatElastoPlasticQPot::Cartesian2d;

int main()
{
    xt::xtensor<double, 1> epsy = xt::linspace<double>(1e-5, 1, 20001);
    GMat::Tensor2 Eps = xt::zeros<double>({2, 2});
    GMat::Tensor2 Sig = xt::zeros<double>({2, 2});

    GMat::Cusp cusp(1.0, 1.0, epsy);

    for (size_t i = 0; i < 100000; ++i) {
        Eps(0, 1) = 0.5 / 100000.0 * static_cast<double>(i + 1);
        Eps(1, 0) = 0.5 / 100000.0 * static_cast<double>(i + 1);
        cusp.stress(Eps, Sig);
    }

    return 0;
}
