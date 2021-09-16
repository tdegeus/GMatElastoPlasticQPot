#define CATCH_CONFIG_MAIN
#include <GMatElastoPlasticQPot/Cartesian2d.h>
#include <GMatTensor/Cartesian2d.h>
#include <catch2/catch.hpp>
#include <xtensor/xrandom.hpp>

namespace GM = GMatElastoPlasticQPot::Cartesian2d;
namespace GT = GMatTensor::Cartesian2d;

TEST_CASE("GMatElastoPlasticQPot::Cartesian2d", "Cartesian2d.h")
{
    SECTION("Epsd - Tensor2")
    {
        auto A = GT::O2();
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        REQUIRE(GM::Epsd(A)() == Approx(1.0));
    }

    SECTION("Epsd - List")
    {
        auto A = GT::O2();
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        auto M = xt::xtensor<double, 3>::from_shape({3, 2, 2});
        auto R = xt::xtensor<double, 1>::from_shape({M.shape(0)});
        for (size_t i = 0; i < M.shape(0); ++i) {
            xt::view(M, i, xt::all(), xt::all()) = static_cast<double>(i) * A;
            R(i) = static_cast<double>(i);
        }
        REQUIRE(xt::allclose(GM::Epsd(M), R));
    }

    SECTION("Epsd - Matrix")
    {
        auto A = GT::O2();
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        auto M = xt::xtensor<double, 4>::from_shape({3, 4, 2, 2});
        auto R = xt::xtensor<double, 2>::from_shape({M.shape(0), M.shape(1)});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                xt::view(M, i, j, xt::all(), xt::all()) =
                    static_cast<double>(i * M.shape(1) + j) * A;
                R(i, j) = static_cast<double>(i * M.shape(1) + j);
            }
        }
        REQUIRE(xt::allclose(GM::Epsd(M), R));
    }

    SECTION("Sigd - Tensor2")
    {
        auto A = GT::O2();
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        REQUIRE(GM::Sigd(A)() == Approx(2.0));
    }

    SECTION("Sigd - List")
    {
        auto A = GT::O2();
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        auto M = xt::xtensor<double, 3>::from_shape({3, 2, 2});
        auto R = xt::xtensor<double, 1>::from_shape({M.shape(0)});
        for (size_t i = 0; i < M.shape(0); ++i) {
            xt::view(M, i, xt::all(), xt::all()) = static_cast<double>(i) * A;
            R(i) = static_cast<double>(i) * 2.0;
        }
        REQUIRE(xt::allclose(GM::Sigd(M), R));
    }

    SECTION("Sigd - Matrix")
    {
        auto A = GT::O2();
        A(0, 1) = 1.0;
        A(1, 0) = 1.0;
        auto M = xt::xtensor<double, 4>::from_shape({3, 4, 2, 2});
        auto R = xt::xtensor<double, 2>::from_shape({M.shape(0), M.shape(1)});
        for (size_t i = 0; i < M.shape(0); ++i) {
            for (size_t j = 0; j < M.shape(1); ++j) {
                xt::view(M, i, j, xt::all(), xt::all()) =
                    static_cast<double>(i * M.shape(1) + j) * A;
                R(i, j) = static_cast<double>(i * M.shape(1) + j) * 2.0;
            }
        }
        REQUIRE(xt::allclose(GM::Sigd(M), R));
    }

    SECTION("Elastic - stress")
    {
        double K = 12.3;
        double G = 45.6;
        double gamma = 0.02;
        double epsm = 0.12;

        xt::xtensor<double, 2> Eps = {{epsm, gamma}, {gamma, epsm}};

        xt::xtensor<double, 2> Sig = {{K * epsm, G * gamma}, {G * gamma, K * epsm}};

        GM::Elastic mat(K, G);
        mat.setStrain(Eps);

        REQUIRE(xt::allclose(mat.Stress(), Sig));
        REQUIRE(mat.energy() == Approx(K * std::pow(epsm, 2.0) + G * std::pow(gamma, 2.0)));
    }

    SECTION("Cusp - stress (0)")
    {
        double K = 12.3;
        double G = 45.6;
        xt::xtensor<double, 2> Eps = xt::zeros<double>({2, 2});
        xt::xtensor<double, 2> Sig = xt::zeros<double>({2, 2});
        ;

        GM::Cusp mat(K, G, xt::xtensor<double, 1>{0.01, 0.03, 0.05, 0.10});
        mat.setStrain(Eps);

        REQUIRE(xt::allclose(mat.Stress(), Sig));
        REQUIRE(mat.currentIndex() == 0);
        REQUIRE(mat.epsp() == 0);
        REQUIRE(mat.currentYieldLeft() == -0.01);
        REQUIRE(mat.currentYieldRight() == +0.01);
        REQUIRE(mat.currentYieldLeft() == mat.refQPotChunked().yleft());
        REQUIRE(mat.currentYieldRight() == mat.refQPotChunked().yright());
        REQUIRE(mat.checkYieldBoundRight());
        REQUIRE(mat.checkYieldRedraw() == 0);
        REQUIRE(mat.energy() == Approx(-G * std::pow(0.01, 2.0)));
    }

    SECTION("Cusp - stress (1)")
    {
        double K = 12.3;
        double G = 45.6;
        double gamma = 0.02;
        double epsm = 0.12;

        xt::xtensor<double, 2> Eps = {{epsm, gamma}, {gamma, epsm}};

        xt::xtensor<double, 2> Sig = {{K * epsm, 0.0}, {0.0, K * epsm}};

        GM::Cusp mat(K, G, xt::xtensor<double, 1>{0.01, 0.03, 0.05, 0.10});
        mat.setStrain(Eps);

        REQUIRE(xt::allclose(mat.Stress(), Sig));
        REQUIRE(mat.currentIndex() == 1);
        REQUIRE(mat.epsp() == 0.02);
        REQUIRE(mat.currentYieldLeft() == 0.01);
        REQUIRE(mat.currentYieldRight() == 0.03);
        REQUIRE(mat.currentYieldLeft() == mat.refQPotChunked().yleft());
        REQUIRE(mat.currentYieldRight() == mat.refQPotChunked().yright());
        REQUIRE(mat.checkYieldBoundLeft());
        REQUIRE(mat.checkYieldBoundRight());
        REQUIRE(mat.checkYieldRedraw() == 0);
        REQUIRE(mat.energy() == Approx(K * std::pow(epsm, 2.0) + G * (0.0 - std::pow(0.01, 2.0))));
    }

    SECTION("Cusp - stress (2)")
    {
        double K = 12.3;
        double G = 45.6;
        double gamma = 1.9 * 0.02;
        double epsm = 2.0 * 0.12;

        xt::xtensor<double, 2> Eps = {{epsm, gamma}, {gamma, epsm}};

        xt::xtensor<double, 2> Sig = {
            {K * epsm, G * (gamma - 0.04)}, {G * (gamma - 0.04), K * epsm}};

        GM::Cusp mat(K, G, xt::xtensor<double, 1>{0.01, 0.03, 0.05, 0.10});
        mat.setStrain(Eps);

        REQUIRE(xt::allclose(mat.Stress(), Sig));
        REQUIRE(mat.currentIndex() == 2);
        REQUIRE(mat.epsp() == 0.04);
        REQUIRE(mat.currentYieldLeft() == 0.03);
        REQUIRE(mat.currentYieldRight() == 0.05);
        REQUIRE(mat.currentYieldLeft() == mat.refQPotChunked().yleft());
        REQUIRE(mat.currentYieldRight() == mat.refQPotChunked().yright());
        REQUIRE(mat.checkYieldBoundLeft());
        REQUIRE(mat.checkYieldBoundRight());
        REQUIRE(mat.checkYieldRedraw() == 0);
        REQUIRE(
            mat.energy() ==
            Approx(
                K * std::pow(epsm, 2.0) + G * (std::pow(gamma - 0.04, 2.0) - std::pow(0.01, 2.0))));
    }

    SECTION("Smooth - stress")
    {
        double K = 12.3;
        double G = 45.6;
        double gamma = 0.02;
        double epsm = 0.12;

        xt::xtensor<double, 2> Eps = {{epsm, gamma}, {gamma, epsm}};

        xt::xtensor<double, 2> Sig = {{K * epsm, 0.0}, {0.0, K * epsm}};

        GM::Smooth mat(K, G, xt::xtensor<double, 1>{0.01, 0.03, 0.05, 0.10});
        mat.setStrain(Eps);

        REQUIRE(xt::allclose(mat.Stress(), Sig));
        REQUIRE(mat.currentIndex() == 1);
        REQUIRE(mat.epsp() == 0.02);
        REQUIRE(mat.currentYieldLeft() == 0.01);
        REQUIRE(mat.currentYieldRight() == 0.03);
        REQUIRE(mat.currentYieldLeft() == mat.refQPotChunked().yleft());
        REQUIRE(mat.currentYieldRight() == mat.refQPotChunked().yright());
        REQUIRE(mat.checkYieldBoundLeft());
        REQUIRE(mat.checkYieldBoundRight());
        REQUIRE(mat.checkYieldRedraw() == 0);
    }

    SECTION("Tangent (purely elastic response only) - Elastic")
    {
        double K = 12.3;
        double G = 45.6;

        xt::xtensor<double, 2> Eps = xt::random::randn<double>({2, 2});
        xt::xtensor<double, 4> Is = GM::I4s();
        Eps = GT::A4_ddot_B2(Is, Eps);

        GM::Elastic mat(K, G);
        mat.setStrain(Eps);
        auto Sig = mat.Stress();
        auto C = mat.Tangent();
        REQUIRE(xt::allclose(GT::A4_ddot_B2(C, Eps), Sig));
    }

    SECTION("Tangent (purely elastic response only) - Cusp")
    {
        double K = 12.3;
        double G = 45.6;

        xt::xtensor<double, 2> Eps = xt::random::randn<double>({2, 2});
        xt::xtensor<double, 4> Is = GM::I4s();
        Eps = GT::A4_ddot_B2(Is, Eps);

        GM::Cusp mat(K, G, xt::xtensor<double, 1>{10000.0});
        mat.setStrain(Eps);
        auto Sig = mat.Stress();
        auto C = mat.Tangent();
        REQUIRE(xt::allclose(GT::A4_ddot_B2(C, Eps), Sig));
    }

    SECTION("Tangent (purely elastic response only) - Smooth")
    {
        double K = 12.3;
        double G = 45.6;

        xt::xtensor<double, 2> Eps = xt::random::randn<double>({2, 2});
        xt::xtensor<double, 4> Is = GM::I4s();
        Eps = GT::A4_ddot_B2(Is, Eps);

        GM::Smooth mat(K, G, xt::xtensor<double, 1>{10000.0});
        mat.setStrain(Eps);
        auto Sig = mat.Stress();
        auto C = mat.Tangent();
        REQUIRE(xt::allclose(GT::A4_ddot_B2(C, Eps), Sig));
    }

    SECTION("Array")
    {
        double K = 12.3;
        double G = 45.6;
        double gamma = 0.02;
        double epsm = 0.12;

        xt::xtensor<double, 2> Eps = {{epsm, gamma}, {gamma, epsm}};

        xt::xtensor<double, 2> Sig_elas = {{K * epsm, G * gamma}, {G * gamma, K * epsm}};

        xt::xtensor<double, 2> Sig_plas = {{K * epsm, 0.0}, {0.0, K * epsm}};

        size_t nelem = 3;
        size_t nip = 2;
        size_t ndim = 2;

        GM::Array<2> mat({nelem, nip});
        xt::xtensor<bool, 2> isElastic = xt::zeros<bool>({nelem, nip});
        xt::xtensor<bool, 2> isCusp = xt::zeros<bool>({nelem, nip});
        xt::xtensor<bool, 2> isSmooth = xt::zeros<bool>({nelem, nip});

        {
            xt::view(isElastic, 0, xt::all()) = true;
            mat.setElastic(isElastic, K, G);
        }

        {
            xt::xtensor<double, 1> epsy = 0.01 + 0.02 * xt::arange<double>(100);
            xt::view(isCusp, 1, xt::all()) = true;
            mat.setCusp(isCusp, K, G, epsy);
        }

        {
            xt::xtensor<double, 1> epsy = 0.01 + 0.02 * xt::arange<double>(100);
            xt::view(isSmooth, 2, xt::all()) = true;
            mat.setSmooth(isSmooth, K, G, epsy);
        }

        REQUIRE(xt::all(xt::equal(mat.isElastic(), isElastic)));
        REQUIRE(xt::all(xt::equal(mat.isCusp(), isCusp)));
        REQUIRE(xt::all(xt::equal(mat.isSmooth(), isSmooth)));
        REQUIRE(xt::all(xt::equal(mat.isPlastic(), isSmooth || isCusp)));

        xt::xtensor<double, 4> eps = xt::empty<double>({nelem, nip, ndim, ndim});
        xt::xtensor<double, 4> sig = xt::empty<double>({nelem, nip, ndim, ndim});
        xt::xtensor<double, 2> epsp = xt::empty<double>({nelem, nip});

        for (size_t e = 0; e < nelem; ++e) {
            for (size_t q = 0; q < nip; ++q) {
                double fac = static_cast<double>((e + 1) * nip + (q + 1));
                xt::view(eps, e, q) = fac * Eps;
                if (e == 0) {
                    xt::view(sig, e, q) = fac * Sig_elas;
                    epsp(e, q) = 0.0;
                }
                else {
                    xt::view(sig, e, q) = fac * Sig_plas;
                    epsp(e, q) = fac * gamma;
                }
            }
        }

        mat.setStrain(eps);

        REQUIRE(xt::allclose(mat.Stress(), sig));
        REQUIRE(xt::allclose(mat.Epsp(), epsp));
        REQUIRE(mat.checkYieldBoundLeft());
        REQUIRE(mat.checkYieldBoundRight());
        REQUIRE(xt::all(xt::equal(mat.CheckYieldRedraw(), 0)));
    }

    SECTION("Array - reference to underlying model")
    {
        double K = 12.3;
        double G = 45.6;
        double gamma = 0.02;
        double epsm = 0.12;

        xt::xtensor<double, 2> Eps = {{epsm, gamma}, {gamma, epsm}};

        xt::xtensor<double, 2> Sig_elas = {{K * epsm, G * gamma}, {G * gamma, K * epsm}};

        xt::xtensor<double, 2> Sig_plas = {{K * epsm, 0.0}, {0.0, K * epsm}};

        size_t nelem = 3;
        size_t nip = 2;

        GM::Array<2> mat({nelem, nip});

        {
            xt::xtensor<bool, 2> I = xt::zeros<bool>({nelem, nip});
            xt::view(I, 0, xt::all()) = true;
            mat.setElastic(I, K, G);
        }

        {
            xt::xtensor<bool, 2> I = xt::zeros<bool>({nelem, nip});
            xt::xtensor<double, 1> epsy = 0.01 + 0.02 * xt::arange<double>(100);
            xt::view(I, 1, xt::all()) = true;
            mat.setCusp(I, K, G, epsy);
        }

        {
            xt::xtensor<bool, 2> I = xt::zeros<bool>({nelem, nip});
            xt::xtensor<double, 1> epsy = 0.01 + 0.02 * xt::arange<double>(100);
            xt::view(I, 2, xt::all()) = true;
            mat.setSmooth(I, K, G, epsy);
        }

        for (size_t e = 0; e < nelem; ++e) {
            for (size_t q = 0; q < nip; ++q) {
                double fac = static_cast<double>((e + 1) * nip + (q + 1));
                if (e == 0) {
                    auto model = mat.refElastic({e, q});
                    model.setStrain(xt::eval(fac * Eps));
                    REQUIRE(xt::allclose(model.Stress(), fac * Sig_elas));
                }
                else if (e == 1) {
                    auto model = mat.refCusp({e, q});
                    model.setStrain(xt::eval(fac * Eps));
                    REQUIRE(xt::allclose(model.Stress(), fac * Sig_plas));
                    REQUIRE(xt::allclose(model.epsp(), fac * gamma));
                }
                else if (e == 2) {
                    auto model = mat.refSmooth({e, q});
                    model.setStrain(xt::eval(fac * Eps));
                    REQUIRE(xt::allclose(model.Stress(), fac * Sig_plas));
                    REQUIRE(xt::allclose(model.epsp(), fac * gamma));
                }
            }
        }
    }

    SECTION("Equivalent Elastic Array")
    {
        double K = 12.3;
        double G = 45.6;
        double gamma = 0.02;
        double epsm = 0.12;

        xt::xtensor<double, 2> Eps = {{epsm, gamma}, {gamma, epsm}};

        xt::xtensor<double, 2> Sig_elas = {{K * epsm, G * gamma}, {G * gamma, K * epsm}};

        size_t nelem = 3;
        size_t nip = 2;
        size_t ndim = 2;

        GM::Array<2> mat({nelem, nip});

        {
            xt::xtensor<bool, 2> I = xt::zeros<bool>({nelem, nip});
            xt::view(I, 0, xt::all()) = true;
            mat.setElastic(I, K, G);
        }

        {
            xt::xtensor<bool, 2> I = xt::zeros<bool>({nelem, nip});
            xt::xtensor<double, 1> epsy = 0.01 + 0.02 * xt::arange<double>(100);
            xt::view(I, 1, xt::all()) = true;
            mat.setCusp(I, K, G, epsy);
        }

        {
            xt::xtensor<bool, 2> I = xt::zeros<bool>({nelem, nip});
            xt::xtensor<double, 1> epsy = 0.01 + 0.02 * xt::arange<double>(100);
            xt::view(I, 2, xt::all()) = true;
            mat.setSmooth(I, K, G, epsy);
        }

        GM::Array<2> elas(mat.shape());
        elas.setElastic(mat.K(), mat.G());

        xt::xtensor<double, 4> eps = xt::empty<double>({nelem, nip, ndim, ndim});
        xt::xtensor<double, 4> sig = xt::empty<double>({nelem, nip, ndim, ndim});
        xt::xtensor<double, 2> epsp = xt::zeros<double>({nelem, nip});

        for (size_t e = 0; e < nelem; ++e) {
            for (size_t q = 0; q < nip; ++q) {
                double fac = static_cast<double>((e + 1) * nip + (q + 1));
                xt::view(eps, e, q) = fac * Eps;
                xt::view(sig, e, q) = fac * Sig_elas;
            }
        }

        elas.setStrain(eps);

        REQUIRE(xt::allclose(elas.Stress(), sig));
        REQUIRE(xt::allclose(elas.Epsp(), epsp));
        REQUIRE(xt::allclose(elas.K(), mat.K()));
        REQUIRE(xt::allclose(elas.G(), mat.G()));
    }
}
