
#include <catch2/catch.hpp>
#include <xtensor/xrandom.hpp>

#define EQ(a,b) REQUIRE_THAT((a), Catch::WithinAbs((b), 1.e-12));

#include <GMatElastoPlasticQPot/Cartesian2d.h>

namespace GM = GMatElastoPlasticQPot::Cartesian2d;

template <class T, class S>
S A4_ddot_B2(const T& A, const S& B)
{
    S C = xt::empty<double>({2, 2});
    C.fill(0.0);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            for (size_t k = 0; k < 2; k++) {
                for (size_t l = 0; l < 2; l++) {
                    C(i, j) += A(i, j, k, l) * B(l, k);
                }
            }
        }
    }

    return C;
}

TEST_CASE("GMatElastoPlasticQPot::Cartesian2d", "Cartesian2d.h")
{

SECTION("Identity tensor: Id")
{
    GM::Tensor2 A = xt::random::randn<double>({2, 2});
    GM::Tensor2 I = GM::I2();
    GM::Tensor4 Id = GM::I4d();
    GM::Tensor4 Is = GM::I4s();
    A = A4_ddot_B2(Is, A);
    REQUIRE(xt::allclose(A4_ddot_B2(Id, A), A - GM::Hydrostatic(A) * I));
}

SECTION("Tangent (purely elastic response only)")
{
    double K = 12.3;
    double G = 45.6;

    GM::Tensor2 Eps = xt::random::randn<double>({2, 2});
    GM::Tensor2 Sig;
    GM::Tensor4 C;
    GM::Tensor4 Is = GM::I4s();
    Eps = A4_ddot_B2(Is, Eps);

    // Elastic
    {
        GM::Elastic mat(K, G);
        std::tie(Sig, C) = mat.Tangent(Eps);
        REQUIRE(xt::allclose(A4_ddot_B2(C, Eps), Sig));
    }

    // Cusp
    {
        GM::Cusp mat(K, G, {10000.0});
        std::tie(Sig, C) = mat.Tangent(Eps);
        REQUIRE(xt::allclose(A4_ddot_B2(C, Eps), Sig));
    }

    // Smooth
    {
        GM::Smooth mat(K, G, {10000.0});
        std::tie(Sig, C) = mat.Tangent(Eps);
        REQUIRE(xt::allclose(A4_ddot_B2(C, Eps), Sig));
    }
}

SECTION("Elasto-plastic response")
{
    double K = 12.3;
    double G = 45.6;

    auto Eps = GM::Tensor2::from_shape({2, 2});
    auto Sig = GM::Tensor2::from_shape({2, 2});

    double gamma = 0.02;
    double epsm = 0.12;

    Eps(0,0) = Eps(1,1) = epsm;
    Eps(0,1) = Eps(1,0) = gamma;

    // Elastic
    {
        GM::Elastic mat(K, G);
        Sig = mat.Stress(Eps);

        EQ(Sig(0,0), K * epsm);
        EQ(Sig(1,1), K * epsm);
        EQ(Sig(0,1), G * gamma);
        EQ(Sig(1,0), G * gamma);
    }

    // Cusp
    {
        GM::Cusp mat(K, G, {0.01, 0.03, 0.10});
        Sig = mat.Stress(Eps);

        EQ(Sig(0,0), K * epsm);
        EQ(Sig(1,1), K * epsm);
        EQ(Sig(0,1), 0.0);
        EQ(Sig(1,0), 0.0);
        EQ(mat.epsp(Eps), 0.02);
        REQUIRE(mat.find(Eps) == 1);
    }

    // Smooth
    {
        GM::Smooth mat(K, G, {0.01, 0.03, 0.10});
        Sig = mat.Stress(Eps);

        EQ(Sig(0,0), K * epsm);
        EQ(Sig(1,1), K * epsm);
        EQ(Sig(0,1), 0.0);
        EQ(Sig(1,0), 0.0);
        EQ(mat.epsp(Eps), 0.02);
        REQUIRE(mat.find(Eps) == 1);
    }

    // Matrix
    {
        size_t nelem = 3;
        size_t nip = 2;

        GM::Matrix mat(nelem, nip);

        {
            xt::xtensor<size_t,2> I = xt::zeros<size_t>({nelem, nip});
            xt::view(I, 0, xt::all()) = 1;
            mat.setElastic(I, K, G);
        }

        {
            xt::xtensor<size_t,2> I = xt::zeros<size_t>({nelem, nip});
            xt::xtensor<double,1> epsy = {0.01, 0.03, 0.10};
            xt::view(I, 1, xt::all()) = 1;
            mat.setCusp(I, K, G, epsy);
        }

        {
            xt::xtensor<size_t,2> I = xt::zeros<size_t>({nelem, nip});
            xt::xtensor<double,1> epsy = {0.01, 0.03, 0.10};
            xt::view(I, 2, xt::all()) = 1;
            mat.setCusp(I, K, G, epsy);
        }

        auto eps = xt::xtensor<double,4>::from_shape({nelem, nip, 2ul, 2ul});

        for (size_t e = 0; e < nelem; ++e) {
            for (size_t q = 0; q < nip; ++q) {
                xt::view(eps, e, q) = Eps;
            }
        }

        auto sig = mat.Stress(eps);
        auto epsp = mat.Epsp(eps);

        for (size_t q = 0; q < nip; ++q) {

            EQ(sig(0,q,0,0), K * epsm);
            EQ(sig(0,q,1,1), K * epsm);
            EQ(sig(0,q,0,1), G * gamma);
            EQ(sig(0,q,1,0), G * gamma);
            EQ(epsp(0,q), 0.0);

            EQ(sig(1,q,0,0), K * epsm);
            EQ(sig(1,q,1,1), K * epsm);
            EQ(sig(1,q,0,1), 0.0);
            EQ(sig(1,q,1,0), 0.0);
            EQ(epsp(1,q), gamma);

            EQ(sig(2,q,0,0), K * epsm);
            EQ(sig(2,q,1,1), K * epsm);
            EQ(sig(2,q,0,1), 0.0);
            EQ(sig(2,q,1,0), 0.0);
            EQ(epsp(2,q), gamma);
        }
    }
}

}
