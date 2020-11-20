
#include <catch2/catch.hpp>
#include <xtensor/xrandom.hpp>
#include <GMatElastoPlasticQPot/Cartesian2d.h>

#define ISCLOSE(a,b) REQUIRE_THAT((a), Catch::WithinAbs((b), 1.e-12));

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

SECTION("Id")
{
    GM::Tensor2 A = xt::random::randn<double>({2, 2});
    GM::Tensor2 I = GM::I2();
    GM::Tensor4 Id = GM::I4d();
    GM::Tensor4 Is = GM::I4s();
    A = A4_ddot_B2(Is, A);
    REQUIRE(xt::allclose(A4_ddot_B2(Id, A), A - GM::Hydrostatic(A) * I));
}

SECTION("Deviatoric - Tensor2")
{
    GM::Tensor2 A = xt::random::randn<double>({2, 2});
    GM::Tensor2 B = A;
    double tr = B(0, 0) + B(1, 1);
    B(0, 0) -= 0.5 * tr;
    B(1, 1) -= 0.5 * tr;
    REQUIRE(xt::allclose(GM::Deviatoric(A), B));
}

SECTION("Deviatoric - List")
{
    GM::Tensor2 A = xt::random::randn<double>({2, 2});
    GM::Tensor2 B = A;
    double tr = B(0, 0) + B(1, 1);
    B(0, 0) -= 0.5 * tr;
    B(1, 1) -= 0.5 * tr;
    auto M = xt::xtensor<double,3>::from_shape({3, 2, 2});
    auto R = xt::xtensor<double,3>::from_shape(M.shape());
    for (size_t i = 0; i < M.shape(0); ++i) {
        xt::view(M, i, xt::all(), xt::all()) = static_cast<double>(i) * A;
        xt::view(R, i, xt::all(), xt::all()) = static_cast<double>(i) * B;
    }
    REQUIRE(xt::allclose(GM::Deviatoric(M), R));
}

SECTION("Deviatoric - Matrix")
{
    GM::Tensor2 A = xt::random::randn<double>({2, 2});
    GM::Tensor2 B = A;
    double tr = B(0, 0) + B(1, 1);
    B(0, 0) -= 0.5 * tr;
    B(1, 1) -= 0.5 * tr;
    auto M = xt::xtensor<double,4>::from_shape({3, 4, 2, 2});
    auto R = xt::xtensor<double,4>::from_shape(M.shape());
    for (size_t i = 0; i < M.shape(0); ++i) {
        for (size_t j = 0; j < M.shape(1); ++j) {
            xt::view(M, i, j, xt::all(), xt::all()) = static_cast<double>(i * M.shape(1) + j) * A;
            xt::view(R, i, j, xt::all(), xt::all()) = static_cast<double>(i * M.shape(1) + j) * B;
        }
    }
    REQUIRE(xt::allclose(GM::Deviatoric(M), R));
}

SECTION("Hydrostatic - Tensor2")
{
    GM::Tensor2 A = xt::random::randn<double>({2, 2});
    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    REQUIRE(GM::Hydrostatic(A)() == Approx(1.0));
}

SECTION("Hydrostatic - List")
{
    GM::Tensor2 A = xt::random::randn<double>({2, 2});
    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    auto M = xt::xtensor<double,3>::from_shape({3, 2, 2});
    auto R = xt::xtensor<double,1>::from_shape({M.shape(0)});
    for (size_t i = 0; i < M.shape(0); ++i) {
        xt::view(M, i, xt::all(), xt::all()) = static_cast<double>(i) * A;
        R(i) = static_cast<double>(i);
    }
    REQUIRE(xt::allclose(GM::Hydrostatic(M), R));
}

SECTION("Hydrostatic - Matrix")
{
    GM::Tensor2 A = xt::random::randn<double>({2, 2});
    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    auto M = xt::xtensor<double,4>::from_shape({3, 4, 2, 2});
    auto R = xt::xtensor<double,2>::from_shape({M.shape(0), M.shape(1)});
    for (size_t i = 0; i < M.shape(0); ++i) {
        for (size_t j = 0; j < M.shape(1); ++j) {
            xt::view(M, i, j, xt::all(), xt::all()) = static_cast<double>(i * M.shape(1) + j) * A;
            R(i, j) = static_cast<double>(i * M.shape(1) + j);
        }
    }
    REQUIRE(xt::allclose(GM::Hydrostatic(M), R));
}

SECTION("Epsd - Tensor2")
{
    GM::Tensor2 A = xt::zeros<double>({2, 2});
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    REQUIRE(GM::Epsd(A)() == Approx(1.0));
}

SECTION("Epsd - List")
{
    GM::Tensor2 A = xt::zeros<double>({2, 2});
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    auto M = xt::xtensor<double,3>::from_shape({3, 2, 2});
    auto R = xt::xtensor<double,1>::from_shape({M.shape(0)});
    for (size_t i = 0; i < M.shape(0); ++i) {
        xt::view(M, i, xt::all(), xt::all()) = static_cast<double>(i) * A;
        R(i) = static_cast<double>(i);
    }
    REQUIRE(xt::allclose(GM::Epsd(M), R));
}

SECTION("Epsd - Matrix")
{
    GM::Tensor2 A = xt::zeros<double>({2, 2});
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    auto M = xt::xtensor<double,4>::from_shape({3, 4, 2, 2});
    auto R = xt::xtensor<double,2>::from_shape({M.shape(0), M.shape(1)});
    for (size_t i = 0; i < M.shape(0); ++i) {
        for (size_t j = 0; j < M.shape(1); ++j) {
            xt::view(M, i, j, xt::all(), xt::all()) = static_cast<double>(i * M.shape(1) + j) * A;
            R(i, j) = static_cast<double>(i * M.shape(1) + j);
        }
    }
    REQUIRE(xt::allclose(GM::Epsd(M), R));
}

SECTION("Sigd - Tensor2")
{
    GM::Tensor2 A = xt::zeros<double>({2, 2});
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    REQUIRE(GM::Sigd(A)() == Approx(2.0));
}

SECTION("Sigd - List")
{
    GM::Tensor2 A = xt::zeros<double>({2, 2});
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    auto M = xt::xtensor<double,3>::from_shape({3, 2, 2});
    auto R = xt::xtensor<double,1>::from_shape({M.shape(0)});
    for (size_t i = 0; i < M.shape(0); ++i) {
        xt::view(M, i, xt::all(), xt::all()) = static_cast<double>(i) * A;
        R(i) = static_cast<double>(i) * 2.0;
    }
    REQUIRE(xt::allclose(GM::Sigd(M), R));
}

SECTION("Sigd - Matrix")
{
    GM::Tensor2 A = xt::zeros<double>({2, 2});
    A(0, 1) = 1.0;
    A(1, 0) = 1.0;
    auto M = xt::xtensor<double,4>::from_shape({3, 4, 2, 2});
    auto R = xt::xtensor<double,2>::from_shape({M.shape(0), M.shape(1)});
    for (size_t i = 0; i < M.shape(0); ++i) {
        for (size_t j = 0; j < M.shape(1); ++j) {
            xt::view(M, i, j, xt::all(), xt::all()) = static_cast<double>(i * M.shape(1) + j) * A;
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
    auto Eps = GM::Tensor2::from_shape({2, 2});
    Eps(0, 0) = Eps(1, 1) = epsm;
    Eps(0, 1) = Eps(1, 0) = gamma;

    GM::Elastic mat(K, G);
    mat.setStrain(Eps);
    auto Sig = mat.Stress();

    REQUIRE(Sig(0, 0) == Approx(K * epsm));
    REQUIRE(Sig(1, 1) == Approx(K * epsm));
    REQUIRE(Sig(0, 1) == Approx(G * gamma));
    REQUIRE(Sig(1, 0) == Approx(G * gamma));
    REQUIRE(mat.energy() == Approx(K * std::pow(epsm, 2.0) + G * std::pow(gamma, 2.0)));
}

SECTION("Cusp - stress")
{
    double K = 12.3;
    double G = 45.6;
    double gamma = 0.02;
    double epsm = 0.12;
    auto Eps = GM::Tensor2::from_shape({2, 2});
    Eps(0, 0) = Eps(1, 1) = epsm;
    Eps(0, 1) = Eps(1, 0) = gamma;

    GM::Cusp mat(K, G, {0.01, 0.03, 0.05, 0.10});
    mat.setStrain(Eps);
    auto Sig = mat.Stress();

    REQUIRE(Sig(0, 0) == Approx(K * epsm));
    REQUIRE(Sig(1, 1) == Approx(K * epsm));
    REQUIRE(Sig(0, 1) == Approx(0.0));
    REQUIRE(Sig(1, 0) == Approx(0.0));
    REQUIRE(mat.epsp() == Approx(0.02));
    REQUIRE(mat.currentIndex() == 1);
    REQUIRE(mat.checkYieldBoundLeft());
    REQUIRE(mat.checkYieldBoundRight());
    REQUIRE(mat.energy() == Approx(K * std::pow(epsm, 2.0) + G * (0.0 - std::pow(0.01, 2.0))));

    epsm *= 2.0;
    gamma *= 1.9;

    Eps(0, 0) = Eps(1, 1) = epsm;
    Eps(0, 1) = Eps(1, 0) = gamma;

    mat.setStrain(Eps);
    Sig = mat.Stress();

    REQUIRE(Sig(0, 0) == Approx(K * epsm));
    REQUIRE(Sig(1, 1) == Approx(K * epsm));
    REQUIRE(Sig(0, 1) == Approx(G * (gamma - 0.04)));
    REQUIRE(Sig(1, 0) == Approx(G * (gamma - 0.04)));
    REQUIRE(mat.epsp() == Approx(0.04));
    REQUIRE(mat.currentIndex() == 2);
    REQUIRE(mat.checkYieldBoundLeft());
    REQUIRE(mat.checkYieldBoundRight());
    REQUIRE(mat.energy() == Approx(K * std::pow(epsm, 2.0) + G * (std::pow(gamma - 0.04, 2.0) - std::pow(0.01, 2.0))));
}

SECTION("Smooth - stress")
{
    double K = 12.3;
    double G = 45.6;
    double gamma = 0.02;
    double epsm = 0.12;
    auto Eps = GM::Tensor2::from_shape({2, 2});
    Eps(0, 0) = Eps(1, 1) = epsm;
    Eps(0, 1) = Eps(1, 0) = gamma;

    GM::Smooth mat(K, G, {0.01, 0.03, 0.05, 0.10});
    mat.setStrain(Eps);
    auto Sig = mat.Stress();

    REQUIRE(Sig(0, 0) == Approx(K * epsm));
    REQUIRE(Sig(1, 1) == Approx(K * epsm));
    REQUIRE(Sig(0, 1) == Approx(0.0));
    REQUIRE(Sig(1, 0) == Approx(0.0));
    REQUIRE(mat.epsp() == Approx(0.02));
    REQUIRE(mat.currentIndex() == 1);
    REQUIRE(mat.checkYieldBoundLeft());
    REQUIRE(mat.checkYieldBoundRight());
}

SECTION("Tangent (purely elastic response only)")
{
    double K = 12.3;
    double G = 45.6;

    GM::Tensor2 Eps = xt::random::randn<double>({2, 2});
    GM::Tensor4 Is = GM::I4s();
    Eps = A4_ddot_B2(Is, Eps);

    // Elastic
    {
        GM::Elastic mat(K, G);
        mat.setStrain(Eps);
        auto Sig = mat.Stress();
        auto C = mat.Tangent();
        REQUIRE(xt::allclose(A4_ddot_B2(C, Eps), Sig));
    }

    // Cusp
    {
        GM::Cusp mat(K, G, {10000.0});
        mat.setStrain(Eps);
        auto Sig = mat.Stress();
        auto C = mat.Tangent();
        REQUIRE(xt::allclose(A4_ddot_B2(C, Eps), Sig));
    }

    // Smooth
    {
        GM::Smooth mat(K, G, {10000.0});
        mat.setStrain(Eps);
        auto Sig = mat.Stress();
        auto C = mat.Tangent();
        REQUIRE(xt::allclose(A4_ddot_B2(C, Eps), Sig));
    }
}

SECTION("Array")
{
    double K = 12.3;
    double G = 45.6;
    double gamma = 0.02;
    double epsm = 0.12;
    size_t nelem = 3;
    size_t nip = 2;
    auto Eps = GM::Tensor2::from_shape({2, 2});
    Eps(0, 0) = Eps(1, 1) = epsm;
    Eps(0, 1) = Eps(1, 0) = gamma;

    GM::Array<2> mat({nelem, nip});

    {
        xt::xtensor<size_t,2> I = xt::zeros<size_t>({nelem, nip});
        xt::view(I, 0, xt::all()) = 1;
        mat.setElastic(I, K, G);
    }

    {
        xt::xtensor<size_t,2> I = xt::zeros<size_t>({nelem, nip});
        xt::xtensor<double,1> epsy = 0.01 + 0.02 * xt::arange<double>(100);
        xt::view(I, 1, xt::all()) = 1;
        mat.setCusp(I, K, G, epsy);
    }

    {
        xt::xtensor<size_t,2> I = xt::zeros<size_t>({nelem, nip});
        xt::xtensor<double,1> epsy = 0.01 + 0.02 * xt::arange<double>(100);
        xt::view(I, 2, xt::all()) = 1;
        mat.setCusp(I, K, G, epsy);
    }

    auto eps = xt::xtensor<double,4>::from_shape({nelem, nip, 2ul, 2ul});

    for (size_t e = 0; e < nelem; ++e) {
        for (size_t q = 0; q < nip; ++q) {
            double fac = static_cast<double>((e + 1) * nip + (q + 1));
            xt::view(eps, e, q) = fac * Eps;
        }
    }

    mat.setStrain(eps);
    auto sig = mat.Stress();
    auto epsp = mat.Epsp();

    for (size_t q = 0; q < nip; ++q) {

        size_t e = 0;
        double fac = static_cast<double>((e + 1) * nip + (q + 1));
        ISCLOSE(sig(e, q, 0, 0), fac * K * epsm);
        ISCLOSE(sig(e, q, 1, 1), fac * K * epsm);
        ISCLOSE(sig(e, q, 0, 1), fac * G * gamma);
        ISCLOSE(sig(e, q, 1, 0), fac * G * gamma);
        ISCLOSE(epsp(e, q), 0.0);

        e = 1;
        fac = static_cast<double>((e + 1) * nip + (q + 1));
        ISCLOSE(sig(e, q, 0, 0), fac * K * epsm);
        ISCLOSE(sig(e, q, 1, 1), fac * K * epsm);
        ISCLOSE(sig(e, q, 0, 1), 0.0);
        ISCLOSE(sig(e, q, 1, 0), 0.0);
        ISCLOSE(epsp(e, q), fac * gamma);

        e = 2;
        fac = static_cast<double>((e + 1) * nip + (q + 1));
        ISCLOSE(sig(e, q, 0, 0), fac * K * epsm);
        ISCLOSE(sig(e, q, 1, 1), fac * K * epsm);
        ISCLOSE(sig(e, q, 0, 1), 0.0);
        ISCLOSE(sig(e, q, 1, 0), 0.0);
        ISCLOSE(epsp(e, q), fac * gamma);
    }

    REQUIRE(mat.checkYieldBoundLeft());
    REQUIRE(mat.checkYieldBoundRight());
}

}
