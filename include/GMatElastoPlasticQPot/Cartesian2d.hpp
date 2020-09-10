/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatElastoPlasticQPot

*/

#ifndef GMATELASTOPLASTICQPOT_CARTESIAN2D_HPP
#define GMATELASTOPLASTICQPOT_CARTESIAN2D_HPP

#include "Cartesian2d.h"

namespace GMatElastoPlasticQPot {
namespace Cartesian2d {

inline Tensor2 I2()
{
    return Tensor2({{1.0, 0.0},
                    {0.0, 1.0}});
}

inline Tensor4 II()
{
    Tensor4 out = Tensor4::from_shape({2, 2, 2, 2});
    out.fill(0.0);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                for (size_t l = 0; l < 2; ++l) {
                    if (i == j && k == l) {
                        out(i, j, k, l) = 1.0;
                    }
                }
            }
        }
    }

    return out;
}

inline Tensor4 I4()
{
    Tensor4 out = Tensor4::from_shape({2, 2, 2, 2});
    out.fill(0.0);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                for (size_t l = 0; l < 2; ++l) {
                    if (i == l && j == k) {
                        out(i, j, k, l) = 1.0;
                    }
                }
            }
        }
    }

    return out;
}

inline Tensor4 I4rt()
{
    Tensor4 out = Tensor4::from_shape({2, 2, 2, 2});
    out.fill(0.0);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                for (size_t l = 0; l < 2; ++l) {
                    if (i == k && j == l) {
                        out(i, j, k, l) = 1.0;
                    }
                }
            }
        }
    }

    return out;
}

inline Tensor4 I4s()
{
    return 0.5 * (I4() + I4rt());
}

inline Tensor4 I4d()
{
    return I4s() - 0.5 * II();
}

inline double Hydrostatic(const Tensor2& A)
{
    return 0.5 * trace(A);
}

inline Tensor2 Deviatoric(const Tensor2& A)
{
    return A - 0.5 * trace(A) * I2();
}

inline double Epsd(const Tensor2& Eps)
{
    Tensor2 Epsd = Eps - 0.5 * trace(Eps) * I2();
    return std::sqrt(0.5 * A2_ddot_B2(Epsd, Epsd));
}

inline double Sigd(const Tensor2& Sig)
{
    Tensor2 Sigd = Sig - 0.5 * trace(Sig) * I2();
    return std::sqrt(2.0 * A2_ddot_B2(Sigd, Sigd));
}

inline void hydrostatic(const xt::xtensor<double,3>& A, xt::xtensor<double,1>& Am)
{
    GMATELASTOPLASTICQPOT_ASSERT(
        A.shape() == std::decay_t<decltype(A)>::shape_type({Am.shape(0), 2, 2}));

    #pragma omp parallel
    {
        #pragma omp for
        for (size_t e = 0; e < A.shape(0); ++e) {
            auto Ai = xt::adapt(&A(e, 0, 0), xt::xshape<2, 2>());
            Am(e) = 0.5 * trace(Ai);
        }
    }
}

inline void deviatoric(const xt::xtensor<double,3>& A, xt::xtensor<double,3>& Ad)
{
    GMATELASTOPLASTICQPOT_ASSERT(
        A.shape() == std::decay_t<decltype(A)>::shape_type({Ad.shape(0), 2, 2}));

    #pragma omp parallel
    {
        Tensor2 I = I2();

        #pragma omp for
        for (size_t e = 0; e < A.shape(0); ++e) {
            auto Ai = xt::adapt(&A(e, 0, 0), xt::xshape<2, 2>());
            auto Aid = xt::adapt(&Ad(e, 0, 0), xt::xshape<2, 2>());
            xt::noalias(Aid) = Ai - 0.5 * trace(Ai) * I;
        }
    }
}

inline void epsd(const xt::xtensor<double,3>& A, xt::xtensor<double,1>& Aeq)
{
    GMATELASTOPLASTICQPOT_ASSERT(
        A.shape() == std::decay_t<decltype(A)>::shape_type({Aeq.shape(0), 2, 2}));

    #pragma omp parallel
    {
        Tensor2 I = I2();

        #pragma omp for
        for (size_t e = 0; e < A.shape(0); ++e) {
            auto Ai = xt::adapt(&A(e, 0, 0), xt::xshape<2, 2>());
            auto Aid = Ai - 0.5 * trace(Ai) * I;
            Aeq(e) = std::sqrt(0.5 * A2_ddot_B2(Aid, Aid));
        }
    }
}

inline void sigd(const xt::xtensor<double,3>& A, xt::xtensor<double,1>& Aeq)
{
    GMATELASTOPLASTICQPOT_ASSERT(
        A.shape() == std::decay_t<decltype(A)>::shape_type({Aeq.shape(0), 2, 2}));

    #pragma omp parallel
    {
        Tensor2 I = I2();

        #pragma omp for
        for (size_t e = 0; e < A.shape(0); ++e) {
            auto Ai = xt::adapt(&A(e, 0, 0), xt::xshape<2, 2>());
            auto Aid = Ai - 0.5 * trace(Ai) * I;
            Aeq(e) = std::sqrt(2.0 * A2_ddot_B2(Aid, Aid));
        }
    }
}

inline xt::xtensor<double,1> Hydrostatic(const xt::xtensor<double,3>& A)
{
    xt::xtensor<double,1> Am = xt::empty<double>({A.shape(0)});
    hydrostatic(A, Am);
    return Am;
}

inline xt::xtensor<double,3> Deviatoric(const xt::xtensor<double,3>& A)
{
    xt::xtensor<double,3> Ad = xt::empty<double>(A.shape());
    deviatoric(A, Ad);
    return Ad;
}

inline xt::xtensor<double,1> Epsd(const xt::xtensor<double,3>& A)
{
    xt::xtensor<double,1> Aeq = xt::empty<double>({A.shape(0)});
    epsd(A, Aeq);
    return Aeq;
}

inline xt::xtensor<double,1> Sigd(const xt::xtensor<double,3>& A)
{
    xt::xtensor<double,1> Aeq = xt::empty<double>({A.shape(0)});
    sigd(A, Aeq);
    return Aeq;
}

inline void hydrostatic(const xt::xtensor<double,4>& A, xt::xtensor<double,2>& Am)
{
    GMATELASTOPLASTICQPOT_ASSERT(
        A.shape() == std::decay_t<decltype(A)>::shape_type({Am.shape(0), Am.shape(1), 2, 2}));

    #pragma omp parallel
    {
        #pragma omp for
        for (size_t e = 0; e < A.shape(0); ++e) {
            for (size_t q = 0; q < A.shape(1); ++q) {
                auto Ai = xt::adapt(&A(e, q, 0, 0), xt::xshape<2, 2>());
                Am(e, q) = 0.5 * trace(Ai);
            }
        }
    }
}

inline void deviatoric(const xt::xtensor<double,4>& A, xt::xtensor<double,4>& Ad)
{
    GMATELASTOPLASTICQPOT_ASSERT(
        A.shape() == std::decay_t<decltype(A)>::shape_type({Ad.shape(0), Ad.shape(1), 2, 2}));

    #pragma omp parallel
    {
        Tensor2 I = I2();

        #pragma omp for
        for (size_t e = 0; e < A.shape(0); ++e) {
            for (size_t q = 0; q < A.shape(1); ++q) {
                auto Ai = xt::adapt(&A(e, q, 0, 0), xt::xshape<2, 2>());
                auto Aid = xt::adapt(&Ad(e, q, 0, 0), xt::xshape<2, 2>());
                xt::noalias(Aid) = Ai - 0.5 * trace(Ai) * I;
            }
        }
    }
}

inline void epsd(const xt::xtensor<double,4>& A, xt::xtensor<double,2>& Aeq)
{
    GMATELASTOPLASTICQPOT_ASSERT(
        A.shape() == std::decay_t<decltype(A)>::shape_type({Aeq.shape(0), Aeq.shape(1), 2, 2}));

    #pragma omp parallel
    {
        Tensor2 I = I2();

        #pragma omp for
        for (size_t e = 0; e < A.shape(0); ++e) {
            for (size_t q = 0; q < A.shape(1); ++q) {
                auto Ai = xt::adapt(&A(e, q, 0, 0), xt::xshape<2, 2>());
                auto Aid = Ai - 0.5 * trace(Ai) * I;
                Aeq(e, q) = std::sqrt(0.5 * A2_ddot_B2(Aid, Aid));
            }
        }
    }
}

inline void sigd(const xt::xtensor<double,4>& A, xt::xtensor<double,2>& Aeq)
{
    GMATELASTOPLASTICQPOT_ASSERT(
        A.shape() == std::decay_t<decltype(A)>::shape_type({Aeq.shape(0), Aeq.shape(1), 2, 2}));

    #pragma omp parallel
    {
        Tensor2 I = I2();

        #pragma omp for
        for (size_t e = 0; e < A.shape(0); ++e) {
            for (size_t q = 0; q < A.shape(1); ++q) {
                auto Ai = xt::adapt(&A(e, q, 0, 0), xt::xshape<2, 2>());
                auto Aid = Ai - 0.5 * trace(Ai) * I;
                Aeq(e, q) = std::sqrt(2.0 * A2_ddot_B2(Aid, Aid));
            }
        }
    }
}

inline xt::xtensor<double,2> Hydrostatic(const xt::xtensor<double,4>& A)
{
    xt::xtensor<double,2> Am = xt::empty<double>({A.shape(0), A.shape(1)});
    hydrostatic(A, Am);
    return Am;
}

inline xt::xtensor<double,4> Deviatoric(const xt::xtensor<double,4>& A)
{
    xt::xtensor<double,4> Ad = xt::empty<double>(A.shape());
    deviatoric(A, Ad);
    return Ad;
}

inline xt::xtensor<double,2> Epsd(const xt::xtensor<double,4>& A)
{
    xt::xtensor<double,2> Aeq = xt::empty<double>({A.shape(0), A.shape(1)});
    epsd(A, Aeq);
    return Aeq;
}

inline xt::xtensor<double,2> Sigd(const xt::xtensor<double,4>& A)
{
    xt::xtensor<double,2> Aeq = xt::empty<double>({A.shape(0), A.shape(1)});
    sigd(A, Aeq);
    return Aeq;
}

template <class U>
inline double trace(const U& A)
{
    return A(0,0) + A(1,1);
}

template <class U, class V>
inline double A2_ddot_B2(const U& A, const V& B)
{
    return A(0,0) * B(0,0) + 2.0 * A(0,1) * B(0,1) + A(1,1) * B(1,1);
}

} // namespace Cartesian2d
} // namespace GMatElastoPlasticQPot

#endif
