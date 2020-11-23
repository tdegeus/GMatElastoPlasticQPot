/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatElastoPlasticQPot

*/

#ifndef GMATELASTOPLASTICQPOT_CARTESIAN2D_HPP
#define GMATELASTOPLASTICQPOT_CARTESIAN2D_HPP

#include "Cartesian2d.h"

namespace GMatElastoPlasticQPot {
namespace Cartesian2d {

namespace detail {
namespace xtensor {

    template <class T>
    inline auto trace(const T& A)
    {
        return A(0, 0) + A(1, 1);
    }

    template <class T, class U>
    inline auto A2_ddot_B2(const T& A, const U& B)
    {
        return A(0, 0) * B(0, 0) + 2.0 * A(0, 1) * B(0, 1) + A(1, 1) * B(1, 1);
    }

} // namespace xtensor
} // namespace detail

namespace detail {
namespace pointer {

    template <class T>
    inline auto trace(const T A)
    {
        return A[0] + A[3];
    }

    template <class T, class U>
    inline void deviatoric(const T A, U ret)
    {
        auto m = 0.5 * (A[0] + A[3]);
        ret[0] = A[0] - m;
        ret[1] = A[1];
        ret[2] = A[2];
        ret[3] = A[3] - m;
    }

    template <class T>
    inline auto deviatoric_ddot_deviatoric(const T A)
    {
        auto m = 0.5 * (A[0] + A[3]);
        return (A[0] - m) * (A[0] - m) + 2.0 * A[1] * A[1] + (A[3] - m) * (A[3] - m);
    }

} // namespace pointer
} // namespace detail

namespace detail {

    template <class T>
    inline T trace(const std::array<T, 4>& A)
    {
        return A[0] + A[3];
    }

    template <class T>
    inline T hydrostatic_deviator(const std::array<T, 4>& A, std::array<T, 4>& ret)
    {
        T m = 0.5 * (A[0] + A[3]);
        ret[0] = A[0] - m;
        ret[1] = A[1];
        ret[2] = A[2];
        ret[3] = A[3] - m;
        return m;
    }

    template <class T>
    inline T A2_ddot_B2(const std::array<T, 4>& A, const std::array<T, 4>& B)
    {
        return A[0] * B[0] + 2.0 * A[1] * B[1] + A[3] * B[3];
    }

} // namespace detail

template <class T, class U>
inline void epsd(const T& A, U& B)
{
    GMatTensor::Cartesian2d::equivalent_deviatoric(A, B);
    B *= std::sqrt(0.5);
}

template <class T>
inline auto Epsd(const T& A)
{
    return xt::eval(std::sqrt(0.5) * GMatTensor::Cartesian2d::Equivalent_deviatoric(A));
}

template <class T, class U>
inline void sigd(const T& A, U& B)
{
    GMatTensor::Cartesian2d::equivalent_deviatoric(A, B);
    B *= std::sqrt(2.0);
}

template <class T>
inline auto Sigd(const T& A)
{
    return xt::eval(std::sqrt(2.0) * GMatTensor::Cartesian2d::Equivalent_deviatoric(A));
}

} // namespace Cartesian2d
} // namespace GMatElastoPlasticQPot

#endif
