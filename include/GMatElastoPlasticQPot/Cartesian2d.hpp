/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatElastoPlasticQPot

*/

#ifndef GMATELASTOPLASTICQPOT_CARTESIAN2D_HPP
#define GMATELASTOPLASTICQPOT_CARTESIAN2D_HPP

#include "Cartesian2d.h"

namespace GMatElastoPlasticQPot {
namespace Cartesian2d {

template <class T, class U>
inline void epsd(const T& A, U& B)
{
    GMatTensor::Cartesian2d::norm_deviatoric(A, B);
    B *= std::sqrt(0.5);
}

template <class T>
inline auto Epsd(const T& A)
{
    return xt::eval(std::sqrt(0.5) * GMatTensor::Cartesian2d::Norm_deviatoric(A));
}

template <class T, class U>
inline void sigd(const T& A, U& B)
{
    GMatTensor::Cartesian2d::norm_deviatoric(A, B);
    B *= std::sqrt(2.0);
}

template <class T>
inline auto Sigd(const T& A)
{
    return xt::eval(std::sqrt(2.0) * GMatTensor::Cartesian2d::Norm_deviatoric(A));
}

} // namespace Cartesian2d
} // namespace GMatElastoPlasticQPot

#endif
