/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatElastoPlasticQPot

*/

#ifndef GMATELASTOPLASTICQPOT_CARTESIAN2D_ELASTIC_HPP
#define GMATELASTOPLASTICQPOT_CARTESIAN2D_ELASTIC_HPP

#include "Cartesian2d.h"

namespace GMatElastoPlasticQPot {
namespace Cartesian2d {

inline Elastic::Elastic(double K, double G) : m_K(K), m_G(G)
{
}

inline double Elastic::K() const
{
    return m_K;
}

inline double Elastic::G() const
{
    return m_G;
}

template <class T>
inline void Elastic::stress(const Tensor2& Eps, T&& Sig) const
{
    auto I = Cartesian2d::I2();
    auto epsm = 0.5 * trace(Eps);
    auto Epsd = Eps - epsm * I;
    xt::noalias(Sig) = m_K * epsm * I + m_G * Epsd;
}

inline Tensor2 Elastic::Stress(const Tensor2& Eps) const
{
    Tensor2 Sig;
    this->stress(Eps, Sig);
    return Sig;
}

template <class T, class S>
inline void Elastic::tangent(const Tensor2& Eps, T&& Sig, S&& C) const
{
    auto II = Cartesian2d::II();
    auto I4d = Cartesian2d::I4d();
    this->stress(Eps, Sig);
    xt::noalias(C) = 0.5 * m_K * II + m_G * I4d;
}

inline std::tuple<Tensor2, Tensor4> Elastic::Tangent(const Tensor2& Eps) const
{
    Tensor2 Sig;
    Tensor4 C;
    this->tangent(Eps, Sig, C);
    return std::make_tuple(Sig, C);
}

inline double Elastic::energy(const Tensor2& Eps) const
{
    auto I = Cartesian2d::I2();
    auto epsm = 0.5 * trace(Eps);
    auto Epsd = Eps - epsm * I;
    auto epsd = std::sqrt(0.5 * A2_ddot_B2(Epsd, Epsd));
    auto U = m_K * std::pow(epsm, 2.0);
    auto V = m_G * std::pow(epsd, 2.0);
    return U + V;
}

} // namespace Cartesian2d
} // namespace GMatElastoPlasticQPot

#endif
