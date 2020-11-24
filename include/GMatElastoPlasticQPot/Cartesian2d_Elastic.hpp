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
inline void Elastic::setStrain(const T& a)
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(a, {2, 2}));
    return this->setStrainIterator(a.cbegin());
}

template <class T>
inline void Elastic::setStrainIterator(const T& begin)
{
    std::copy(begin, begin + 4, m_Eps.begin());

    double epsm = 0.5 * GMatTensor::Cartesian2d::pointer::trace(&m_Eps[0]);

    m_Sig[0] = (m_K - m_G) * epsm + m_G * m_Eps[0];
    m_Sig[1] = m_G * m_Eps[1];
    m_Sig[2] = m_G * m_Eps[2];
    m_Sig[3] = (m_K - m_G) * epsm + m_G * m_Eps[3];
}

template <class T>
inline void Elastic::stress(T& a) const
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(a, {2, 2}));
    return this->stressIterator(a.begin());
}

template <class T>
inline void Elastic::stressIterator(const T& begin) const
{
    std::copy(m_Sig.begin(), m_Sig.end(), begin);
}

inline xt::xtensor<double, 2> Elastic::Stress() const
{
    xt::xtensor<double, 2> ret = xt::empty<double>({2, 2});
    this->stressIterator(ret.begin());
    return ret;
}

template <class T>
inline void Elastic::tangent(T& C) const
{
    auto II = Cartesian2d::II();
    auto I4d = Cartesian2d::I4d();
    xt::noalias(C) = 0.5 * m_K * II + m_G * I4d;
}

inline xt::xtensor<double, 4> Elastic::Tangent() const
{
    xt::xtensor<double, 4> ret = xt::zeros<double>({2, 2, 2, 2});
    this->tangent(ret);
    return ret;
}

inline double Elastic::energy() const
{
    std::array<double, 4> Epsd;
    double epsm = GMatTensor::Cartesian2d::pointer::hydrostatic_deviatoric(&m_Eps[0], &Epsd[0]);
    double epsd = std::sqrt(0.5 * GMatTensor::Cartesian2d::pointer::A2_ddot_B2(&Epsd[0], &Epsd[0]));
    double U = m_K * std::pow(epsm, 2.0);
    double V = m_G * std::pow(epsd, 2.0);
    return U + V;
}

} // namespace Cartesian2d
} // namespace GMatElastoPlasticQPot

#endif
