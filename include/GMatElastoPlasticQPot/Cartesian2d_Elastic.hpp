/**
Partial implementation of GMatElastoPlasticQPot/Cartesian2d.h

\file GMatElastoPlasticQPot/Cartesian2d_Elastic.hpp
\copyright Copyright 2018. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
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

inline double Elastic::energy() const
{
    namespace GT = GMatTensor::Cartesian2d::pointer;
    std::array<double, 4> Epsd;
    double epsm = GT::Hydrostatic_deviatoric(&m_Eps[0], &Epsd[0]);
    double epsd = std::sqrt(0.5 * GT::A2s_ddot_B2s(&Epsd[0], &Epsd[0]));
    double U = m_K * std::pow(epsm, 2.0);
    double V = m_G * std::pow(epsd, 2.0);
    return U + V;
}

template <class T>
inline void Elastic::setStrainPtr(const T* arg)
{
    namespace GT = GMatTensor::Cartesian2d::pointer;
    std::copy(arg, arg + 4, m_Eps.begin());

    double epsm = GT::Hydrostatic(&m_Eps[0]);

    m_Sig[0] = (m_K - m_G) * epsm + m_G * m_Eps[0];
    m_Sig[1] = m_G * m_Eps[1];
    m_Sig[2] = m_G * m_Eps[2];
    m_Sig[3] = (m_K - m_G) * epsm + m_G * m_Eps[3];
}

template <class T>
inline void Elastic::strainPtr(T* ret) const
{
    std::copy(m_Eps.begin(), m_Eps.end(), ret);
}

template <class T>
inline void Elastic::stressPtr(T* ret) const
{
    std::copy(m_Sig.begin(), m_Sig.end(), ret);
}

template <class T>
inline void Elastic::tangentPtr(T* ret) const
{
    auto II = Cartesian2d::II();
    auto I4d = Cartesian2d::I4d();
    auto C = 0.5 * m_K * II + m_G * I4d;
    std::copy(C.cbegin(), C.cend(), ret);
}

template <class T>
inline void Elastic::setStrain(const T& arg)
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(arg, {2, 2}));
    return this->setStrainPtr(arg.data());
}

template <class T>
inline void Elastic::strain(T& ret) const
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(ret, {2, 2}));
    return this->strainPtr(ret.data());
}

template <class T>
inline void Elastic::stress(T& ret) const
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(ret, {2, 2}));
    return this->stressPtr(ret.data());
}

template <class T>
inline void Elastic::tangent(T& ret) const
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(ret, {2, 2, 2, 2}));
    return this->tangentPtr(ret.data());
}

inline xt::xtensor<double, 2> Elastic::Strain() const
{
    xt::xtensor<double, 2> ret = xt::empty<double>({2, 2});
    this->strainPtr(ret.data());
    return ret;
}

inline xt::xtensor<double, 2> Elastic::Stress() const
{
    xt::xtensor<double, 2> ret = xt::empty<double>({2, 2});
    this->stressPtr(ret.data());
    return ret;
}

inline xt::xtensor<double, 4> Elastic::Tangent() const
{
    xt::xtensor<double, 4> ret = xt::empty<double>({2, 2, 2, 2});
    this->tangentPtr(ret.data());
    return ret;
}

} // namespace Cartesian2d
} // namespace GMatElastoPlasticQPot

#endif
