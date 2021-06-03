/**
Partial implementation of GMatElastoPlasticQPot/Cartesian2d.h

\file
\copyright Copyright 2018. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
*/

#ifndef GMATELASTOPLASTICQPOT_CARTESIAN2D_CUSP_HPP
#define GMATELASTOPLASTICQPOT_CARTESIAN2D_CUSP_HPP

#include "Cartesian2d.h"

namespace GMatElastoPlasticQPot {
namespace Cartesian2d {

template <class Y>
inline Cusp::Cusp(double K, double G, const Y& epsy, bool init_elastic) : m_K(K), m_G(G)
{
    GMATELASTOPLASTICQPOT_ASSERT(epsy.dimension() == 1);
    GMATELASTOPLASTICQPOT_ASSERT(epsy.size() > 0);
    // assertion on epsy being sorted is done by QPot::Chunked

    if (!init_elastic) {
        m_yield = QPot::Chunked(0.0, epsy, 0);
        return;
    }

    GMATELASTOPLASTICQPOT_ASSERT(epsy[0] > 0);
    std::vector<double> y(epsy.size() + 1);
    y[0] = -epsy[0];
    std::copy(epsy.cbegin(), epsy.cend(), y.begin() + 1);
    m_yield = QPot::Chunked(0.0, y, 0);
}

inline double Cusp::K() const
{
    return m_K;
}

inline double Cusp::G() const
{
    return m_G;
}

inline xt::xtensor<double, 1> Cusp::epsy() const
{
    return xt::adapt(m_yield.y());
}

inline QPot::Chunked& Cusp::refQPotChunked()
{
    return m_yield;
}

inline auto Cusp::currentIndex() const
{
    return m_yield.i();
}

inline auto Cusp::currentYieldLeft() const
{
    return m_yield.yleft();
}

inline auto Cusp::currentYieldRight() const
{
    return m_yield.yright();
}

inline auto Cusp::currentYieldLeft(size_t offset) const
{
    return m_yield.yleft(offset);
}

inline auto Cusp::currentYieldRight(size_t offset) const
{
    return m_yield.yright(offset);
}

inline double Cusp::epsp() const
{
    return 0.5 * (m_yield.yleft() + m_yield.yright());
}

inline double Cusp::energy() const
{
    namespace GT = GMatTensor::Cartesian2d::pointer;
    std::array<double, 4> Epsd;
    double epsm = GT::Hydrostatic_deviatoric(&m_Eps[0], &Epsd[0]);
    double epsd = std::sqrt(0.5 * GT::A2s_ddot_B2s(&Epsd[0], &Epsd[0]));
    double U = m_K * std::pow(epsm, 2.0);

    double eps_min = 0.5 * (m_yield.yright() + m_yield.yleft());
    double deps_y = 0.5 * (m_yield.yright() - m_yield.yleft());

    double V = m_G * (std::pow(epsd - eps_min, 2.0) - std::pow(deps_y, 2.0));

    return U + V;
}

inline bool Cusp::checkYieldBoundLeft(size_t n) const
{
    return m_yield.boundcheck_left(n);
}

inline bool Cusp::checkYieldBoundRight(size_t n) const
{
    return m_yield.boundcheck_right(n);
}

template <class T>
inline void Cusp::setStrainPtr(const T* arg)
{
    namespace GT = GMatTensor::Cartesian2d::pointer;
    std::copy(arg, arg + 4, m_Eps.begin());

    std::array<double, 4> Epsd;
    double epsm = GT::Hydrostatic_deviatoric(&m_Eps[0], &Epsd[0]);
    double epsd = std::sqrt(0.5 * GT::A2s_ddot_B2s(&Epsd[0], &Epsd[0]));
    m_yield.set_x(epsd);

    m_Sig[0] = m_Sig[3] = m_K * epsm;

    if (epsd <= 0.0) {
        m_Sig[1] = m_Sig[2] = 0.0;
        return;
    }

    double eps_min = 0.5 * (m_yield.yright() + m_yield.yleft());

    double g = m_G * (1.0 - eps_min / epsd);
    m_Sig[0] += g * Epsd[0];
    m_Sig[1] = g * Epsd[1];
    m_Sig[2] = g * Epsd[2];
    m_Sig[3] += g * Epsd[3];
}

template <class T>
inline void Cusp::strainPtr(T* ret) const
{
    std::copy(m_Eps.begin(), m_Eps.end(), ret);
}

template <class T>
inline void Cusp::stressPtr(T* ret) const
{
    std::copy(m_Sig.begin(), m_Sig.end(), ret);
}

template <class T>
inline void Cusp::tangentPtr(T* ret) const
{
    auto II = Cartesian2d::II();
    auto I4d = Cartesian2d::I4d();
    auto C = 0.5 * m_K * II + m_G * I4d;
    std::copy(C.cbegin(), C.cend(), ret);
}

template <class T>
inline void Cusp::setStrain(const T& arg)
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(arg, {2, 2}));
    return this->setStrainPtr(arg.data());
}

template <class T>
inline void Cusp::strain(T& ret) const
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(ret, {2, 2}));
    return this->strainPtr(ret.data());
}

template <class T>
inline void Cusp::stress(T& ret) const
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(ret, {2, 2}));
    return this->stressPtr(ret.data());
}

template <class T>
inline void Cusp::tangent(T& ret) const
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(ret, {2, 2, 2, 2}));
    return this->tangentPtr(ret.data());
}

inline xt::xtensor<double, 2> Cusp::Strain() const
{
    xt::xtensor<double, 2> ret = xt::empty<double>({2, 2});
    this->strainPtr(ret.data());
    return ret;
}

inline xt::xtensor<double, 2> Cusp::Stress() const
{
    xt::xtensor<double, 2> ret = xt::empty<double>({2, 2});
    this->stressPtr(ret.data());
    return ret;
}

inline xt::xtensor<double, 4> Cusp::Tangent() const
{
    xt::xtensor<double, 4> ret = xt::empty<double>({2, 2, 2, 2});
    this->tangentPtr(ret.data());
    return ret;
}

} // namespace Cartesian2d
} // namespace GMatElastoPlasticQPot

#endif
