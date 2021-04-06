/**
Partial implementation of GMatElastoPlasticQPot/Cartesian2d.h

\file
\copyright Copyright 2018. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
*/

#ifndef GMATELASTOPLASTICQPOT_CARTESIAN2D_ARRAY_HPP
#define GMATELASTOPLASTICQPOT_CARTESIAN2D_ARRAY_HPP

#include "Cartesian2d.h"

namespace GMatElastoPlasticQPot {
namespace Cartesian2d {

template <size_t N>
inline Array<N>::Array(const std::array<size_t, N>& shape)
{
    this->init(shape);
    m_type = xt::ones<size_t>(m_shape) * Type::Unset;
    m_index = xt::empty<size_t>(m_shape);
}

template <size_t N>
inline xt::xtensor<double, N> Array<N>::K() const
{
    xt::xtensor<double, N> ret = xt::empty<double>(m_shape);

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
        case Type::Unset:
            ret.data()[i] = 0.0;
            break;
        case Type::Elastic:
            ret.data()[i] = m_Elastic[m_index.data()[i]].K();
            break;
        case Type::Cusp:
            ret.data()[i] = m_Cusp[m_index.data()[i]].K();
            break;
        case Type::Smooth:
            ret.data()[i] = m_Smooth[m_index.data()[i]].K();
            break;
        }
    }

    return ret;
}

template <size_t N>
inline xt::xtensor<double, N> Array<N>::G() const
{
    xt::xtensor<double, N> ret = xt::empty<double>(m_shape);

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
        case Type::Unset:
            ret.data()[i] = 0.0;
            break;
        case Type::Elastic:
            ret.data()[i] = m_Elastic[m_index.data()[i]].G();
            break;
        case Type::Cusp:
            ret.data()[i] = m_Cusp[m_index.data()[i]].G();
            break;
        case Type::Smooth:
            ret.data()[i] = m_Smooth[m_index.data()[i]].G();
            break;
        }
    }

    return ret;
}

template <size_t N>
inline void Array<N>::currentIndex(xt::xtensor<size_t, N>& ret) const
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(ret, m_shape));

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
        case Type::Unset:
            ret.data()[i] = 0;
            break;
        case Type::Elastic:
            ret.data()[i] = 0;
            break;
        case Type::Cusp:
            ret.data()[i] = m_Cusp[m_index.data()[i]].currentIndex();
            break;
        case Type::Smooth:
            ret.data()[i] = m_Smooth[m_index.data()[i]].currentIndex();
            break;
        }
    }
}

template <size_t N>
inline bool Array<N>::checkYieldBoundLeft(size_t n) const
{

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
        case Type::Unset:
            break;
        case Type::Elastic:
            break;
        case Type::Cusp:
            if (!m_Cusp[m_index.data()[i]].checkYieldBoundLeft(n)) {
                return false;
            }
            break;
        case Type::Smooth:
            if (!m_Smooth[m_index.data()[i]].checkYieldBoundLeft(n)) {
                return false;
            }
            break;
        }
    }

    return true;
}

template <size_t N>
inline bool Array<N>::checkYieldBoundRight(size_t n) const
{

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
        case Type::Unset:
            break;
        case Type::Elastic:
            break;
        case Type::Cusp:
            if (!m_Cusp[m_index.data()[i]].checkYieldBoundRight(n)) {
                return false;
            }
            break;
        case Type::Smooth:
            if (!m_Smooth[m_index.data()[i]].checkYieldBoundRight(n)) {
                return false;
            }
            break;
        }
    }

    return true;
}

template <size_t N>
inline void Array<N>::currentYieldLeft(xt::xtensor<double, N>& ret) const
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(ret, m_shape));

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
        case Type::Unset:
            ret.data()[i] = 0.0;
            break;
        case Type::Elastic:
            ret.data()[i] = std::numeric_limits<double>::infinity();
            break;
        case Type::Cusp:
            ret.data()[i] = m_Cusp[m_index.data()[i]].currentYieldLeft();
            break;
        case Type::Smooth:
            ret.data()[i] = m_Smooth[m_index.data()[i]].currentYieldLeft();
            break;
        }
    }
}

template <size_t N>
inline void Array<N>::currentYieldRight(xt::xtensor<double, N>& ret) const
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(ret, m_shape));

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
        case Type::Unset:
            ret.data()[i] = 0.0;
            break;
        case Type::Elastic:
            ret.data()[i] = std::numeric_limits<double>::infinity();
            break;
        case Type::Cusp:
            ret.data()[i] = m_Cusp[m_index.data()[i]].currentYieldRight();
            break;
        case Type::Smooth:
            ret.data()[i] = m_Smooth[m_index.data()[i]].currentYieldRight();
            break;
        }
    }
}

template <size_t N>
inline void Array<N>::currentYieldLeft(xt::xtensor<double, N>& ret, size_t offset) const
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(ret, m_shape));

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
        case Type::Unset:
            ret.data()[i] = 0.0;
            break;
        case Type::Elastic:
            ret.data()[i] = std::numeric_limits<double>::infinity();
            break;
        case Type::Cusp:
            ret.data()[i] = m_Cusp[m_index.data()[i]].currentYieldLeft(offset);
            break;
        case Type::Smooth:
            ret.data()[i] = m_Smooth[m_index.data()[i]].currentYieldLeft(offset);
            break;
        }
    }
}

template <size_t N>
inline void Array<N>::currentYieldRight(xt::xtensor<double, N>& ret, size_t offset) const
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(ret, m_shape));

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
        case Type::Unset:
            ret.data()[i] = 0.0;
            break;
        case Type::Elastic:
            ret.data()[i] = std::numeric_limits<double>::infinity();
            break;
        case Type::Cusp:
            ret.data()[i] = m_Cusp[m_index.data()[i]].currentYieldRight(offset);
            break;
        case Type::Smooth:
            ret.data()[i] = m_Smooth[m_index.data()[i]].currentYieldRight(offset);
            break;
        }
    }
}

template <size_t N>
inline void Array<N>::nextYield(xt::xtensor<double, N>& ret, int offset) const
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(ret, m_shape));

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
        case Type::Unset:
            ret.data()[i] = 0.0;
            break;
        case Type::Elastic:
            ret.data()[i] = std::numeric_limits<double>::infinity();
            break;
        case Type::Cusp:
            ret.data()[i] = m_Cusp[m_index.data()[i]].nextYield(offset);
            break;
        case Type::Smooth:
            ret.data()[i] = m_Smooth[m_index.data()[i]].nextYield(offset);
            break;
        }
    }
}

template <size_t N>
inline void Array<N>::epsp(xt::xtensor<double, N>& ret) const
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(ret, m_shape));

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
        case Type::Unset:
            ret.data()[i] = 0.0;
            break;
        case Type::Elastic:
            ret.data()[i] = 0.0;
            break;
        case Type::Cusp:
            ret.data()[i] = m_Cusp[m_index.data()[i]].epsp();
            break;
        case Type::Smooth:
            ret.data()[i] = m_Smooth[m_index.data()[i]].epsp();
            break;
        }
    }
}

template <size_t N>
inline void Array<N>::energy(xt::xtensor<double, N>& ret) const
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(ret, m_shape));

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
        case Type::Unset:
            ret.data()[i] = 0.0;
            break;
        case Type::Elastic:
            ret.data()[i] = m_Elastic[m_index.data()[i]].energy();
            break;
        case Type::Cusp:
            ret.data()[i] = m_Cusp[m_index.data()[i]].energy();
            break;
        case Type::Smooth:
            ret.data()[i] = m_Smooth[m_index.data()[i]].energy();
            break;
        }
    }
}

template <size_t N>
inline xt::xtensor<size_t, N> Array<N>::type() const
{
    return m_type;
}

template <size_t N>
inline xt::xtensor<size_t, N> Array<N>::isElastic() const
{
    xt::xtensor<size_t, N> ret = xt::where(xt::equal(m_type, Type::Elastic), 1ul, 0ul);
    return ret;
}

template <size_t N>
inline xt::xtensor<size_t, N> Array<N>::isPlastic() const
{
    xt::xtensor<size_t, N> ret = xt::where(xt::not_equal(m_type, Type::Elastic), 1ul, 0ul);
    return ret;
}

template <size_t N>
inline xt::xtensor<size_t, N> Array<N>::isCusp() const
{
    xt::xtensor<size_t, N> ret = xt::where(xt::equal(m_type, Type::Cusp), 1ul, 0ul);
    return ret;
}

template <size_t N>
inline xt::xtensor<size_t, N> Array<N>::isSmooth() const
{
    xt::xtensor<size_t, N> ret = xt::where(xt::equal(m_type, Type::Cusp), 1ul, 0ul);
    return ret;
}

template <size_t N>
inline void Array<N>::setElastic(const xt::xtensor<double, N>& K, const xt::xtensor<double, N>& G)
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(m_type, K.shape()));
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(m_type, G.shape()));
    GMATELASTOPLASTICQPOT_ASSERT(xt::all(xt::equal(m_type, m_type)));

    for (size_t i = 0; i < m_size; ++i) {
        m_type.data()[i] = Type::Elastic;
        m_index.data()[i] = m_Elastic.size();
        m_Elastic.push_back(Elastic(K.data()[i], G.data()[i]));
    }
}

template <size_t N>
inline void Array<N>::setElastic(const xt::xtensor<size_t, N>& I, double K, double G)
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(m_type, I.shape()));
    GMATELASTOPLASTICQPOT_ASSERT(xt::all(xt::equal(I, 0ul) || xt::equal(I, 1ul)));
    GMATELASTOPLASTICQPOT_ASSERT(
        xt::all(xt::equal(xt::where(xt::equal(I, 1ul), m_type, Type::Unset), Type::Unset)));

    for (size_t i = 0; i < m_size; ++i) {
        if (I.data()[i] == 1ul) {
            m_type.data()[i] = Type::Elastic;
            m_index.data()[i] = m_Elastic.size();
            m_Elastic.push_back(Elastic(K, G));
        }
    }
}

template <size_t N>
inline void Array<N>::setCusp(
    const xt::xtensor<size_t, N>& I,
    double K,
    double G,
    const xt::xtensor<double, 1>& epsy,
    bool init_elastic)
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(m_type, I.shape()));
    GMATELASTOPLASTICQPOT_ASSERT(xt::all(xt::equal(I, 0ul) || xt::equal(I, 1ul)));
    GMATELASTOPLASTICQPOT_ASSERT(
        xt::all(xt::equal(xt::where(xt::equal(I, 1ul), m_type, Type::Unset), Type::Unset)));

    for (size_t i = 0; i < m_size; ++i) {
        if (I.data()[i] == 1ul) {
            m_type.data()[i] = Type::Cusp;
            m_index.data()[i] = m_Cusp.size();
            m_Cusp.push_back(Cusp(K, G, epsy, init_elastic));
        }
    }
}

template <size_t N>
inline void Array<N>::setSmooth(
    const xt::xtensor<size_t, N>& I,
    double K,
    double G,
    const xt::xtensor<double, 1>& epsy,
    bool init_elastic)
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(m_type, I.shape()));
    GMATELASTOPLASTICQPOT_ASSERT(xt::all(xt::equal(I, 0ul) || xt::equal(I, 1ul)));
    GMATELASTOPLASTICQPOT_ASSERT(
        xt::all(xt::equal(xt::where(xt::equal(I, 1ul), m_type, Type::Unset), Type::Unset)));

    for (size_t i = 0; i < m_size; ++i) {
        if (I.data()[i] == 1ul) {
            m_type.data()[i] = Type::Smooth;
            m_index.data()[i] = m_Smooth.size();
            m_Smooth.push_back(Smooth(K, G, epsy, init_elastic));
        }
    }
}

template <size_t N>
inline void Array<N>::setElastic(
    const xt::xtensor<size_t, N>& I,
    const xt::xtensor<size_t, N>& idx,
    const xt::xtensor<double, 1>& K,
    const xt::xtensor<double, 1>& G)
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::amax(idx)() == K.size() - 1);
    GMATELASTOPLASTICQPOT_ASSERT(K.size() == G.size());
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(m_type, I.shape()));
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(m_type, idx.shape()));
    GMATELASTOPLASTICQPOT_ASSERT(xt::all(xt::equal(I, 0ul) || xt::equal(I, 1ul)));
    GMATELASTOPLASTICQPOT_ASSERT(
        xt::all(xt::equal(xt::where(xt::equal(I, 1ul), m_type, Type::Unset), Type::Unset)));

    for (size_t i = 0; i < m_size; ++i) {
        if (I.data()[i] == 1ul) {
            size_t j = idx.data()[i];
            m_type.data()[i] = Type::Elastic;
            m_index.data()[i] = m_Elastic.size();
            m_Elastic.push_back(Elastic(K(j), G(j)));
        }
    }
}

template <size_t N>
inline void Array<N>::setCusp(
    const xt::xtensor<size_t, N>& I,
    const xt::xtensor<size_t, N>& idx,
    const xt::xtensor<double, 1>& K,
    const xt::xtensor<double, 1>& G,
    const xt::xtensor<double, 2>& epsy,
    bool init_elastic)
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::amax(idx)() == K.size() - 1);
    GMATELASTOPLASTICQPOT_ASSERT(K.size() == G.size());
    GMATELASTOPLASTICQPOT_ASSERT(K.size() == epsy.shape(0));
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(m_type, I.shape()));
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(m_type, idx.shape()));
    GMATELASTOPLASTICQPOT_ASSERT(xt::all(xt::equal(I, 0ul) || xt::equal(I, 1ul)));
    GMATELASTOPLASTICQPOT_ASSERT(
        xt::all(xt::equal(xt::where(xt::equal(I, 1ul), m_type, Type::Unset), Type::Unset)));

    for (size_t i = 0; i < m_size; ++i) {
        if (I.data()[i] == 1ul) {
            size_t j = idx.data()[i];
            m_type.data()[i] = Type::Cusp;
            m_index.data()[i] = m_Cusp.size();
            m_Cusp.push_back(Cusp(K(j), G(j), xt::view(epsy, j, xt::all()), init_elastic));
        }
    }
}

template <size_t N>
inline void Array<N>::setSmooth(
    const xt::xtensor<size_t, N>& I,
    const xt::xtensor<size_t, N>& idx,
    const xt::xtensor<double, 1>& K,
    const xt::xtensor<double, 1>& G,
    const xt::xtensor<double, 2>& epsy,
    bool init_elastic)
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::amax(idx)() == K.size() - 1);
    GMATELASTOPLASTICQPOT_ASSERT(K.size() == G.size());
    GMATELASTOPLASTICQPOT_ASSERT(K.size() == epsy.shape(0));
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(m_type, I.shape()));
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(m_type, idx.shape()));
    GMATELASTOPLASTICQPOT_ASSERT(xt::all(xt::equal(I, 0ul) || xt::equal(I, 1ul)));
    GMATELASTOPLASTICQPOT_ASSERT(
        xt::all(xt::equal(xt::where(xt::equal(I, 1ul), m_type, Type::Unset), Type::Unset)));

    for (size_t i = 0; i < m_size; ++i) {
        if (I.data()[i] == 1ul) {
            size_t j = idx.data()[i];
            m_type.data()[i] = Type::Smooth;
            m_index.data()[i] = m_Smooth.size();
            m_Smooth.push_back(Smooth(K(j), G(j), xt::view(epsy, j, xt::all()), init_elastic));
        }
    }
}

template <size_t N>
inline void Array<N>::setStrain(const xt::xtensor<double, N + 2>& arg)
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(arg, m_shape_tensor2));

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
        case Type::Unset:
            break;
        case Type::Elastic:
            m_Elastic[m_index.data()[i]].setStrainPtr(&arg.data()[i * m_stride_tensor2]);
            break;
        case Type::Cusp:
            m_Cusp[m_index.data()[i]].setStrainPtr(&arg.data()[i * m_stride_tensor2]);
            break;
        case Type::Smooth:
            m_Smooth[m_index.data()[i]].setStrainPtr(&arg.data()[i * m_stride_tensor2]);
            break;
        }
    }
}

template <size_t N>
inline void Array<N>::strain(xt::xtensor<double, N + 2>& ret) const
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(ret, m_shape_tensor2));

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
        case Type::Unset:
            GMatTensor::Cartesian2d::pointer::O2(&ret.data()[i * m_stride_tensor2]);
            break;
        case Type::Elastic:
            m_Elastic[m_index.data()[i]].strainPtr(&ret.data()[i * m_stride_tensor2]);
            break;
        case Type::Cusp:
            m_Cusp[m_index.data()[i]].strainPtr(&ret.data()[i * m_stride_tensor2]);
            break;
        case Type::Smooth:
            m_Smooth[m_index.data()[i]].strainPtr(&ret.data()[i * m_stride_tensor2]);
            break;
        }
    }
}

template <size_t N>
inline void Array<N>::stress(xt::xtensor<double, N + 2>& ret) const
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(ret, m_shape_tensor2));

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
        case Type::Unset:
            GMatTensor::Cartesian2d::pointer::O2(&ret.data()[i * m_stride_tensor2]);
            break;
        case Type::Elastic:
            m_Elastic[m_index.data()[i]].stressPtr(&ret.data()[i * m_stride_tensor2]);
            break;
        case Type::Cusp:
            m_Cusp[m_index.data()[i]].stressPtr(&ret.data()[i * m_stride_tensor2]);
            break;
        case Type::Smooth:
            m_Smooth[m_index.data()[i]].stressPtr(&ret.data()[i * m_stride_tensor2]);
            break;
        }
    }
}

template <size_t N>
inline void Array<N>::tangent(xt::xtensor<double, N + 4>& ret) const
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(ret, m_shape_tensor4));

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
        case Type::Unset:
            GMatTensor::Cartesian2d::pointer::O4(&ret.data()[i * m_stride_tensor4]);
            break;
        case Type::Elastic:
            m_Elastic[m_index.data()[i]].tangentPtr(&ret.data()[i * m_stride_tensor4]);
            break;
        case Type::Cusp:
            m_Cusp[m_index.data()[i]].tangentPtr(&ret.data()[i * m_stride_tensor4]);
            break;
        case Type::Smooth:
            m_Smooth[m_index.data()[i]].tangentPtr(&ret.data()[i * m_stride_tensor4]);
            break;
        }
    }
}

template <size_t N>
inline xt::xtensor<double, N + 2> Array<N>::Strain() const
{
    xt::xtensor<double, N + 2> ret = xt::empty<double>(m_shape_tensor2);
    this->strain(ret);
    return ret;
}

template <size_t N>
inline xt::xtensor<double, N + 2> Array<N>::Stress() const
{
    xt::xtensor<double, N + 2> ret = xt::empty<double>(m_shape_tensor2);
    this->stress(ret);
    return ret;
}

template <size_t N>
inline xt::xtensor<double, N + 4> Array<N>::Tangent() const
{
    xt::xtensor<double, N + 4> ret = xt::empty<double>(m_shape_tensor4);
    this->tangent(ret);
    return ret;
}

template <size_t N>
inline xt::xtensor<size_t, N> Array<N>::CurrentIndex() const
{
    xt::xtensor<size_t, N> ret = xt::empty<size_t>(m_shape);
    this->currentIndex(ret);
    return ret;
}

template <size_t N>
inline xt::xtensor<double, N> Array<N>::CurrentYieldLeft() const
{
    xt::xtensor<double, N> ret = xt::empty<double>(m_shape);
    this->currentYieldLeft(ret);
    return ret;
}

template <size_t N>
inline xt::xtensor<double, N> Array<N>::CurrentYieldRight() const
{
    xt::xtensor<double, N> ret = xt::empty<double>(m_shape);
    this->currentYieldRight(ret);
    return ret;
}

template <size_t N>
inline xt::xtensor<double, N> Array<N>::CurrentYieldLeft(size_t offset) const
{
    xt::xtensor<double, N> ret = xt::empty<double>(m_shape);
    this->currentYieldLeft(ret, offset);
    return ret;
}

template <size_t N>
inline xt::xtensor<double, N> Array<N>::CurrentYieldRight(size_t offset) const
{
    xt::xtensor<double, N> ret = xt::empty<double>(m_shape);
    this->currentYieldRight(ret, offset);
    return ret;
}

template <size_t N>
inline xt::xtensor<double, N> Array<N>::NextYield(int offset) const
{
    xt::xtensor<double, N> ret = xt::empty<double>(m_shape);
    this->nextYield(ret, offset);
    return ret;
}

template <size_t N>
inline xt::xtensor<double, N> Array<N>::Epsp() const
{
    xt::xtensor<double, N> ret = xt::empty<double>(m_shape);
    this->epsp(ret);
    return ret;
}

template <size_t N>
inline xt::xtensor<double, N> Array<N>::Energy() const
{
    xt::xtensor<double, N> ret = xt::empty<double>(m_shape);
    this->energy(ret);
    return ret;
}

template <size_t N>
inline auto Array<N>::getElastic(const std::array<size_t, N>& index) const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_type[index] == Type::Elastic);
    return m_Elastic[m_index[index]];
}

template <size_t N>
inline auto Array<N>::getCusp(const std::array<size_t, N>& index) const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_type[index] == Type::Cusp);
    return m_Cusp[m_index[index]];
}

template <size_t N>
inline auto Array<N>::getSmooth(const std::array<size_t, N>& index) const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_type[index] == Type::Smooth);
    return m_Smooth[m_index[index]];
}

template <size_t N>
inline auto* Array<N>::refElastic(const std::array<size_t, N>& index)
{
    GMATELASTOPLASTICQPOT_ASSERT(m_type[index] == Type::Elastic);
    return &m_Elastic[m_index[index]];
}

template <size_t N>
inline auto* Array<N>::refCusp(const std::array<size_t, N>& index)
{
    GMATELASTOPLASTICQPOT_ASSERT(m_type[index] == Type::Cusp);
    return &m_Cusp[m_index[index]];
}

template <size_t N>
inline auto* Array<N>::refSmooth(const std::array<size_t, N>& index)
{
    GMATELASTOPLASTICQPOT_ASSERT(m_type[index] == Type::Smooth);
    return &m_Smooth[m_index[index]];
}

} // namespace Cartesian2d
} // namespace GMatElastoPlasticQPot

#endif
