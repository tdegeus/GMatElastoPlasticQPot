/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatElastoPlasticQPot

*/

#ifndef GMATELASTOPLASTICQPOT_CARTESIAN2D_ARRAY_HPP
#define GMATELASTOPLASTICQPOT_CARTESIAN2D_ARRAY_HPP

#include "Cartesian2d.h"

namespace GMatElastoPlasticQPot {
namespace Cartesian2d {

template <size_t rank>
inline Array<rank>::Array(const std::array<size_t, rank>& shape) : m_shape(shape)
{
    m_type = xt::ones<size_t>(m_shape) * Type::Unset;
    m_index = xt::empty<size_t>(m_shape);
    m_allSet = false;
    size_t nd = m_ndim;
    std::copy(shape.begin(), shape.end(), m_shape_tensor2.begin());
    std::copy(shape.begin(), shape.end(), m_shape_tensor4.begin());
    std::fill(m_shape_tensor2.begin() + rank, m_shape_tensor2.end(), nd);
    std::fill(m_shape_tensor4.begin() + rank, m_shape_tensor4.end(), nd);
    m_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
}

template <size_t rank>
inline std::array<size_t, rank> Array<rank>::shape() const
{
    return m_shape;
}

template <size_t rank>
inline xt::xtensor<double, rank> Array<rank>::K() const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    xt::xtensor<double, rank> ret = xt::empty<double>(m_shape);

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
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

template <size_t rank>
inline xt::xtensor<double, rank> Array<rank>::G() const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    xt::xtensor<double, rank> ret = xt::empty<double>(m_shape);

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
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

template <size_t rank>
inline xt::xtensor<double, rank + 2> Array<rank>::I2() const
{
    xt::xtensor<double, rank + 2> ret = xt::empty<double>(m_shape_tensor2);

    #pragma omp parallel
    {
        Tensor2 unit = Cartesian2d::I2();
        size_t stride = m_ndim * m_ndim;

        #pragma omp for
        for (size_t i = 0; i < m_size; ++i) {
            auto view = xt::adapt(&ret.data()[i * stride], xt::xshape<m_ndim, m_ndim>());
            xt::noalias(view) = unit;
        }
    }

    return ret;
}

template <size_t rank>
inline xt::xtensor<double, rank + 4> Array<rank>::II() const
{
    xt::xtensor<double, rank + 4> ret = xt::empty<double>(m_shape_tensor4);

    #pragma omp parallel
    {
        Tensor4 unit = Cartesian2d::II();
        size_t stride = m_ndim * m_ndim * m_ndim * m_ndim;

        #pragma omp for
        for (size_t i = 0; i < m_size; ++i) {
            auto view = xt::adapt(&ret.data()[i * stride], xt::xshape<m_ndim, m_ndim, m_ndim, m_ndim>());
            xt::noalias(view) = unit;
        }
    }

    return ret;
}

template <size_t rank>
inline xt::xtensor<double, rank + 4> Array<rank>::I4() const
{
    xt::xtensor<double, rank + 4> ret = xt::empty<double>(m_shape_tensor4);

    #pragma omp parallel
    {
        Tensor4 unit = Cartesian2d::I4();
        size_t stride = m_ndim * m_ndim * m_ndim * m_ndim;

        #pragma omp for
        for (size_t i = 0; i < m_size; ++i) {
            auto view = xt::adapt(&ret.data()[i * stride], xt::xshape<m_ndim, m_ndim, m_ndim, m_ndim>());
            xt::noalias(view) = unit;
        }
    }

    return ret;
}

template <size_t rank>
inline xt::xtensor<double, rank + 4> Array<rank>::I4rt() const
{
    xt::xtensor<double, rank + 4> ret = xt::empty<double>(m_shape_tensor4);

    #pragma omp parallel
    {
        Tensor4 unit = Cartesian2d::I4rt();
        size_t stride = m_ndim * m_ndim * m_ndim * m_ndim;

        #pragma omp for
        for (size_t i = 0; i < m_size; ++i) {
            auto view = xt::adapt(&ret.data()[i * stride], xt::xshape<m_ndim, m_ndim, m_ndim, m_ndim>());
            xt::noalias(view) = unit;
        }
    }

    return ret;
}

template <size_t rank>
inline xt::xtensor<double, rank + 4> Array<rank>::I4s() const
{
    xt::xtensor<double, rank + 4> ret = xt::empty<double>(m_shape_tensor4);

    #pragma omp parallel
    {
        Tensor4 unit = Cartesian2d::I4s();
        size_t stride = m_ndim * m_ndim * m_ndim * m_ndim;

        #pragma omp for
        for (size_t i = 0; i < m_size; ++i) {
            auto view = xt::adapt(&ret.data()[i * stride], xt::xshape<m_ndim, m_ndim, m_ndim, m_ndim>());
            xt::noalias(view) = unit;
        }
    }

    return ret;
}

template <size_t rank>
inline xt::xtensor<double, rank + 4> Array<rank>::I4d() const
{
    xt::xtensor<double, rank + 4> ret = xt::empty<double>(m_shape_tensor4);

    #pragma omp parallel
    {
        Tensor4 unit = Cartesian2d::I4d();
        size_t stride = m_ndim * m_ndim * m_ndim * m_ndim;

        #pragma omp for
        for (size_t i = 0; i < m_size; ++i) {
            auto view = xt::adapt(&ret.data()[i * stride], xt::xshape<m_ndim, m_ndim, m_ndim, m_ndim>());
            xt::noalias(view) = unit;
        }
    }

    return ret;
}

template <size_t rank>
inline xt::xtensor<size_t, rank> Array<rank>::type() const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    return m_type;
}

template <size_t rank>
inline xt::xtensor<size_t, rank> Array<rank>::isElastic() const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    xt::xtensor<size_t, rank> ret = xt::where(xt::equal(m_type, Type::Elastic), 1ul, 0ul);
    return ret;
}

template <size_t rank>
inline xt::xtensor<size_t, rank> Array<rank>::isPlastic() const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    xt::xtensor<size_t, rank> ret = xt::where(xt::not_equal(m_type, Type::Elastic), 1ul, 0ul);
    return ret;
}

template <size_t rank>
inline xt::xtensor<size_t, rank> Array<rank>::isCusp() const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    xt::xtensor<size_t, rank> ret = xt::where(xt::equal(m_type, Type::Cusp), 1ul, 0ul);
    return ret;
}

template <size_t rank>
inline xt::xtensor<size_t, rank> Array<rank>::isSmooth() const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    xt::xtensor<size_t, rank> ret = xt::where(xt::equal(m_type, Type::Cusp), 1ul, 0ul);
    return ret;
}

template <size_t rank>
inline void Array<rank>::check() const
{
    if (xt::any(xt::equal(m_type, Type::Unset))) {
        throw std::runtime_error("Points without material found");
    }
}

template <size_t rank>
inline void Array<rank>::checkAllSet()
{
    if (xt::any(xt::equal(m_type, Type::Unset))) {
        m_allSet = false;
    }
    else {
        m_allSet = true;
    }
}

template <size_t rank>
inline void Array<rank>::setElastic(const xt::xtensor<size_t, rank>& I, double K, double G)
{
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == I.shape());
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

    this->checkAllSet();
}

template <size_t rank>
inline void Array<rank>::setCusp(
    const xt::xtensor<size_t, rank>& I,
    double K,
    double G,
    const xt::xtensor<double, 1>& epsy,
    bool init_elastic)
{
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == I.shape());
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

    this->checkAllSet();
}

template <size_t rank>
inline void Array<rank>::setSmooth(
    const xt::xtensor<size_t, rank>& I,
    double K,
    double G,
    const xt::xtensor<double, 1>& epsy,
    bool init_elastic)
{
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == I.shape());
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

    this->checkAllSet();
}

template <size_t rank>
inline void Array<rank>::setElastic(
    const xt::xtensor<size_t, rank>& I,
    const xt::xtensor<size_t, rank>& idx,
    const xt::xtensor<double, 1>& K,
    const xt::xtensor<double, 1>& G)
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::amax(idx)() == K.size() - 1);
    GMATELASTOPLASTICQPOT_ASSERT(K.size() == G.size());
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == idx.shape());
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == I.shape());
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

    this->checkAllSet();
}

template <size_t rank>
inline void Array<rank>::setCusp(
    const xt::xtensor<size_t, rank>& I,
    const xt::xtensor<size_t, rank>& idx,
    const xt::xtensor<double, 1>& K,
    const xt::xtensor<double, 1>& G,
    const xt::xtensor<double, 2>& epsy,
    bool init_elastic)
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::amax(idx)() == K.size() - 1);
    GMATELASTOPLASTICQPOT_ASSERT(K.size() == G.size());
    GMATELASTOPLASTICQPOT_ASSERT(K.size() == epsy.shape(0));
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == idx.shape());
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == I.shape());
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

    this->checkAllSet();
}

template <size_t rank>
inline void Array<rank>::setSmooth(
    const xt::xtensor<size_t, rank>& I,
    const xt::xtensor<size_t, rank>& idx,
    const xt::xtensor<double, 1>& K,
    const xt::xtensor<double, 1>& G,
    const xt::xtensor<double, 2>& epsy,
    bool init_elastic)
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::amax(idx)() == K.size() - 1);
    GMATELASTOPLASTICQPOT_ASSERT(K.size() == G.size());
    GMATELASTOPLASTICQPOT_ASSERT(K.size() == epsy.shape(0));
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == idx.shape());
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == I.shape());
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

    this->checkAllSet();
}

template <size_t rank>
inline void Array<rank>::setStrain(const xt::xtensor<double, rank + 2>& A)
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(A, m_shape_tensor2));

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
        case Type::Elastic:
            m_Elastic[m_index.data()[i]].setStrainIterator(&A.data()[i * m_ndim * m_ndim]);
            break;
        case Type::Cusp:
            m_Cusp[m_index.data()[i]].setStrainIterator(&A.data()[i * m_ndim * m_ndim]);
            break;
        case Type::Smooth:
            m_Smooth[m_index.data()[i]].setStrainIterator(&A.data()[i * m_ndim * m_ndim]);
            break;
        }
    }
}

template <size_t rank>
inline void Array<rank>::stress(xt::xtensor<double, rank + 2>& A) const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(A, m_shape_tensor2));

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
        case Type::Elastic:
            m_Elastic[m_index.data()[i]].stressIterator(&A.data()[i * m_ndim * m_ndim]);
            break;
        case Type::Cusp:
            m_Cusp[m_index.data()[i]].stressIterator(&A.data()[i * m_ndim * m_ndim]);
            break;
        case Type::Smooth:
            m_Smooth[m_index.data()[i]].stressIterator(&A.data()[i * m_ndim * m_ndim]);
            break;
        }
    }
}

template <size_t rank>
inline void Array<rank>::tangent(xt::xtensor<double, rank + 4>& A) const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(A, m_shape_tensor4));
    size_t stride = m_ndim * m_ndim * m_ndim * m_ndim;

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        auto c = xt::adapt(&A.data()[i * stride], xt::xshape<m_ndim, m_ndim, m_ndim, m_ndim>());
        switch (m_type.data()[i]) {
        case Type::Elastic:
            m_Elastic[m_index.data()[i]].tangent(c);
            break;
        case Type::Cusp:
            m_Cusp[m_index.data()[i]].tangent(c);
            break;
        case Type::Smooth:
            m_Smooth[m_index.data()[i]].tangent(c);
            break;
        }
    }
}

template <size_t rank>
inline void Array<rank>::currentIndex(xt::xtensor<size_t, rank>& A) const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(A, m_shape));

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
        case Type::Elastic:
            A.data()[i] = 0;
            break;
        case Type::Cusp:
            A.data()[i] = m_Cusp[m_index.data()[i]].currentIndex();
            break;
        case Type::Smooth:
            A.data()[i] = m_Smooth[m_index.data()[i]].currentIndex();
            break;
        }
    }
}

template <size_t rank>
inline bool Array<rank>::checkYieldBoundLeft(size_t n) const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
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

template <size_t rank>
inline bool Array<rank>::checkYieldBoundRight(size_t n) const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
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

template <size_t rank>
inline void Array<rank>::currentYieldLeft(xt::xtensor<double, rank>& A) const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(A, m_shape));

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
        case Type::Elastic:
            A.data()[i] = std::numeric_limits<double>::infinity();
            break;
        case Type::Cusp:
            A.data()[i] = m_Cusp[m_index.data()[i]].currentYieldLeft();
            break;
        case Type::Smooth:
            A.data()[i] = m_Smooth[m_index.data()[i]].currentYieldLeft();
            break;
        }
    }
}

template <size_t rank>
inline void Array<rank>::currentYieldRight(xt::xtensor<double, rank>& A) const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(A, m_shape));

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
        case Type::Elastic:
            A.data()[i] = std::numeric_limits<double>::infinity();
            break;
        case Type::Cusp:
            A.data()[i] = m_Cusp[m_index.data()[i]].currentYieldRight();
            break;
        case Type::Smooth:
            A.data()[i] = m_Smooth[m_index.data()[i]].currentYieldRight();
            break;
        }
    }
}

template <size_t rank>
inline void Array<rank>::epsp(xt::xtensor<double, rank>& A) const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(A, m_shape));

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
        case Type::Elastic:
            A.data()[i] = 0.0;
            break;
        case Type::Cusp:
            A.data()[i] = m_Cusp[m_index.data()[i]].epsp();
            break;
        case Type::Smooth:
            A.data()[i] = m_Smooth[m_index.data()[i]].epsp();
            break;
        }
    }
}

template <size_t rank>
inline void Array<rank>::energy(xt::xtensor<double, rank>& A) const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    GMATELASTOPLASTICQPOT_ASSERT(xt::has_shape(A, m_shape));

    #pragma omp parallel for
    for (size_t i = 0; i < m_size; ++i) {
        switch (m_type.data()[i]) {
        case Type::Elastic:
            A.data()[i] = m_Elastic[m_index.data()[i]].energy();
            break;
        case Type::Cusp:
            A.data()[i] = m_Cusp[m_index.data()[i]].energy();
            break;
        case Type::Smooth:
            A.data()[i] = m_Smooth[m_index.data()[i]].energy();
            break;
        }
    }
}

template <size_t rank>
inline xt::xtensor<double, rank + 2> Array<rank>::Stress() const
{
    xt::xtensor<double, rank + 2> ret = xt::empty<double>(m_shape_tensor2);
    this->stress(ret);
    return ret;
}

template <size_t rank>
inline xt::xtensor<double, rank + 4> Array<rank>::Tangent() const
{
    xt::xtensor<double, rank + 4> ret = xt::empty<double>(m_shape_tensor4);
    this->tangent(ret);
    return ret;
}

template <size_t rank>
inline xt::xtensor<size_t, rank> Array<rank>::CurrentIndex() const
{
    xt::xtensor<size_t, rank> ret = xt::empty<size_t>(m_shape);
    this->currentIndex(ret);
    return ret;
}

template <size_t rank>
inline xt::xtensor<double, rank> Array<rank>::CurrentYieldLeft() const
{
    xt::xtensor<double, rank> ret = xt::empty<double>(m_shape);
    this->currentYieldLeft(ret);
    return ret;
}

template <size_t rank>
inline xt::xtensor<double, rank> Array<rank>::CurrentYieldRight() const
{
    xt::xtensor<double, rank> ret = xt::empty<double>(m_shape);
    this->currentYieldRight(ret);
    return ret;
}

template <size_t rank>
inline xt::xtensor<double, rank> Array<rank>::Epsp() const
{
    xt::xtensor<double, rank> ret = xt::empty<double>(m_shape);
    this->epsp(ret);
    return ret;
}

template <size_t rank>
inline xt::xtensor<double, rank> Array<rank>::Energy() const
{
    xt::xtensor<double, rank> ret = xt::empty<double>(m_shape);
    this->energy(ret);
    return ret;
}

} // namespace Cartesian2d
} // namespace GMatElastoPlasticQPot

#endif
