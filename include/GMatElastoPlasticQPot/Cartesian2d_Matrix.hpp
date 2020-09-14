/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatElastoPlasticQPot

*/

#ifndef GMATELASTOPLASTICQPOT_CARTESIAN2D_MATRIX_HPP
#define GMATELASTOPLASTICQPOT_CARTESIAN2D_MATRIX_HPP

#include "Cartesian2d.h"

namespace GMatElastoPlasticQPot {
namespace Cartesian2d {

inline Matrix::Matrix(size_t nelem, size_t nip) : m_nelem(nelem), m_nip(nip)
{
    m_size = m_nelem * m_nip;
    m_type = xt::ones<size_t>({m_nelem, m_nip}) * Type::Unset;
    m_index = xt::empty<size_t>({m_nelem, m_nip});
    m_allSet = false;
}

inline size_t Matrix::ndim() const
{
    return m_ndim;
}

inline size_t Matrix::nelem() const
{
    return m_nelem;
}

inline size_t Matrix::nip() const
{
    return m_nip;
}

inline xt::xtensor<size_t,2> Matrix::type() const
{
    return m_type;
}

inline xt::xtensor<double,2> Matrix::K() const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    xt::xtensor<double,2> ret = xt::empty<double>({m_nelem, m_nip});

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

inline xt::xtensor<double,2> Matrix::G() const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    xt::xtensor<double,2> ret = xt::empty<double>({m_nelem, m_nip});

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

inline xt::xtensor<double,4> Matrix::I2() const
{
    xt::xtensor<double,4> ret = xt::empty<double>({m_nelem, m_nip, m_ndim, m_ndim});

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

inline xt::xtensor<double,6> Matrix::II() const
{
    xt::xtensor<double,6> ret =
        xt::empty<double>({m_nelem, m_nip, m_ndim, m_ndim, m_ndim, m_ndim});

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

inline xt::xtensor<double,6> Matrix::I4() const
{
    xt::xtensor<double,6> ret =
        xt::empty<double>({m_nelem, m_nip, m_ndim, m_ndim, m_ndim, m_ndim});

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

inline xt::xtensor<double,6> Matrix::I4rt() const
{
    xt::xtensor<double,6> ret =
        xt::empty<double>({m_nelem, m_nip, m_ndim, m_ndim, m_ndim, m_ndim});

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

inline xt::xtensor<double,6> Matrix::I4s() const
{
    xt::xtensor<double,6> ret =
        xt::empty<double>({m_nelem, m_nip, m_ndim, m_ndim, m_ndim, m_ndim});

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

inline xt::xtensor<double,6> Matrix::I4d() const
{
    xt::xtensor<double,6> ret =
        xt::empty<double>({m_nelem, m_nip, m_ndim, m_ndim, m_ndim, m_ndim});

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

inline xt::xtensor<size_t,2> Matrix::isElastic() const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    xt::xtensor<size_t,2> ret = xt::where(xt::equal(m_type, Type::Elastic), 1ul, 0ul);
    return ret;
}

inline xt::xtensor<size_t,2> Matrix::isPlastic() const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    xt::xtensor<size_t,2> ret = xt::where(xt::not_equal(m_type, Type::Elastic), 1ul, 0ul);
    return ret;
}

inline xt::xtensor<size_t,2> Matrix::isCusp() const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    xt::xtensor<size_t,2> ret = xt::where(xt::equal(m_type, Type::Cusp), 1ul, 0ul);
    return ret;
}

inline xt::xtensor<size_t,2> Matrix::isSmooth() const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    xt::xtensor<size_t,2> ret = xt::where(xt::equal(m_type, Type::Cusp), 1ul, 0ul);
    return ret;
}

inline void Matrix::check() const
{
    if (xt::any(xt::equal(m_type, Type::Unset))) {
        throw std::runtime_error("Points without material found");
    }
}

inline void Matrix::checkAllSet()
{
    if (xt::any(xt::equal(m_type, Type::Unset))) {
        m_allSet = false;
    }
    else {
        m_allSet = true;
    }
}

inline void Matrix::setElastic(const xt::xtensor<size_t,2>& I, double K, double G)
{
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == I.shape());
    GMATELASTOPLASTICQPOT_ASSERT(xt::all(xt::equal(I, 0ul) || xt::equal(I, 1ul)));
    GMATELASTOPLASTICQPOT_ASSERT(
        xt::all(xt::equal(xt::where(xt::equal(I, 1ul), m_type, Type::Unset), Type::Unset)));

    m_type = xt::where(xt::equal(I, 1ul), Type::Elastic, m_type);
    m_index = xt::where(xt::equal(I, 1ul), m_Elastic.size(), m_index);
    this->checkAllSet();
    m_Elastic.push_back(Elastic(K, G));
}

inline void Matrix::setCusp(
    const xt::xtensor<size_t,2>& I,
    double K,
    double G,
    const xt::xtensor<double,1>& epsy,
    bool init_elastic)
{
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == I.shape());
    GMATELASTOPLASTICQPOT_ASSERT(xt::all(xt::equal(I, 0ul) || xt::equal(I, 1ul)));
    GMATELASTOPLASTICQPOT_ASSERT(
        xt::all(xt::equal(xt::where(xt::equal(I, 1ul), m_type, Type::Unset), Type::Unset)));

    m_type = xt::where(xt::equal(I, 1ul), Type::Cusp, m_type);
    m_index = xt::where(xt::equal(I, 1ul), m_Cusp.size(), m_index);
    this->checkAllSet();
    m_Cusp.push_back(Cusp(K, G, epsy, init_elastic));
}

inline void Matrix::setSmooth(
    const xt::xtensor<size_t,2>& I,
    double K,
    double G,
    const xt::xtensor<double,1>& epsy,
    bool init_elastic)
{
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == I.shape());
    GMATELASTOPLASTICQPOT_ASSERT(xt::all(xt::equal(I, 0ul) || xt::equal(I, 1ul)));
    GMATELASTOPLASTICQPOT_ASSERT(
        xt::all(xt::equal(xt::where(xt::equal(I, 1ul), m_type, Type::Unset), Type::Unset)));

    m_type = xt::where(xt::equal(I, 1ul), Type::Smooth, m_type);
    m_index = xt::where(xt::equal(I, 1ul), m_Smooth.size(), m_index);
    this->checkAllSet();
    m_Smooth.push_back(Smooth(K, G, epsy, init_elastic));
}

inline void Matrix::setElastic(
    const xt::xtensor<size_t,2>& I,
    const xt::xtensor<size_t,2>& idx,
    const xt::xtensor<double,1>& K,
    const xt::xtensor<double,1>& G)
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::amax(idx)[0] == K.size() - 1);
    GMATELASTOPLASTICQPOT_ASSERT(K.size() == G.size());
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == idx.shape());
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == I.shape());
    GMATELASTOPLASTICQPOT_ASSERT(xt::all(xt::equal(I, 0ul) || xt::equal(I, 1ul)));
    GMATELASTOPLASTICQPOT_ASSERT(
        xt::all(xt::equal(xt::where(xt::equal(I, 1ul), m_type, Type::Unset), Type::Unset)));

    m_type = xt::where(xt::equal(I, 1ul), Type::Elastic, m_type);
    m_index = xt::where(xt::equal(I, 1ul), m_Elastic.size() + idx, m_index);
    this->checkAllSet();

    for (size_t i = 0; i < K.size(); ++i) {
        m_Elastic.push_back(Elastic(K(i), G(i)));
    }
}

inline void Matrix::setCusp(
    const xt::xtensor<size_t,2>& I,
    const xt::xtensor<size_t,2>& idx,
    const xt::xtensor<double,1>& K,
    const xt::xtensor<double,1>& G,
    const xt::xtensor<double,2>& epsy,
    bool init_elastic)
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::amax(idx)[0] == K.size() - 1);
    GMATELASTOPLASTICQPOT_ASSERT(K.size() == G.size());
    GMATELASTOPLASTICQPOT_ASSERT(K.size() == epsy.shape()[0]);
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == idx.shape());
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == I.shape());
    GMATELASTOPLASTICQPOT_ASSERT(xt::all(xt::equal(I, 0ul) || xt::equal(I, 1ul)));
    GMATELASTOPLASTICQPOT_ASSERT(
        xt::all(xt::equal(xt::where(xt::equal(I, 1ul), m_type, Type::Unset), Type::Unset)));

    m_type = xt::where(xt::equal(I, 1ul), Type::Cusp, m_type);
    m_index = xt::where(xt::equal(I, 1ul), m_Cusp.size() + idx, m_index);
    this->checkAllSet();

    for (size_t i = 0; i < K.size(); ++i) {
        m_Cusp.push_back(Cusp(K(i), G(i), xt::view(epsy, i, xt::all()), init_elastic));
    }
}

inline void Matrix::setSmooth(
    const xt::xtensor<size_t,2>& I,
    const xt::xtensor<size_t,2>& idx,
    const xt::xtensor<double,1>& K,
    const xt::xtensor<double,1>& G,
    const xt::xtensor<double,2>& epsy,
    bool init_elastic)
{
    GMATELASTOPLASTICQPOT_ASSERT(xt::amax(idx)[0] == K.size() - 1);
    GMATELASTOPLASTICQPOT_ASSERT(K.size() == G.size());
    GMATELASTOPLASTICQPOT_ASSERT(K.size() == epsy.shape()[0]);
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == idx.shape());
    GMATELASTOPLASTICQPOT_ASSERT(m_type.shape() == I.shape());
    GMATELASTOPLASTICQPOT_ASSERT(xt::all(xt::equal(I, 0ul) || xt::equal(I, 1ul)));
    GMATELASTOPLASTICQPOT_ASSERT(
        xt::all(xt::equal(xt::where(xt::equal(I, 1ul), m_type, Type::Unset), Type::Unset)));

    m_type = xt::where(xt::equal(I, 1ul), Type::Smooth, m_type);
    m_index = xt::where(xt::equal(I, 1ul), m_Smooth.size() + idx, m_index);
    this->checkAllSet();

    for (size_t i = 0; i < K.size(); ++i) {
        m_Smooth.push_back(Smooth(K(i), G(i), xt::view(epsy, i, xt::all()), init_elastic));
    }
}

inline void Matrix::setStrain(const xt::xtensor<double,4>& A)
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    GMATELASTOPLASTICQPOT_ASSERT(
        A.shape() == std::decay_t<decltype(A)>::shape_type({m_nelem, m_nip, m_ndim, m_ndim}));

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

inline void Matrix::stress(xt::xtensor<double,4>& A) const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    GMATELASTOPLASTICQPOT_ASSERT(
        A.shape() == std::decay_t<decltype(A)>::shape_type({m_nelem, m_nip, m_ndim, m_ndim}));

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

inline void Matrix::tangent(xt::xtensor<double,6>& A) const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    GMATELASTOPLASTICQPOT_ASSERT(A.shape() ==
        std::decay_t<decltype(A)>::shape_type({m_nelem, m_nip, m_ndim, m_ndim, m_ndim, m_ndim}));
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

inline void Matrix::currentIndex(xt::xtensor<size_t,2>& A) const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    GMATELASTOPLASTICQPOT_ASSERT(A.shape() == std::decay_t<decltype(A)>::shape_type({m_nelem, m_nip}));

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

inline void Matrix::currentYieldLeft(xt::xtensor<double,2>& A) const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    GMATELASTOPLASTICQPOT_ASSERT(A.shape() == std::decay_t<decltype(A)>::shape_type({m_nelem, m_nip}));

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

inline void Matrix::currentYieldRight(xt::xtensor<double,2>& A) const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    GMATELASTOPLASTICQPOT_ASSERT(A.shape() == std::decay_t<decltype(A)>::shape_type({m_nelem, m_nip}));

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

inline void Matrix::epsp(xt::xtensor<double,2>& A) const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    GMATELASTOPLASTICQPOT_ASSERT(A.shape() == std::decay_t<decltype(A)>::shape_type({m_nelem, m_nip}));

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

inline void Matrix::energy(xt::xtensor<double,2>& A) const
{
    GMATELASTOPLASTICQPOT_ASSERT(m_allSet);
    GMATELASTOPLASTICQPOT_ASSERT(A.shape() == std::decay_t<decltype(A)>::shape_type({m_nelem, m_nip}));

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

inline xt::xtensor<double,4> Matrix::Stress() const
{
    xt::xtensor<double,4> ret = xt::empty<double>({m_nelem, m_nip, m_ndim, m_ndim});
    this->stress(ret);
    return ret;
}

inline xt::xtensor<double,6> Matrix::Tangent() const
{
    xt::xtensor<double,6> ret = xt::empty<double>({m_nelem, m_nip, m_ndim, m_ndim, m_ndim, m_ndim});
    this->tangent(ret);
    return ret;
}

inline xt::xtensor<double,2> Matrix::Energy() const
{
    xt::xtensor<double,2> ret = xt::empty<double>({m_nelem, m_nip});
    this->energy(ret);
    return ret;
}

inline xt::xtensor<size_t,2> Matrix::CurrentIndex() const
{
    xt::xtensor<size_t,2> ret = xt::empty<size_t>({m_nelem, m_nip});
    this->currentIndex(ret);
    return ret;
}

inline xt::xtensor<double,2> Matrix::CurrentYieldLeft() const
{
    xt::xtensor<double,2> ret = xt::empty<double>({m_nelem, m_nip});
    this->currentYieldLeft(ret);
    return ret;
}

inline xt::xtensor<double,2> Matrix::CurrentYieldRight() const
{
    xt::xtensor<double,2> ret = xt::empty<double>({m_nelem, m_nip});
    this->currentYieldRight(ret);
    return ret;
}

inline xt::xtensor<double,2> Matrix::Epsp() const
{
    xt::xtensor<double,2> ret = xt::empty<double>({m_nelem, m_nip});
    this->epsp(ret);
    return ret;
}

} // namespace Cartesian2d
} // namespace GMatElastoPlasticQPot

#endif
