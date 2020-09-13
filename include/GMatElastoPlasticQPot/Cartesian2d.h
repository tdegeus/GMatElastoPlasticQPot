/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatElastoPlasticQPot

*/

#ifndef GMATELASTOPLASTICQPOT_CARTESIAN2D_H
#define GMATELASTOPLASTICQPOT_CARTESIAN2D_H

#include "config.h"

namespace GMatElastoPlasticQPot {
namespace Cartesian2d {

// Alias

#if defined(_WIN32) || defined(_WIN64)
    using Tensor2 = xt::xtensor<double, 2>;
    using Tensor4 = xt::xtensor<double, 4>;
#else
    using Tensor2 = xt::xtensor_fixed<double, xt::xshape<2, 2>>;
    using Tensor4 = xt::xtensor_fixed<double, xt::xshape<2, 2, 2, 2>>;
#endif

// Unit tensors

inline Tensor2 I2();
inline Tensor4 II();
inline Tensor4 I4();
inline Tensor4 I4rt();
inline Tensor4 I4s();
inline Tensor4 I4d();

// Tensor decomposition

template <class T, class U>
inline void hydrostatic(const T& A, U& B);

template <class T>
inline auto Hydrostatic(const T& A);

template <class T, class U>
inline void deviatoric(const T& A, U& B);

template <class T>
inline auto Deviatoric(const T& A);

// Equivalent strain

template <class T, class U>
inline void epsd(const T& A, U& B);

template <class T>
inline auto Epsd(const T& A);

// Equivalent stress

template <class T, class U>
inline void sigd(const T& A, U& B);

template <class T>
inline auto Sigd(const T& A);

// Material point

class Elastic
{
public:

    // Constructors
    Elastic() = default;
    Elastic(double K, double G);

    // Parameters
    double K() const;
    double G() const;

    // Set strain
    template <class T>
    void setStrain(const T& Eps);

    template <class T>
    void setStrainIterator(const T&& begin); // presumes contiguous storage in row-major

    // Stress (no allocation, overwrites "Sig")
    template <class T>
    void stress(T& Sig) const;

    template <class T>
    void stressIterator(T&& begin) const; // presumes contiguous storage in row-major

    // Tangent (no allocation, overwrites "C")
    template <class T>
    void tangent(T& C) const;

    template <class T>
    void tangentIterator(T& begin) const;

    // Auto-allocation
    Tensor2 Stress() const;
    Tensor4 Tangent() const;

    // Return current state
    double energy() const; // potential energy

private:

    double m_K; // bulk modulus
    double m_G; // shear modulus
    std::array<double,4> m_Eps; // strain tensor
    std::array<double,4> m_Sig; // stress tensor

};

// Material point

class Cusp
{
public:

    // Constructors
    Cusp() = default;
    Cusp(double K, double G, const xt::xtensor<double,1>& epsy, bool init_elastic = true);

    // Parameters
    double K() const;
    double G() const;
    xt::xtensor<double,1> epsy() const;

    // Set strain
    template <class T>
    void setStrain(const T& Eps);

    template <class T>
    void setStrainIterator(const T&& begin); // presumes contiguous storage in row-major

    // Stress (no allocation, overwrites "Sig")
    template <class T>
    void stress(T& Sig) const;

    template <class T>
    void stressIterator(T&& begin) const; // presumes contiguous storage in row-major

    // Tangent (no allocation, overwrites "C")
    template <class T>
    void tangent(T& C) const;

    template <class T>
    void tangentIterator(T& begin) const;

    // Auto-allocation
    Tensor2 Stress() const;
    Tensor4 Tangent() const;

    // Return current state
    size_t currentIndex() const; // yield index
    size_t currentYieldLeft() const; // yield strain left epsy[index]
    size_t currentYieldRight() const; // yield strain right epsy[index + 1]
    double epsp() const; // "plastic strain" (mean of currentYieldLeft and currentYieldRight)
    double energy() const; // potential energy

private:

    double m_K; // bulk modulus
    double m_G; // shear modulus
    QPot::Static m_yield; // potential energy landscape
    std::array<double,4> m_Eps; // strain tensor
    std::array<double,4> m_Sig; // stress tensor

};

// Material point

class Smooth
{
public:

    // Constructors
    Smooth() = default;
    Smooth(double K, double G, const xt::xtensor<double,1>& epsy, bool init_elastic = true);

    // Parameters
    double K() const;
    double G() const;
    xt::xtensor<double,1> epsy() const;

    // Set strain
    template <class T>
    void setStrain(const T& Eps);

    template <class T>
    void setStrainIterator(const T&& begin); // presumes contiguous storage in row-major

    // Stress (no allocation, overwrites "Sig")
    template <class T>
    void stress(T& Sig) const;

    template <class T>
    void stressIterator(T&& begin) const; // presumes contiguous storage in row-major

    // Tangent (no allocation, overwrites "C")
    template <class T>
    void tangent(T& C) const;

    template <class T>
    void tangentIterator(T& begin) const;

    // Auto-allocation
    Tensor2 Stress() const;
    Tensor4 Tangent() const;

    // Return current state
    size_t currentIndex() const; // yield index
    size_t currentYieldLeft() const; // yield strain left epsy[index]
    size_t currentYieldRight() const; // yield strain right epsy[index + 1]
    double epsp() const; // "plastic strain" (mean of currentYieldLeft and currentYieldRight)
    double energy() const; // potential energy

private:

    double m_K; // bulk modulus
    double m_G; // shear modulus
    QPot::Static m_yield; // potential energy landscape
    std::array<double,4> m_Eps; // strain tensor
    std::array<double,4> m_Sig; // stress tensor

};

// Material identifier

struct Type {
    enum Value {
        Unset,
        Elastic,
        Cusp,
        Smooth,
    };
};

// Matrix of material points

class Matrix
{
public:

    // Constructors

    Matrix() = default;
    Matrix(size_t nelem, size_t nip);

    // Shape

    size_t ndim() const;
    size_t nelem() const;
    size_t nip() const;

    // Type

    xt::xtensor<size_t,2> type() const;
    xt::xtensor<size_t,2> isElastic() const;
    xt::xtensor<size_t,2> isPlastic() const;
    xt::xtensor<size_t,2> isCusp() const;
    xt::xtensor<size_t,2> isSmooth() const;

    // Parameters

    xt::xtensor<double,2> K() const;
    xt::xtensor<double,2> G() const;

    // Matrix of unit tensors

    xt::xtensor<double,4> I2() const;
    xt::xtensor<double,6> II() const;
    xt::xtensor<double,6> I4() const;
    xt::xtensor<double,6> I4rt() const;
    xt::xtensor<double,6> I4s() const;
    xt::xtensor<double,6> I4d() const;

    // Check that a type has been set everywhere (throws if unset points are found)

    void check() const;

    // Set parameters for a batch of points
    // (uniform for all points specified: that have "I(i,j) == 1")

    void setElastic(
        const xt::xtensor<size_t,2>& I,
        double K,
        double G);

    void setCusp(
        const xt::xtensor<size_t,2>& I,
        double K,
        double G,
        const xt::xtensor<double,1>& epsy,
        bool init_elastic = true);

    void setSmooth(
        const xt::xtensor<size_t,2>& I,
        double K,
        double G,
        const xt::xtensor<double,1>& epsy,
        bool init_elastic = true);

    // Set parameters for a batch of points:
    // each to the same material, but with different parameters:
    // the matrix "idx" refers to a which entry to use: "K[idx]", "G[idx]", or "epsy[idx,:]"

    void setElastic(
        const xt::xtensor<size_t,2>& I,
        const xt::xtensor<size_t,2>& idx,
        const xt::xtensor<double,1>& K,
        const xt::xtensor<double,1>& G);

    void setCusp(
        const xt::xtensor<size_t,2>& I,
        const xt::xtensor<size_t,2>& idx,
        const xt::xtensor<double,1>& K,
        const xt::xtensor<double,1>& G,
        const xt::xtensor<double,2>& epsy,
        bool init_elastic = true);

    void setSmooth(
        const xt::xtensor<size_t,2>& I,
        const xt::xtensor<size_t,2>& idx,
        const xt::xtensor<double,1>& K,
        const xt::xtensor<double,1>& G,
        const xt::xtensor<double,2>& epsy,
        bool init_elastic = true);

    // Compute (no allocation, overwrites last argument)

    void stress(
        const xt::xtensor<double,4>& Eps,
              xt::xtensor<double,4>& Sig) const;

    void tangent(
        const xt::xtensor<double,4>& Eps,
              xt::xtensor<double,4>& Sig,
              xt::xtensor<double,6>& C) const;

    void energy(
        const xt::xtensor<double,4>& Eps,
              xt::xtensor<double,2>& energy) const;

    void find(
        const xt::xtensor<double,4>& Eps,
              xt::xtensor<size_t,2>& idx) const;

    void epsy(
        const xt::xtensor<size_t,2>& idx,
              xt::xtensor<double,2>& epsy) const;

    void epsp(
        const xt::xtensor<double,4>& Eps,
              xt::xtensor<double,2>& epsp) const;

    // Auto-allocation of the functions above

    xt::xtensor<double,4> Stress(const xt::xtensor<double,4>& Eps) const;
    xt::xtensor<double,2> Energy(const xt::xtensor<double,4>& Eps) const;
    xt::xtensor<size_t,2> Find(const xt::xtensor<double,4>& Eps) const;
    xt::xtensor<double,2> Epsy(const xt::xtensor<size_t,2>& idx) const;
    xt::xtensor<double,2> Epsp(const xt::xtensor<double,4>& Eps) const;

    std::tuple<xt::xtensor<double,4>, xt::xtensor<double,6>>
    Tangent(const xt::xtensor<double,4>& Eps) const;

private:

    // Material vectors
    std::vector<Elastic> m_Elastic;
    std::vector<Cusp> m_Cusp;
    std::vector<Smooth> m_Smooth;

    // Identifiers for each matrix entry
    xt::xtensor<size_t,2> m_type; // type (e.g. "Type::Elastic")
    xt::xtensor<size_t,2> m_index; // index from the relevant material vector (e.g. "m_Elastic")

    // Shape
    size_t m_nelem;
    size_t m_nip;
    static const size_t m_ndim = 2;

    // Internal check
    bool m_allSet = false; // true if all points have a material assigned
    void checkAllSet(); // check if all points have a material assigned (modifies "m_allSet")
};

// Internal support functions

// Trace: "c = A_ii"
template <class U>
inline double trace(const U& A);

// Tensor contraction: "c = A_ij * B_ji"
// Symmetric tensors only, no assertion
template <class U, class V>
inline double A2_ddot_B2(const U& A, const V& B);

} // namespace Cartesian2d
} // namespace GMatElastoPlasticQPot

#include "Cartesian2d.hpp"
#include "Cartesian2d_Elastic.hpp"
#include "Cartesian2d_Cusp.hpp"
#include "Cartesian2d_Smooth.hpp"
#include "Cartesian2d_Matrix.hpp"


#endif
