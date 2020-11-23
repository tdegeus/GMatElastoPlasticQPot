/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatElastoPlasticQPot

*/

#ifndef GMATELASTOPLASTICQPOT_CARTESIAN2D_H
#define GMATELASTOPLASTICQPOT_CARTESIAN2D_H

// use "M_PI" from "math.h"
#define _USE_MATH_DEFINES

#include <QPot/Static.hpp>
#include <GMatTensor/Cartesian2d.h>
#include <math.h>
#include <xtensor/xsort.hpp>

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

using GMatTensor::Cartesian2d::I2;
using GMatTensor::Cartesian2d::II;
using GMatTensor::Cartesian2d::I4;
using GMatTensor::Cartesian2d::I4rt;
using GMatTensor::Cartesian2d::I4s;
using GMatTensor::Cartesian2d::I4d;

// Tensor decomposition

using GMatTensor::Cartesian2d::hydrostatic;
using GMatTensor::Cartesian2d::Hydrostatic;
using GMatTensor::Cartesian2d::deviatoric;
using GMatTensor::Cartesian2d::Deviatoric;

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
    void setStrainIterator(const T& begin); // presumes: contiguous + row-major & symmetric

    // Stress (no allocation, overwrites "Sig" / writes to "begin")
    template <class T>
    void stress(T& Sig) const;

    template <class T>
    void stressIterator(const T& begin) const; // presumes: contiguous + row-major

    // Tangent (no allocation, overwrites "C")
    template <class T>
    void tangent(T& C) const;

    // Auto-allocation
    Tensor2 Stress() const;
    Tensor4 Tangent() const;

    // Return current state
    double energy() const; // potential energy

private:
    double m_K;                  // bulk modulus
    double m_G;                  // shear modulus
    std::array<double, 4> m_Eps; // strain tensor [xx, xy, yx, yy]
    std::array<double, 4> m_Sig; // stress tensor [xx, xy, yx, yy]
};

// Material point

class Cusp
{
public:
    // Constructors
    Cusp() = default;
    Cusp(double K, double G, const xt::xtensor<double, 1>& epsy, bool init_elastic = true);

    // Parameters
    double K() const;
    double G() const;
    xt::xtensor<double, 1> epsy() const;

    // Set strain
    template <class T>
    void setStrain(const T& Eps);

    template <class T>
    void setStrainIterator(const T& begin); // presumes: contiguous + row-major & symmetric

    // Stress (no allocation, overwrites "Sig" / writes to "begin")
    template <class T>
    void stress(T& Sig) const;

    template <class T>
    void stressIterator(const T& begin) const; // presumes: contiguous + row-major

    // Tangent (no allocation, overwrites "C")
    template <class T>
    void tangent(T& C) const;

    // Auto-allocation
    Tensor2 Stress() const;
    Tensor4 Tangent() const;

    // Return current state
    size_t currentIndex() const;      // yield index
    double currentYieldLeft() const;  // yield strain left epsy[index]
    double currentYieldRight() const; // yield strain right epsy[index + 1]
    double epsp() const;   // "plastic strain" (mean of currentYieldLeft and currentYieldRight)
    double energy() const; // potential energy

    // Check that 'the particle' is at least "n" wells from the far-left/right
    bool checkYieldBoundLeft(size_t n = 0) const;
    bool checkYieldBoundRight(size_t n = 0) const;

private:
    double m_K;                  // bulk modulus
    double m_G;                  // shear modulus
    QPot::Static m_yield;        // potential energy landscape
    std::array<double, 4> m_Eps; // strain tensor [xx, xy, yx, yy]
    std::array<double, 4> m_Sig; // stress tensor [xx, xy, yx, yy]
};

// Material point

class Smooth
{
public:
    // Constructors
    Smooth() = default;
    Smooth(double K, double G, const xt::xtensor<double, 1>& epsy, bool init_elastic = true);

    // Parameters
    double K() const;
    double G() const;
    xt::xtensor<double, 1> epsy() const;

    // Set strain
    template <class T>
    void setStrain(const T& Eps);

    template <class T>
    void setStrainIterator(const T& begin); // presumes: contiguous + row-major & symmetric

    // Stress (no allocation, overwrites "Sig" / writes to "begin")
    template <class T>
    void stress(T& Sig) const;

    template <class T>
    void stressIterator(const T& begin) const; // presumes: contiguous + row-major

    // Tangent (no allocation, overwrites "C")
    template <class T>
    void tangent(T& C) const;

    // Auto-allocation
    Tensor2 Stress() const;
    Tensor4 Tangent() const;

    // Return current state
    size_t currentIndex() const;      // yield index
    double currentYieldLeft() const;  // yield strain left epsy[index]
    double currentYieldRight() const; // yield strain right epsy[index + 1]
    double epsp() const;   // "plastic strain" (mean of currentYieldLeft and currentYieldRight)
    double energy() const; // potential energy

    // Check that 'the particle' is at least "n" wells from the far-left/right
    bool checkYieldBoundLeft(size_t n = 0) const;
    bool checkYieldBoundRight(size_t n = 0) const;

private:
    double m_K;                  // bulk modulus
    double m_G;                  // shear modulus
    QPot::Static m_yield;        // potential energy landscape
    std::array<double, 4> m_Eps; // strain tensor [xx, xy, yx, yy]
    std::array<double, 4> m_Sig; // stress tensor [xx, xy, yx, yy]
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

// Array of material points

template <size_t N>
class Array : public GMatTensor::Cartesian2d::Array<N>
{
public:
    // Constructors

    Array() = default;
    Array(const std::array<size_t, N>& shape);

    // Overloaded methods

    /*
    std::array<size_t, N> shape() const;

    xt::xtensor<double, N + 2> I2() const;
    xt::xtensor<double, N + 4> II() const;
    xt::xtensor<double, N + 4> I4() const;
    xt::xtensor<double, N + 4> I4rt() const;
    xt::xtensor<double, N + 4> I4s() const;
    xt::xtensor<double, N + 4> I4d() const;
    */

    // Type

    xt::xtensor<size_t, N> type() const;
    xt::xtensor<size_t, N> isElastic() const;
    xt::xtensor<size_t, N> isPlastic() const;
    xt::xtensor<size_t, N> isCusp() const;
    xt::xtensor<size_t, N> isSmooth() const;

    // Parameters

    xt::xtensor<double, N> K() const;
    xt::xtensor<double, N> G() const;

    // Check that a type has been set everywhere (throws if unset points are found)

    void check() const;

    // Set parameters for a batch of points
    // (uniform for all points specified: that have "I(i, j) == 1")

    void setElastic(
        const xt::xtensor<size_t, N>& I,
        double K,
        double G);

    void setCusp(
        const xt::xtensor<size_t, N>& I,
        double K,
        double G,
        const xt::xtensor<double, 1>& epsy,
        bool init_elastic = true);

    void setSmooth(
        const xt::xtensor<size_t, N>& I,
        double K,
        double G,
        const xt::xtensor<double, 1>& epsy,
        bool init_elastic = true);

    // Set parameters for a batch of points:
    // each to the same material, but with different parameters:
    // the matrix "idx" refers to a which entry to use: "K(idx)", "G(idx)", or "epsy(idx,:)"

    void setElastic(
        const xt::xtensor<size_t, N>& I,
        const xt::xtensor<size_t, N>& idx,
        const xt::xtensor<double, 1>& K,
        const xt::xtensor<double, 1>& G);

    void setCusp(
        const xt::xtensor<size_t, N>& I,
        const xt::xtensor<size_t, N>& idx,
        const xt::xtensor<double, 1>& K,
        const xt::xtensor<double, 1>& G,
        const xt::xtensor<double, 2>& epsy,
        bool init_elastic = true);

    void setSmooth(
        const xt::xtensor<size_t, N>& I,
        const xt::xtensor<size_t, N>& idx,
        const xt::xtensor<double, 1>& K,
        const xt::xtensor<double, 1>& G,
        const xt::xtensor<double, 2>& epsy,
        bool init_elastic = true);

    // Set strain tensor, get the response

    void setStrain(const xt::xtensor<double, N + 2>& Eps);
    void stress(xt::xtensor<double, N + 2>& Sig) const;
    void tangent(xt::xtensor<double, N + 4>& C) const;
    void currentIndex(xt::xtensor<size_t, N>& arg) const;
    void currentYieldLeft(xt::xtensor<double, N>& arg) const;
    void currentYieldRight(xt::xtensor<double, N>& arg) const;
    bool checkYieldBoundLeft(size_t n = 0) const;
    bool checkYieldBoundRight(size_t n = 0) const;
    void epsp(xt::xtensor<double, N>& arg) const;
    void energy(xt::xtensor<double, N>& arg) const;

    // Auto-allocation of the functions above

    xt::xtensor<double, N + 2> Stress() const;
    xt::xtensor<double, N + 4> Tangent() const;
    xt::xtensor<size_t, N> CurrentIndex() const;
    xt::xtensor<double, N> CurrentYieldLeft() const;
    xt::xtensor<double, N> CurrentYieldRight() const;
    xt::xtensor<double, N> Epsp() const;
    xt::xtensor<double, N> Energy() const;

private:
    // Material vectors
    std::vector<Elastic> m_Elastic;
    std::vector<Cusp> m_Cusp;
    std::vector<Smooth> m_Smooth;

    // Identifiers for each matrix entry
    xt::xtensor<size_t, N> m_type;  // type (e.g. "Type::Elastic")
    xt::xtensor<size_t, N> m_index; // index from the relevant material vector (e.g. "m_Elastic")

    // Shape
    static const size_t m_ndim = 2;
    using GMatTensor::Cartesian2d::Array<N>::m_size;
    using GMatTensor::Cartesian2d::Array<N>::m_shape;
    using GMatTensor::Cartesian2d::Array<N>::m_shape_tensor2;
    using GMatTensor::Cartesian2d::Array<N>::m_shape_tensor4;

    // Internal check
    bool m_allSet = false; // true if all points have a material assigned
    void checkAllSet();    // check if all points have a material assigned (modifies "m_allSet")
};

} // namespace Cartesian2d
} // namespace GMatElastoPlasticQPot

#include "Cartesian2d.hpp"
#include "Cartesian2d_Array.hpp"
#include "Cartesian2d_Cusp.hpp"
#include "Cartesian2d_Elastic.hpp"
#include "Cartesian2d_Smooth.hpp"

#endif
