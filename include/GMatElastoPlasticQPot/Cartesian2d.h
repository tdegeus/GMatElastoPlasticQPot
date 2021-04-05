/**
Implementation in a 2-d Cartesian coordinate frame.

\file GMatElastoPlasticQPot/Cartesian2d.h
\copyright Copyright 2018. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
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

/**
Implementation in a 2-d Cartesian coordinate frame.

Note that for convenience this namespace include aliases to:
-   GMatTensor::Cartesian2d::O2()
-   GMatTensor::Cartesian2d::O4()
-   GMatTensor::Cartesian2d::I2()
-   GMatTensor::Cartesian2d::II()
-   GMatTensor::Cartesian2d::I4()
-   GMatTensor::Cartesian2d::I4rt()
-   GMatTensor::Cartesian2d::I4s()
-   GMatTensor::Cartesian2d::I4d()
-   GMatTensor::Cartesian2d::Hydrostatic()
-   GMatTensor::Cartesian2d::hydrostatic()
-   GMatTensor::Cartesian2d::Deviatoric()
-   GMatTensor::Cartesian2d::deviatoric()
*/
namespace Cartesian2d {

using GMatTensor::Cartesian2d::O2;
using GMatTensor::Cartesian2d::O4;
using GMatTensor::Cartesian2d::I2;
using GMatTensor::Cartesian2d::II;
using GMatTensor::Cartesian2d::I4;
using GMatTensor::Cartesian2d::I4rt;
using GMatTensor::Cartesian2d::I4s;
using GMatTensor::Cartesian2d::I4d;
using GMatTensor::Cartesian2d::Hydrostatic;
using GMatTensor::Cartesian2d::hydrostatic;
using GMatTensor::Cartesian2d::Deviatoric;
using GMatTensor::Cartesian2d::deviatoric;

/**
Equivalent strain: norm of strain deviator

\f$ \sqrt{\frac{1}{2} (dev(A))_{ij} (dev(A))_{ji}} \f$

To write to allocated data use epsd().

\param A [..., 2, 2] array.
\return [...] array.
*/
template <class T>
inline auto Epsd(const T& A);

/**
Same as Epsd(), but writes to externally allocated output.

\param A [..., 2, 2] array.
\param ret output [...] array
*/
template <class T, class U>
inline void epsd(const T& A, U& ret);

/**
Equivalent strain: norm of strain deviator

\f$ \sqrt{2 (dev(A))_{ij} (dev(A))_{ji}} \f$

To write to allocated data use sigd().

\param A [..., 2, 2] array.
\return [...] array.
*/
template <class T>
inline auto Sigd(const T& A);

/**
Same as Sigd(), but writes to externally allocated output.

\param A [..., 2, 2] array.
\param ret output [...] array
*/
template <class T, class U>
inline void sigd(const T& A, U& ret);

/**
Elastic material point.
*/
class Elastic
{
public:

    Elastic() = default;

    /**
    Constructor.

    \param K Bulk modulus.
    \param G Shear modulus.
    */
    Elastic(double K, double G);

    /**
    \return Bulk modulus.
    */
    double K() const;

    /**
    \return Shear modulus.
    */
    double G() const;

    /**
    Get the potential energy for the current state (see setStrain()).

    \return Potential energy.
    */
    double energy() const;

    /**
    Set the current strain tensor.

    \param arg xtensor array [2, 2].
    */
    template <class T>
    void setStrain(const T& arg);

    /**
    Same as setStrain(), but reads from a pointer assuming row-major storage (no bound check).

    \param arg Pointer to array (xx, xy, yx, yy).
    */
    template <class T>
    void setStrainPtr(const T* arg);

    /**
    Get the current strain tensor.

    \return [2, 2] array.
    */
    xt::xtensor<double, 2> Strain() const;

    /**
    Same as Strain(), but write to allocated data.

    \param arg xtensor array [2, 2], overwritten.
    */
    template <class T>
    void strain(T& ret) const;

    /**
    Same as Strain(), but write to a pointer assuming row-major storage (no bound check).

    \param ret Pointer to array (xx, xy, yx, yy), overwritten.
    */
    template <class T>
    void strainPtr(T* ret) const;

    /**
    Get the current stress tensor.

    \return [2, 2] array.
    */
    xt::xtensor<double, 2> Stress() const;

    /**
    Same as Stress(), but write to allocated data.

    \param arg xtensor array [2, 2], overwritten.
    */
    template <class T>
    void stress(T& ret) const;

    /**
    Same as Stress(), but write to a pointer assuming row-major storage (no bound check).

    \param ret Pointer to array (xx, xy, yx, yy), overwritten.
    */
    template <class T>
    void stressPtr(T* ret) const;

    /**
    Get the tangent tensor (strain independent).

    \return [2, 2, 2, 2] array.
    */
    xt::xtensor<double, 4> Tangent() const;

    /**
    Same as Tangent(), but write to allocated data.

    \param arg xtensor array [2, 2, 2, 2], overwritten.
    */
    template <class T>
    void tangent(T& ret) const;

    /**
    Same as Tangent(), but write to a pointer assuming row-major storage (no bound check).

    \param ret Pointer to array of size 2 * 2 * 2 * 2, overwritten.
    */
    template <class T>
    void tangentPtr(T* ret) const;

private:
    double m_K; ///< bulk modulus
    double m_G; ///< shear modulus
    std::array<double, 4> m_Eps; ///< strain tensor [xx, xy, yx, yy]
    std::array<double, 4> m_Sig; ///< stress tensor ,,
};

// Material point

class Cusp
{
public:
    Cusp() = default;
    Cusp(double K, double G, const xt::xtensor<double, 1>& epsy, bool init_elastic = true);

    /**
    \return Bulk modulus.
    */
    double K() const;

    /**
    \return Shear modulus.
    */
    double G() const;

    /**
    \return Yield strains.
    */
    xt::xtensor<double, 1> epsy() const;

    auto getQPot() const; // underlying QPot model
    auto* refQPot();      // reference to underlying QPot model

    size_t currentIndex() const;      // yield index
    double currentYieldLeft() const;  // epsy[current_index]
    double currentYieldRight() const; // epsy[current_index + 1]
    double currentYieldLeft(size_t offset) const;  // epsy[current_index - offset]
    double currentYieldRight(size_t offset) const; // epsy[current_index + offset + 1]
    double nextYield(int offset) const; // offset > 0: epsy[current_index + offset], offset < 0: epsy[current_index + offset + 1]
    double epsp() const;   // "plastic strain" = 0.5 * (currentYieldLeft + currentYieldRight)
    double energy() const; // potential energy

    // Check that 'the particle' is at least "n" wells from the far-left/right
    bool checkYieldBoundLeft(size_t n = 0) const;
    bool checkYieldBoundRight(size_t n = 0) const;

    /**
    Set the current strain tensor.

    \param arg xtensor array [2, 2].
    */
    template <class T>
    void setStrain(const T& arg);

    /**
    Same as setStrain(), but reads from a pointer assuming row-major storage (no bound check).

    \param arg Pointer to array (xx, xy, yx, yy).
    */
    template <class T>
    void setStrainPtr(const T* arg);

    /**
    Get the current strain tensor.

    \return [2, 2] array.
    */
    xt::xtensor<double, 2> Strain() const;

    /**
    Same as Strain(), but write to allocated data.

    \param arg xtensor array [2, 2], overwritten.
    */
    template <class T>
    void strain(T& ret) const;

    /**
    Same as Strain(), but write to a pointer assuming row-major storage (no bound check).

    \param ret Pointer to array (xx, xy, yx, yy), overwritten.
    */
    template <class T>
    void strainPtr(T* ret) const;

    /**
    Get the current stress tensor.

    \return [2, 2] array.
    */
    xt::xtensor<double, 2> Stress() const;

    /**
    Same as Stress(), but write to allocated data.

    \param arg xtensor array [2, 2], overwritten.
    */
    template <class T>
    void stress(T& ret) const;

    /**
    Same as Stress(), but write to a pointer assuming row-major storage (no bound check).

    \param ret Pointer to array (xx, xy, yx, yy), overwritten.
    */
    template <class T>
    void stressPtr(T* ret) const;

    /**
    Get the tangent tensor (strain independent).

    \return [2, 2, 2, 2] array.
    */
    xt::xtensor<double, 4> Tangent() const;

    /**
    Same as Tangent(), but write to allocated data.

    \param arg xtensor array [2, 2, 2, 2], overwritten.
    */
    template <class T>
    void tangent(T& ret) const;

    /**
    Same as Tangent(), but write to a pointer assuming row-major storage (no bound check).

    \param ret Pointer to array of size 2 * 2 * 2 * 2, overwritten.
    */
    template <class T>
    void tangentPtr(T* ret) const;

private:
    double m_K;                  // bulk modulus
    double m_G;                  // shear modulus
    QPot::Static m_yield;        // potential energy landscape
    std::array<double, 4> m_Eps; // strain tensor [xx, xy, yx, yy]
    std::array<double, 4> m_Sig; // stress tensor ,,
};

// Material point

class Smooth
{
public:
    Smooth() = default;
    Smooth(double K, double G, const xt::xtensor<double, 1>& epsy, bool init_elastic = true);

    /**
    \return Bulk modulus.
    */
    double K() const;

    /**
    \return Shear modulus.
    */
    double G() const;

    /**
    \return Yield strains.
    */
    xt::xtensor<double, 1> epsy() const;

    auto getQPot() const; // underlying QPot model
    auto* refQPot();      // reference to underlying QPot model

    size_t currentIndex() const;      // yield index
    double currentYieldLeft() const;  // epsy[current_index]
    double currentYieldRight() const; // epsy[current_index + 1]
    double currentYieldLeft(size_t offset) const;  // epsy[current_index - offset]
    double currentYieldRight(size_t offset) const; // epsy[current_index + offset + 1]
    double nextYield(int offset) const; // offset > 0: epsy[current_index + offset], offset < 0: epsy[current_index + offset + 1]
    double epsp() const;   // "plastic strain" = 0.5 * (currentYieldLeft + currentYieldRight)
    double energy() const; // potential energy

    // Check that 'the particle' is at least "n" wells from the far-left/right
    bool checkYieldBoundLeft(size_t n = 0) const;
    bool checkYieldBoundRight(size_t n = 0) const;

    /**
    Set the current strain tensor.

    \param arg xtensor array [2, 2].
    */
    template <class T>
    void setStrain(const T& arg);

    /**
    Same as setStrain(), but reads from a pointer assuming row-major storage (no bound check).

    \param arg Pointer to array (xx, xy, yx, yy).
    */
    template <class T>
    void setStrainPtr(const T* arg);

    /**
    Get the current strain tensor.

    \return [2, 2] array.
    */
    xt::xtensor<double, 2> Strain() const;

    /**
    Same as Strain(), but write to allocated data.

    \param arg xtensor array [2, 2], overwritten.
    */
    template <class T>
    void strain(T& ret) const;

    /**
    Same as Strain(), but write to a pointer assuming row-major storage (no bound check).

    \param ret Pointer to array (xx, xy, yx, yy), overwritten.
    */
    template <class T>
    void strainPtr(T* ret) const;

    /**
    Get the current stress tensor.

    \return [2, 2] array.
    */
    xt::xtensor<double, 2> Stress() const;

    /**
    Same as Stress(), but write to allocated data.

    \param arg xtensor array [2, 2], overwritten.
    */
    template <class T>
    void stress(T& ret) const;

    /**
    Same as Stress(), but write to a pointer assuming row-major storage (no bound check).

    \param ret Pointer to array (xx, xy, yx, yy), overwritten.
    */
    template <class T>
    void stressPtr(T* ret) const;

    /**
    Get the tangent tensor (strain independent).

    \return [2, 2, 2, 2] array.
    */
    xt::xtensor<double, 4> Tangent() const;

    /**
    Same as Tangent(), but write to allocated data.

    \param arg xtensor array [2, 2, 2, 2], overwritten.
    */
    template <class T>
    void tangent(T& ret) const;

    /**
    Same as Tangent(), but write to a pointer assuming row-major storage (no bound check).

    \param ret Pointer to array of size 2 * 2 * 2 * 2, overwritten.
    */
    template <class T>
    void tangentPtr(T* ret) const;

private:
    double m_K;                  // bulk modulus
    double m_G;                  // shear modulus
    QPot::Static m_yield;        // potential energy landscape
    std::array<double, 4> m_Eps; // strain tensor [xx, xy, yx, yy]
    std::array<double, 4> m_Sig; // stress tensor ,,
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
    using GMatTensor::Cartesian2d::Array<N>::rank;

    // Constructors

    Array() = default;
    Array(const std::array<size_t, N>& shape);

    // Overloaded methods:
    // - "shape"
    // - unit tensors: "I2", "II", "I4", "I4rt", "I4s", "I4d"

    // Type

    xt::xtensor<size_t, N> type() const;
    xt::xtensor<size_t, N> isElastic() const;
    xt::xtensor<size_t, N> isPlastic() const;
    xt::xtensor<size_t, N> isCusp() const;
    xt::xtensor<size_t, N> isSmooth() const;

    // Parameters

    xt::xtensor<double, N> K() const;
    xt::xtensor<double, N> G() const;

    // Set purely elastic

    void setElastic(
        const xt::xtensor<double, N>& K,
        const xt::xtensor<double, N>& G);

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

    void setStrain(const xt::xtensor<double, N + 2>& arg);
    void strain(xt::xtensor<double, N + 2>& ret) const;
    void stress(xt::xtensor<double, N + 2>& ret) const;
    void tangent(xt::xtensor<double, N + 4>& ret) const;
    void currentIndex(xt::xtensor<size_t, N>& ret) const;
    void currentYieldLeft(xt::xtensor<double, N>& ret) const;
    void currentYieldRight(xt::xtensor<double, N>& ret) const;
    void currentYieldLeft(xt::xtensor<double, N>& ret, size_t offset) const;
    void currentYieldRight(xt::xtensor<double, N>& ret, size_t offset) const;
    void nextYield(xt::xtensor<double, N>& ret, int offset) const;
    bool checkYieldBoundLeft(size_t n = 0) const;
    bool checkYieldBoundRight(size_t n = 0) const;
    void epsp(xt::xtensor<double, N>& ret) const;

    /**
    Return the elastic energy.

    \returns [shape()]
    */
    void energy(xt::xtensor<double, N>& ret) const;

    // Auto-allocation of the functions above

    xt::xtensor<double, N + 2> Strain() const;
    xt::xtensor<double, N + 2> Stress() const;
    xt::xtensor<double, N + 4> Tangent() const;
    xt::xtensor<size_t, N> CurrentIndex() const;
    xt::xtensor<double, N> CurrentYieldLeft() const;
    xt::xtensor<double, N> CurrentYieldRight() const;
    xt::xtensor<double, N> CurrentYieldLeft(size_t offset) const;
    xt::xtensor<double, N> CurrentYieldRight(size_t offset) const;
    xt::xtensor<double, N> NextYield(int offset) const;
    xt::xtensor<double, N> Epsp() const;

    /**
    Same as energy(), but returns allocated output.
    */
    xt::xtensor<double, N> Energy() const;

    // Get copy or reference to the underlying model at on point

    auto getElastic(const std::array<size_t, N>& index) const;
    auto getCusp(const std::array<size_t, N>& index) const;
    auto getSmooth(const std::array<size_t, N>& index) const;
    auto* refElastic(const std::array<size_t, N>& index);
    auto* refCusp(const std::array<size_t, N>& index);
    auto* refSmooth(const std::array<size_t, N>& index);

private:
    // Material vectors
    std::vector<Elastic> m_Elastic;
    std::vector<Cusp> m_Cusp;
    std::vector<Smooth> m_Smooth;

    // Identifiers for each matrix entry
    xt::xtensor<size_t, N> m_type;  // type (e.g. "Type::Elastic")
    xt::xtensor<size_t, N> m_index; // index from the relevant material vector (e.g. "m_Elastic")

    // Shape
    using GMatTensor::Cartesian2d::Array<N>::m_ndim;
    using GMatTensor::Cartesian2d::Array<N>::m_stride_tensor2;
    using GMatTensor::Cartesian2d::Array<N>::m_stride_tensor4;
    using GMatTensor::Cartesian2d::Array<N>::m_size;
    using GMatTensor::Cartesian2d::Array<N>::m_shape;
    using GMatTensor::Cartesian2d::Array<N>::m_shape_tensor2;
    using GMatTensor::Cartesian2d::Array<N>::m_shape_tensor4;
};

} // namespace Cartesian2d
} // namespace GMatElastoPlasticQPot

#include "Cartesian2d.hpp"
#include "Cartesian2d_Array.hpp"
#include "Cartesian2d_Cusp.hpp"
#include "Cartesian2d_Elastic.hpp"
#include "Cartesian2d_Smooth.hpp"

#endif
