/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatElastoPlasticQPot

*/

#ifndef GMATELASTOPLASTICQPOT_CARTESIAN2D_H
#define GMATELASTOPLASTICQPOT_CARTESIAN2D_H

#include "config.h"

namespace GMatElastoPlasticQPot {
namespace Cartesian2d {

// Alias

using Tensor2 = xt::xtensor_fixed<double, xt::xshape<2, 2>>;

// Unit tensors

inline Tensor2 I2();

// Hydrostatic part of a tensor

inline double Hydrostatic(const Tensor2& A);

// Deviatoric part of a tensor

inline Tensor2 Deviatoric(const Tensor2& A);

// Equivalent deviatoric stress/stress

inline double Sigd(const Tensor2& Sig);
inline double Epsd(const Tensor2& Eps);

// List version of the functions above (no allocation)

inline void hydrostatic(const xt::xtensor<double,3>& A, xt::xtensor<double,1>& Am);
inline void deviatoric(const xt::xtensor<double,3>& A, xt::xtensor<double,3>& Ad);
inline void sigd(const xt::xtensor<double,3>& Sig, xt::xtensor<double,1>& Sigeq);
inline void epsd(const xt::xtensor<double,3>& Eps, xt::xtensor<double,1>& Epseq);

// Auto-allocation allocation of the functions above

inline xt::xtensor<double,1> Hydrostatic(const xt::xtensor<double,3>& A);
inline xt::xtensor<double,3> Deviatoric(const xt::xtensor<double,3>& A);
inline xt::xtensor<double,1> Sigd(const xt::xtensor<double,3>& Sig);
inline xt::xtensor<double,1> Epsd(const xt::xtensor<double,3>& Eps);

// Matrix version of the functions above (no allocation)

inline void hydrostatic(const xt::xtensor<double,4>& A, xt::xtensor<double,2>& Am);
inline void deviatoric(const xt::xtensor<double,4>& A, xt::xtensor<double,4>& Ad);
inline void sigd(const xt::xtensor<double,4>& Sig, xt::xtensor<double,2>& Sigeq);
inline void epsd(const xt::xtensor<double,4>& Eps, xt::xtensor<double,2>& Epseq);

// Auto-allocation allocation of the functions above

inline xt::xtensor<double,2> Hydrostatic(const xt::xtensor<double,4>& A);
inline xt::xtensor<double,4> Deviatoric(const xt::xtensor<double,4>& A);
inline xt::xtensor<double,2> Sigd(const xt::xtensor<double,4>& Sig);
inline xt::xtensor<double,2> Epsd(const xt::xtensor<double,4>& Eps);

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

    // Stress (no allocation, overwrites "Sig")
    template <class U>
    void stress(const Tensor2& Eps, U&& Sig) const;

    // Stress (auto allocation)
    Tensor2 Stress(const Tensor2& Eps) const;

    // Energy
    double energy(const Tensor2& Eps) const;

private:

    double m_K; // bulk modulus
    double m_G; // shear modulus
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
    double epsy(size_t idx) const;

    // Stress (no allocation, overwrites "Sig")
    template <class U>
    void stress(const Tensor2& Eps, U&& Sig) const;

    // Stress (auto allocation)
    Tensor2 Stress(const Tensor2& Eps) const;

    // Energy
    double energy(const Tensor2& Eps) const;

    // Index of the current yield strain
    size_t find(const Tensor2& Eps) const; // strain tensor
    size_t find(double epsd) const; // equivalent deviatoric strain (epsd == Deviatoric(Eps))

    // Equivalent plastic strain
    double epsp(const Tensor2& Eps) const; // strain tensor
    double epsp(double epsd) const; // equivalent deviatoric strain (epsd == Deviatoric(Eps))

private:

    double m_K; // bulk modulus
    double m_G; // shear modulus
    xt::xtensor<double,1> m_epsy; // yield strains
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
    double epsy(size_t idx) const;

    // Stress (no allocation, overwrites "Sig")
    template <class U>
    void stress(const Tensor2& Eps, U&& Sig) const;

    // Stress (auto allocation)
    Tensor2 Stress(const Tensor2& Eps) const;

    // Energy
    double energy(const Tensor2& Eps) const;

    // Index of the current yield strain
    size_t find(const Tensor2& Eps) const; // strain tensor
    size_t find(double epsd) const; // equivalent deviatoric strain (epsd == Deviatoric(Eps))

    // Equivalent plastic strain
    double epsp(const Tensor2& Eps) const; // strain tensor
    double epsp(double epsd) const; // equivalent deviatoric strain (epsd == Deviatoric(Eps))

private:

    double m_K; // bulk modulus
    double m_G; // shear modulus
    xt::xtensor<double,1> m_epsy; // yield strains
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

    void stress(const xt::xtensor<double,4>& Eps, xt::xtensor<double,4>& Sig) const;
    void energy(const xt::xtensor<double,4>& Eps, xt::xtensor<double,2>& energy) const;
    void find(const xt::xtensor<double,4>& Eps, xt::xtensor<size_t,2>& find) const;
    void epsy(const xt::xtensor<size_t,2>& idx, xt::xtensor<double,2>& epsy) const;
    void epsp(const xt::xtensor<double,4>& Eps, xt::xtensor<double,2>& epsp) const;

    // Auto-allocation of the functions above

    xt::xtensor<double,4> Stress(const xt::xtensor<double,4>& Eps) const;
    xt::xtensor<double,2> Energy(const xt::xtensor<double,4>& Eps) const;
    xt::xtensor<size_t,2> Find(const xt::xtensor<double,4>& Eps) const;
    xt::xtensor<double,2> Epsy(const xt::xtensor<size_t,2>& idx) const;
    xt::xtensor<double,2> Epsp(const xt::xtensor<double,4>& Eps) const;

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
