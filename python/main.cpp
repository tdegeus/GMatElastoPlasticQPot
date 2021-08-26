/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatElastoPlasticQPot

*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pytensor.hpp>

#include <GMatElastoPlasticQPot/Cartesian2d.h>

namespace py = pybind11;

template <class S, class T>
auto construct_Array(T& cls)
{
    cls.def(py::init<std::array<size_t, S::rank>>(), "Array of material points.", py::arg("shape"));
    cls.def("shape", &S::shape, "Shape of array.");
    cls.def("I2", &S::I2, "Array with 2nd-order unit tensors.");
    cls.def("II", &S::II, "Array with 4th-order tensors = dyadic(I2, I2).");
    cls.def("I4", &S::I4, "Array with 4th-order unit tensors.");
    cls.def("I4rt", &S::I4rt, "Array with 4th-order right-transposed unit tensors.");
    cls.def("I4s", &S::I4s, "Array with 4th-order symmetric projection tensors.");
    cls.def("I4d", &S::I4d, "Array with 4th-order deviatoric projection tensors.");
    cls.def("K", &S::K, "Array with bulk moduli.");
    cls.def("G", &S::G, "Array with shear moduli.");
    cls.def("type", &S::type, "Array with material types.");
    cls.def("isElastic", &S::isElastic, "Boolean-matrix: true for Elastic.");
    cls.def("isPlastic", &S::isPlastic, "Boolean-matrix: true for Cusp/Smooth.");
    cls.def("isCusp", &S::isCusp, "Boolean-matrix: true for Cusp.");
    cls.def("isSmooth", &S::isSmooth, "Boolean-matrix: true for Smooth.");

    cls.def("setElastic",
             static_cast<void (S::*)(
                const xt::pytensor<double, S::rank>&,
                const xt::pytensor<double, S::rank>&)>(&S::template setElastic),
            "Set all points 'Elastic'.",
            py::arg("K"),
            py::arg("G"));

    cls.def("setElastic",
            static_cast<void (S::*)(
                const xt::pytensor<bool, S::rank>&,
                const xt::pytensor<size_t, S::rank>&,
                const xt::pytensor<double, 1>&,
                const xt::pytensor<double, 1>&)>(&S::template setElastic),
            "Set specific entries 'Elastic'.",
            py::arg("I"),
            py::arg("idx"),
            py::arg("K"),
            py::arg("G"));

    cls.def("setCusp",
            static_cast<void (S::*)(
                const xt::pytensor<bool, S::rank>&,
                const xt::pytensor<size_t, S::rank>&,
                const xt::pytensor<double, 1>&,
                const xt::pytensor<double, 1>&,
                const xt::pytensor<double, 2>&, bool)>(&S::template setCusp),
            "Set specific entries 'Cusp'.",
            py::arg("I"),
            py::arg("idx"),
            py::arg("K"),
            py::arg("G"),
            py::arg("epsy"),
            py::arg("init_elastic") = true);

    cls.def("setSmooth",
            static_cast<void (S::*)(const xt::pytensor<bool, S::rank>&,
                const xt::pytensor<size_t, S::rank>&,
                const xt::pytensor<double, 1>&,
                const xt::pytensor<double, 1>&,
                const xt::pytensor<double, 2>&, bool)>(&S::template setSmooth),
            "Set specific entries 'Smooth'.",
            py::arg("I"),
            py::arg("idx"),
            py::arg("K"),
            py::arg("G"),
            py::arg("epsy"),
            py::arg("init_elastic") = true);

    cls.def("setElastic",
            static_cast<void (S::*)(
                const xt::pytensor<bool, S::rank>&,
                double,
                double)>(&S::template setElastic),
            "Set specific entries 'Elastic'.",
            py::arg("I"),
            py::arg("K"),
            py::arg("G"));

    cls.def("setCusp",
            static_cast<void (S::*)(
                const xt::pytensor<bool, S::rank>&,
                double,
                double,
                const xt::pytensor<double, 1>&,
                bool)>(&S::template setCusp),
            "Set specific entries 'Cusp'.",
            py::arg("I"),
            py::arg("K"),
            py::arg("G"),
            py::arg("epsy"),
            py::arg("init_elastic") = true);

    cls.def("setSmooth",
            static_cast<void (S::*)(
                const xt::pytensor<bool, S::rank>&,
                double,
                double,
                const xt::pytensor<double, 1>&,
                bool)>(&S::template setSmooth),
            "Set specific entries 'Smooth'.",
            py::arg("I"),
            py::arg("K"),
            py::arg("G"),
            py::arg("epsy"),
            py::arg("init_elastic") = true);

    cls.def("setStrain",
            &S::template setStrain<xt::pytensor<double, S::rank + 2>>,
            "Set strain tensors.",
            py::arg("Eps"));

    cls.def("Strain",
            &S::Strain,
            "Get strain tensors.");

    cls.def("strain",
            &S::template strain<xt::pytensor<double, S::rank + 2>>,
            "Get strain tensors.");

    cls.def("Stress",
            &S::Stress,
            "Get stress tensors.");

    cls.def("stress",
            &S::template stress<xt::pytensor<double, S::rank + 2>>,
            "Get stress tensors.");

    cls.def("Tangent",
            &S::Tangent,
            "Get stiffness tensors.");

    cls.def("tangent",
            &S::template tangent<xt::pytensor<double, S::rank + 4>>,
            "Get stiffness tensors.");

    cls.def("CurrentIndex",
            &S::CurrentIndex,
            "Get potential indices.");

    cls.def("currentIndex",
            &S::template currentIndex<xt::pytensor<long, S::rank>>,
            "Get potential indices.");

    cls.def("CurrentYieldLeft",
            py::overload_cast<>(&S::CurrentYieldLeft, py::const_),
            "Returns the yield strain to the left, for last known strain.");

    cls.def("currentYieldLeft",
            static_cast<void (S::*)(xt::pytensor<double, S::rank>&) const>(
                &S::template currentYieldLeft),
            "Returns the yield strain to the left, for last known strain.",
            py::arg("ret"));

    cls.def("CurrentYieldRight",
            py::overload_cast<>(&S::CurrentYieldRight, py::const_),
            "Returns the yield strain to the right, for last known strain.");

    cls.def("currentYieldRight",
            static_cast<void (S::*)(xt::pytensor<double, S::rank>&) const>(
                &S::template currentYieldRight),
            "Returns the yield strain to the right, for last known strain.",
            py::arg("ret"));

    cls.def("CurrentYieldLeft",
            py::overload_cast<size_t>(&S::CurrentYieldLeft, py::const_),
            "Returns the yield strain to the left, for last known strain.",
            py::arg("offset"));

    cls.def("currentYieldLeft",
            static_cast<void (S::*)(xt::pytensor<double, S::rank>&, size_t) const>(
                &S::template currentYieldLeft),
            "Returns the yield strain to the left, for last known strain.",
            py::arg("ret"),
            py::arg("offset"));

    cls.def("CurrentYieldRight",
            py::overload_cast<size_t>(&S::CurrentYieldRight, py::const_),
            "Returns the yield strain to the right, for last known strain.",
            py::arg("offset"));

    cls.def("currentYieldRight",
            static_cast<void (S::*)(xt::pytensor<double, S::rank>&, size_t) const>(
                &S::template currentYieldRight),
            "Returns the yield strain to the right, for last known strain.",
            py::arg("ret"),
            py::arg("offset"));

    cls.def("CheckYieldRedraw",
            py::overload_cast<>(&S::CheckYieldRedraw, py::const_),
            "Check to redraw the chunk of yield strains.");

    cls.def("checkYieldRedraw",
            static_cast<void (S::*)(xt::pytensor<int, S::rank>&) const>(
                &S::template checkYieldRedraw),
            "Check to redraw the chunk of yield strains.",
            py::arg("ret"));

    cls.def("checkYieldBoundLeft",
            &S::checkYieldBoundLeft,
            "Check that 'the particle' is at least 'n' wells from the far-left.",
            py::arg("n") = 0);

    cls.def("checkYieldBoundRight",
            &S::checkYieldBoundRight,
            "Check that 'the particle' is at least 'n' wells from the far-right.",
            py::arg("n") = 0);

    cls.def("Epsp",
            &S::Epsp,
            "Get equivalent plastic strains.");

    cls.def("epsp",
            &S::template epsp<xt::pytensor<long, S::rank>>,
            "Get equivalent plastic strains.");

    cls.def("Energy",
            &S::Energy,
            "Get energies.");

    cls.def("energy",
            &S::template energy<xt::pytensor<long, S::rank>>,
            "Get energies.");

    cls.def("refElastic",
             &S::refElastic,
             "Returns a reference to the underlying Elastic model.",
             py::return_value_policy::reference_internal);

    cls.def("refCusp",
             &S::refCusp,
             "Returns a reference to the underlying Cusp model.",
             py::return_value_policy::reference_internal);

    cls.def("refSmooth",
             &S::refSmooth,
             "Returns a reference to the underlying Smooth model.",
             py::return_value_policy::reference_internal);

    cls.def("__repr__", [](const S&) { return "<GMatElastoPlasticQPot.Cartesian2d.Array>"; });
}

template <class R, class T, class M>
void add_Deviatoric(M& mod)
{
    mod.def("Deviatoric",
            static_cast<R (*)(const T&)>(&GMatElastoPlasticQPot::Cartesian2d::Deviatoric),
            "Deviatoric part of a(n) (array of) tensor(s).",
            py::arg("A"));
}

template <class R, class T, class M>
void add_deviatoric(M& mod)
{
    mod.def("deviatoric",
            static_cast<void (*)(const T&, R&)>(&GMatElastoPlasticQPot::Cartesian2d::deviatoric),
            "Deviatoric part of a(n) (array of) tensor(s).",
            py::arg("A"),
            py::arg("ret"));
}

template <class R, class T, class M>
void add_Hydrostatic(M& mod)
{
    mod.def("Hydrostatic",
            static_cast<R (*)(const T&)>(&GMatElastoPlasticQPot::Cartesian2d::Hydrostatic),
            "Hydrostatic part of a(n) (array of) tensor(s).",
            py::arg("A"));
}

template <class R, class T, class M>
void add_hydrostatic(M& mod)
{
    mod.def("hydrostatic",
            static_cast<void (*)(const T&, R&)>(&GMatElastoPlasticQPot::Cartesian2d::hydrostatic),
            "Hydrostatic part of a(n) (array of) tensor(s).",
            py::arg("A"),
            py::arg("ret"));
}

template <class R, class T, class M>
void add_Epsd(M& mod)
{
    mod.def("Epsd",
            static_cast<R (*)(const T&)>(&GMatElastoPlasticQPot::Cartesian2d::Epsd),
            "Equivalent strain of a(n) (array of) tensor(s).",
            py::arg("A"));
}

template <class R, class T, class M>
void add_epsd(M& mod)
{
    mod.def("epsd",
            static_cast<void (*)(const T&, R&)>(&GMatElastoPlasticQPot::Cartesian2d::epsd),
            "Equivalent strain of a(n) (array of) tensor(s).",
            py::arg("A"),
            py::arg("ret"));
}

template <class R, class T, class M>
void add_Sigd(M& mod)
{
    mod.def("Sigd",
            static_cast<R (*)(const T&)>(&GMatElastoPlasticQPot::Cartesian2d::Sigd),
            "Equivalent stress of a(n) (array of) tensor(s).",
            py::arg("A"));
}

template <class R, class T, class M>
void add_sigd(M& mod)
{
    mod.def("sigd",
            static_cast<void (*)(const T&, R&)>(&GMatElastoPlasticQPot::Cartesian2d::sigd),
            "Equivalent stress of a(n) (array of) tensor(s).",
            py::arg("A"),
            py::arg("ret"));
}

PYBIND11_MODULE(_GMatElastoPlasticQPot, m)
{
    xt::import_numpy();

    m.doc() = "Elasto-plastic material model";

    // ---------------------------------
    // GMatElastoPlasticQPot.Cartesian2d
    // ---------------------------------

    py::module sm = m.def_submodule("Cartesian2d", "2d Cartesian coordinates");

    namespace SM = GMatElastoPlasticQPot::Cartesian2d;

    // Unit tensors

    sm.def("I2", &SM::I2, "Second order unit tensor.");
    sm.def("II", &SM::II, "Fourth order tensor with the result of the dyadic product II.");
    sm.def("I4", &SM::I4, "Fourth order unit tensor.");
    sm.def("I4rt", &SM::I4rt, "Fourth right-transposed order unit tensor.");
    sm.def("I4s", &SM::I4s, "Fourth order symmetric projection tensor.");
    sm.def("I4d", &SM::I4d, "Fourth order deviatoric projection tensor.");

    // Tensor algebra

    add_Deviatoric<xt::pytensor<double, 4>, xt::pytensor<double, 4>>(sm);
    add_Deviatoric<xt::pytensor<double, 3>, xt::pytensor<double, 3>>(sm);
    add_Deviatoric<xt::pytensor<double, 2>, xt::pytensor<double, 2>>(sm);

    add_deviatoric<xt::pytensor<double, 4>, xt::pytensor<double, 4>>(sm);
    add_deviatoric<xt::pytensor<double, 3>, xt::pytensor<double, 3>>(sm);
    add_deviatoric<xt::pytensor<double, 2>, xt::pytensor<double, 2>>(sm);

    add_Hydrostatic<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(sm);
    add_Hydrostatic<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(sm);
    add_Hydrostatic<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(sm);

    add_hydrostatic<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(sm);
    add_hydrostatic<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(sm);
    add_hydrostatic<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(sm);

    add_Epsd<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(sm);
    add_Epsd<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(sm);
    add_Epsd<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(sm);

    add_epsd<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(sm);
    add_epsd<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(sm);
    add_epsd<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(sm);

    add_Sigd<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(sm);
    add_Sigd<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(sm);
    add_Sigd<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(sm);

    add_sigd<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(sm);
    add_sigd<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(sm);
    add_sigd<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(sm);

    // Material point: Elastic

    {
        py::class_<SM::Elastic> cls(sm, "Elastic");

        cls.def(py::init<double, double>(),
                "Elastic material point.",
                py::arg("K"),
                py::arg("G"));

        cls.def("K",
                &SM::Elastic::K,
                "Returns the bulk modulus.");

        cls.def("G",
                &SM::Elastic::G,
                "Returns the shear modulus.");

        cls.def("setStrain",
                &SM::Elastic::setStrain<xt::pytensor<double, 2>>,
                "Set current strain tensor.");

        cls.def("Strain",
                &SM::Elastic::Strain,
                "Returns strain tensor.");

        cls.def("strain",
                &SM::Elastic::strain<xt::pytensor<double, 2>>,
                "Returns strain tensor.",
                py::arg("ret"));

        cls.def("Stress",
                &SM::Elastic::Stress,
                "Returns stress tensor.");

        cls.def("stress",
                &SM::Elastic::stress<xt::pytensor<double, 2>>,
                "Returns stress tensor.",
                py::arg("ret"));

        cls.def("Tangent",
                &SM::Elastic::Tangent,
                "Returns tangent stiffness.");

        cls.def("tangent",
                &SM::Elastic::tangent<xt::pytensor<double, 4>>,
                "Returns tangent stiffness.",
                py::arg("ret"));

        cls.def("energy",
                &SM::Elastic::energy,
                "Returns the energy, for last known strain.");

        cls.def("__repr__", [](const SM::Elastic&) {
            return "<GMatElastoPlasticQPot.Cartesian2d.Elastic>"; });
    }

    // Material point: Cusp

    {
        py::class_<SM::Cusp> cls(sm, "Cusp");

        cls.def(py::init<double, double, const xt::pytensor<double, 1>&, bool>(),
                "Elasto-plastic material point, with 'cusp' potentials.",
                py::arg("K"),
                py::arg("G"),
                py::arg("epsy"),
                py::arg("init_elastic") = true);

        cls.def("K",
                &SM::Cusp::K,
                "Returns the bulk modulus.");

        cls.def("G",
                &SM::Cusp::G,
                "Returns the shear modulus.");

        cls.def("epsy",
                &SM::Cusp::epsy,
                "Returns the yield strains.");

        cls.def("refQPotChunked",
                &SM::Cusp::refQPotChunked,
                "Returns a reference underlying QPot::Chunked model.",
                py::return_value_policy::reference_internal);

        cls.def("setStrain",
                &SM::Cusp::setStrain<xt::pytensor<double, 2>>,
                "Set current strain tensor.");

        cls.def("Strain",
                &SM::Cusp::Strain,
                "Returns strain tensor.");

        cls.def("strain",
                &SM::Cusp::strain<xt::pytensor<double, 2>>,
                "Returns strain tensor.",
                py::arg("ret"));

        cls.def("Stress",
                &SM::Cusp::Stress,
                "Returns stress tensor.");

        cls.def("stress",
                &SM::Cusp::stress<xt::pytensor<double, 2>>,
                "Returns stress tensor.",
                py::arg("ret"));

        cls.def("Tangent",
                &SM::Cusp::Tangent,
                "Returns tangent stiffness.");

        cls.def("tangent",
                &SM::Cusp::tangent<xt::pytensor<double, 4>>,
                "Returns tangent stiffness.",
                py::arg("ret"));

        cls.def("currentIndex",
                &SM::Cusp::currentIndex,
                "Returns the potential index, for last known strain.");

        cls.def("currentYieldLeft",
                py::overload_cast<>(&SM::Cusp::currentYieldLeft, py::const_),
                "Returns the yield strain to the left, for last known strain.");

        cls.def("currentYieldRight",
                py::overload_cast<>(&SM::Cusp::currentYieldRight, py::const_),
                "Returns the yield strain to the right, for last known strain.");

        cls.def("currentYieldLeft",
                py::overload_cast<size_t>(&SM::Cusp::currentYieldLeft, py::const_),
                "Returns the yield strain to the left, for last known strain.",
                py::arg("shift"));

        cls.def("currentYieldRight",
                py::overload_cast<size_t>(&SM::Cusp::currentYieldRight, py::const_),
                "Returns the yield strain to the right, for last known strain.",
                py::arg("shift"));

        cls.def("checkYieldBoundLeft",
                &SM::Cusp::checkYieldBoundLeft,
                "Check that 'the particle' is at least 'n' wells from the far-left.",
                py::arg("n") = 0);

        cls.def("checkYieldBoundRight",
                &SM::Cusp::checkYieldBoundRight,
                "Check that 'the particle' is at least 'n' wells from the far-right.",
                py::arg("n") = 0);

        cls.def("checkYieldRedraw",
                &SM::Cusp::checkYieldRedraw,
                "Check to redraw the chunk of yield strains.");

        cls.def("epsp",
                &SM::Cusp::epsp,
                "Returns equivalent plastic strain.");

        cls.def("energy",
                &SM::Cusp::energy,
                "Returns the energy, for last known strain.");

        cls.def("__repr__", [](const SM::Cusp&) {
            return "<GMatElastoPlasticQPot.Cartesian2d.Cusp>"; });
    }

    // Material point: Smooth

    {
        py::class_<SM::Smooth> cls(sm, "Smooth");

        cls.def(py::init<double, double, const xt::pytensor<double, 1>&, bool>(),
                "Elasto-plastic material point, with 'smooth' potentials.",
                py::arg("K"),
                py::arg("G"),
                py::arg("epsy"),
                py::arg("init_elastic") = true);

        cls.def("K",
                &SM::Smooth::K,
                "Returns the bulk modulus.");

        cls.def("G",
                &SM::Smooth::G,
                "Returns the shear modulus.");

        cls.def("epsy",
                &SM::Smooth::epsy,
                "Returns the yield strains.");

        cls.def("refQPotChunked",
                 &SM::Smooth::refQPotChunked,
                 "Returns a reference underlying QPot::Chunked model.",
                 py::return_value_policy::reference_internal);

        cls.def("setStrain",
                &SM::Smooth::setStrain<xt::pytensor<double, 2>>,
                "Set current strain tensor.");

        cls.def("Strain",
                &SM::Smooth::Strain,
                "Returns strain tensor.");

        cls.def("strain",
                &SM::Smooth::strain<xt::pytensor<double, 2>>,
                "Returns strain tensor.",
                py::arg("ret"));

        cls.def("Stress",
                &SM::Smooth::Stress,
                "Returns stress tensor.");

        cls.def("stress",
                &SM::Smooth::stress<xt::pytensor<double, 2>>,
                "Returns stress tensor.",
                py::arg("ret"));

        cls.def("Tangent",
                &SM::Smooth::Tangent,
                "Returns tangent stiffness.");

        cls.def("tangent",
                &SM::Smooth::tangent<xt::pytensor<double, 4>>,
                "Returns tangent stiffness.",
                py::arg("ret"));

        cls.def("currentIndex",
                &SM::Smooth::currentIndex,
                "Returns the potential index, for last known strain.");

        cls.def("currentYieldLeft",
                py::overload_cast<>(&SM::Smooth::currentYieldLeft, py::const_),
                "Returns the yield strain to the left, for last known strain.");

        cls.def("currentYieldRight",
                py::overload_cast<>(&SM::Smooth::currentYieldRight, py::const_),
                "Returns the yield strain to the right, for last known strain.");

        cls.def("currentYieldLeft",
                py::overload_cast<size_t>(&SM::Smooth::currentYieldLeft, py::const_),
                "Returns the yield strain to the left, for last known strain.",
                py::arg("shift"));

        cls.def("currentYieldRight",
                py::overload_cast<size_t>(&SM::Smooth::currentYieldRight, py::const_),
                "Returns the yield strain to the right, for last known strain.",
                py::arg("shift"));

        cls.def("checkYieldBoundLeft",
                &SM::Smooth::checkYieldBoundLeft,
                "Check that 'the particle' is at least 'n' wells from the far-left.",
                py::arg("n") = 0);

        cls.def("checkYieldBoundRight",
                &SM::Smooth::checkYieldBoundRight,
                "Check that 'the particle' is at least 'n' wells from the far-right.",
                py::arg("n") = 0);

        cls.def("checkYieldRedraw",
                &SM::Smooth::checkYieldRedraw,
                "Check to redraw the chunk of yield strains.");

        cls.def("epsp",
                &SM::Smooth::epsp,
                "Returns equivalent plastic strain.");

        cls.def("energy",
                &SM::Smooth::energy,
                "Returns the energy, for last known strain.");

        cls.def("__repr__", [](const SM::Smooth&) {
                return "<GMatElastoPlasticQPot.Cartesian2d.Smooth>"; });
    }

    // Material identifier

    py::module smm = sm.def_submodule("Type", "Type enumerator");

    py::enum_<SM::Type::Value>(smm, "Type")
        .value("Unset", SM::Type::Unset)
        .value("Elastic", SM::Type::Elastic)
        .value("Cusp", SM::Type::Cusp)
        .value("Smooth", SM::Type::Smooth)
        .export_values();

    // Array

    py::class_<SM::Array<1>> array1d(sm, "Array1d");
    py::class_<SM::Array<2>> array2d(sm, "Array2d");
    py::class_<SM::Array<3>> array3d(sm, "Array3d");

    construct_Array<SM::Array<1>>(array1d);
    construct_Array<SM::Array<2>>(array2d);
    construct_Array<SM::Array<3>>(array3d);
}
