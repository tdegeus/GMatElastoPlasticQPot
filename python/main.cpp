/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatElastoPlasticQPot

*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pytensor.hpp>

#ifdef _WIN32
#include <pyxtensor/pyxtensor.hpp>
#endif

// Enable basic assertions on matrix shape
// (doesn't cost a lot of time, but avoids segmentation faults)
#define QPOT_ENABLE_ASSERT
#define GMATELASTOPLASTICQPOT_ENABLE_ASSERT

#include <GMatElastoPlasticQPot/Cartesian2d.h>

namespace py = pybind11;

template <class S, class T>
auto construct_Array(T& self)
{
    self.def(py::init<std::array<size_t, S::rank>>(), "Array of material points.", py::arg("shape"))

        .def("shape", &S::shape, "Shape of array.")
        .def("I2", &S::I2, "Array with 2nd-order unit tensors.")
        .def("II", &S::II, "Array with 4th-order tensors = dyadic(I2, I2).")
        .def("I4", &S::I4, "Array with 4th-order unit tensors.")
        .def("I4rt", &S::I4rt, "Array with 4th-order right-transposed unit tensors.")
        .def("I4s", &S::I4s, "Array with 4th-order symmetric projection tensors.")
        .def("I4d", &S::I4d, "Array with 4th-order deviatoric projection tensors.")
        .def("K", &S::K, "Array with bulk moduli.")
        .def("G", &S::G, "Array with shear moduli.")
        .def("type", &S::type, "Array with material types.")
        .def("isElastic", &S::isElastic, "Boolean-matrix: true for Elastic.")
        .def("isPlastic", &S::isPlastic, "Boolean-matrix: true for Cusp/Smooth.")
        .def("isCusp", &S::isCusp, "Boolean-matrix: true for Cusp.")
        .def("isSmooth", &S::isSmooth, "Boolean-matrix: true for Smooth.")

        .def(
            "setElastic",
             static_cast<void (S::*)(const xt::pytensor<double, S::rank>&, const xt::pytensor<double, S::rank>&)>(&S::template setElastic),
            "Set all points 'Elastic'.",
            py::arg("K"),
            py::arg("G"))

        .def(
            "setElastic",
            static_cast<void (S::*)(const xt::pytensor<bool, S::rank>&, const xt::pytensor<size_t, S::rank>&, const xt::pytensor<double, 1>&, const xt::pytensor<double, 1>&)>(&S::template setElastic),
            "Set specific entries 'Elastic'.",
            py::arg("I"),
            py::arg("idx"),
            py::arg("K"),
            py::arg("G"))

        .def(
            "setCusp",
            static_cast<void (S::*)(const xt::pytensor<bool, S::rank>&, const xt::pytensor<size_t, S::rank>&, const xt::pytensor<double, 1>&, const xt::pytensor<double, 1>&, const xt::pytensor<double, 2>&, bool)>(&S::template setCusp),
            "Set specific entries 'Cusp'.",
            py::arg("I"),
            py::arg("idx"),
            py::arg("K"),
            py::arg("G"),
            py::arg("epsy"),
            py::arg("init_elastic") = true)

        .def(
            "setSmooth",
            static_cast<void (S::*)(const xt::pytensor<bool, S::rank>&, const xt::pytensor<size_t, S::rank>&, const xt::pytensor<double, 1>&, const xt::pytensor<double, 1>&, const xt::pytensor<double, 2>&, bool)>(&S::template setSmooth),
            "Set specific entries 'Smooth'.",
            py::arg("I"),
            py::arg("idx"),
            py::arg("K"),
            py::arg("G"),
            py::arg("epsy"),
            py::arg("init_elastic") = true)

        .def(
            "setElastic",
            static_cast<void (S::*)(const xt::pytensor<bool, S::rank>&, double, double)>(&S::template setElastic),
            "Set specific entries 'Elastic'.",
            py::arg("I"),
            py::arg("K"),
            py::arg("G"))

        .def(
            "setCusp",
            static_cast<void (S::*)(const xt::pytensor<bool, S::rank>&, double, double, const xt::pytensor<double, 1>&, bool)>(&S::template setCusp),
            "Set specific entries 'Cusp'.",
            py::arg("I"),
            py::arg("K"),
            py::arg("G"),
            py::arg("epsy"),
            py::arg("init_elastic") = true)

        .def(
            "setSmooth",
            static_cast<void (S::*)(const xt::pytensor<bool, S::rank>&, double, double, const xt::pytensor<double, 1>&, bool)>(&S::template setSmooth),
            "Set specific entries 'Smooth'.",
            py::arg("I"),
            py::arg("K"),
            py::arg("G"),
            py::arg("epsy"),
            py::arg("init_elastic") = true)

        .def("setStrain", &S::template setStrain<xt::pytensor<double, S::rank + 2>>, "Set strain tensors.", py::arg("Eps"))

        .def("strain", &S::template strain<xt::pytensor<double, S::rank + 2>>, "Get strain tensors.")
        .def("stress", &S::template stress<xt::pytensor<double, S::rank + 2>>, "Get stress tensors.")
        .def("tangent", &S::template tangent<xt::pytensor<double, S::rank + 4>>, "Get stiffness tensors.")
        .def("currentIndex", &S::template currentIndex<xt::pytensor<long, S::rank>>, "Get potential indices.")

        .def(
            "currentYieldLeft",
            static_cast<void (S::*)(xt::pytensor<double, S::rank>&) const>(&S::template currentYieldLeft),
            "Returns the yield strain to the left, for last known strain.",
            py::arg("ret"))

        .def(
            "currentYieldRight",
            static_cast<void (S::*)(xt::pytensor<double, S::rank>&) const>(&S::template currentYieldRight),
            "Returns the yield strain to the right, for last known strain.",
            py::arg("ret"))

        .def(
            "currentYieldLeft",
            static_cast<void (S::*)(xt::pytensor<double, S::rank>&, size_t) const>(&S::template currentYieldLeft),
            "Returns the yield strain to the left, for last known strain.",
            py::arg("ret"),
            py::arg("offset"))

        .def(
            "currentYieldRight",
            static_cast<void (S::*)(xt::pytensor<double, S::rank>&, size_t) const>(&S::template currentYieldRight),
            "Returns the yield strain to the right, for last known strain.",
            py::arg("ret"),
            py::arg("offset"))

        .def("epsp", &S::template epsp<xt::pytensor<long, S::rank>>, "Get equivalent plastic strains.")
        .def("energy", &S::template energy<xt::pytensor<long, S::rank>>, "Get energies.")

        .def("Strain", &S::Strain, "Get strain tensors.")
        .def("Stress", &S::Stress, "Get stress tensors.")
        .def("Tangent", &S::Tangent, "Get stiffness tensors.")
        .def("CurrentIndex", &S::CurrentIndex, "Get potential indices.")

        .def(
            "CurrentYieldLeft",
            py::overload_cast<>(&S::CurrentYieldLeft, py::const_),
            "Returns the yield strain to the left, for last known strain.")

        .def(
            "CurrentYieldRight",
            py::overload_cast<>(&S::CurrentYieldRight, py::const_),
            "Returns the yield strain to the right, for last known strain.")

        .def(
            "CurrentYieldLeft",
            py::overload_cast<size_t>(&S::CurrentYieldLeft, py::const_),
            "Returns the yield strain to the left, for last known strain.",
            py::arg("offset"))

        .def(
            "CurrentYieldRight",
            py::overload_cast<size_t>(&S::CurrentYieldRight, py::const_),
            "Returns the yield strain to the right, for last known strain.",
            py::arg("offset"))

        .def("Epsp", &S::Epsp, "Get equivalent plastic strains.")
        .def("Energy", &S::Energy, "Get energies.")

        .def("refElastic",
             &S::refElastic,
             "Returns a reference to the underlying Elastic model.",
             py::return_value_policy::reference_internal)

        .def("refCusp",
             &S::refCusp,
             "Returns a reference to the underlying Cusp model.",
             py::return_value_policy::reference_internal)

        .def("refSmooth",
             &S::refSmooth,
             "Returns a reference to the underlying Smooth model.",
             py::return_value_policy::reference_internal)

        .def(
            "checkYieldBoundLeft",
            &S::checkYieldBoundLeft,
            "Check that 'the particle' is at least 'n' wells from the far-left.",
            py::arg("n") = 0)

        .def(
            "checkYieldBoundRight",
            &S::checkYieldBoundRight,
            "Check that 'the particle' is at least 'n' wells from the far-right.",
            py::arg("n") = 0)

        .def("__repr__", [](const S&) {
            return "<GMatElastoPlasticQPot.Cartesian2d.Array>";
        });
}

template <class R, class T, class M>
void add_deviatoric(M& module)
{
    module.def(
        "deviatoric",
        static_cast<void (*)(const T&, R&)>(&GMatElastoPlasticQPot::Cartesian2d::deviatoric),
        "Deviatoric part of a(n) (array of) tensor(s).",
        py::arg("A"),
        py::arg("ret"));
}

template <class R, class T, class M>
void add_hydrostatic(M& module)
{
    module.def(
        "hydrostatic",
        static_cast<void (*)(const T&, R&)>(&GMatElastoPlasticQPot::Cartesian2d::hydrostatic),
        "Hydrostatic part of a(n) (array of) tensor(s).",
        py::arg("A"),
        py::arg("ret"));
}

template <class R, class T, class M>
void add_epsd(M& module)
{
    module.def(
        "epsd",
        static_cast<void (*)(const T&, R&)>(&GMatElastoPlasticQPot::Cartesian2d::epsd),
        "Equivalent strain of a(n) (array of) tensor(s).",
        py::arg("A"),
        py::arg("ret"));
}

template <class R, class T, class M>
void add_sigd(M& module)
{
    module.def(
        "sigd",
        static_cast<void (*)(const T&, R&)>(&GMatElastoPlasticQPot::Cartesian2d::sigd),
        "Equivalent stress of a(n) (array of) tensor(s).",
        py::arg("A"),
        py::arg("ret"));
}

template <class R, class T, class M>
void add_Deviatoric(M& module)
{
    module.def(
        "Deviatoric",
        static_cast<R (*)(const T&)>(&GMatElastoPlasticQPot::Cartesian2d::Deviatoric),
        "Deviatoric part of a(n) (array of) tensor(s).",
        py::arg("A"));
}

template <class R, class T, class M>
void add_Hydrostatic(M& module)
{
    module.def(
        "Hydrostatic",
        static_cast<R (*)(const T&)>(&GMatElastoPlasticQPot::Cartesian2d::Hydrostatic),
        "Hydrostatic part of a(n) (array of) tensor(s).",
        py::arg("A"));
}

template <class R, class T, class M>
void add_Epsd(M& module)
{
    module.def(
        "Epsd",
        static_cast<R (*)(const T&)>(&GMatElastoPlasticQPot::Cartesian2d::Epsd),
        "Equivalent strain of a(n) (array of) tensor(s).",
        py::arg("A"));
}

template <class R, class T, class M>
void add_Sigd(M& module)
{
    module.def(
        "Sigd",
        static_cast<R (*)(const T&)>(&GMatElastoPlasticQPot::Cartesian2d::Sigd),
        "Equivalent stress of a(n) (array of) tensor(s).",
        py::arg("A"));
}

PYBIND11_MODULE(GMatElastoPlasticQPot, m)
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

    add_deviatoric<xt::pytensor<double, 4>, xt::pytensor<double, 4>>(sm);
    add_deviatoric<xt::pytensor<double, 3>, xt::pytensor<double, 3>>(sm);
    add_deviatoric<xt::pytensor<double, 2>, xt::pytensor<double, 2>>(sm);

    add_hydrostatic<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(sm);
    add_hydrostatic<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(sm);
    #ifdef _WIN32
    // todo: switch to xt::pytensor when https://github.com/xtensor-stack/xtensor-python/pull/263 is fixed
    add_hydrostatic<xt::xtensor<double, 0>, xt::xtensor<double, 2>>(sm);
    #else
    add_hydrostatic<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(sm);
    #endif

    add_epsd<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(sm);
    add_epsd<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(sm);
    #ifdef _WIN32
    // todo: switch to xt::pytensor when https://github.com/xtensor-stack/xtensor-python/pull/263 is fixed
    add_epsd<xt::xtensor<double, 0>, xt::xtensor<double, 2>>(sm);
    #else
    add_epsd<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(sm);
    #endif

    add_sigd<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(sm);
    add_sigd<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(sm);
    #ifdef _WIN32
    // todo: switch to xt::pytensor when https://github.com/xtensor-stack/xtensor-python/pull/263 is fixed
    add_sigd<xt::xtensor<double, 0>, xt::xtensor<double, 2>>(sm);
    #else
    add_sigd<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(sm);
    #endif

    add_Deviatoric<xt::pytensor<double, 4>, xt::pytensor<double, 4>>(sm);
    add_Deviatoric<xt::pytensor<double, 3>, xt::pytensor<double, 3>>(sm);
    add_Deviatoric<xt::pytensor<double, 2>, xt::pytensor<double, 2>>(sm);

    add_Hydrostatic<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(sm);
    add_Hydrostatic<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(sm);
    #ifdef _WIN32
    // todo: switch to xt::pytensor when https://github.com/xtensor-stack/xtensor-python/pull/263 is fixed
    add_Hydrostatic<xt::xtensor<double, 0>, xt::xtensor<double, 2>>(sm);
    #else
    add_Hydrostatic<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(sm);
    #endif

    add_Epsd<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(sm);
    add_Epsd<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(sm);
    #ifdef _WIN32
    // todo: switch to xt::pytensor when https://github.com/xtensor-stack/xtensor-python/pull/263 is fixed
    add_Epsd<xt::xtensor<double, 0>, xt::xtensor<double, 2>>(sm);
    #else
    add_Epsd<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(sm);
    #endif

    add_Sigd<xt::pytensor<double, 2>, xt::pytensor<double, 4>>(sm);
    add_Sigd<xt::pytensor<double, 1>, xt::pytensor<double, 3>>(sm);
    #ifdef _WIN32
    // todo: switch to xt::pytensor when https://github.com/xtensor-stack/xtensor-python/pull/263 is fixed
    add_Sigd<xt::xtensor<double, 0>, xt::xtensor<double, 2>>(sm);
    #else
    add_Sigd<xt::pytensor<double, 0>, xt::pytensor<double, 2>>(sm);
    #endif

    // Material point: Elastic

    py::class_<SM::Elastic>(sm, "Elastic")

        .def(py::init<double, double>(), "Elastic material point.", py::arg("K"), py::arg("G"))

        .def("K", &SM::Elastic::K, "Returns the bulk modulus.")
        .def("G", &SM::Elastic::G, "Returns the shear modulus.")
        .def("setStrain", &SM::Elastic::setStrain<xt::pytensor<double, 2>>, "Set current strain tensor.")
        .def("strain", &SM::Elastic::strain<xt::pytensor<double, 2>>, "Returns strain tensor.", py::arg("ret"))
        .def("stress", &SM::Elastic::stress<xt::pytensor<double, 2>>, "Returns stress tensor.", py::arg("ret"))
        .def("tangent", &SM::Elastic::tangent<xt::pytensor<double, 4>>, "Returns tangent stiffness.", py::arg("ret"))
        .def("Strain", &SM::Elastic::Strain, "Returns strain tensor.")
        .def("Stress", &SM::Elastic::Stress, "Returns stress tensor.")
        .def("Tangent", &SM::Elastic::Tangent, "Returns tangent stiffness.")
        .def("energy", &SM::Elastic::energy, "Returns the energy, for last known strain.")

        .def("__repr__", [](const SM::Elastic&) {
            return "<GMatElastoPlasticQPot.Cartesian2d.Elastic>";
        });

    // Material point: Cusp

    py::class_<SM::Cusp>(sm, "Cusp")

        .def(
            py::init<double, double, const xt::pytensor<double, 1>&, bool>(),
            "Elasto-plastic material point, with 'cusp' potentials.",
            py::arg("K"),
            py::arg("G"),
            py::arg("epsy"),
            py::arg("init_elastic") = true)

        .def("K", &SM::Cusp::K, "Returns the bulk modulus.")
        .def("G", &SM::Cusp::G, "Returns the shear modulus.")
        .def("epsy", &SM::Cusp::epsy, "Returns the yield strains.")

        .def("refQPotChunked",
             &SM::Cusp::refQPotChunked,
             "Returns a reference underlying QPot::Chunked model.",
             py::return_value_policy::reference_internal)

        .def("setStrain", &SM::Cusp::setStrain<xt::pytensor<double, 2>>, "Set current strain tensor.")
        .def("strain", &SM::Cusp::strain<xt::pytensor<double, 2>>, "Returns strain tensor.", py::arg("ret"))
        .def("stress", &SM::Cusp::stress<xt::pytensor<double, 2>>, "Returns stress tensor.", py::arg("ret"))
        .def("tangent", &SM::Cusp::tangent<xt::pytensor<double, 4>>, "Returns tangent stiffness.", py::arg("ret"))
        .def("Strain", &SM::Cusp::Strain, "Returns strain tensor.")
        .def("Stress", &SM::Cusp::Stress, "Returns stress tensor.")
        .def("Tangent", &SM::Cusp::Tangent, "Returns tangent stiffness.")

        .def(
            "currentIndex",
            &SM::Cusp::currentIndex,
            "Returns the potential index, for last known strain.")

        .def(
            "currentYieldLeft",
            py::overload_cast<>(&SM::Cusp::currentYieldLeft, py::const_),
            "Returns the yield strain to the left, for last known strain.")

        .def(
            "currentYieldRight",
            py::overload_cast<>(&SM::Cusp::currentYieldRight, py::const_),
            "Returns the yield strain to the right, for last known strain.")

        .def(
            "currentYieldLeft",
            py::overload_cast<size_t>(&SM::Cusp::currentYieldLeft, py::const_),
            "Returns the yield strain to the left, for last known strain.",
            py::arg("shift"))

        .def(
            "currentYieldRight",
            py::overload_cast<size_t>(&SM::Cusp::currentYieldRight, py::const_),
            "Returns the yield strain to the right, for last known strain.",
            py::arg("shift"))

        .def(
            "checkYieldBoundLeft",
            &SM::Cusp::checkYieldBoundLeft,
            "Check that 'the particle' is at least 'n' wells from the far-left.",
            py::arg("n") = 0)

        .def(
            "checkYieldBoundRight",
            &SM::Cusp::checkYieldBoundRight,
            "Check that 'the particle' is at least 'n' wells from the far-right.",
            py::arg("n") = 0)

        .def("epsp", &SM::Cusp::epsp, "Returns equivalent plastic strain.")
        .def("energy", &SM::Cusp::energy, "Returns the energy, for last known strain.")

        .def("__repr__", [](const SM::Cusp&) {
            return "<GMatElastoPlasticQPot.Cartesian2d.Cusp>";
        });

    // Material point: Smooth

    py::class_<SM::Smooth>(sm, "Smooth")

        .def(
            py::init<double, double, const xt::pytensor<double, 1>&, bool>(),
            "Elasto-plastic material point, with 'smooth' potentials.",
            py::arg("K"),
            py::arg("G"),
            py::arg("epsy"),
            py::arg("init_elastic") = true)

        .def("K", &SM::Smooth::K, "Returns the bulk modulus.")
        .def("G", &SM::Smooth::G, "Returns the shear modulus.")
        .def("epsy", &SM::Smooth::epsy, "Returns the yield strains.")

        .def("refQPotChunked",
             &SM::Smooth::refQPotChunked,
             "Returns a reference underlying QPot::Chunked model.",
             py::return_value_policy::reference_internal)

        .def("setStrain", &SM::Smooth::setStrain<xt::pytensor<double, 2>>, "Set current strain tensor.")
        .def("strain", &SM::Smooth::strain<xt::pytensor<double, 2>>, "Returns strain tensor.", py::arg("ret"))
        .def("stress", &SM::Smooth::stress<xt::pytensor<double, 2>>, "Returns stress tensor.", py::arg("ret"))
        .def("tangent", &SM::Smooth::tangent<xt::pytensor<double, 4>>, "Returns tangent stiffness.", py::arg("ret"))
        .def("Strain", &SM::Smooth::Strain, "Returns strain tensor.")
        .def("Stress", &SM::Smooth::Stress, "Returns stress tensor.")
        .def("Tangent", &SM::Smooth::Tangent, "Returns tangent stiffness.")

        .def(
            "currentIndex",
            &SM::Smooth::currentIndex,
            "Returns the potential index, for last known strain.")

        .def(
            "currentYieldLeft",
            py::overload_cast<>(&SM::Smooth::currentYieldLeft, py::const_),
            "Returns the yield strain to the left, for last known strain.")

        .def(
            "currentYieldRight",
            py::overload_cast<>(&SM::Smooth::currentYieldRight, py::const_),
            "Returns the yield strain to the right, for last known strain.")

        .def(
            "currentYieldLeft",
            py::overload_cast<size_t>(&SM::Smooth::currentYieldLeft, py::const_),
            "Returns the yield strain to the left, for last known strain.",
            py::arg("shift"))

        .def(
            "currentYieldRight",
            py::overload_cast<size_t>(&SM::Smooth::currentYieldRight, py::const_),
            "Returns the yield strain to the right, for last known strain.",
            py::arg("shift"))

        .def(
            "checkYieldBoundLeft",
            &SM::Smooth::checkYieldBoundLeft,
            "Check that 'the particle' is at least 'n' wells from the far-left.",
            py::arg("n") = 0)

        .def(
            "checkYieldBoundRight",
            &SM::Smooth::checkYieldBoundRight,
            "Check that 'the particle' is at least 'n' wells from the far-right.",
            py::arg("n") = 0)

        .def("epsp", &SM::Smooth::epsp, "Returns equivalent plastic strain.")
        .def("energy", &SM::Smooth::energy, "Returns the energy, for last known strain.")

        .def("__repr__", [](const SM::Smooth&) {
            return "<GMatElastoPlasticQPot.Cartesian2d.Smooth>";
        });

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
