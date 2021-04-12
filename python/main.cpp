/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/GMatElastoPlasticQPot

*/

#include <pybind11/pybind11.h>
#include <pyxtensor/pyxtensor.hpp>

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
            py::overload_cast<
                const xt::xtensor<double, S::rank>&,
                const xt::xtensor<double, S::rank>&>(&S::setElastic),
            "Set all points 'Elastic'.",
            py::arg("K"),
            py::arg("G"))

        .def(
            "setElastic",
            py::overload_cast<
                const xt::xtensor<bool, S::rank>&,
                const xt::xtensor<size_t, S::rank>&,
                const xt::xtensor<double, 1>&,
                const xt::xtensor<double, 1>&>(&S::setElastic),
            "Set specific entries 'Elastic'.",
            py::arg("I"),
            py::arg("idx"),
            py::arg("K"),
            py::arg("G"))

        .def(
            "setCusp",
            py::overload_cast<
                const xt::xtensor<bool, S::rank>&,
                const xt::xtensor<size_t, S::rank>&,
                const xt::xtensor<double, 1>&,
                const xt::xtensor<double, 1>&,
                const xt::xtensor<double, 2>&,
                bool>(&S::setCusp),
            "Set specific entries 'Cusp'.",
            py::arg("I"),
            py::arg("idx"),
            py::arg("K"),
            py::arg("G"),
            py::arg("epsy"),
            py::arg("init_elastic") = true)

        .def(
            "setSmooth",
            py::overload_cast<
                const xt::xtensor<bool, S::rank>&,
                const xt::xtensor<size_t, S::rank>&,
                const xt::xtensor<double, 1>&,
                const xt::xtensor<double, 1>&,
                const xt::xtensor<double, 2>&,
                bool>(&S::setSmooth),
            "Set specific entries 'Smooth'.",
            py::arg("I"),
            py::arg("idx"),
            py::arg("K"),
            py::arg("G"),
            py::arg("epsy"),
            py::arg("init_elastic") = true)

        .def(
            "setElastic",
            py::overload_cast<const xt::xtensor<bool, S::rank>&, double, double>(
                &S::setElastic),
            "Set specific entries 'Elastic'.",
            py::arg("I"),
            py::arg("K"),
            py::arg("G"))

        .def(
            "setCusp",
            py::overload_cast<
                const xt::xtensor<bool, S::rank>&,
                double,
                double,
                const xt::xtensor<double, 1>&,
                bool>(&S::setCusp),
            "Set specific entries 'Cusp'.",
            py::arg("I"),
            py::arg("K"),
            py::arg("G"),
            py::arg("epsy"),
            py::arg("init_elastic") = true)

        .def(
            "setSmooth",
            py::overload_cast<
                const xt::xtensor<bool, S::rank>&,
                double,
                double,
                const xt::xtensor<double, 1>&,
                bool>(&S::setSmooth),
            "Set specific entries 'Smooth'.",
            py::arg("I"),
            py::arg("K"),
            py::arg("G"),
            py::arg("epsy"),
            py::arg("init_elastic") = true)

        .def("setStrain", &S::setStrain, "Set strain tensors.", py::arg("Eps"))
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
            py::arg("shift"))

        .def(
            "CurrentYieldRight",
            py::overload_cast<size_t>(&S::CurrentYieldRight, py::const_),
            "Returns the yield strain to the right, for last known strain.",
            py::arg("shift"))

        .def(
            "NextYield",
            &S::NextYield,
            "Returns the next yield strain to either direction, for last known strain.",
            py::arg("shift"))

        .def("Epsp", &S::Epsp, "Get equivalent plastic strains.")
        .def("Energy", &S::Energy, "Get energies.")
        .def("getElastic", &S::getElastic, "Returns underlying Elastic model.")
        .def("getCusp", &S::getCusp, "Returns underlying Cusp model.")
        .def("getSmooth", &S::getSmooth, "Returns underlying Smooth model.")

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

template <class S, class T>
void add_deviatoric_overloads(T& module)
{
    module.def(
        "Deviatoric",
        static_cast<S (*)(const S&)>(&GMatElastoPlasticQPot::Cartesian2d::Deviatoric<S>),
        "Deviatoric part of a(n) (array of) tensor(s).",
        py::arg("A"));
}

template <class R, class S, class T>
void add_hydrostatic_overloads(T& module)
{
    module.def(
        "Hydrostatic",
        static_cast<R (*)(const S&)>(&GMatElastoPlasticQPot::Cartesian2d::Hydrostatic<S>),
        "Hydrostatic part of a(n) (array of) tensor(s).",
        py::arg("A"));
}

template <class R, class S, class T>
void add_epsd_overloads(T& module)
{
    module.def(
        "Epsd",
        static_cast<R (*)(const S&)>(
            &GMatElastoPlasticQPot::Cartesian2d::Epsd<S>),
        "Equivalent strain of a(n) (array of) tensor(s).",
        py::arg("A"));
}

template <class R, class S, class T>
void add_sigd_overloads(T& module)
{
    module.def(
        "Sigd",
        static_cast<R (*)(const S&)>(
            &GMatElastoPlasticQPot::Cartesian2d::Sigd<S>),
        "Equivalent stress of a(n) (array of) tensor(s).",
        py::arg("A"));
}

PYBIND11_MODULE(GMatElastoPlasticQPot, m)
{

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

    add_deviatoric_overloads<xt::xtensor<double, 4>>(sm);
    add_deviatoric_overloads<xt::xtensor<double, 3>>(sm);
    add_deviatoric_overloads<xt::xtensor<double, 2>>(sm);
    add_hydrostatic_overloads<xt::xtensor<double, 2>, xt::xtensor<double, 4>>(sm);
    add_hydrostatic_overloads<xt::xtensor<double, 1>, xt::xtensor<double, 3>>(sm);
    add_hydrostatic_overloads<xt::xtensor<double, 0>, xt::xtensor<double, 2>>(sm);
    add_epsd_overloads<xt::xtensor<double, 2>, xt::xtensor<double, 4>>(sm);
    add_epsd_overloads<xt::xtensor<double, 1>, xt::xtensor<double, 3>>(sm);
    add_epsd_overloads<xt::xtensor<double, 0>, xt::xtensor<double, 2>>(sm);
    add_sigd_overloads<xt::xtensor<double, 2>, xt::xtensor<double, 4>>(sm);
    add_sigd_overloads<xt::xtensor<double, 1>, xt::xtensor<double, 3>>(sm);
    add_sigd_overloads<xt::xtensor<double, 0>, xt::xtensor<double, 2>>(sm);

    // Material point: Elastic

    py::class_<SM::Elastic>(sm, "Elastic")

        .def(py::init<double, double>(), "Elastic material point.", py::arg("K"), py::arg("G"))

        .def("K", &SM::Elastic::K, "Returns the bulk modulus.")
        .def("G", &SM::Elastic::G, "Returns the shear modulus.")
        .def("setStrain", &SM::Elastic::setStrain<xt::xtensor<double, 2>>, "Set current strain tensor.")
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
            py::init<double, double, const xt::xtensor<double, 1>&, bool>(),
            "Elasto-plastic material point, with 'cusp' potentials.",
            py::arg("K"),
            py::arg("G"),
            py::arg("epsy"),
            py::arg("init_elastic") = true)

        .def("K", &SM::Cusp::K, "Returns the bulk modulus.")
        .def("G", &SM::Cusp::G, "Returns the shear modulus.")
        .def("epsy", &SM::Cusp::epsy, "Returns the yield strains.")
        .def("getQPot", &SM::Cusp::getQPot, "Returns underlying QPot model.")

        .def("refQPotStatic",
             &SM::Cusp::refQPotStatic,
             "Returns a reference underlying QPot::Static model.",
             py::return_value_policy::reference_internal)

        .def("setStrain", &SM::Cusp::setStrain<xt::xtensor<double, 2>>, "Set current strain tensor.")
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
            "nextYield",
            &SM::Cusp::nextYield,
            "Returns the next yield strain to either direction, for last known strain.",
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
            py::init<double, double, const xt::xtensor<double, 1>&, bool>(),
            "Elasto-plastic material point, with 'smooth' potentials.",
            py::arg("K"),
            py::arg("G"),
            py::arg("epsy"),
            py::arg("init_elastic") = true)

        .def("K", &SM::Smooth::K, "Returns the bulk modulus.")
        .def("G", &SM::Smooth::G, "Returns the shear modulus.")
        .def("epsy", &SM::Smooth::epsy, "Returns the yield strains.")
        .def("getQPot", &SM::Smooth::getQPot, "Returns underlying QPot model.")

        .def("refQPotStatic",
             &SM::Smooth::refQPotStatic,
             "Returns a reference underlying QPot::Static model.",
             py::return_value_policy::reference_internal)

        .def("setStrain", &SM::Smooth::setStrain<xt::xtensor<double, 2>>, "Set current strain tensor.")
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
            "nextYield",
            &SM::Smooth::nextYield,
            "Returns the next yield strain to either direction, for last known strain.",
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
