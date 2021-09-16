
#include <GMatElastoPlasticQPot/Cartesian2d.h>
#include <GooseFEM/GooseFEM.h>
#include <GooseFEM/Matrix.h>
#include <fstream>
#include <xtensor/xcsv.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>

namespace GF = GooseFEM;
namespace QD = GooseFEM::Element::Quad4;
namespace GM = GMatElastoPlasticQPot::Cartesian2d;

// #define CHECK_MYSHORTCUT

class System {

private:
    // mesh parameters
    xt::xtensor<size_t, 2> m_conn;
    xt::xtensor<size_t, 2> m_conn_elastic;
    xt::xtensor<size_t, 2> m_conn_plastic;
    xt::xtensor<double, 2> m_coor;
    xt::xtensor<size_t, 2> m_dofs;
    xt::xtensor<size_t, 1> m_iip;

    // mesh dimensions
    size_t m_N; // == nelem_plas
    size_t m_nelem;
    size_t m_nelem_elas;
    size_t m_nelem_plas;
    size_t m_nne;
    size_t m_nnode;
    size_t m_ndim;
    size_t m_nip;

    // element sets
    xt::xtensor<size_t, 1> m_elastic;
    xt::xtensor<size_t, 1> m_plastic;

    // numerical quadrature
    QD::Quadrature m_quad;
    QD::Quadrature m_quad_elas;
    QD::Quadrature m_quad_plas;

    // convert vectors between 'nodevec', 'elemvec', ...
    GF::VectorPartitioned m_vector;
    GF::VectorPartitioned m_vector_elas;
    GF::VectorPartitioned m_vector_plas;

    // mass matrix
    GF::MatrixDiagonalPartitioned m_M;

    // damping matrix
    GF::MatrixDiagonal m_D;

    // material definition
    GM::Matrix m_material;
    GM::Matrix m_material_elas;
    GM::Matrix m_material_plas;

    // convergence check
    GF::Iterate::StopList m_stop = GF::Iterate::StopList(20);

    // time evolution
    double m_t = 0.0; // current time
    double m_dt; // time step

    // nodal displacements, velocities, and accelerations (current and last time-step)
    xt::xtensor<double, 2> m_u;
    xt::xtensor<double, 2> m_v;
    xt::xtensor<double, 2> m_a;
    xt::xtensor<double, 2> m_v_n;
    xt::xtensor<double, 2> m_a_n;

    // element vectors
    xt::xtensor<double, 3> m_ue;
    xt::xtensor<double, 3> m_fe;
    xt::xtensor<double, 3> m_ue_plas;
    xt::xtensor<double, 3> m_fe_plas;

    // nodal forces
#ifdef CHECK_MYSHORTCUT
    xt::xtensor<double, 2> m_fmaterial;
#endif
    xt::xtensor<double, 2> m_felas;
    xt::xtensor<double, 2> m_fplas;
    xt::xtensor<double, 2> m_fdamp;
    xt::xtensor<double, 2> m_fint;
    xt::xtensor<double, 2> m_fext;
    xt::xtensor<double, 2> m_fres;

    // integration point tensors
    xt::xtensor<double, 4> m_Eps;
    xt::xtensor<double, 4> m_Eps_elas;
    xt::xtensor<double, 4> m_Eps_plas;
    xt::xtensor<double, 4> m_Sig;
    xt::xtensor<double, 4> m_Sig_elas;
    xt::xtensor<double, 4> m_Sig_plas;

    // stiffness matrix
    GF::Matrix m_K_elas;

public:
    System(size_t N) : m_N(N)
    {

        // ----
        // mesh
        // ----

        double h = xt::numeric_constants<double>::PI;
        double L = h * static_cast<double>(m_N);

        GF::Mesh::Quad4::FineLayer mesh(m_N, m_N, h);

        m_coor = mesh.coor();
        m_conn = mesh.conn();
        m_dofs = mesh.dofs();

        m_nnode = m_coor.shape(0);
        m_ndim = m_coor.shape(1);
        m_nelem = m_conn.shape(0);
        m_nne = m_conn.shape(1);

        m_plastic = mesh.elementsMiddleLayer();
        m_elastic = xt::setdiff1d(xt::arange(m_nelem), m_plastic);

        m_nelem_elas = m_elastic.size();
        m_nelem_plas = m_plastic.size();
        m_conn_elastic = xt::view(m_conn, xt::keep(m_elastic), xt::all());
        m_conn_plastic = xt::view(m_conn, xt::keep(m_plastic), xt::all());

        auto left = mesh.nodesLeftOpenEdge();
        auto right = mesh.nodesRightOpenEdge();
        xt::view(m_dofs, xt::keep(right), 0) = xt::view(m_dofs, xt::keep(left), 0);
        xt::view(m_dofs, xt::keep(right), 1) = xt::view(m_dofs, xt::keep(left), 1);

        auto top = mesh.nodesTopEdge();
        auto bottom = mesh.nodesBottomEdge();
        size_t nfix = top.size();
        m_iip = xt::empty<decltype(m_iip)::value_type>({2 * m_ndim * nfix});
        xt::view(m_iip, xt::range(0 * nfix, 1 * nfix)) = xt::view(m_dofs, xt::keep(bottom), 0);
        xt::view(m_iip, xt::range(1 * nfix, 2 * nfix)) = xt::view(m_dofs, xt::keep(bottom), 1);
        xt::view(m_iip, xt::range(2 * nfix, 3 * nfix)) = xt::view(m_dofs, xt::keep(top), 0);
        xt::view(m_iip, xt::range(3 * nfix, 4 * nfix)) = xt::view(m_dofs, xt::keep(top), 1);

        m_vector = GF::VectorPartitioned(m_conn, m_dofs, m_iip);
        m_vector_elas = GF::VectorPartitioned(m_conn_elastic, m_dofs, m_iip);
        m_vector_plas = GF::VectorPartitioned(m_conn_plastic, m_dofs, m_iip);

        m_quad = QD::Quadrature(m_vector.AsElement(m_coor));
        m_quad_elas = QD::Quadrature(m_vector_elas.AsElement(m_coor));
        m_quad_plas = QD::Quadrature(m_vector_plas.AsElement(m_coor));
        m_nip = m_quad.nip();

        m_u = m_vector.AllocateNodevec(0.0);
        m_v = m_vector.AllocateNodevec(0.0);
        m_a = m_vector.AllocateNodevec(0.0);
        m_v_n = m_vector.AllocateNodevec(0.0);
        m_a_n = m_vector.AllocateNodevec(0.0);

        m_ue = m_vector.AllocateElemvec(0.0);
        m_fe = m_vector.AllocateElemvec(0.0);
        m_ue_plas = m_vector_plas.AllocateElemvec(0.0);
        m_fe_plas = m_vector_plas.AllocateElemvec(0.0);

#ifdef CHECK_MYSHORTCUT
        m_fmaterial = m_vector.AllocateNodevec(0.0);
#endif
        m_felas = m_vector.AllocateNodevec(0.0);
        m_fplas = m_vector.AllocateNodevec(0.0);
        m_fdamp = m_vector.AllocateNodevec(0.0);
        m_fint = m_vector.AllocateNodevec(0.0);
        m_fext = m_vector.AllocateNodevec(0.0);
        m_fres = m_vector.AllocateNodevec(0.0);

        m_Eps = m_quad.AllocateQtensor<2>(0.0);
        m_Sig = m_quad.AllocateQtensor<2>(0.0);
        m_Eps_elas = m_quad_elas.AllocateQtensor<2>(0.0);
        m_Sig_elas = m_quad_elas.AllocateQtensor<2>(0.0);
        m_Eps_plas = m_quad_plas.AllocateQtensor<2>(0.0);
        m_Sig_plas = m_quad_plas.AllocateQtensor<2>(0.0);

        // --------
        // material
        // --------

        // material parameters
        double c = 1.0;
        double G = 1.0;
        double K = 10.0 * G;
        double rho = G / std::pow(c, 2.0);
        double qL = 2.0 * xt::numeric_constants<double>::PI / L;
        double qh = 2.0 * xt::numeric_constants<double>::PI / h;
        double alpha = std::sqrt(2.0) * qL * c * rho;
        m_dt = 1.0 / (c * qh) / 10.0;

        // material definition
        m_material = GM::Matrix(m_nelem, m_nip);
        m_material_elas = GM::Matrix(m_nelem_elas, m_nip);
        m_material_plas = GM::Matrix(m_nelem_plas, m_nip);

        // assign elastic material points
        {
            xt::xtensor<size_t, 2> I = xt::zeros<size_t>({m_nelem, m_nip});
            xt::view(I, xt::keep(m_elastic), xt::all()) = 1ul;
            m_material.setElastic(I, K, G);

            I = xt::ones<size_t>({m_nelem_elas, m_nip});
            m_material_elas.setElastic(I, K, G);
        }

        // assign plastic material points
        {
            double k = 2.0;
            std::array<size_t, 2> shape = {m_N, 1000};
            xt::xtensor<double, 2> epsy = 1e-5 + 1e-3 * xt::random::weibull<double>(shape, k, 1.0);
            xt::view(epsy, xt::all(), 0) = 1e-5 + 1e-3 * xt::random::rand<double>({m_N});
            epsy = xt::cumsum(epsy, 1);

            xt::xtensor<size_t, 2> I = xt::zeros<size_t>({m_nelem, m_nip});
            xt::xtensor<size_t, 2> idx = xt::zeros<size_t>({m_nelem, m_nip});
            xt::view(I, xt::keep(m_plastic), xt::all()) = 1ul;
            xt::view(idx, xt::keep(m_plastic), xt::all()) =
                xt::arange<size_t>(m_N).reshape({-1, 1});
            xt::xtensor<double, 1> unit = xt::ones<double>({m_N});
            m_material.setCusp(I, idx, K * unit, G * unit, epsy);

            I = xt::ones<size_t>({m_N, m_nip});
            idx = xt::zeros<size_t>({m_N, m_nip});
            xt::view(idx, xt::all(), xt::all()) = xt::arange<size_t>(m_N).reshape({-1, 1});
            m_material_plas.setCusp(I, idx, K * unit, G * unit, epsy);
        }

        m_material.check();
        m_material_elas.check();
        m_material_plas.check();

        // -----------
        // mass matrix
        // -----------

        {
            auto x = m_vector.AsElement(m_coor);
            QD::Quadrature nodalQuad(x, QD::Nodal::xi(), QD::Nodal::w());
            xt::xtensor<double, 2> val_quad = rho * xt::ones<double>({m_nelem, nodalQuad.nip()});
            m_M = GF::MatrixDiagonalPartitioned(m_conn, m_dofs, m_iip);
            m_M.assemble(nodalQuad.Int_N_scalar_NT_dV(val_quad));
        }

        // --------------
        // damping matrix
        // --------------

        {
            auto x = m_vector.AsElement(m_coor);
            QD::Quadrature nodalQuad(x, QD::Nodal::xi(), QD::Nodal::w());
            xt::xtensor<double, 2> val_quad = alpha * xt::ones<double>({m_nelem, nodalQuad.nip()});
            m_D = GF::MatrixDiagonal(m_conn, m_dofs);
            m_D.assemble(nodalQuad.Int_N_scalar_NT_dV(val_quad));
        }

        // ------------------------
        // partial stiffness matrix
        // ------------------------

        xt::xtensor<double, 4> stmp;
        xt::xtensor<double, 6> ctmp;
        std::tie(stmp, ctmp) = m_material_elas.Tangent(m_Eps_elas);
        m_K_elas = GF::Matrix(m_conn_elastic, m_dofs);
        m_K_elas.assemble(m_quad_elas.Int_gradN_dot_tensor4_dot_gradNT_dV(ctmp));
    }

public:
    void computeStrainStress()
    {
        m_vector.asElement(m_u, m_ue);
        m_quad.symGradN_vector(m_ue, m_Eps);
        m_material.stress(m_Eps, m_Sig);
    }

public:
    void computeStrainStressForcesWeakLayer()
    {
        m_vector_plas.asElement(m_u, m_ue_plas);
        m_quad_plas.symGradN_vector(m_ue_plas, m_Eps_plas);
        m_material_plas.stress(m_Eps_plas, m_Sig_plas);
        m_quad_plas.int_gradN_dot_tensor2_dV(m_Sig_plas, m_fe_plas);
        m_vector_plas.assembleNode(m_fe_plas, m_fplas);
        m_K_elas.dot(m_u, m_felas);
    }

public:
    void timeStep()
    {
        // history

        m_t += m_dt;

        xt::noalias(m_v_n) = m_v;
        xt::noalias(m_a_n) = m_a;

        // new displacement

        xt::noalias(m_u) = m_u + m_dt * m_v + 0.5 * std::pow(m_dt, 2.) * m_a;

        computeStrainStressForcesWeakLayer();

#ifdef CHECK_MYSHORTCUT
        computeStrainStress();
        m_quad.int_gradN_dot_tensor2_dV(m_Sig, m_fe);
        m_vector.assembleNode(m_fe, m_fmaterial);

        if (!xt::allclose(m_fmaterial, m_felas + m_fplas)) {
            throw std::runtime_error("Forces");
        }
#endif

        // estimate new velocity, update corresponding force

        xt::noalias(m_v) = m_v_n + m_dt * m_a_n;

        m_D.dot(m_v, m_fdamp);

        // compute residual force & solve

        xt::noalias(m_fint) = m_felas + m_fplas + m_fdamp;

        m_vector.copy_p(m_fint, m_fext);

        xt::noalias(m_fres) = m_fext - m_fint;

        m_M.solve(m_fres, m_a);

        // re-estimate new velocity, update corresponding force

        xt::noalias(m_v) = m_v_n + .5 * m_dt * (m_a_n + m_a);

        m_D.dot(m_v, m_fdamp);

        // compute residual force & solve

        xt::noalias(m_fint) = m_felas + m_fplas + m_fdamp;

        m_vector.copy_p(m_fint, m_fext);

        xt::noalias(m_fres) = m_fext - m_fint;

        m_M.solve(m_fres, m_a);

        // new velocity, update corresponding force

        xt::noalias(m_v) = m_v_n + .5 * m_dt * (m_a_n + m_a);

        m_D.dot(m_v, m_fdamp);

        // compute residual force & solve

        xt::noalias(m_fint) = m_felas + m_fplas + m_fdamp;

        m_vector.copy_p(m_fint, m_fext);

        xt::noalias(m_fres) = m_fext - m_fint;

        m_M.solve(m_fres, m_a);
    }

public:
    xt::xtensor<double, 2> run()
    {
        xt::xtensor<double, 3> dF = xt::zeros<double>({1001, 2, 2});
        xt::view(dF, xt::range(1, dF.shape(0)), 0, 1) = 0.004 / 1000.0;

        xt::xtensor<double, 2> ret = xt::zeros<double>(std::array<size_t, 2>{dF.shape(0), 2});
        auto dV = m_quad.AsTensor<2>(m_quad.dV());

        for (size_t inc = 0; inc < dF.shape(0); ++inc) {

            for (size_t i = 0; i < m_nnode; ++i) {
                for (size_t j = 0; j < m_ndim; ++j) {
                    for (size_t k = 0; k < m_ndim; ++k) {
                        m_u(i, j) += dF(inc, j, k) * (m_coor(i, k) - m_coor(0, k));
                    }
                }
            }

            computeStrainStressForcesWeakLayer();

            for (size_t iiter = 0; iiter < 99999; ++iiter) {

                timeStep();

                if (m_stop.stop(xt::norm_l2(m_fres)() / xt::norm_l2(m_fext)(), 1e-5)) {
                    std::cout << inc << ", " << iiter << std::endl;
                    break;
                }
            }

            m_v.fill(0.0);
            m_a.fill(0.0);
            m_stop.reset();

            computeStrainStress();
            xt::xtensor_fixed<double, xt::xshape<2, 2>> Epsbar = xt::average(m_Eps, dV, {0, 1});
            xt::xtensor_fixed<double, xt::xshape<2, 2>> Sigbar = xt::average(m_Sig, dV, {0, 1});

            ret(inc, 0) = GM::Epsd(Epsbar);
            ret(inc, 1) = GM::Epsd(Sigbar);
        }

        return ret;
    }
};

int main(void)
{
    size_t N = std::pow(3, 4);

    System sys(N);

    auto ret = sys.run();

    std::ofstream outfile("friction.txt");
    xt::dump_csv(outfile, ret);
    outfile.close();

#ifdef CHECK_MYSHORTCUT
    std::cout << "Shortcut successfully checked" << std::endl;
#endif

    return 0;
}
