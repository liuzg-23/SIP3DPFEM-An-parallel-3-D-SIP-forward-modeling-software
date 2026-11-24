

#include "goafem.h"
#include "mfemplus.h"
#include "post.h"

#include "fgmres.h"
#include "hypre_amg_handler.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>

GOAFEM::GOAFEM(std::vector<std::complex<double>> &att_cond,
               ParaHandler &para_handler_, std::shared_ptr<ParMesh> pmesh_,
               std::string output_file_)
    : cond_att(att_cond), para_handler(para_handler_), pmesh(pmesh_),
      output_file(output_file_) {
  this->init();
}

GOAFEM::~GOAFEM() {}

void GOAFEM::init() {
  sn = para_handler.sources.size();
  pfec.reset(new H1_FECollection(1, pmesh->Dimension()));
  pfes.reset(new ParFiniteElementSpace(pmesh.get(), pfec.get()));
  U.resize(sn);
  F.resize(sn);
  up.resize(sn);
  f.resize(sn);
}

void GOAFEM::reconstruct_FEM_terms() {

  a.reset(
      new ParSesquilinearForm(pfes.get(), ComplexOperator::BLOCK_SYMMETRIC));
  for (int i = 0; i < sn; i++) {
    up[i].reset(new ParComplexGridFunction(pfes.get()));
    *(up[i]) = 0.0;
    f[i].reset(
        new ParComplexLinearForm(pfes.get(), ComplexOperator::BLOCK_SYMMETRIC));
    f[i]->Vector::operator=(0.0);
  }
}

void GOAFEM::one_forward_iteration() {
  MPI_Comm mycomm = pmesh->GetComm();
  int myid, n_procs;
  MPI_Status status;
  MPI_Comm_size(mycomm, &n_procs);
  MPI_Comm_rank(mycomm, &myid);
  double timea1, timea2, timetotal1, timetotal2;
  timetotal1 = MPI_Wtime();
  this->set_bdr();
  this->set_sigma0();
  para_handler.find_point_tets(pmesh, para_handler.sites,
                               para_handler.local_sites_tets);
  ess_tdof_list.SetSize(0);
  timea1 = MPI_Wtime();
  int real_or_imag = 1; // 1:for real part;2:for imaginary part
  LVolume_coef_r.reset(new LVCoefficient(cond_att, real_or_imag));
  real_or_imag = 2;
  LVolume_coef_i.reset(new LVCoefficient(cond_att, real_or_imag));
  Vector sources_center = para_handler.get_sources_center(); //源的中心
  real_or_imag = 1;
  LGamma1_coef_r.reset(
      new LGamma1Coefficient(cond_att, sources_center, pmesh, real_or_imag));
  real_or_imag = 2;
  LGamma1_coef_i.reset(
      new LGamma1Coefficient(cond_att, sources_center, pmesh, real_or_imag));
  a->AddDomainIntegrator(new DiffusionIntegrator(*LVolume_coef_r),
                         new DiffusionIntegrator(*LVolume_coef_i));
  a->AddBoundaryIntegrator(new MassIntegrator(*LGamma1_coef_r),
                           new MassIntegrator(*LGamma1_coef_i), gamma1_bdr);
  a->Assemble();
  a->Finalize();

  if (myid == 0)
    std::cout << "  The number of the sources： " << para_handler.sources.size()
              << std::endl;
  for (int i = 0; i < para_handler.sources.size(); i++) {
    Vector source_i(3);
    source_i = 0.0;
    for (int j = 0; j < 3; j++) {
      source_i[j] = para_handler.sources[i](j);
    }
    int if_s_on_surface = para_handler.is_source_on_surface[i];
    real_or_imag = 1;
    RGamma0_coef_r.reset(new RGamma0Coefficient(sigma0[i], source_i,
                                                if_s_on_surface, real_or_imag));
    RGamma1_coef_r.reset(new RGamma1Coefficient(
        cond_att, sigma0[i], source_i, if_s_on_surface, pmesh, real_or_imag));
    real_or_imag = 2;
    RGamma0_coef_i.reset(new RGamma0Coefficient(sigma0[i], source_i,
                                                if_s_on_surface, real_or_imag));
    RGamma1_coef_i.reset(new RGamma1Coefficient(
        cond_att, sigma0[i], source_i, if_s_on_surface, pmesh, real_or_imag));
    f[i]->AddBoundaryIntegrator(new BoundaryLFIntegrator(*RGamma0_coef_r),
                                new BoundaryLFIntegrator(*RGamma0_coef_i),
                                gamma0_bdr);
    f[i]->AddBoundaryIntegrator(new BoundaryLFIntegrator(*RGamma1_coef_r),
                                new BoundaryLFIntegrator(*RGamma1_coef_i),
                                gamma1_bdr);
    f[i]->Assemble();
  }

  std::vector<Vector> source(para_handler.sources.size());
  for (int i = 0; i < para_handler.sources.size(); i++) {
    source[i].SetSize(3);
    for (int j = 0; j < 3; j++) {
      source[i][j] = para_handler.sources[i](j);
    }
  }
  assemble_RV_linearform(f, source);
  for (int i = 0; i < para_handler.sources.size(); i++) {
    a->FormLinearSystem(ess_tdof_list, *(up[i]), *(f[i]), A, U[i], F[i]);
  }
  timetotal2 = MPI_Wtime();
  if (myid == 0)
    std::cout << "  Runtime for forming the linear system："
              << timetotal2 - timetotal1 << std::endl;
  if (myid == 0) {
    std::cout << "  Solving the linear system of equation ..." << std::endl;
  }
  double time_s1, time_s2;
  time_s1 = MPI_Wtime();
  this->solve();
  time_s2 = MPI_Wtime();
  if (myid == 0)
    std::cout << "  Runtime for solving the linear system："
              << time_s2 - time_s1 << std::endl;
  if (myid == 0) {
    std::cout << "  Post processing...\n";
  }
  Post post(pmesh, pfes, up, para_handler);
  post.compute_sites_up(cond_att, sigma0, para_handler.is_source_on_surface);
  post.save_as_one(sigma0, para_handler.is_source_on_surface);
  if (myid == 0) {
    post.compute_rho(para_handler.survey_input_file, output_file);
  }
}

void GOAFEM::solve() {
  MPI_Comm comm = pmesh->GetComm();
  int myid;
  MPI_Comm_rank(comm, &myid);
  const std::string linear_solver = para_handler.linear_solver;

  if (linear_solver == "mumps") {
    if (myid == 0) {
      std::cout << "  Using MUMPS direct solver ..." << std::endl;
    }
    HypreParMatrix *_A = A.As<ComplexHypreParMatrix>()->GetSystemMatrix();
    MUMPSSolver mumps_solver = MUMPSSolver(comm);
    mumps_solver.SetMatrixSymType(MUMPSSolver::MatType::SYMMETRIC_INDEFINITE);
    mumps_solver.SetPrintLevel(0);
    mumps_solver.SetOperator(*_A);
    for (int i = 0; i < F.size(); i++) {
      mumps_solver.Mult(F[i], U[i]);
      a->RecoverFEMSolution(U[i], *(f[i]), *(up[i]));
    }
    delete _A;
  } else if (linear_solver == "fgmres_amg") {
    if (myid == 0) {
      std::cout << "  Using FGMRES + block-diagonal BoomerAMG solver ..."
                << std::endl;
    }
    // Preconditioner matrix: the real part of the operator.
    ParBilinearForm pc_bform(pfes.get());
    pc_bform.AddDomainIntegrator(new DiffusionIntegrator(*LVolume_coef_r));
    pc_bform.AddBoundaryIntegrator(new MassIntegrator(*LGamma1_coef_r),
                                   gamma1_bdr);
    pc_bform.Assemble();
    pc_bform.Finalize();
    OperatorHandle pc_mat;
    pc_bform.FormSystemMatrix(ess_tdof_list, pc_mat);

    // Block preconditioner for the 2x2 real block system [Ar -Ai; Ai Ar].
    HypreAMGHandler amg_handler(para_handler.gmres_amg_file);
    Array<int> blockTrueOffsets(3);
    blockTrueOffsets[0] = 0;
    blockTrueOffsets[1] = pc_mat.Ptr()->Height();
    blockTrueOffsets[2] = pc_mat.Ptr()->Height();
    blockTrueOffsets.PartialSum();

    // Each SPD diagonal block Ar is approximately inverted by BoomerAMG
    // running `amg_max_iter` V-cycles.
    std::unique_ptr<HypreBoomerAMG> pc0(
        new HypreBoomerAMG(*pc_mat.As<HypreParMatrix>()));
    std::unique_ptr<HypreBoomerAMG> pc1(
        new HypreBoomerAMG(*pc_mat.As<HypreParMatrix>()));
    setup_amg_opts(*pc0, *pc1);

    // Choose the block preconditioner:
    //   0 = block diagonal diag(Ar, Ar),
    //   1 = block lower triangular [Ar 0; Ai Ar] (captures the -Ai/Ai coupling,
    //       so FGMRES converges substantially faster).
    std::unique_ptr<Solver> BlockPrec;
    if (amg_handler.prec_type == 1) {
      std::unique_ptr<BlockLowerTriangularPreconditioner> p(
          new BlockLowerTriangularPreconditioner(blockTrueOffsets));
      p->SetDiagonalBlock(0, pc0.get());
      p->SetDiagonalBlock(1, pc1.get());
      p->SetBlock(1, 0, &(A.As<ComplexHypreParMatrix>()->imag()));
      p->owns_blocks = 0; // pc0/pc1 owned by unique_ptr, Ai owned by A
      BlockPrec = std::move(p);
      if (myid == 0) {
        std::cout << "  Block preconditioner: lower triangular" << std::endl;
      }
    } else {
      std::unique_ptr<BlockDiagonalPreconditioner> p(
          new BlockDiagonalPreconditioner(blockTrueOffsets));
      p->SetDiagonalBlock(0, pc0.get());
      p->SetDiagonalBlock(1, pc1.get());
      p->owns_blocks = 0; // pc0/pc1 owned by unique_ptr
      BlockPrec = std::move(p);
      if (myid == 0) {
        std::cout << "  Block preconditioner: diagonal" << std::endl;
      }
    }

    FGMRES fgmres(comm, myid);
    fgmres.SetOperator(*A.Ptr());
    fgmres.SetPreconditioner(*BlockPrec);
    fgmres.SetMaxIter(amg_handler.fgmres_maxit);
    fgmres.SetRelTol(amg_handler.fgmres_primal_tol);
    fgmres.SetAbsTol(0.0);
    fgmres.SetKDim(amg_handler.fgmres_restart);
    fgmres.PrintLevel = amg_handler.fgmres_print_level;

    for (int i = 0; i < F.size(); i++) {
      fgmres.Mult(F[i], U[i]);
      a->RecoverFEMSolution(U[i], *(f[i]), *(up[i]));
    }
  } else {
    if (myid == 0) {
      std::cerr << "Unknown linear solver '" << linear_solver
                << "'. Expected \"mumps\" or \"fgmres_amg\"." << std::endl;
    }
    MPI_Abort(comm, 1);
  }
}

void GOAFEM::setup_amg_opts(HypreBoomerAMG &pc0, HypreBoomerAMG &pc1) {
  HypreAMGHandler amg_handler(para_handler.gmres_amg_file);

  pc0.SetPrintLevel(amg_handler.amg_print_level);
  pc0.SetMaxIter(amg_handler.amg_max_iter);
  // IMPORTANT: the BoomerAMG blocks act as a preconditioner (a few V-cycles),
  // not as a standalone solver. Setting a nonzero tolerance would make hypre
  // flag HYPRE_ERROR_CONV (256) when the tolerance is not reached and MFEM
  // would abort. Convergence of the overall system is controlled by FGMRES.
  pc0.SetTol(0.0);
  pc0.SetCoarsening(amg_handler.amg_coarsen_type);
  pc0.SetAggressiveCoarsening(amg_handler.amg_agg_levels);
  pc0.SetRelaxType(amg_handler.amg_relax_type);
  pc0.SetStrengthThresh(amg_handler.amg_strong_threshold);
  pc0.SetInterpolation(amg_handler.amg_interp_type);

  pc1.SetPrintLevel(amg_handler.amg_print_level);
  pc1.SetMaxIter(amg_handler.amg_max_iter);
  pc1.SetTol(0.0);
  pc1.SetCoarsening(amg_handler.amg_coarsen_type);
  pc1.SetAggressiveCoarsening(amg_handler.amg_agg_levels);
  pc1.SetRelaxType(amg_handler.amg_relax_type);
  pc1.SetStrengthThresh(amg_handler.amg_strong_threshold);
  pc1.SetInterpolation(amg_handler.amg_interp_type);
}

void GOAFEM::set_sigma0() {
  // find sources
  Array<int> local_sources_tets;
  para_handler.find_point_tets(pmesh, para_handler.sources, local_sources_tets);
  MPI_Status status;
  MPI_Comm mycomm = pmesh->GetComm();
  int myid, n_procs;
  MPI_Comm_size(mycomm, &n_procs);
  MPI_Comm_rank(mycomm, &myid);
  global_initial_sources_tets.SetSize(sn);
  std::unique_ptr<int> msg_tets;
  if (myid == 0) {
    for (int i = 0; i < sn; i++) {
      int tet_id = local_sources_tets[i];
      if (tet_id != -2) {
        global_initial_sources_tets[i] = pmesh->GetAttribute(tet_id) - 1;
      } else {
        global_initial_sources_tets[i] = 0;
      }
    }
    for (int p = 1; p < n_procs; p++) {
      msg_tets.reset(new int[sn]);
      MPI_Recv(msg_tets.get(), sn, MPI_INT, p, 105, mycomm, &status);
      // Append the data of other threads
      for (int i = 0; i < sn; i++) {
        global_initial_sources_tets[i] += msg_tets.get()[i];
      }
    }
  } else {
    msg_tets.reset(new int[sn]);
    for (int i = 0; i < sn; i++) {
      int tet_id = local_sources_tets[i];
      if (tet_id != -2) {
        msg_tets.get()[i] = pmesh->GetAttribute(tet_id) - 1;
      } else {
        msg_tets.get()[i] = 0;
      }
    }
    MPI_Send(msg_tets.get(), sn, MPI_INT, 0, 105, mycomm);
  }
  std::unique_ptr<int> sources_tets_g(new int[sn]);
  if (myid == 0) {
    for (int j = 0; j < sn; j++) {
      sources_tets_g.get()[j] = global_initial_sources_tets[j];
    }
    for (int p = 1; p < n_procs; p++) {
      MPI_Send(sources_tets_g.get(), sn, MPI_INT, p, 6, mycomm);
    }
  } else {
    MPI_Recv(sources_tets_g.get(), sn, MPI_INT, 0, 6, mycomm, &status);
    for (int j = 0; j < sn; j++) {
      global_initial_sources_tets[j] = sources_tets_g.get()[j];
    }
  }
  sigma0.clear();
  sigma0.resize(para_handler.sources.size());
  for (int k = 0; k < para_handler.sources.size(); k++) {
    sigma0[k] = cond_att[global_initial_sources_tets[k]];
  }
}

void GOAFEM::set_bdr() {
  myassert(pmesh->bdr_attributes[0] == 1);
  myassert(pmesh->bdr_attributes[1] == 2);
  gamma0_bdr.SetSize(pmesh->bdr_attributes.Size());
  gamma1_bdr.SetSize(pmesh->bdr_attributes.Size());
  gamma0_bdr = 0;
  gamma1_bdr = 0;
  gamma0_bdr[0] = 1;
  gamma1_bdr[1] = 1;
}

void GOAFEM::update() {
  pfes->Update();
  a->Update();
  for (int i = 0; i < para_handler.sources.size(); i++) {
    f[i]->Update();
    up[i]->Update();
  }
}

HYPRE_Int GOAFEM::get_problem_dofs() { return pfes->GlobalTrueVSize(); }

void GOAFEM::print_problem_size() {
  int myid;
  MPI_Comm_rank(pmesh->GetComm(), &myid);
  HYPRE_Int size = pfes->GlobalTrueVSize();
  long n_global_elems = pmesh->GetGlobalNE();
  if (myid == 0) {
    std::cout << "  Number of tetrahedral elements: " << n_global_elems << "\n";
    std::cout << "  Degrees of Freedom (DOFs): " << size << "\n";
  }
}

void GOAFEM::set_source_underground(std::vector<int> &is_source_on_surface) {
  myassert(sn > 0);
  is_source_on_surface.resize(sn);
  int count_invalid = 0;
  int NBE = pmesh->GetNBE();
  for (int k = 0; k < NBE; k++) {
    int attrite = pmesh->GetBdrAttribute(k);
    if (attrite == 1) {
      ElementTransformation *f = pmesh->GetBdrElementTransformation(k);
      Vector normal(3);
      normal = 0.0;
      compute_normal(*f, normal);
      Vector face_center(3);
      face_center = 0.0;
      f->Transform(Geometries.GetCenter(f->GetGeometryType()), face_center);
      for (int s = 0; s < sn; s++) {
        int tet_id = global_initial_sources_tets[s];
        Vector center(3);
        center = 0.0;
        pmesh->GetElementCenter(tet_id, center);
        double distance_face_to_tet = length_two_point(face_center, center);

        if (distance_face_to_tet < 4) {
          Vector c(3);
          c = 0.0;
          for (int ii = 0; ii < 3; ii++) {
            c(ii) = para_handler.sources[s](ii);
          }
          Vector rc(3);
          rc = 0.0;
          subtract(c, face_center, rc);
          double value = std::abs(rc * normal);
          // if (value == 0.0) {
          if (value <= 1e-12) {
            count_invalid++;
            is_source_on_surface[s] = 1;
          }
        }
      }
    }
  }
  if (is_source_on_surface[0] == 1)
    std::cout << "count_invalid: " << count_invalid << std::endl;
  int myid;
  MPI_Comm_rank(pmesh->GetComm(), &myid);
  if (myid == 0) {
    std::cout << "is_source_on_surface: " << std::endl;
    for (int k = 0; k < sn; k++) {
      std::cout << is_source_on_surface[k] << " ";
    }
    std::cout << std::endl;
  }
}

void GOAFEM::check_source_underground(std::vector<int> &is_source_on_surface) {
  myassert(is_source_on_surface.size() == sn);
  std::map<int, int> bdrTets2Faces;
  bdrTets2Faces.clear();
  int NBE = pmesh->GetNBE();
  for (int k = 0; k < NBE; k++) {
    int attrite = pmesh->GetBdrAttribute(k);
    if (attrite == 1) {
      int tet_id = get_tetId_bdr(pmesh, k);
      bdrTets2Faces[tet_id] = k;
    }
  }
  for (int s = 0; s < sn; s++) {
    int tet_id = global_initial_sources_tets[s];
    auto it = bdrTets2Faces.find(tet_id);
    if (is_source_on_surface[s] == 1) {
      // myassert(it != bdrTets2Faces.end());
      if (it == bdrTets2Faces.end()) {
        std::cout << "The found tet containing the source " << s
                  << " at: " << para_handler.sources[s](0) << " "
                  << para_handler.sources[s](1) << " "
                  << para_handler.sources[s](2) << " is not on the surface bdr"
                  << std::endl;
      }
    }
  }
}

void GOAFEM::assemble_RV_linearform(
    std::vector<std::shared_ptr<ParComplexLinearForm>> f_,
    std::vector<Vector> source) {
  for (int e = 0; e < pmesh->GetNE(); e++) {
    const FiniteElement *fe = pfes->GetFE(e);
    const IntegrationRule *ir =
        &IntRules.Get(fe->GetGeomType(), 2 * fe->GetOrder()); // 4
    ElementTransformation *eltrans = pfes->GetElementTransformation(e);
    const int ndof = fe->GetDof();
    const int dim = fe->GetDim();
    std::vector<Vector> elemvec_r(f_.size()), elemvec_i(f_.size());
    for (int i = 0; i < f_.size(); i++) {
      elemvec_r[i].SetSize(ndof);
      elemvec_i[i].SetSize(ndof);
      elemvec_r[i] = 0.0;
      elemvec_i[i] = 0.0;
    }
    for (int q = 0; q < ir->GetNPoints(); ++q) {
      // std::cout << "T.Attribute: " << e << std::endl;
      const IntegrationPoint &ip = ir->IntPoint(q);
      eltrans->SetIntPoint(&ip);
      Vector x(3);
      x = 0.0;
      eltrans->Transform(ip, x);
      double WJ = eltrans->Weight() * ip.weight;
      // derivative shape function
      DenseMatrix shapeV(ndof, dim);
      fe->CalcPhysDShape(*eltrans, shapeV);
      int sn = para_handler.sources.size();
      for (int i = 0; i < sn; i++) {
        std::complex<double> sigma_0 = sigma0[i];
        Point_double xp, xi;
        for (int j = 0; j < 3; j++) {
          xp(j) = x(j);
          xi(j) = source[i](j);
        }
        Vector V_r(3), V_i(3);
        V_r = 0.0;
        V_i = 0.0;
        std::complex<double> temp = sigma0[i];
        temp -= cond_att[eltrans->Attribute - 1];
        int real_or_imag = 1;
        Point_double grad_U_i_s_r =
            gradient_Uis(sigma_0, xp, xi, para_handler.is_source_on_surface[i],
                         real_or_imag);
        real_or_imag = 2;
        Point_double grad_U_i_s_i =
            gradient_Uis(sigma_0, xp, xi, para_handler.is_source_on_surface[i],
                         real_or_imag);
        Point_double point_temp_r =
            grad_U_i_s_r * temp.real() - grad_U_i_s_i * temp.imag();
        Point_double point_temp_i =
            grad_U_i_s_i * temp.real() + grad_U_i_s_r * temp.imag();
        for (int k = 0; k < 3; k++) {
          V_r(k) = point_temp_r(k);
          V_i(k) = -1.0 * point_temp_i(k);
        }
        // 4*3 * 3*3 * 3*1
        for (int k = 0; k < ndof; k++) // loop all dofs
        {
          elemvec_r[i][k] +=
              WJ * (shapeV(k, 0) * V_r(0) + shapeV(k, 1) * V_r(1) +
                    shapeV(k, 2) * V_r(2));
          elemvec_i[i][k] +=
              WJ * (shapeV(k, 0) * V_i(0) + shapeV(k, 1) * V_i(1) +
                    shapeV(k, 2) * V_i(2));
        }
      }
    } // q loop
    Array<int> vdofs;
    pfes->GetElementDofs(e, vdofs);
    for (int i = 0; i < sn; i++) {
      f[i]->real().AddElementVector(vdofs, elemvec_r[i]);
      f[i]->imag().AddElementVector(vdofs, elemvec_i[i]);
    }
  }
}