

#ifndef _GOAFEM_H
#define _GOAFEM_H

#include "em.h"
#include "mfem.hpp"
#include "parahandler.h"

#include <iostream>
#include <mpi.h>
#include <vector>

using namespace EM;
// using namespace std;
using namespace mfem;

class GOAFEM {
public:
  std::vector<std::complex<double>> cond_att;

  std::vector<std::complex<double>> sigma0; // size = number of sources

  ParaHandler &para_handler;

  std::shared_ptr<ParMesh> pmesh;
  std::string    output_file;
  std::shared_ptr<FiniteElementCollection> pfec;
  std::shared_ptr<ParFiniteElementSpace> pfes;

  // The weak formulation: a(u,V) = f(V)
  std::shared_ptr<ParSesquilinearForm> a;
  std::vector<std::shared_ptr<ParComplexLinearForm>> f;    // multiple sources
  std::vector<std::shared_ptr<ParComplexGridFunction>> up; // multiple sources

  // Create "marker arrays" to define the portions of boundary associated
  //    with each type of boundary condition. These arrays have an entry
  //    corresponding to each boundary attribute.  Placing a '1' in entry i
  //    marks attribute i+1 as being active, '0' is inactive.
  Array<int> gamma0_bdr;
  Array<int> gamma1_bdr;
  Array<int> ess_tdof_list;

  // Ccefficients of governing electric field  equation
  std::shared_ptr<Coefficient> LVolume_coef_r,LVolume_coef_i;  // For A
  std::shared_ptr<Coefficient> LGamma1_coef_r,LGamma1_coef_i;  // For A
  std::shared_ptr<Coefficient> RGamma0_coef_r,RGamma0_coef_i;  // For F
  std::shared_ptr<Coefficient> RGamma1_coef_r,RGamma1_coef_i;  // For F
  OperatorHandle A;
  std::vector<Vector> U, F; // primal problem AU = F

  // the number of the sources
  int sn;

  Array<int> global_initial_sources_tets;

public:
  GOAFEM(std::vector<std::complex<double>> &att_cond, ParaHandler &para_handler_,
         std::shared_ptr<ParMesh> pmesh, std::string output_file_);
  ~GOAFEM();

  void init();
  void reconstruct_FEM_terms();
  void one_forward_iteration();

  void set_sigma0();
  /*   Setup coefficient for left volume integral
  Setup coefficient for left distant surface integral
  Setup coefficient for right volume integral
  Setup coefficient for right distant surface integral
  Setup coefficient for air-earth surface integral */
  // void setup_integral_coefficient();
  void set_bdr();

  // Solve the linear system for all sources using either the direct MUMPS
  // solver or the FGMRES + block-triangular/diagonal BoomerAMG iterative solver
  // (selected through ParaHandler::linear_solver).
  void solve();

  // Configure the two BoomerAMG diagonal blocks of the block preconditioner.
  void setup_amg_opts(HypreBoomerAMG &pc0, HypreBoomerAMG &pc1);

  void solve_with_pcg(OperatorHandle &A, Vector &X, Vector &B, std::string pre,
                      int maxit);
  void solve_with_fgmres(OperatorHandle &A, Vector &X, Vector &B,
        std::string pre, int maxit);

  // Update all objects based on pmesh
  void update();

  HYPRE_Int get_problem_dofs();

  void print_problem_size();

  void set_source_underground(std::vector<int> &is_source_on_surface);
  void check_source_underground(std::vector<int> &is_source_on_surface);

  void assemble_RV_linearform(std::vector<std::shared_ptr<ParComplexLinearForm>> f_,
                              std::vector<Vector> source_);
};

#endif // _GOAFEM_H
