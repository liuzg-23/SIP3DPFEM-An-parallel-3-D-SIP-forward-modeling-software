// hypre's BoomerAMG parameters handler.
//
// This handler is the H1-scalar analogue of the AMS handler used in the
// FGMRES+AMS_EM project. The SIP forward problem is a scalar Poisson-type
// problem (nabla . (sigma grad phi)), discretized with H1 nodal elements, so
// the Auxiliary-space Maxwell Solver (AMS, which is designed for H(curl)
// curl-curl operators) cannot be applied here. Instead, each real/imaginary
// diagonal block is preconditioned with hypre's BoomerAMG.

#ifndef _HypreAMGHandler_H
#define _HypreAMGHandler_H

#include "mfem.hpp"

#include <string>

using namespace mfem;

class HypreAMGHandler
{
public:
   //------------------------------ FGMRES options -----------------------------
   int fgmres_maxit;
   double fgmres_primal_tol;  // relative tolerance for primal problem, e.g. 1e-8
   int fgmres_restart;
   int fgmres_print_level;

   //--------------------- BoomerAMG preconditioner options --------------------
   int amg_print_level;     // 0 = silent, 1 = basic, 2+ = verbose
   int amg_max_iter;        // V-cycles per block application (default 3; 2-5 typical)
   double amg_tol;          // reserved: the preconditioner always uses tol = 0
                            // (convergence is controlled by FGMRES, not BoomerAMG)
   int amg_coarsen_type;    // 10 = HMIS, 8 = PMIS, 6 = Falgout, 0 = CLJP
   int amg_agg_levels;      // number of aggressive coarsening levels
   int amg_relax_type;      // 6 = symm. GS, 8 = l1-GS,
                            // 3 = GS, 18 = l1-Jacobi
   double amg_strong_threshold; // strength threshold: 0.25, 0.5, 0.8
   int amg_interp_type;     // 6 = extended+i, 0 = classical

   //------------------- Reserved (currently unused) options ------------------
   int pcg_maxit;           // reserved: inner PCG scheme was reverted
   double pcg_rel_tol;      // reserved: inner PCG scheme was reverted

   //------------------- Block preconditioner type ----------------------------
   int prec_type;           // 0 = block diagonal, 1 = block lower triangular

public:
   HypreAMGHandler(std::string amg_opts_file = "");
   ~HypreAMGHandler();

public:
   // load BoomerAMG options from file; if the file cannot be opened the
   // built-in defaults are kept.
   void load_amg_opts(std::string amg_opts_file);
};

#endif // _HypreAMGHandler_H
