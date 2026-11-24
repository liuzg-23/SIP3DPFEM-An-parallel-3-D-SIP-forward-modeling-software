// hypre's BoomerAMG parameters handler.

#include "hypre_amg_handler.h"
#include <fstream>

HypreAMGHandler::HypreAMGHandler(std::string amg_opts_file)
{
   // Default options (sensible for a scalar Poisson-type SIP problem).
   fgmres_maxit       = 1000;
   fgmres_primal_tol  = 1e-8;
   fgmres_restart     = 50;
   fgmres_print_level = 1;

   amg_print_level     = 0;
   amg_max_iter        = 3;    // a few V-cycles per block application (stronger)
   amg_tol              = 0.0;   // reserved: ignored, preconditioner uses tol = 0
   amg_coarsen_type     = 10;   // HMIS
   amg_agg_levels       = 1;
   amg_relax_type       = 8;    // l1-GS
   amg_strong_threshold = 0.25;
   amg_interp_type      = 6;    // extended+i

   pcg_maxit           = 10;
   pcg_rel_tol         = 1e-2;

   prec_type           = 1;    // 1 = block lower-triangular (default), 0 = diagonal

   if (!amg_opts_file.empty())
   {
      load_amg_opts(amg_opts_file);
   }
}

HypreAMGHandler::~HypreAMGHandler()
{

}

void HypreAMGHandler::load_amg_opts(std::string amg_opts_file)
{
   std::ifstream amg_opts_ifs(amg_opts_file);
   if (!amg_opts_ifs.good())
   {
      return; // keep the built-in defaults
   }
   std::string tmp;

   //------------------------------ FGMRES options -----------------------------
   amg_opts_ifs >> tmp >> fgmres_maxit;
   amg_opts_ifs >> tmp >> fgmres_primal_tol;
   amg_opts_ifs >> tmp >> fgmres_restart;
   amg_opts_ifs >> tmp >> fgmres_print_level;

   //--------------------- BoomerAMG preconditioner options --------------------
   amg_opts_ifs >> tmp >> amg_print_level;
   amg_opts_ifs >> tmp >> amg_max_iter;
   amg_opts_ifs >> tmp >> amg_tol;
   amg_opts_ifs >> tmp >> amg_coarsen_type;
   amg_opts_ifs >> tmp >> amg_agg_levels;
   amg_opts_ifs >> tmp >> amg_relax_type;
   amg_opts_ifs >> tmp >> amg_strong_threshold;
   amg_opts_ifs >> tmp >> amg_interp_type;
   amg_opts_ifs >> tmp >> pcg_maxit;
   amg_opts_ifs >> tmp >> pcg_rel_tol;
   amg_opts_ifs >> tmp >> prec_type;

   amg_opts_ifs.close();
}
