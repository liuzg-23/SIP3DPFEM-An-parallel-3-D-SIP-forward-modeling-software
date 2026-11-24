// Flexible GMRES solver. This file is almost an exact copy of MFEM's FGMRESSolver
// in linalg/solvers.cpp. I do this for extending the FGMRESSolver::Mult function,
// so I can compute and output the relative residual during the iteration procedure.

// The following three inline functions are the direct copies of MFEM's FGMRESSolver
// in linalg/solvers.cpp. For the FGMRESSolver::Mult function, I just did a minor
// modifications to compute and output the relative residual.

// Author:     Hongbo Yao
// Institute:  School of Geosciences and Info-Physics,
//             Central South University (CSU)
// Email:      yaohongbo@csu.edu.cn
// Date:       2020/9/6

// GitHub Page: https://github.com/hongbo-yao
// Researchgate Page: https://www.researchgate.net/profile/Hongbo_Yao2


// The following are a copy of the header's introduction in MFEM's linalg/solvers.cpp:

// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.


#include "fgmres.h"

#include <iostream>
#include <iomanip>

FGMRES::FGMRES():FGMRESSolver()
{

}



FGMRES::FGMRES(MPI_Comm _comm, int _myid):FGMRESSolver(_comm)
{
   myid=_myid;
}


void FGMRES::Mult(const Vector &b, Vector &x) const
{
   DenseMatrix H(m+1,m);
   Vector s(m+1), cs(m+1), sn(m+1);
   Vector r(b.Size());

   int i, j, k;


   if (iterative_mode)
   {
      oper->Mult(x, r);
      subtract(b,r,r);
   }
   else
   {
      x = 0.;
      r = b;
   }
   double beta = Norm(r);  // beta = ||r||
   MFEM_ASSERT(IsFinite(beta), "beta = " << beta);

   final_norm = std::max(rel_tol*beta, abs_tol);

   if (beta <= final_norm)
   {
      final_norm = beta;
      final_iter = 0;
      converged = 1;
      return;
   }

   /*
   if (PrintLevel == 1)
   {
      mfem::out << "   Pass: " << std::setw(2) << 1
                << "   Iteration: " << std::setw(3) << 0
                << "  || r || = " << beta << std::endl;
   }
   */

   // Hongbo Yao, 2020/09/06
   double norm_b = Norm(b);
   if (PrintLevel == 1)
   {
      if(myid==0){
         mfem::out << "   Pass: " << std::setw(2) << 1
                << "   Iteration: " << std::setw(3) << 0
                << "   ||r|| = " << beta << "\t"
                << "||r||/||b|| = " << beta/norm_b << std::endl;
      }
   }

   Monitor(0, beta, r, x);

   Array<Vector*> v(m+1);
   Array<Vector*> z(m+1);
   for (i= 0; i<=m; i++)
   {
      v[i] = NULL;
      z[i] = NULL;
   }

   j = 1;
   while (j <= max_iter)
   {
      if (v[0] == NULL) { v[0] = new Vector(b.Size()); }
      (*v[0]) = 0.0;
      v[0] -> Add (1.0/beta, r);   // v[0] = r / ||r||
      s = 0.0; s(0) = beta;

      for (i = 0; i < m && j <= max_iter; i++, j++)
      {

         if (z[i] == NULL) { z[i] = new Vector(b.Size()); }
         (*z[i]) = 0.0;

         if (prec)
         {
            prec->Mult(*v[i], *z[i]);
         }
         else
         {
            (*z[i]) = (*v[i]);
         }
         oper->Mult(*z[i], r);

         for (k = 0; k <= i; k++)
         {
            H(k,i) = Dot( r, *v[k]); // H(k,i) = r * v[k]
            r.Add(-H(k,i), (*v[k])); // r -= H(k,i) * v[k]
         }

         H(i+1,i)  = Norm(r);       // H(i+1,i) = ||r||
         if (v[i+1] == NULL) { v[i+1] = new Vector(b.Size()); }
         (*v[i+1]) = 0.0;
         v[i+1] -> Add (1.0/H(i+1,i), r); // v[i+1] = r / H(i+1,i)

         for (k = 0; k < i; k++)
         {
            ApplyPlaneRotation(H(k,i), H(k+1,i), cs(k), sn(k));
         }

         GeneratePlaneRotation(H(i,i), H(i+1,i), cs(i), sn(i));
         ApplyPlaneRotation(H(i,i), H(i+1,i), cs(i), sn(i));
         ApplyPlaneRotation(s(i), s(i+1), cs(i), sn(i));

         double resid = fabs(s(i+1));
         MFEM_ASSERT(IsFinite(resid), "resid = " << resid);
         /*
         if (PrintLevel == 1)
         {
            mfem::out << "   Pass: " << std::setw(2) << (j-1)/m+1
                      << "   Iteration: " << std::setw(3) << j
                      << "  || r || = " << resid << std::endl;
         }
         */
         // Hongbo Yao, 2020/09/06
         if (PrintLevel == 1)
         {
            if(myid==0){
               mfem::out << "   Pass: " << std::setw(2) << (j-1)/m+1
                        << "   Iteration: " << std::setw(3) << j
                        << "   ||r|| = " << resid << "\t"
                        << "||r||/||b|| = " << resid/norm_b << std::endl;
            }
         }
         Monitor(j, resid, r, x, resid <= final_norm);

         if (resid <= final_norm)
         {
            Update(x, i, H, s, z);
            final_norm = resid;
            final_iter = j;
            converged = 1;

            if (PrintLevel == 2)
            {
               if(myid==0){
                  mfem::out << "Number of FGMRES iterations: " << final_iter << std::endl;
               }
            }

            for (i= 0; i<=m; i++)
            {
               if (v[i]) { delete v[i]; }
               if (z[i]) { delete z[i]; }
            }
            return;
         }
      }

      if (PrintLevel == 1)
      {
         if(myid==0){
            mfem::out << "Restarting..." << std::endl;
         }
      }

      Update(x, i-1, H, s, z);

      oper->Mult(x, r);
      subtract(b,r,r);
      beta = Norm(r);
      MFEM_ASSERT(IsFinite(beta), "beta = " << beta);
      if (beta <= final_norm)
      {
         final_norm = beta;
         final_iter = j;
         converged = 1;

         if (PrintLevel == 2)
         {
            if(myid==0){
               mfem::out << "Number of FGMRES iterations: " << final_iter << std::endl;
            }
         }

         for (i= 0; i<=m; i++)
         {
            if (v[i]) { delete v[i]; }
            if (z[i]) { delete z[i]; }
         }
         return;
      }
   }

   for (i = 0; i <= m; i++)
   {
      if (v[i]) { delete v[i]; }
      if (z[i]) { delete z[i]; }
   }
   converged = 0;

   if (PrintLevel >= 0)
   {
      if(myid==0){
         mfem::out << "FGMRES: No convergence!" << std::endl;
      }
   }

   return;
}
