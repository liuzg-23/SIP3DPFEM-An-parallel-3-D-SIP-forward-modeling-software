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

#ifndef _FGMRES_H
#define _FGMRES_H

#include "mfem.hpp"
using namespace mfem;

#include <mpi.h>
class FGMRES: public FGMRESSolver
{
public:
   FGMRES();
   FGMRES(MPI_Comm _comm,int _myid);
   int PrintLevel;
   int myid;
   void Mult(const Vector &b, Vector &x) const;
};



// ---------------------------------------------------------------------------------
// The following three inline functions are the direct copies of MFEM's FGMRESSolver
// in linalg/solvers.cpp.
inline void GeneratePlaneRotation(double &dx, double &dy,
                                  double &cs, double &sn)
{
   if (dy == 0.0)
   {
      cs = 1.0;
      sn = 0.0;
   }
   else if (fabs(dy) > fabs(dx))
   {
      double temp = dx / dy;
      sn = 1.0 / sqrt( 1.0 + temp*temp );
      cs = temp * sn;
   }
   else
   {
      double temp = dy / dx;
      cs = 1.0 / sqrt( 1.0 + temp*temp );
      sn = temp * cs;
   }
}



inline void ApplyPlaneRotation(double &dx, double &dy, double &cs, double &sn)
{
   double temp = cs * dx + sn * dy;
   dy = -sn * dx + cs * dy;
   dx = temp;
}



inline void Update(Vector &x, int k, DenseMatrix &h, Vector &s,
                   Array<Vector*> &v)
{
   Vector y(s);

   // Backsolve:
   for (int i = k; i >= 0; i--)
   {
      y(i) /= h(i,i);
      for (int j = i - 1; j >= 0; j--)
      {
         y(j) -= h(j,i) * y(i);
      }
   }

   for (int j = 0; j <= k; j++)
   {
      x.Add(y(j), *v[j]);
   }
}

#endif // _FGMRES_H
