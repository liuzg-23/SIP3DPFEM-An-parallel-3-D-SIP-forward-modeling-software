

#include "em.h"
#include "mfem.hpp"
using namespace mfem;

//------------------common functions------

inline void compute_normal(mfem::ElementTransformation &T,
                           mfem::Vector &normal) {
  CalcOrtho(T.Jacobian(), normal);
  normal /= normal.Norml2(); // normal.unit();
}

inline int get_tetId_bdr(std::shared_ptr<ParMesh> pmesh, int ElementNo) {
  int tet_id = -1;
  int flag = -1;
  mfem::FaceElementTransformations *face_trans =
      pmesh->GetBdrFaceTransformations(ElementNo); //(int BdrElemNo)
  mfem::ElementTransformation *e1 = face_trans->Elem1;
  if (e1 != NULL) {
    tet_id = e1->Attribute - 1;
    flag *= -1;
  }
  mfem::ElementTransformation *e2 = face_trans->Elem2;
  if (e2 != NULL) {
    tet_id = e2->Attribute - 1;
    flag *= -1;
  }
  myassert(flag == 1 && tet_id != -1);
  return tet_id;
}

std::complex<double> B(Point_double &x,
                       Point_double &x_c,
                       std::complex<double> &tet_sigma);

std::complex<double> Beta(std::complex<double> B,
                          Point_double &x,
                          Point_double &x_c,
                          Point_double &n);
std::complex<double> Uis(std::complex<double> &sigma, Point_double &x, Point_double &x_s,
           int if_s_on_surface);
Point_double gradient_Uis(std::complex<double> &sigma, Point_double &x, Point_double &x_s,
                          int if_s_on_surface,int real_or_imag);

// This class inherits MFEM's class in linalg/solvers.h. I do this for modify
// the print part of CGSolver::Mult function. For more
// information and source code availability visit https://mfem.org. MFEM is free
// software; you can redistribute it and/or modify it under the terms of the
// BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#ifndef _SOLVERSPLUS_h
#define _SOLVERSPLUS_h
class SolversPlus : public CGSolver {

public:
  SolversPlus();
  SolversPlus(MPI_Comm comm_);

  // aim to print ||r|| instead of (B r, r) , respond to pcg_tol
  void Mult(const Vector &b, Vector &x);
};

#endif // _SOLVERSPLUS_h

// This class inherits MFEM's class in mesh/mesh.hpp. I do this for extending
// the print function This file is part of the MFEM library. For more
// information and source code availability visit https://mfem.org. MFEM is free
// software; you can redistribute it and/or modify it under the terms of the
// BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef _MESHPLUS_H
#define _MESHPLUS_H

// #include "mfem.hpp"
// using namespace mfem;

class Meshplus : public Mesh {

public:
  Meshplus();
  Meshplus(ParMesh *pmesh_);
  Meshplus(Mesh *mesh_);
  void PrintVTK_eta(std::ostream &out, Vector &other_attri);
  void write_out_node_info_vtk(std::ostream &out, Vector &node_data);
};

#endif // _MESHPLUS_H

// The following classes are the direct copies and modifies of MFEM's classes
// in fem/coefficient.hpp. I just did a minor
// modifications to compute different coeffients for assembling equations.

#ifndef MFEM_COEFFICIENTS_H
#define MFEM_COEFFICIENTS_H

// #include "em.h"
// #include "mfem.hpp"
// using namespace mfem;

class LVCoefficient : public Coefficient {
private:
  std::vector<double> mat;

public:
  /// Construct a coefficeint using matrix @a m for left side volume integral
  int real_or_imag_;
  LVCoefficient(std::vector<std::complex<double>> &m,int &real_or_imag) {
    mat.resize(m.size());
    real_or_imag_=real_or_imag;
    if(real_or_imag_==1){
      for(int i=0;i<m.size();i++){
        mat[i]=m[i].real();
      }
    }else{
      for(int i=0;i<m.size();i++){
        mat[i]=m[i].imag();
      }
    }
  }
  using Coefficient::Eval;
  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class LGamma1Coefficient : public Coefficient {
private:
  std::vector<std::complex<double>> mat;
  Vector x_c;
  // ParMesh *pmesh;
  std::shared_ptr<ParMesh> pmesh;

public:
  /// Construct a double coefficient for the surface integral in distant
  /// boundary face
  int real_or_imag_;
  LGamma1Coefficient(std::vector<std::complex<double>> &m, Vector sources_center,
                     //  ParMesh *pmesh_
                     std::shared_ptr<ParMesh> pmesh_,int &real_or_imag)
      : mat(m), x_c(sources_center), pmesh(pmesh_), real_or_imag_(real_or_imag) {}
  using Coefficient::Eval;
  /// Evaluate the coefficient at @a ip.
  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class RVCoefficient : public VectorCoefficient {
private:
  std::vector<double> mat;
  double sigma0;
  Vector x_i;
  int if_s_on_surface;
  // ParMesh *pmesh;

public:
  /// Construct a coeffient  using vector @a V for right side volume integral
  RVCoefficient(std::vector<double> &m, double sigma0_, Vector source_i,
                int if_s_on_surface_
                // ,ParMesh* pmesh_
                )
      : VectorCoefficient(3), mat(m), sigma0(sigma0_), x_i(source_i),
        if_s_on_surface(if_s_on_surface_)
  // ,pmesh(pmesh_)
  {}
  using VectorCoefficient::Eval;
  /// Evaluate the vector coefficient at @a ip.
  virtual void Eval(Vector &V, ElementTransformation &T,
                    const IntegrationPoint &ip);
};

class RGamma0Coefficient : public Coefficient {
private:
  std::complex<double> sigma0;
  Vector x_i;
  int if_s_on_surface;
  int real_or_imag_;
public:
  /// Construct a coeffient for right side volume integral
  RGamma0Coefficient(std::complex<double> sigma0_, Vector source_i, int if_s_on_surface_,int real_or_imag)
      : sigma0(sigma0_), x_i(source_i), if_s_on_surface(if_s_on_surface_),real_or_imag_(real_or_imag) {}
  using Coefficient::Eval;
  /// Evaluate the double coefficient at @a ip.
  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

class RGamma1Coefficient : public Coefficient {
private:
  std::vector<std::complex<double>> mat;
  std::complex<double> sigma0;
  Vector x_i;
  int if_s_on_surface;
  std::shared_ptr<ParMesh> pmesh;
  int real_or_imag_;

public:
  /// Construct a coeffient for right side volume integral
  RGamma1Coefficient(std::vector<std::complex<double>> &m, std::complex<double> sigma0_, Vector source_i,
                     int if_s_on_surface_, std::shared_ptr<ParMesh> pmesh_, int real_or_imag)
      : mat(m), sigma0(sigma0_), x_i(source_i),
        if_s_on_surface(if_s_on_surface_), pmesh(pmesh_),real_or_imag_(real_or_imag) {}
  using Coefficient::Eval;
  /// Evaluate the double coefficient at @a ip.
  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

#endif