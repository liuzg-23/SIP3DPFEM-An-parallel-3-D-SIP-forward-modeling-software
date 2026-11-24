/*
 * @Description:
 * @version:
 * @Author: luowen
 * @Date: 2025-12-07 20:03:46
 * @LastEditTime: 2025-12-07 20:03:47
 * @FilePath: /DC/dcforward/src/post.h
 */

#ifndef _POST_H
#define _POST_H

#include <fstream>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "parahandler.h"

// using namespace std;
using namespace mfem;

class Post {
public:
  // ParMesh *pmesh;
  std::shared_ptr<ParMesh> pmesh;

  // ParFiniteElementSpace *pfes;
  std::shared_ptr<ParFiniteElementSpace> pfes;

  std::vector<std::shared_ptr<ParComplexGridFunction>> up; // multiple sources

  ParaHandler &para_handler;

  // potential Up(i,m) of source i at measuring site m owned by my rank
  std::vector<std::vector<std::complex<double>>> local_Up;

  // total potential Ut(i,m) of source i at measuring sites m for global mesh
  std::vector<std::vector<std::complex<double>>> Ut;

public:
  Post(std::shared_ptr<ParMesh> pmesh_,
       std::shared_ptr<ParFiniteElementSpace> pfes_,
       std::vector<std::shared_ptr<ParComplexGridFunction>> up_,
       ParaHandler &para_handler_);

  ~Post();

public:
  void compute_sites_up(std::vector<std::complex<double>> &cond_att,
                        std::vector<std::complex<double>> &sigma0,
                        std ::vector<int> is_source_on_surface);
  // Save results of global sites in rank 0 (root rank)
  void save_as_one(std::vector<std::complex<double>> &sigma0,
                   std ::vector<int> is_source_on_surface);

  void compute_rho(std::string survery_mode_file, std::string output_file);

  // void compute_U_E_J_averagely(std::vector<DenseMatrix> &cond_att,
  //                              std::vector<DenseMatrix> &sigma0,
  //                              std ::vector<int> is_source_on_surface);
};

#endif // _POST_H
