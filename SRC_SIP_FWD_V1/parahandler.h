/*
 * @Description:
 * @version:
 * @Author: luowen
 * @Date: 2025-12-07 19:59:41
 * @LastEditTime: 2026-01-17 01:34:06
 * @FilePath: /DC/dcforward/SRC_SIP_FWD/parahandler.h
 */

#ifndef _PARAHANDLER_H
#define _PARAHANDLER_H

#include <fstream>
#include <map>
#include <math.h>
#include <string>
#include <vector>

#include "em.h" //REV

#include "mfem.hpp"

using namespace EM;
using namespace mfem;
class ParaHandler {
public:
  std::string survey_input_file;
  std::string frequencies_file;

  /******************** forward parameters ********************/

  int N_electrodes;
  std::vector<Point_double> electrodes;
  std::vector<int> is_electrode_on_surface;
  // Source parameters
  int source_number; // The number of source
  std::vector<Vertex> sources;
  // size is same to the sources
  std ::vector<int> is_source_on_surface;

  std::vector<Vertex> sites;
  std::vector<double> frequencies;

  std::map<Point_double, int> c_map;
  std::map<Point_double, int> p_map;

  // used in post.cpp
  Array<int> local_sites_tets;

  // Model parameters
  int n_regions;
  std::string model_parameters_file;
  std::vector<int> marker_vec;

  std::vector<std::map<int, std::complex<double>>>
      region2conductivity; // atrribute map cond

  // Mesh parameters
  std::string mesh_file;

  // FEM parameters
  int N_refine_local_tets;
  int survey_mode;

  // Linear solver selection: "mumps" (direct) or "fgmres_amg" (FGMRES +
  // block-diagonal BoomerAMG preconditioner).
  std::string linear_solver;
  // Optional file holding FGMRES/BoomerAMG options; empty -> built-in defaults.
  std::string gmres_amg_file;

  // The coefficients of the setup
  REV K_coef;
  std::vector<bool> if_invalid;
  int n_source_pair = -1;

public:
  ParaHandler(char *model_file);
  ~ParaHandler();

public:
  // skip comments
  void skip_comments(std::istream &in, std::vector<std::string> &para_str_vec,
                     std::string comment_str = "#");

  // read model information from file
  void read_model_info(char *model_file);
  std::complex<double> get_elem_conductivity(int N_f, int marker);

  // get center coordinates of sources
  Vector get_sources_center();

  // Refining the receivers in advance
  void refine_local_tets(
      // Mesh *mesh,
      std::shared_ptr<Mesh> mesh, int myid);

  void refine_local_tets(
      // Mesh *mesh,
      std::shared_ptr<ParMesh> pmesh, int myid);

  void find_point_tets(
      // ParMesh *pmesh,
      std::shared_ptr<ParMesh> pmesh, const std::vector<Vertex> &points,
      Array<int> &find_tets);

  // Eigen::Matrix3d eulerAnglesToRotationMatrix(Eigen::Vector3d &theta);
};

#endif // _PARAHANDLER_H
