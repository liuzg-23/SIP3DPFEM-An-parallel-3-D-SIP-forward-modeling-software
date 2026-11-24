

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include "parahandler.h"

ParaHandler::ParaHandler(char *model_file) {
  this->read_model_info(model_file);
}

ParaHandler::~ParaHandler() {}
void ParaHandler::skip_comments(std::istream &in,
                                std::vector<std::string> &para_str_vec,
                                std::string comment_str) {
  std::string line;
  while (std::getline(in, line)) {
    for (char &c : line) // loop string in C++11 grammar
    {
      if (c == '\t' || c == ',' || c == ';' || c == '\r' || c == '\n') {
        c = ' ';
      }
    }
    // delete the space at the beginning of the line
    line.erase(0, line.find_first_not_of(" "));
    // delete the space at the ending of the line
    line.erase(line.find_last_not_of(" ") + 1);

    int n_comment_start = line.find_first_of(comment_str);
    if (n_comment_start != std::string::npos) {
      line.erase(n_comment_start); // delete comments
    }

    if (line.empty())
      continue;

    para_str_vec.push_back(line);
  }
}

void ParaHandler::read_model_info(char *model_file) {

  std::ifstream in_stream(model_file);
  myassert(in_stream.good());
  int para_lines = 5; // minimum number of parameters' lines
  std::vector<std::string> para_str_vec;
  skip_comments(in_stream, para_str_vec);
  myassert(para_str_vec.size() >= para_lines);
  in_stream.close();
  std::stringstream ss;

  // survey mode file
  ss << para_str_vec[0];
  ss >> survey_input_file;
  ss.clear();

  // Frequencies parameters
  ss << para_str_vec[1];
  ss >> frequencies_file;
  ss.clear();

  // Model parameters
  ss << para_str_vec[2];
  ss >> model_parameters_file;
  ss.clear();

  // FEM parameter
  ss << para_str_vec[3];
  ss >> N_refine_local_tets;
  ss.clear();
  myassert(N_refine_local_tets >= 0);

  // Mesh parameters
  ss << para_str_vec[4];
  ss >> mesh_file;
  ss.clear();

  // Optional parameters (lines 6 and 7). They default to the direct solver so
  // that existing 5-line input files remain valid.
  linear_solver = "mumps";
  gmres_amg_file = "";
  if (para_str_vec.size() >= 6) {
    ss << para_str_vec[5];
    ss >> linear_solver;
    ss.clear();
  }
  if (para_str_vec.size() >= 7) {
    ss << para_str_vec[6];
    ss >> gmres_amg_file;
    ss.clear();
  }

  /**Read and handle the survey setup in terms of the sources and receivers from
   * the file**/
  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  std::ifstream survey_instream(survey_input_file.c_str());
  myassert(survey_instream.good());
  survey_instream >> N_electrodes;
  myassert(N_electrodes > 0);
  for (int i = 0; i < N_electrodes; i++) {
    Point_double electrode;
    int is_on_earth_surface;
    survey_instream >> electrode(0) >> electrode(1) >> electrode(2) >>
        is_on_earth_surface;
    myassert(is_on_earth_surface == 0 || is_on_earth_surface == 1);
    electrodes.push_back(electrode);
    is_electrode_on_surface.push_back(is_on_earth_surface);
  }
  myassert(electrodes.size() == N_electrodes);

  unsigned int n_data = 0;
  int temp_mode = 0;

  survey_instream >> n_data // nunmber of measurments
      >> temp_mode;         // measurment mode
  myassert(n_data > 0);
  myassert(temp_mode == 11 || temp_mode == 12 || temp_mode == 21 ||
           temp_mode == 22 || (temp_mode % 10 == 2));

  survey_mode = temp_mode;
  double obs_u_temp,noise_u_temp;
  Point_double point_s, point_m, point_s2, point_m2;
  std::set<Point_double> temp_s, temp_m;
  typedef std::pair<std::set<Point_double>::iterator, bool> stlpair;
  K_coef.resize(n_data);
  if_invalid.resize(n_data);

  // build the source pairs, avoid repeated cost for every measurement
  int mark_s = survey_mode / 10;
  // int n_source_pair = mark_s / 2;
  n_source_pair = mark_s / 2;
  myassert(n_source_pair > 0);
  // myassert(mark_s % 2 == 0);
  std::vector<Point_double> points_sourceA, points_sourceB;
  points_sourceA.resize(n_source_pair);
  points_sourceB.resize(n_source_pair);
  std::vector<int> index_sourceA(n_source_pair);
  std::vector<int> index_sourceB(n_source_pair);

  // std::cout << "n_source_pair00: " << n_source_pair << std::endl;

  for (int i = 0; i < n_data; i++) {
    if (survey_mode == 11) {
      int s_id = -1, m_id = -1;
      survey_instream >> s_id >> m_id;
      for (int j = 0; j < 3; j++) {
        point_s(j) = electrodes[s_id - 1](j);
        point_m(j) = electrodes[m_id - 1](j);
      }

      double a = length_two_point(point_s, point_m);
      double k = 2.0 * PI * a;
      K_coef(i) = k;
      stlpair pair_s = temp_s.insert(point_s);
      stlpair pair_m = temp_m.insert(point_m);

    // } else if (survey_mode == 22) {
    //   int s1_id = -1, s2_id = -1, m1_id = -1, m2_id = -1;
    //   survey_instream >> s1_id >> s2_id >> m1_id >> m2_id;
    //   for (int j = 0; j < 3; j++) {
    //     point_s(j) = electrodes[s1_id - 1](j);
    //     point_s2(j) = electrodes[s2_id - 1](j);
    //     point_m(j) = electrodes[m1_id - 1](j);
    //     point_m2(j) = electrodes[m2_id - 1](j);
    //   }

    //   Point_double PA = point_s;
    //   Point_double PB = point_s2;
    //   Point_double PM = point_m;
    //   Point_double PN = point_m2;

    //   double AM = length_two_point(PA, PM);
    //   double BM = length_two_point(PB, PM);
    //   double AN = length_two_point(PA, PN);
    //   double BN = length_two_point(PB, PN);
    //   double factor = 1.0 / AM - 1.0 / BM - 1.0 / AN + 1.0 / BN;
    //   double k = 2 * PI * 1.0 / factor;
    //   K_coef(i) = k;
    //   stlpair pair_s1 = temp_s.insert(point_s);
    //   stlpair pair_s2 = temp_s.insert(point_s2);
    //   stlpair pair_m1 = temp_m.insert(point_m);
    //   stlpair pair_m2 = temp_m.insert(point_m2);

    } else if (survey_mode % 2 == 0) {
      for (int sn = 0; sn < n_source_pair; sn++) {
        index_sourceA[sn] = -1, index_sourceB[sn] = -1;
        survey_instream >> index_sourceA[sn] >> index_sourceB[sn];
      }
      // int mark_m = survey_mode % 10;
      int index_M = -1, index_N = -1;
      survey_instream >> index_M >> index_N;
      for (int k = 0; k < 3; k++) {
        for (int sn = 0; sn < n_source_pair; sn++) {
          points_sourceA[sn](k) = electrodes[index_sourceA[sn] - 1](k);
          points_sourceB[sn](k) = electrodes[index_sourceB[sn] - 1](k);
        }
        point_m(k) = electrodes[index_M - 1](k);
        point_m2(k) = electrodes[index_N - 1](k);
      }

      Point_double PA_average, PB_average;
      for (int sn = 0; sn < n_source_pair; sn++) {
        PA_average = PA_average + points_sourceA[sn];
        PB_average = PB_average + points_sourceB[sn];
      }
      PA_average = PA_average * (1.0 / n_source_pair);
      PB_average = PB_average * (1.0 / n_source_pair);
      // add all the sources and sites, used to build the map of source and
      // measurement
      for (int sn = 0; sn < n_source_pair; sn++) {
        stlpair pair_sA = temp_s.insert(points_sourceA[sn]);
        stlpair pair_sB = temp_s.insert(points_sourceB[sn]);
      }
      stlpair pair_M = temp_m.insert(point_m);
      stlpair pair_N = temp_m.insert(point_m2);
      double AM_average = length_two_point(PA_average, point_m);
      double BM_average = length_two_point(PB_average, point_m);
      double AN_average = length_two_point(PA_average, point_m2);
      double BN_average = length_two_point(PB_average, point_m2);
      double factor = 1.0 / AM_average - 1.0 / BM_average - 1.0 / AN_average +
                      1.0 / BN_average;
      double k = 2 * PI * 1.0 / factor;
      K_coef(i) = k;

    }

    else if (survey_mode == 21) {
      int s1_id = -1, s2_id = -1, m_id = -1;
      survey_instream >> s1_id >> s2_id >> m_id;
      for (int j = 0; j < 3; j++) {
        point_s(j) = electrodes[s1_id - 1](j);
        point_s2(j) = electrodes[s2_id - 1](j);
        point_m(j) = electrodes[m_id - 1](j);
      }
      double AM = length_two_point(point_s, point_m);
      double BM = length_two_point(point_s2, point_m);
      double factor = 1.0 / AM - 1.0 / BM - 0 + 0;
      double k = 2 * PI * 1.0 / factor;

      K_coef(i) = k;
      stlpair pair_s1 = temp_s.insert(point_s);
      stlpair pair_s2 = temp_s.insert(point_s2);
      stlpair pair_m1 = temp_m.insert(point_m);

    } else if (survey_mode == 12) {

      int s_id = -1, m1_id = -1, m2_id = -1;
      survey_instream >> s_id >> m1_id >> m2_id;
      for (int j = 0; j < 3; j++) {
        point_s(j) = electrodes[s_id - 1](j);
        point_m(j) = electrodes[m1_id - 1](j);
        point_m2(j) = electrodes[m2_id - 1](j);
      }

      double AM = length_two_point(point_s, point_m);
      double AN = length_two_point(point_s, point_m2);
      double factor = 1.0 / AM - 1.0 / AN;
      double k = 2 * PI * 1.0 / factor;

      K_coef(i) = k;

      stlpair pair_s1 = temp_s.insert(point_s);
      stlpair pair_m1 = temp_m.insert(point_m);
      stlpair pair_m2 = temp_m.insert(point_m2);

    } else {
      std::cout << "fail to get measure_sources, unsopported mode!\n";
      std::abort();
    }
  }

  survey_instream.close();

  for (int i = 0; i < K_coef.size(); i++) {
    // if (!std::isfinite(K_coef(i))) {
    if ((!std::isfinite(K_coef(i))) || (std::abs(K_coef(i) > 1e8))) {
      K_coef(i) = 0.0;
    }
  }

  std::vector<Point_double> tempC(temp_s.begin(), temp_s.end());
  int ns = tempC.size();
  sources.resize(ns);
  is_source_on_surface.resize(ns);
  for (int i = 0; i < ns; i++) {
    sources[i](0) = tempC[i](0);
    sources[i](1) = tempC[i](1);
    sources[i](2) = tempC[i](2);
    Point_double source;
    source(0) = tempC[i](0);
    source(1) = tempC[i](1);
    source(2) = tempC[i](2);
    auto it = std::find(electrodes.begin(), electrodes.end(), source);
    if (it == electrodes.end()) {
      std::cout << "source = " << source << " not found in electrodes!"
                << std::endl;
    } else {
      int index = std::distance(electrodes.begin(), it);
      is_source_on_surface[i] = is_electrode_on_surface[index];
    }
  }
  source_number = sources.size();
  myassert(source_number > 0);
  std::vector<Point_double> tempP(temp_m.begin(), temp_m.end());
  // sites.clear();
  sites.resize(tempP.size());
  for (int i = 0; i < tempP.size(); i++) {
    sites[i](0) = tempP[i](0);
    sites[i](1) = tempP[i](1);
    sites[i](2) = tempP[i](2);

    Point_double site;
    site(0) = tempP[i](0);
    site(1) = tempP[i](1);
    site(2) = tempP[i](2);
    auto it = std::find(electrodes.begin(), electrodes.end(), site);
    if (it == electrodes.end()) {
      std::cout << "source = " << site << " not found in electrodes!"
                << std::endl;
    }
  }
  int sites_number = sites.size();
  myassert(sites_number > 0);

  // two maps to spead up the computation
  // mapping the position of c/p to its local id in "C" and "P" vectors
  for (int i = 0; i < source_number; i++)
    c_map[tempC[i]] = i;
  for (int i = 0; i < sites_number; i++)
    p_map[tempP[i]] = i;
  /*******************************************************************************************/

  /********************Read and handle the model parameters from the
   * file*********************/
  // Read frequencies
  std::ifstream frequency_in_stream(frequencies_file.c_str());
  myassert(frequency_in_stream.good());
  int frequency_number;
  frequency_in_stream >> frequency_number;
  frequencies.resize(frequency_number);
  for (int i = 0; i < frequency_number; i++) {
    frequency_in_stream >> frequencies[i];
  }
  frequency_in_stream.close();
  region2conductivity.resize(frequencies.size());

  // Read the electrical parameters
  std::ifstream cond_in_stream(model_parameters_file.c_str());
  myassert(cond_in_stream.good());
  cond_in_stream >> n_regions;
  for (int i = 0; i < n_regions; i++) {
    int marker;
    double cond_0, polar, time_c, fre_factor;
    cond_in_stream >> marker >> cond_0 >> polar >> time_c >> fre_factor;
    marker_vec.push_back(marker);
    for (int j = 0; j < frequencies.size(); j++) {
      double omega = 2 * 3.1415926535897932384626433832795 * frequencies[j];
      std::complex<double> mid_1(0, omega * time_c);
      std::complex<double> mid_2 = 1.0 / (1.0 + std::pow(mid_1, fre_factor));
      std::complex<double> cond_complex =
          cond_0 / (1.0 - polar * (1.0 - mid_2));
      region2conductivity[j][marker] = cond_complex;
    }
  }
  cond_in_stream.close();

  // check mesh file stream
  std::ifstream msh_stream(mesh_file);
  myassert(msh_stream.good());
  msh_stream.close();
  /*******************************************************************************************/
}

std::complex<double> ParaHandler::get_elem_conductivity(int N_f, int marker) {
  myassert(!region2conductivity[N_f].empty());
  std::map<int, std::complex<double>>::iterator it =
      region2conductivity[N_f].find(marker);
  myassert(it != region2conductivity[N_f].end());
  return (*it).second;
}

Vector ParaHandler::get_sources_center() {
  Vector gpoint(3);
  gpoint = 0.0;
  for (int i = 0; i < sources.size(); i++) {
    for (int j = 0; j < 3; j++) {
      gpoint[j] += sources[i](j);
    }
  }
  gpoint /= sources.size();
  return gpoint;
}

void ParaHandler::refine_local_tets(
    // Mesh *mesh,
    std::shared_ptr<Mesh> mesh, int myid) {
  // Refining the receivers in advance
  if (N_refine_local_tets != 0) {
    if (myid == 0)
      std::cout << "N_refine_local_tets: " << N_refine_local_tets << std::endl;
    std::set<Point_double> temp_SM;
    std::vector<Vertex> s_plus_m;
    // 注意虚拟场的方法，源附近不需要加密，但是源附近的电阻率取值有可能影响结果，通常妥善的处理是将源附近所有单元的电导率取平均，但是MFEM通常只找到周围的一个单元，所以可能影响结果。
    for (int j = 0; j < sites.size(); j++) {
      Point_double temp;
      temp(0) = sites[j](0);
      temp(1) = sites[j](1);
      temp(2) = sites[j](2);
      temp_SM.insert(temp);
    }
    typedef std::set<Point_double>::iterator SIT;
    for (SIT sit = temp_SM.begin(); sit != temp_SM.end(); sit++) {
      Vertex temp;
      temp(0) = (*sit)(0);
      temp(1) = (*sit)(1);
      temp(2) = (*sit)(2);
      s_plus_m.push_back(temp);
    }
    // all measurements and sources after removing duplicates
    DenseMatrix point_mat(3, s_plus_m.size());
    for (int i = 0; i < s_plus_m.size(); i++) {
      point_mat(0, i) = s_plus_m[i](0);
      point_mat(1, i) = s_plus_m[i](1);
      point_mat(2, i) = s_plus_m[i](2);
    }

    Array<int> sites_tets;
    Array<IntegrationPoint> ips;
    for (int i_r = 0; i_r < N_refine_local_tets; i_r++) {
      mesh->FindPoints(point_mat, sites_tets, ips);
      for (int i = 0; i < sites_tets.Size(); i++) {
        if (sites_tets[i] == -1 || sites_tets[i] == -2) {
          std::cout << "The code cannot find the tetrahedra that contain the "
                       "receiver !!!"
                    << std::endl;
          std::abort();
        }
      }
      mesh->GeneralRefinement(sites_tets);
    }
  }
}

void ParaHandler::refine_local_tets(std::shared_ptr<ParMesh> pmesh, int myid) {
  // Refining the receivers in advance
  if (N_refine_local_tets != 0) {
    if (myid == 0)
      std::cout << "N_refine_local_tets: " << N_refine_local_tets << std::endl;
    DenseMatrix point_mat(3, sites.size());
    for (int i = 0; i < sites.size(); i++) {
      point_mat(0, i) = sites[i](0);
      point_mat(1, i) = sites[i](1);
      point_mat(2, i) = sites[i](2);
    }

    Array<int> sites_tets;
    Array<IntegrationPoint> ips;
    for (int i_r = 0; i_r < N_refine_local_tets; i_r++) {
      pmesh->FindPoints(point_mat, sites_tets, ips);
      for (int i = 0; i < sites_tets.Size(); i++) {
        // if (sites_tets[i] == -1 || sites_tets[i] == -2) {
        if (sites_tets[i] == -1) {
          std::cout << "The code cannot find the tetrahedra that contain the "
                       "receiver !!!"
                    << std::endl;
          std::abort();
        }
      }
      pmesh->GeneralRefinement(sites_tets);
    }
  }
}

void ParaHandler::find_point_tets(
    // ParMesh *pmesh,
    std::shared_ptr<ParMesh> pmesh, const std::vector<Vertex> &points,
    Array<int> &find_tets) {
  find_tets.DeleteAll();

  // Find the potential tets which contain sites
  int npts = points.size();
  // 找不到点的可能原因，测点输入错误；网格过于稀疏，所找的点附近没有网格点
  DenseMatrix point_mat(3, npts);
  for (int j = 0; j < npts; j++) {
    point_mat(0, j) = points[j](0);
    point_mat(1, j) = points[j](1);
    point_mat(2, j) = points[j](2);
  }
  Array<int> temp_tet_id;
  Array<IntegrationPoint> ips;
  pmesh->FindPoints(point_mat, temp_tet_id, ips);
  // pmesh->FindPoints(point_mat, temp_tet_id, ips);
  for (int j = 0; j < temp_tet_id.Size(); j++) {
    int tid = temp_tet_id[j];
    if (tid == -1) {
      std::cout << "measurements point not found! please check your parameters";
      std::cout << "point: " << point_mat(0, j) << " " << point_mat(1, j)
                << "  " << point_mat(2, j) << std::endl;
      // std::abort();
    } else {
      // local_sites.Append(sites[j]);
      find_tets.Append(tid); // 按顺序添加局部单元编号
    }
  }
  myassert(find_tets.Size() == npts);
}