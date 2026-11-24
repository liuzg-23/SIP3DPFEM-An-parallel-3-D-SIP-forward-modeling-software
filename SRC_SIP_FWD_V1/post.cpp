

#include "post.h"
#include "mfemplus.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <math.h>
#include <mpi.h>

Post::Post(std::shared_ptr<ParMesh> pmesh_,
           std::shared_ptr<ParFiniteElementSpace> pfes_,
           std::vector<std::shared_ptr<ParComplexGridFunction>> up_,
           ParaHandler &para_handler_)
    : pmesh(pmesh_), pfes(pfes_), up(up_), para_handler(para_handler_) {}
Post::~Post() {}

void Post::compute_sites_up(std::vector<std::complex<double>> &cond_att,
                            std::vector<std::complex<double>> &sigma0,
                            std::vector<int> is_source_on_surface) {

  MPI_Comm mycomm = pmesh->GetComm();
  int myid, n_procs;
  MPI_Comm_rank(mycomm, &myid);

  int Ns = para_handler.sources.size();
  // local_Up.clear();
  local_Up.resize(Ns);

  int nsite = para_handler.sites.size();
  myassert(nsite > 0);
  for (int k = 0; k < Ns; k++) {
    myassert(sigma0.size() == Ns);
    ParGridFunction Up_r = up[k]->real();
    ParGridFunction Up_i = up[k]->imag();
    local_Up[k].resize(nsite);
    for (int i = 0; i < nsite; i++) {
      myassert(para_handler.local_sites_tets.Size() != 0);
      int tet_id = para_handler.local_sites_tets[i];
      if (tet_id != -2) { // find in this process
        const FiniteElement *fe = pfes->GetFE(tet_id);
        int ndof = fe->GetDof();
        Vector pt(3);
        pt = 0.0;
        pt[0] = para_handler.sites[i](0);
        pt[1] = para_handler.sites[i](1);
        pt[2] = para_handler.sites[i](2);
        IntegrationPoint ip;
        ElementTransformation *tr = pfes->GetElementTransformation(tet_id);
        tr->TransformBack(pt, ip);
        tr->SetIntPoint(&ip);
        // Compute shape function
        Vector shape(ndof);
        shape = 0.0;
        fe->CalcPhysShape(*tr, shape);
        Array<int> vdofs;
        pfes->GetElementDofs(tet_id, vdofs);
        Vector tempUp_r, tempUp_i;
        tempUp_r = 0.0;
        tempUp_i = 0.0;
        Up_r.GetSubVector(vdofs, tempUp_r);
        Up_i.GetSubVector(vdofs, tempUp_i);
        double potential_r = 0.0, potential_i = 0.0;
        for (int j = 0; j < ndof; j++) {
          potential_r += shape[j] * tempUp_r[j];
          potential_i += shape[j] * tempUp_i[j];
          // if(tempUp_i[j]!=0.0){
          //   std::cout<<tempUp_i[j]<<std::endl;
          // }
        }
        local_Up[k][i] =
            std::complex<double>(potential_r, potential_i); // local Up
      } else {
        myassert(local_Up[k][i] == 0.0);
        // local_Up[k][i] = 0.0;
      }
    }
  }
}

void Post::save_as_one(std::vector<std::complex<double>> &sigma0,
                       std ::vector<int> is_source_on_surface) {
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Comm mycomm = pmesh->GetComm();
  MPI_Status status;
  int myid, n_procs;
  MPI_Comm_size(mycomm, &n_procs);
  MPI_Comm_rank(mycomm, &myid);

  int count = -1;
  double *msg_Up;
  int Ns = para_handler.sources.size();

  Ut.resize(Ns);
  for (int k = 0; k < Ns; k++) {
    if (myid == 0) {
      for (int p = 1; p < n_procs; p++) {
        MPI_Recv(&count, 1, MPI_INT, p, 101, mycomm, &status);
        myassert(count != -1);
        msg_Up = new double[count];
        // msg_Up.reset(new double[count]);
        MPI_Recv(msg_Up, count, MPI_DOUBLE, p, 105, mycomm, &status);
        // Append the data of other threads
        for (int i = 0; i < count / 2; i++) {
          local_Up[k][i].real(local_Up[k][i].real() + msg_Up[2 * i]);
          local_Up[k][i].imag(local_Up[k][i].imag() + msg_Up[2 * i + 1]);
        }
        delete msg_Up;
      }

      // compute total potential Ut(sources,measures)
      int nsite = para_handler.sites.size();
      Ut[k].resize(nsite);

      Point_double pi;
      for (int kk = 0; kk < 3; kk++) {
        pi(kk) = para_handler.sources[k](kk);
      }
      std::complex<double> sigma_0 = sigma0[k];
      for (int h = 0; h < nsite; h++) {
        Point_double pj;
        for (int gg = 0; gg < 3; gg++) {
          pj(gg) = para_handler.sites[h](gg);
        }
        if (pj(0) == pi(0) && pj(1) == pi(1) && pj(2) == pi(2)) {
          Ut[k][h] = 1E10;
        } else {
          std::complex<double> Us =
              Uis(sigma_0, pj, pi, is_source_on_surface[k]);
          Ut[k][h] = local_Up[k][h] + Us;
        }
      }
    } else {
      count = 2 * local_Up[k].size();
      // array类型的函数 返回指针数组 不需要new和delete
      msg_Up = new double[count];
      std::vector<std::complex<double>> msg_Up_c = local_Up[k];
      for (int i = 0; i < local_Up[k].size(); i++) {
        msg_Up[2 * i] = msg_Up_c[i].real();
        msg_Up[2 * i + 1] = msg_Up_c[i].imag();
      }
      MPI_Send(&count, 1, MPI_INT, 0, 101, mycomm);
      MPI_Send(msg_Up, count, MPI_DOUBLE, 0, 105, mycomm);
    }
  } // source loop
}

// checked
void Post::compute_rho(std::string survery_mode_file, std::string output_file) {

  unsigned int n_data = 0;
  unsigned int mode = 0;
  // check survey mode file
  std::ifstream mode_stream(survery_mode_file.c_str());
  myassert(mode_stream.good());
  std::ofstream out_stream(output_file.c_str(), std::ios::app);
  myassert(out_stream.good());

  int N_electrodes;
  std::vector<Point_double> electrodes;
  std::vector<int> is_electrode_on_surface;
  mode_stream >> N_electrodes;
  myassert(N_electrodes > 0);
  for (int i = 0; i < N_electrodes; i++) {
    Point_double electrode;
    int is_on_earth_surface;
    mode_stream >> electrode(0) >> electrode(1) >> electrode(2) >>
        is_on_earth_surface;
    myassert(is_on_earth_surface == 0 || is_on_earth_surface == 1);
    electrodes.push_back(electrode);
    is_electrode_on_surface.push_back(is_on_earth_surface);
  }
  myassert(electrodes.size() == N_electrodes);

  mode_stream >> n_data // nunmber of measurments
      >> mode;          // measurment mode
  myassert(n_data > 0);

  // int n_source_pair = para_handler.source_number / 2;
  // myassert(para_handler.source_number % 2 == 0);
  int n_source_pair = para_handler.n_source_pair;
  // myassert(n_source_pair % 2 == 0);
  std::vector<Point_double> CA_nodes(n_source_pair);
  std::vector<Point_double> CB_nodes(n_source_pair);
  std::vector<int> index_sourceA(n_source_pair);
  std::vector<int> index_sourceB(n_source_pair);
  Point_double PM_node, PN_node;
  std::vector<unsigned int> local_index_sA(n_source_pair);
  std::vector<unsigned int> local_index_sB(n_source_pair);
  unsigned int local_index_m, local_index_n;

  // K_coef.resize(n_data);
  for (int i = 0; i < n_data; i++) {
    if (mode == 11) { // pole-pole configuration
      Point_double C_node, P_node;

      int s_id = -1, m_id = -1;
      mode_stream >> s_id >> m_id;
      for (int j = 0; j < 3; j++) {
        C_node(j) = electrodes[s_id - 1](j);
        P_node(j) = electrodes[m_id - 1](j);
      }

      unsigned int C_local_id = para_handler.c_map[C_node];
      unsigned int P_local_id = para_handler.p_map[P_node];

      std::complex<double> U = this->Ut[C_local_id][P_local_id];
      double k = para_handler.K_coef(i);
      std::complex<double> rho = k * U;

      if (k != 0.0) {
        rho = k * U;
      } else {
        rho = 0.0;
      }

      out_stream.precision(8);

      out_stream << s_id << "\t" << m_id << "\t" << k << "\t" << U.real()
                 << "\t" << U.imag() << "\t" << rho.real() << "\t" << rho.imag()
                 << "\n";
    } else if (mode == 12) {
      Point_double C1_node, P1_node, P2_node;

      int s1_id = -1, m1_id = -1, m2_id = -1;
      mode_stream >> s1_id >> m1_id >> m2_id;
      for (int j = 0; j < 3; j++) {
        C1_node(j) = electrodes[s1_id - 1](j);
        P1_node(j) = electrodes[m1_id - 1](j);
        P2_node(j) = electrodes[m2_id - 1](j);
      }
      unsigned int C1_local_id = para_handler.c_map[C1_node];
      unsigned int P1_local_id = para_handler.p_map[P1_node];
      unsigned int P2_local_id = para_handler.p_map[P2_node];
      std::complex<double> U = this->Ut[C1_local_id][P1_local_id] -
                               this->Ut[C1_local_id][P2_local_id];

      double k = para_handler.K_coef(i);
      std::complex<double> rho = k * U;

      if (k != 0.0) {
        rho = k * U;
        if (rho.real() < 0.0) {
          std::cout << "存在rho < 0.0" << std::endl;
          para_handler.if_invalid[i] = true;
        }
        if (std::abs(rho) > 1e6) {
          std::cout << "存在std::abs(rho) > 1e-6" << std::endl;
          para_handler.if_invalid[i] = true;
        }
      } else {
        // std::cout << "存在k = 0.0" << std::endl;
        // cout_invalid_data++;
        // rho = 0.0;
        para_handler.if_invalid[i] = true;
      }

      if (para_handler.if_invalid[i] == true) {
        rho = 0.0;
      }

      out_stream.precision(8);

      out_stream << s1_id << "\t" << m1_id << "\t" << m2_id << "\t" << k << "\t"
                 << U.real() << "\t" << U.imag() << "\t" << rho.real() << "\t"
                 << rho.imag() << "\n";

    } else if (mode == 21) {
      Point_double C1_node, C2_node, P1_node;

      int s1_id = -1, s2_id = -1, m1_id = -1;
      mode_stream >> s1_id >> s2_id >> m1_id;
      for (int j = 0; j < 3; j++) {
        C1_node(j) = electrodes[s1_id - 1](j);
        C2_node(j) = electrodes[s2_id - 1](j);
        P1_node(j) = electrodes[m1_id - 1](j);
      }

      unsigned int C1_local_id = para_handler.c_map[C1_node];
      unsigned int C2_local_id = para_handler.c_map[C2_node];
      unsigned int P1_local_id = para_handler.p_map[P1_node];
      std::complex<double> U = (this->Ut[C1_local_id][P1_local_id] -
                                this->Ut[C2_local_id][P1_local_id]);
      double k = para_handler.K_coef(i);
      std::complex<double> rho = k * U;

      if (k != 0.0) {
        rho = k * U;
      } else {
        // std::cout << "K_coef is infinite! \n";
        // U = 0.0;
        rho = 0.0;
      }

      out_stream.precision(8);

      out_stream << s1_id << "\t" << s2_id << "\t" << m1_id << "\t" << k << "\t"
                 << U.real() << "\t" << U.imag() << "\t" << rho.real() << "\t"
                 << rho.imag() << "\n";
    // } else if (mode == 22) { // dipole-dipole
    //   Point_double C1_node, C2_node, P1_node, P2_node;

    //   int s1_id = -1, s2_id = -1, m1_id = -1, m2_id = -1;
    //   mode_stream >> s1_id >> s2_id >> m1_id >> m2_id;
    //   for (int j = 0; j < 3; j++) {
    //     C1_node(j) = electrodes[s1_id - 1](j);
    //     C2_node(j) = electrodes[s2_id - 1](j);
    //     P1_node(j) = electrodes[m1_id - 1](j);
    //     P2_node(j) = electrodes[m2_id - 1](j);
    //   }

    //   unsigned int C1_local_id = para_handler.c_map[C1_node];
    //   unsigned int C2_local_id = para_handler.c_map[C2_node];
    //   unsigned int P1_local_id = para_handler.p_map[P1_node];
    //   unsigned int P2_local_id = para_handler.p_map[P2_node];
    //   std::complex<double> U = (this->Ut[C1_local_id][P1_local_id] -
    //                             this->Ut[C2_local_id][P1_local_id]) -
    //                            (this->Ut[C1_local_id][P2_local_id] -
    //                             this->Ut[C2_local_id][P2_local_id]);
    //   double k = para_handler.K_coef(i);
    //   std::complex<double> rho = k * U;

    //   if (k != 0.0) {
    //     rho = k * U;

    //   } else {
    //     rho = 0.0;
    //   }

    //   out_stream.precision(8);
    //   // output.setf(std::ios::scientific);
    //   out_stream << s1_id << "\t" << s2_id << "\t" << m1_id << "\t" << m2_id
    //              << "\t" << k << "\t" << U.real() << "\t" << U.imag() << "\t"
    //              << rho.real() << "\t" << rho.imag() << "\n";
    //   // } else if (mode == 122) {
    } else if (mode % 2 == 0) {
      for (int sn = 0; sn < n_source_pair; sn++) {
        index_sourceA[sn] = -1, index_sourceB[sn] = -1;
        mode_stream >> index_sourceA[sn] >> index_sourceB[sn];
      }
      int m1_id = -1, m2_id = -1;
      mode_stream >> m1_id >> m2_id;
      for (int k = 0; k < 3; k++) {
        for (int sn = 0; sn < n_source_pair; sn++) {
          CA_nodes[sn](k) = electrodes[index_sourceA[sn] - 1](k);
          CB_nodes[sn](k) = electrodes[index_sourceB[sn] - 1](k);
        }
        PM_node(k) = electrodes[m1_id - 1](k);
        PN_node(k) = electrodes[m2_id - 1](k);
      }
      for (int sn = 0; sn < n_source_pair; sn++) {
        local_index_sA[sn] = para_handler.c_map[CA_nodes[sn]];
        local_index_sB[sn] = para_handler.c_map[CB_nodes[sn]];
      }
      local_index_m = para_handler.p_map[PM_node];
      local_index_n = para_handler.p_map[PN_node];
      std::complex<double> U = 0.0;
      for (int sn = 0; sn < n_source_pair; sn++) {
        U += (this->Ut[local_index_sA[sn]][local_index_m] -
              this->Ut[local_index_sB[sn]][local_index_m]) -
             (this->Ut[local_index_sA[sn]][local_index_n] -
              this->Ut[local_index_sB[sn]][local_index_n]);
      }
      double k = para_handler.K_coef(i);
      std::complex<double> rho = -1;
      if (k != 0.0) {
        rho = k * U * (1.0 / n_source_pair);
      } else {
        std::cout << "k is zero\n";
        rho = 0.0;
      }
      out_stream.precision(8);
      // std::cout << "n_source_pair: " << n_source_pair << std::endl;
      for (int sn = 0; sn < n_source_pair; sn++) {
        out_stream << index_sourceA[sn] << "\t" << index_sourceB[sn] << "\t";
      }
      out_stream << m1_id << "\t" << m2_id << "\t" << k << "\t" << U.real()
                 << "\t" << U.imag() << "\t" << rho.real() << "\t" << rho.imag()
                 << "\n";
    }
  }
  mode_stream.close();

  return;
}
