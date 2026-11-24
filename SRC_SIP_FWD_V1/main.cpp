#include "goafem.h"
#include "mfemplus.h"

int main(int argc, char *argv[]) {
  // MPI handler
  MPI_Init(&argc, &argv);
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  double main_tic, main_toc;
  main_tic = MPI_Wtime();
  // Check program usage
  if (argc < 2) {
    if (world_rank == 0) {
      std::cerr << "Usage: " << argv[0] << " input_parameters_filename\n";
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }

  // Reading input parameters
  ParaHandler inv_para(argv[1]);

  std::string result_path = "solutions/";
  if(world_rank==0){
    if (access(result_path.c_str(), 0)) {
    std::ostringstream command;
    command.str("");
    command << "mkdir " + result_path;
    int status = std::system(command.str().c_str());
    // assert(status == 0); /* normal exit*/
    }
  }
  std::string output_file = "solutions/solution.txt";

  std::shared_ptr<Mesh> mesh;
  mesh.reset(new Mesh(inv_para.mesh_file.c_str(), 1, 1));
  inv_para.refine_local_tets(mesh, world_rank);
  std::vector<std::vector<std::complex<double>>> cond_att(inv_para.frequencies.size());
  for (int i = 0; i < mesh->GetNE(); i++) {
    for (int j=0; j<inv_para.frequencies.size(); j++){
      cond_att[j].push_back(inv_para.get_elem_conductivity(j,mesh->GetAttribute(i)));
    }  
    mesh->SetAttribute(i, i + 1);
  }
  mesh->SetAttributes();
  std::shared_ptr<ParMesh> pmesh;
  pmesh.reset(new ParMesh(MPI_COMM_WORLD, *mesh));
  myassert(pmesh != NULL);

  for(int fre=0;fre<inv_para.frequencies.size();fre++){
    if (world_rank == 0){
      std::cout<<"Forward modeling of the "<<fre+1<<"-th frequency: "<<std::endl;
    }
    std::vector<std::complex<double>> cond_1f=cond_att[fre];
    std::unique_ptr<GOAFEM> goafem(new GOAFEM(cond_1f, inv_para, pmesh, output_file));
    goafem->print_problem_size();
    goafem->reconstruct_FEM_terms();
    goafem->one_forward_iteration();
  }
  // 统计装置系数不合理的值
  int cout_invalid_data = 0;
  for (int i = 0; i < inv_para.K_coef.size(); i++) {
    if (inv_para.if_invalid[i]) {
      cout_invalid_data++;
    }
  }
  if (cout_invalid_data > 0) {
    if (world_rank == 0)
      std::cout << "count_invalid = " << cout_invalid_data << "\n";
  }

  //获取墙上时间并建立日志文件
  main_toc = MPI_Wtime();
  if (world_rank == 0) {
    double local_time;
    double max_time = main_toc - main_tic;
    for (int n = 1; n < world_size; n++) {
      MPI_Recv(&local_time, 1, MPI_DOUBLE, n, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      if (max_time < local_time) {
        max_time = local_time;
      }
    }
    std::cout << "the total calculation time:\t" << max_time << " seconds\n";
  } else {
    double time = main_toc - main_tic;
    MPI_Send(&time, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
  MPI_Finalize();
  return 0;
}
