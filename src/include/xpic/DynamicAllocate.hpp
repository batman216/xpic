#pragma once 

#include <vector>
#include <mpi.h>
#include <thrust/count.h>

#include "ParallelCommunicator.hpp"
#include "gizmos.hpp"

#define RANK0 if(mpi_rank==0)


namespace xpic {

// MPI 相关的整型， 还是用 int 吧
template <typename Particles,typename val_type>
std::vector<int> getParticleNumInChunks(Particles* p,
                                                std::array<val_type,2> bound,
                                                int n) {
  std::vector<int> np_dac;
  np_dac.reserve(n);

  val_type L = bound[1]-bound[0], L_dac = L/n; 

  for (int i=0; i<n; i++) {
    val_type a = bound[0]+i*L_dac, b = a + L_dac;
    np_dac.push_back(thrust::count_if(p->x[0]->begin(),p->x[0]->end(),
                     [a,b]__host__ __device__(val_type x)
                     { return x>=a && x<b;  }));
  }
  assert(std::accumulate(np_dac.begin(),np_dac.end(),0)==p->np);
  return np_dac;

}

template <typename PIC>
struct DynamicAllocate {
  
  using val_type = PIC::value_t;

  /// 将总粒子数分成 n_max*mpi_size 份
  const std::size_t n_max = 10;
  val_type L, L_loc, L_dac;

  const int mpi_size, mpi_rank, n_dac_tot;
  std::vector<val_type> L_loc_vec;

  int n_dac; // # dynamic allocated chunks (DAC)
  std::vector<int> n_dac_global;

  std::vector<int> np_dac_local, np_dac_global; // # particles in each DAC
  std::vector<int> np_loc_vec;

  std::vector<val_type> loc_bound_vec;
  std::array<val_type,2> loc_bound;

  DynamicAllocate(PIC* pic) :
  mpi_size(pic->para->mpi_size),
  mpi_rank(pic->para->mpi_rank),
  n_dac_tot(pic->para->mpi_size*n_max),
  L(pic->cells.upper_bound[0]-pic->cells.lower_bound[0]){
    // initial 
    n_dac = n_max; 

    L_loc = L / mpi_size;
    L_dac = L_loc / n_dac;
  
    np_loc_vec.resize(mpi_size);
    n_dac_global.resize(mpi_size);
    np_dac_local.resize(n_dac);
    np_dac_global.resize(n_dac_tot);
    
    loc_bound_vec.resize(mpi_size*2);

    loc_bound[0] = pic->cells.lower_bound[0] + mpi_rank*L_loc;
    loc_bound[1] = loc_bound[0] + n_dac*L_dac;
    
    std::cout << loc_bound[0] << "," << loc_bound[1] << std::endl;
    np_loc_vec.resize(mpi_size);

  }

  void operator()(PIC* pic) {
      
    int np_tot{}, np_loc{}; // 算出总粒子数，告诉大家
    for (int i=0; i<pic->n_species; i++)
      np_loc += pic->all_particles[i].np;
    MPI_Allreduce(&np_loc,&np_tot,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

    // 理想情况下，每个rank都有np_mean个粒子就最好了
    int np_mean = np_tot / mpi_size;


    np_dac_local = getParticleNumInChunks(&pic->all_particles[0],
                                          loc_bound, n_dac);

    // 告诉rank0每个rank现在有几个DAC, 存在 n_dac_global 里面
    MPI_Gather(&n_dac,1,MPI_INT,n_dac_global.data(),1,MPI_INT,0,MPI_COMM_WORLD);

    /// 告诉rank0每个rank的每个DAC里面都有多少粒子
    MPI_Gather(&np_loc,1,MPI_INT,np_loc_vec.data(),1,MPI_INT,0,MPI_COMM_WORLD);

    int *p; 
    RANK0 {
      std::copy(np_dac_local.begin(),np_dac_local.end(),
                np_dac_global.begin());
      p = np_dac_global.data() + n_dac; // 
    }
    for (int r=1; r<mpi_size; r++) {
      if (mpi_rank == r) 
        MPI_Send(np_dac_local.data(),n_dac,MPI_INT,0,0,MPI_COMM_WORLD);
      if (mpi_rank == 0) {
        MPI_Recv(p,n_dac_global[r],MPI_INT,r,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      }
      p += n_dac_global[r];

    } MPI_Barrier(MPI_COMM_WORLD);


    RANK0 { // 动态规划，重新分配n_dac
      assert(sumArray(np_dac_global)==np_tot);
      int i=0;
      for (int r=0; r<mpi_size; r++) {
        int np_buf=0, n_buf=0;
        while ((np_buf<np_mean) && n_buf<n_max*1.5
                                && i < n_dac_tot) {
          np_buf += np_dac_global[i++];
          n_buf++;
        } 
        n_dac_global[r] = n_buf;
      }
      int rem = n_dac_tot-std::accumulate(n_dac_global.begin(),n_dac_global.end(),0);
      n_dac_global[mpi_size-1] += rem;

      std::puts("\n---- Dynamic allocate:");
      printArray(n_dac_global);

      loc_bound_vec[0] = 0;
      loc_bound_vec[1] = n_dac_global[0]*L_dac;
      for (int i=2; i<2*mpi_size-1; i+=2) {
        loc_bound_vec[i] = loc_bound_vec[i-1];
        loc_bound_vec[i+1] = loc_bound_vec[i] + n_dac_global[i/2]*L_dac;
      }
    } MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_Scatter(n_dac_global.data(),1,MPI_INT,&n_dac,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Scatter(loc_bound_vec.data(),2,mpiTypeTraits<val_type>::type(),
                    loc_bound.data(),2,mpiTypeTraits<val_type>::type(),0,MPI_COMM_WORLD);

  }



};

} // namespace xpic
