#pragma once 

#include <vector>
#include <mpi.h>
#include <thrust/count.h>

#include "ParallelCommunicator.hpp"

#define RANK0 if(mpi_rank==0)


namespace xpic {

// MPI 相关的整形， 还是用 int 吧
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
                     { return x>a && x<b;  }));
  }
  return np_dac;

}

template <typename PIC>
struct DynamicAllocate {
  
  using val_type = PIC::value_t;

  /// 将总粒子数分成 n_max*mpi_size 份
  const std::size_t n_max = 4;
  val_type L, L_loc, L_dac;

  const int mpi_size, mpi_rank, n_dac_tot;
  std::vector<val_type> L_loc_vec;

  int n_dac; // # dynamic allocated chunks (DAC)
  std::vector<int> n_dac_global;

  std::vector<int> np_dac_local, np_dac_global; // # particles in each DAC
  std::vector<int> np_loc_vec;
  std::vector<float> weight_p, weight_dac;

  std::array<val_type,2> loc_bound;

  DynamicAllocate(PIC* pic) :
  mpi_size(pic->para->mpi_size),
  mpi_rank(pic->para->mpi_rank),
  n_dac_tot(pic->para->mpi_size*4),
  L(pic->cells.upper_bound[0]-pic->cells.lower_bound[0]){
    // initial 
    n_dac = 4; 

    L_loc = L / mpi_size;
    L_dac = L_loc / n_dac;
  
    np_loc_vec.resize(mpi_size);
    n_dac_global.resize(mpi_size);
    weight_p.resize(n_dac_tot); weight_dac.resize(n_dac_tot);
    np_dac_local.resize(n_dac);
    np_dac_global.resize(n_dac_tot);

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

    RANK0
      std::cout << "#particle in all ranks: " << np_tot << std::endl;

    np_dac_local = getParticleNumInChunks(&pic->all_particles[0],
                                          loc_bound, n_dac);

    // 告诉rank0每个rank现在有几个DAC, 存在 n_dac_global 里面
    MPI_Gather(&n_dac,1,MPI_INT,n_dac_global.data(),1,MPI_INT,0,MPI_COMM_WORLD);

    /// 告诉rank0每个rank的每个DAC里面都有多少粒子
    MPI_Gather(&np_loc,1,MPI_INT,np_loc_vec.data(),1,MPI_INT,0,MPI_COMM_WORLD);

    int *p; 
    if (mpi_rank==0) {
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
      MPI_Barrier(MPI_COMM_WORLD);
      p += n_dac_global[r];
    }


    RANK0 { // 重新分配n_dac
      std::copy(np_dac_global.begin(),np_dac_global.end(),
                std::ostream_iterator<int>(std::cout," "));
      int i=0;
      for (int r=0; r<mpi_size; r++) {
        int np_buf=0, n_buf=0;
        while ((np_buf<np_mean*0.95) && i<n_dac_tot) {
          np_buf += np_dac_global[i++];
          n_buf++;
          std::cout<<std::endl;
          std::cout << i << "," << np_buf << "," << r<<std::endl;
        } 
        n_dac_global[r] = n_buf;
      }
      int rem = n_dac_tot-std::accumulate(n_dac_global.begin(),n_dac_global.end(),0);
      n_dac_global[mpi_size-1] += rem;

      std::cout << "--" << std::endl;
      std::copy(n_dac_global.begin(),n_dac_global.end(),
                std::ostream_iterator<int>(std::cout," "));
      std::cout << "--" << std::endl;
    } MPI_Barrier(MPI_COMM_WORLD);

  }



};

} // namespace xpic
