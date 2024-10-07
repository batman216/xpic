#pragma once
#include <thrust/partition.h>
#include <nccl.h>



template <typename Array, std::size_t...idx>
auto make_zip_iterator_from_pointer_array(Array &arr,std::index_sequence<idx...>) {
  return thrust::make_zip_iterator(arr[idx]->begin()...); 
}


template <typename Array, std::size_t...idx>
auto make_zip_iterator_from_array(Array &arr,std::index_sequence<idx...>) {
  return thrust::make_zip_iterator(arr[idx].begin()...); 
}


namespace xpic {
namespace particle {

template <typename PIC>
struct Swap {

  using val_type = PIC::value_t;
  const int xdim = PIC::x_dimension;
  nccl_traits<val_type> ncclType;
  ncclComm_t comm;
  cudaStream_t s;

  int n_mpi,r_mpi;
  char flag_mpi;
  std::array<val_type,2> bound;
  
  std::array<thrust::device_vector<val_type>,6> send_buffer;
  std::array<thrust::device_vector<val_type>,6> r_recv_buffer, l_recv_buffer;

  Swap(PIC *pic) : s(pic->s), comm(pic->comm),flag_mpi(pic->flag_mpi), 
  n_mpi(pic->n_mpi), r_mpi(pic->r_mpi) {

    val_type b = pic->cells.upper_bound_loc[xdim-1], 
             a = pic->cells.lower_bound_loc[xdim-1];
    val_type localL = b-a;

    bound[0] = r_mpi * localL;
    bound[1] = bound[0] + localL;

  }


  template <typename Particles>
  void operator()(Particles *p) {

    std::size_t np = p->np;
    auto zit = p->particle_zip_iterator();

    using Tuple = thrust::tuple<val_type,val_type,val_type,val_type,val_type,val_type>;

    val_type a = bound[0], b = bound[1];

    const int D = xdim-1; // 并行的维度
    // 把超出边界[a,b)的粒子都放到数组最后
    auto mid = thrust::stable_partition(zit,zit+np,[a,b]__host__ __device__(Tuple t)
                        { return thrust::get<2>(t)>=a&&thrust::get<2>(t)<b;});

    // 需要保留的粒子数和需要发送走的粒子数
    std::size_t n_remain = static_cast<std::size_t>(mid-zit),
                n_send   = np - n_remain;

    // 在需要发送走的粒子中，往左发送的放前面
    auto midlr = thrust::stable_partition(mid,mid+n_send,[b]__host__ __device__(Tuple t)
                         { return thrust::get<2>(t)<b;});

    // 往左发送的粒子数和往右发送的粒子数
    std::size_t n_l_send = static_cast<std::size_t>(midlr-mid),
                n_r_send = n_send - n_l_send;
 
    for (int d=0; d<6; d++) send_buffer[d].resize(n_send);

    auto zit_buf = make_zip_iterator_from_array(send_buffer,
                                                std::make_index_sequence<6>{});
    thrust::copy(mid,mid+n_send,zit_buf);
 
    int l_rank = r_mpi==0? n_mpi - 1 : r_mpi-1,
        r_rank = r_mpi==n_mpi - 1 ? 0 : r_mpi+1;
    
    int n_r_recv{}, n_l_recv{};

    MPI_Send(&n_l_send,1,MPI_INT,l_rank,0,MPI_COMM_WORLD);
    MPI_Recv(&n_r_recv,1,MPI_INT,r_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    MPI_Send(&n_r_send,1,MPI_INT,r_rank,0,MPI_COMM_WORLD);
    MPI_Recv(&n_l_recv,1,MPI_INT,l_rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    for (int d=0; d<6; d++) r_recv_buffer[d].resize(n_r_recv);
    for (int d=0; d<6; d++) l_recv_buffer[d].resize(n_l_recv);

    auto zit_rrecvbuf = make_zip_iterator_from_array(r_recv_buffer,
                                                     std::make_index_sequence<6>{});
    auto zit_lrecvbuf = make_zip_iterator_from_array(l_recv_buffer,
                                                     std::make_index_sequence<6>{});
    ncclGroupStart();   // <-----向左发送--------
    for (int d=0; d<6; d++) {
      if (r_mpi>0)
        ncclSend(thrust::raw_pointer_cast(send_buffer[d].data()),
                 n_l_send,ncclType.name,l_rank,comm,s);
      if (r_mpi<n_mpi-1)
        ncclRecv(thrust::raw_pointer_cast(r_recv_buffer[d].data()),
                 n_r_recv,ncclType.name,r_rank,comm,s);
    }
    ncclGroupEnd();

    ncclGroupStart();   // -------向右发送-------->
    for (int d=0; d<6; d++) {
      if (r_mpi<n_mpi-1)
        ncclSend(thrust::raw_pointer_cast(send_buffer[d].data()+n_l_send),
                 n_r_send,ncclType.name,r_rank,comm,s);
      if (r_mpi>0)
        ncclRecv(thrust::raw_pointer_cast(l_recv_buffer[d].data()),
                 n_l_recv,ncclType.name,l_rank,comm,s);
    }
    ncclGroupEnd();

    np += - n_send + n_r_recv + n_l_recv;

    for (int d=0; d<3; d++) {
      p->x[d]->resize(np);
      p->v[d]->resize(np);
    }

    p->np = np; assert(p->np==p->x[D]->size());

    int nhere=thrust::count_if(p->x[D]->begin(),p->x[D]->end(),
                     [a,b]__host__ __device__(const val_type &x)
                     { return x>=a && x < b;  });

    /// resize了之后，zip_iterator 需要重新初始化
    zit = p->particle_zip_iterator();
    if (r_mpi>0)
      thrust::copy(zit_lrecvbuf,zit_lrecvbuf+n_l_recv,zit+n_remain);
    if (r_mpi<n_mpi-1)
      thrust::copy(zit_rrecvbuf,zit_rrecvbuf+n_r_recv,zit+n_remain+n_l_recv);
  }

};

} // namespace particle
} // namespace xpic
