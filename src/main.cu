#include <iostream>
#include <vector>
#include <hdf5.h>
#include <mpi.h>
#include <chrono>

#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/generate.h"
#include "thrust/execution_policy.h"
#include "include/xpic/ParticleInCell.hpp"
#include "include/xpic/particle/push_methods.hpp"
#include "include/xpic/loadParticle.hpp"
#include "include/xpic/interpolate.hpp"
#include "include/xpic/ParallelCommunicator.hpp"
#include "include/Timer.h"

#define L_ t0.tick();
#define _L t0.tock();

#define p_ t1.tick();
#define _p t1.tock();

#define I_ t2.tick();
#define _I t2.tock();

using Real = double;

int main(int argc, char** argv) {
  
  MPI_Init(&argc,&argv);

  xpic::ParticleInCell<Real,3,3> pic; 

  std::cout << pic.n_species << std::endl;

  std::size_t np = pic.np_init[0];

  Timer t0("Load particle"),t1("push particle"), t2("Interpolate"); 

  L_ xpic::loadParticle(&pic); _L

  int mpi_rank, mpi_size; 
  MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);
  xpic::ParallelCommunicator<std::size_t,Real>
    para(mpi_rank,mpi_size,MPI_COMM_WORLD);

  xpic::Interpolate interpolate(&pic.all_particles[0],&pic.cells);

  xpic::particle::SymplecticEuler pusher(&pic.all_particles[0],0.2);

  std::ofstream pos("Y.data",std::ios::out);
  std::ofstream fieldos("field.data",std::ios::out);

  int step = 0, total_step = 100;
  while (step++<total_step) {

    p_  pusher.push(&pic.all_particles[0],&pic.cells); _p

    I_ interpolate.go(); _I
/*
    thrust::copy(interpolate.Y.begin(),interpolate.Y.end(),
                 std::ostream_iterator<Real>(fieldos," "));
    fieldos << std::endl;
    thrust::copy(pic.all_particles[0].x[0].begin(),pic.all_particles[0].x[0].end(),
                 std::ostream_iterator<Real>(pos," "));
    thrust::copy(pic.all_particles[0].x[1].begin(),pic.all_particles[0].x[1].end(),
                 std::ostream_iterator<Real>(pos," "));
    thrust::copy(pic.all_particles[0].x[2].begin(),pic.all_particles[0].x[2].end(),
                 std::ostream_iterator<Real>(pos," "));

    thrust::copy(pic.all_particles[0].v[0].begin(),pic.all_particles[0].v[0].end(),
                 std::ostream_iterator<Real>(pos," "));
    thrust::copy(pic.all_particles[0].v[1].begin(),pic.all_particles[0].v[1].end(),
                 std::ostream_iterator<Real>(pos," "));
    thrust::copy(pic.all_particles[0].v[2].begin(),pic.all_particles[0].v[2].end(),
                 std::ostream_iterator<Real>(pos," "));
    pos << std::endl;
*/
  }

  fieldos.close();
  pos.close();

  /*
  pic.cells.Bfield.field.resize(pic.cells.n_cell_tot);
  thrust::generate(pic.cells.Bfield.field.begin(),pic.cells.Bfield.field.end(),
                   [] __host__ __device__ { return xpic::cell::Node<Real,3>{1,2,423};});
  
  
  hsize_t dims[1] {3*pic.cells.n_cell_tot};

  hid_t h5file     = H5Fcreate("test.h5",H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
  hid_t data_type  = H5Tcopy(H5T_NATIVE_Real);
  hid_t data_space = H5Screate_simple(1,dims,NULL);
  hid_t dset_field = H5Dcreate(h5file, "/field", data_type, data_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  //std::vector<Real> buffer(pic.cells.n_cell_tot*3);
  thrust::device_vector<Real> buffer(pic.cells.n_cell_tot*3);
  auto ptr      = thrust::raw_pointer_cast(pic.cells.Bfield.field.data());
  auto it_begin = reinterpret_cast<Real*>(ptr);
  auto it_end   = reinterpret_cast<Real*>(ptr+pic.cells.n_cell_tot);
  
  //thrust::for_each(thrust::device, it_begin, it_end, []__host__ __device__ (Real val) { printf("%f",val);});
  thrust::copy(thrust::device,it_begin, it_end, buffer.data());
  thrust::host_vector<Real> hbuffer = buffer;
  std::puts("here");
  H5Dwrite(dset_field,H5T_NATIVE_Real,H5S_ALL,H5S_ALL,H5P_DEFAULT,hbuffer.data());
 
  H5Sclose(data_space); 
  H5Tclose(data_type);
  H5Dclose(dset_field);
  H5Fclose(h5file);
  */
}

