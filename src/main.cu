#include <iostream>
#include <vector>
#include <hdf5.h>
#include <chrono>

#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/generate.h"
#include "thrust/execution_policy.h"
#include "include/xpic/ParticleInCell.hpp"
#include "include/xpic/loadParticle.hpp"
#include "include/xpic/interpolate.hpp"
#include "include/Timer.h"

int main(int argc, char** argv) {

  xpic::ParticleInCell<double,3,3> pic; 

  std::cout << pic.n_species << std::endl;

  std::size_t np = pic.np_init[0];

  Timer t1("Load partcle"), t2("Interpolate"); 

  t1.tick();
  xpic::loadParticle(&pic);
  t1.tock();

  xpic::Interpolate interpolate(&pic.all_particles[0],&pic.cells);
  t2.tick();
  interpolate.go();
  t2.tock();


  std::ofstream cfs("cols",std::ios::out);
  thrust::copy(interpolate.A_cols.begin(),interpolate.A_cols.end(),
               std::ostream_iterator<double>(cfs," "));
  cfs.close();

  std::ofstream rfs("rows",std::ios::out);
  thrust::copy(interpolate.A_rows.begin(),interpolate.A_rows.end(),
               std::ostream_iterator<double>(rfs," "));
  rfs.close();

  std::ofstream vfs("vals",std::ios::out);
  thrust::copy(interpolate.A_vals.begin(),interpolate.A_vals.end(),
               std::ostream_iterator<double>(vfs," "));
  vfs.close();

  std::ofstream yfs("Y",std::ios::out);
  thrust::copy(interpolate.Y.begin(),interpolate.Y.end(),
               std::ostream_iterator<double>(yfs," "));
  yfs.close();

  /*
  pic.cells.Bfield.field.resize(pic.cells.n_cell_tot);
  thrust::generate(pic.cells.Bfield.field.begin(),pic.cells.Bfield.field.end(),
                   [] __host__ __device__ { return xpic::cell::Node<double,3>{1,2,423};});
  
  
  hsize_t dims[1] {3*pic.cells.n_cell_tot};

  hid_t h5file     = H5Fcreate("test.h5",H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
  hid_t data_type  = H5Tcopy(H5T_NATIVE_DOUBLE);
  hid_t data_space = H5Screate_simple(1,dims,NULL);
  hid_t dset_field = H5Dcreate(h5file, "/field", data_type, data_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  //std::vector<double> buffer(pic.cells.n_cell_tot*3);
  thrust::device_vector<double> buffer(pic.cells.n_cell_tot*3);
  auto ptr      = thrust::raw_pointer_cast(pic.cells.Bfield.field.data());
  auto it_begin = reinterpret_cast<double*>(ptr);
  auto it_end   = reinterpret_cast<double*>(ptr+pic.cells.n_cell_tot);
  
  //thrust::for_each(thrust::device, it_begin, it_end, []__host__ __device__ (double val) { printf("%f",val);});
  thrust::copy(thrust::device,it_begin, it_end, buffer.data());
  thrust::host_vector<double> hbuffer = buffer;
  std::puts("here");
  H5Dwrite(dset_field,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,hbuffer.data());
 
  H5Sclose(data_space); 
  H5Tclose(data_type);
  H5Dclose(dset_field);
  H5Fclose(h5file);
  */
}

