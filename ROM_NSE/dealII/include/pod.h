#ifndef POD_H
#define POD_H

#include "../include/nse.h"
#include "../include/RedSVD.h"
//#include "../include/rom.h"

//#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>

//#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
//#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
//#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

//#include <algorithm>
//#include <cmath>
#include <fstream>
//#include <iostream>
//#include <memory>
#include <string>
//#include <sstream>
#include <vector>
#include <filesystem>
#include <sys/stat.h>
#include <dirent.h>
#include <hdf5.h>
#include <deal.II/base/hdf5.h> // do I need this import? probably not.

// count number of files in directory 'path'
int compute_number_snapshots(std::string path);

// load velocity vector from HDF5 file
dealii::Vector<double> load_h5_vector(std::string file_path);

// load pressure vector from HDF5 file
dealii::Vector<double> load_h5_vector_pressure(std::string file_path);

namespace POD
{
  using namespace dealii;

  template<int dim>
  class ProperOrthogonalDecomposition
  {
  public:
    ProperOrthogonalDecomposition(PDEInfo *pde_info);
    ProperOrthogonalDecomposition();
    void run(int refinements,bool output_files);
    void run_greedy(int refinements,bool output_files, bool mean);
    void run_greedy_vp(int refinements,bool output_files);
    void setup(PDEInfo *pde_info,int refinements);
    void setup_vp(PDEInfo *pde_info,int refinements);
    double quad_weight(int i);
    void compute_pod_basis(double information_content, int pod_basis_size);
    void compute_pod_basis_supremizer(double information_content, int pod_basis_size);
    void compute_pod_basis_no_mean(double information_content, int pod_basis_size);
    void compute_pod_basis_no_mean_press(double information_content, int pod_basis_size);
    void compute_pod_basis_SVD(double information_content, int pod_basis_size);
    void compute_pod_basis_eigen(double information_content, int pod_basis_size);
    void compute_pod_basis_pressure(double information_content, int pod_basis_size);
    void compute_lifting_function();
    void save_h5_vector(const std::string &file_name, const Vector<double> &vector);
    void load_h5_vector_HF(const std::string &file_name, Vector<double> &vector);
    void save_h5_matrix(const std::string &file_name, const dealii::FullMatrix<double> &matrix);
    void output_mean_vector() const;
    void output_mean_vector_pressure() const;
    void output_pod_vectors() const;
    void output_pod_vectors_pressure() const;
    void output_eigenvalues() const;
    void output_eigenvalues_pressure() const;
    void compute_reduced_matrices(double time_step, double theta);
    FullMatrix<double> compute_reduced_matrix(SparseMatrix<double> &fem_matrix);
    FullMatrix<double> compute_reduced_matrix_pressure(SparseMatrix<double> &fem_matrix);
    Vector<double> compute_reduced_vector(Vector<double> &fem_vector);
    Vector<double> compute_reduced_vector_pressure(Vector<double> &fem_vector);
    NSE::NavierStokes<dim> navier_stokes_solver;
    int r = 0; // POD basis size
    int r_p = 0; // pressure POD basis size
    int r_s = 0;
    int m; // number of dofs
    int m_p; // number of pressure dofs
    std::vector<Vector<double>> pod_vectors;
    std::vector<Vector<double>> pod_vectors_p;
    Vector<double> mean_vector;
    Vector<double> mean_vector_p;
    int n; // number of snapshots
//    int n_start = 300;


    // VELOCITY ROM
    FullMatrix<double> reduced_linear_operator_theta;
    FullMatrix<double> reduced_linear_operator_one_minus_theta;
    Vector<double> reduced_mean_vector_contribution_rhs;
    std::vector<FullMatrix<double>> reduced_nonlinearity_tensor;

    FullMatrix<double> reduced_system_matrix;
    FullMatrix<double> reduced_system_matrix_inverse;
    Vector<double> reduced_system_rhs;

    Vector<double> space_weights;

  private:
    PDEInfo *pde_info;
    std::vector<double> eigenvalues;
    std::vector<double> eigenvalues_p;
    std::string filename_h5;

    double fluid_density;
    double viscosity;
	double information_content;
	double timestep;
	int pod_basis_size;

//	ROM::ReducedOrderModel<dim> rom;
  };
} // namespace POD

#endif
