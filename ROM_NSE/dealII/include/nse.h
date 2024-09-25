#ifndef NSE_H
#define NSE_H

#include <deal.II/base/hdf5.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>
#include <iostream>
#include <cmath>
#include <dirent.h>

// load block vector from HDF5 file
dealii::BlockVector<double> load_h5_block_vector(const std::string &file_path);

//
double load_h5_time(const std::string &file_path);

// save block vector to HDF5 file
void save_h5_block_vector(const std::string &file_path, const dealii::BlockVector<double> &block_vector, const double time);

// Struct which holds relevant information about the PDE
struct PDEInfo
{
  std::string test_case;
  double newton_tol;
  double fluid_density;
  double viscosity;
  double start_time;
  std::string intial_solution_snapshot;
  std::string fem_path;
  std::string pod_path;
  std::string rom_path;
  double coarse_timestep;
  double fine_timestep;
  double coarse_endtime;
  double fine_endtime;
  double POD_start_time;
  bool POD_offline;
  double information_content;
  double information_content_greedy;
  int pod_basis_size;
  int pod_greedy_basis_size;
  int pod_greedy_basis_pressure_size;
  int hidden_layers;
  int hidden_neurons;
};

namespace NSE
{
  using namespace dealii;

  template <int dim>
  class NavierStokes
  {
  public:
    // overloading the constructor
    NavierStokes();
    NavierStokes(PDEInfo *pde_info); //std::string test_case="2D-1", double newton_tol=1.0e-8, double fluid_density=1.0, double viscosity=1.0e-3);
    void run(int refinements, bool output_files);
    void run_ROM(int refinements, bool output_files);
    void setup_system(int refinements);
    void setup_system_ROM(int refinements);
    void setup_system_only_velocity(int refinements);
    void setup_system_only_pressure(int refinements);
    void init(PDEInfo *pde_info);
//    void set_viscosity(double viscosity);

    void assemble_first_convection_term_with_mean(Vector<double> &mean_vector);
    void assemble_second_convection_term_with_mean(Vector<double> &mean_vector);
    void assemble_convection_term_with_lifting(BlockVector<double> &lifting);
    void assemble_nonlinearity_tensor_velocity(std::vector<Vector<double>> &pod_vectors);
    void assemble_nonlinearity_tensor_velocity_vp(std::vector<BlockVector<double>> &pod_vector);
    void assemble_nonlinearity_tensor_velocity_vp_i(BlockVector<double> &pod_vector, int i);
    void assemble_mean_vector_contribution_rhs(Vector<double> &mean_vector, double rom_time_step);
    void assemble_mean_vector_contribution_rhs_greedy(Vector<double> &mean_vector, std::vector<Vector<double>> &mean_vector_contribution_rhs_vector);
    void assemble_lifting_contribution_rhs_greedy_vp(BlockVector<double> &lifting_vector, std::vector<BlockVector<double>> &lifting_vector_contribution_rhs_vector);
    void assemble_FOM_matrices_aff_decomp();

    void assemble_nonlinearity_tensor_pressure(std::vector<Vector<double>> &pod_vectors);
    void assemble_nonlinearity_tensor_boundary_pressure(std::vector<Vector<double>> &pod_vectors);
    FullMatrix<double> assemble_boundary_integral_pressure(std::vector<Vector<double>> &pod_vectors, std::vector<Vector<double>> &pod_vectors_p);
    Vector<double> assemble_boundary_integral_with_mean_pressure(std::vector<Vector<double>> &pod_vectors_p, Vector<double> &mean_vector);

    Vector<double> reconstruct_pressure_FEM(Vector<double> &velocity_solution);

    Triangulation<dim> triangulation;
    FESystem<dim>          fe;
    DoFHandler<dim>    dof_handler;
    FESystem<dim>          fe_velocity;
    DoFHandler<dim>    dof_handler_velocity;
    FESystem<dim>          fe_pressure;
    DoFHandler<dim>    dof_handler_pressure;

    BlockSparsityPattern      sparsity_pattern;
    SparsityPattern      sparsity_pattern_velocity;
    SparsityPattern      sparsity_pattern_pressure;
    // POD-ROM objects
    SparseMatrix<double> mass_matrix_velocity;
    SparseMatrix<double> laplace_matrix_velocity;
    SparseMatrix<double> laplace_matrix_velocity_with_transposed_trial_function;
    SparseMatrix<double> boundary_matrix_velocity;
    SparseMatrix<double> indicator_matrix_velocity;
    SparseMatrix<double> first_convection_matrix_velocity_with_mean;
    SparseMatrix<double> second_convection_matrix_velocity_with_mean;
    SparseMatrix<double> gradient_matrix;
    std::vector<SparseMatrix<double>> nonlinear_tensor_velocity;
    Vector<double> mean_vector_contribution_rhs;

    // v-p ROM matrices // lifting (akt mean must be applied !! - matrices !! all in bif v-p blocks
    BlockSparseMatrix<double> mass_matrix_vp;
    BlockSparseMatrix<double> laplace_matrix_vp;
    BlockSparseMatrix<double> boundary_matrix_vp;
    BlockSparseMatrix<double> pressure_matrix_vp;
    BlockSparseMatrix<double> incompressibilty_matrix_vp;
    BlockSparseMatrix<double> first_convection_matrix_velocity_with_mean_vp;
    BlockSparseMatrix<double> second_convection_matrix_velocity_with_mean_vp;
    std::vector<BlockSparseMatrix<double>> nonlinear_tensor_velocity_vp;

    SparseMatrix<double> mass_matrix_pressure;
    SparseMatrix<double> laplace_matrix_pressure;
    SparseMatrix<double> laplace_matrix_pressure_with_bc;
    SparseDirectUMFPACK laplace_matrix_pressure_inverse;
    std::vector<SparseMatrix<double>> nonlinear_tensor_pressure;
    std::vector<SparseMatrix<double>> nonlinear_tensor_boundary_pressure;

    AffineConstraints<double>    constraints;
    AffineConstraints<double>    no_constraints;
    AffineConstraints<double>    pressure_constraints;
    double theta;
    double time_step;
    int dofs_velocity;
    int dofs_pressure;

  private:
    void solve();
    void output_results() const;
    void set_initial_bc(const double time);
    void set_newton_bc();
    void assemble_system_matrix();
    void assemble_system_rhs();
    void newton_iteration(const double time);
    // Evaluation of functional values
    double compute_pressure(Point<dim> p) const;
    void compute_drag_lift_tensor();
    void compute_functional_values();

    void assemble_laplace_with_transposed_trial_function();
    void assemble_boundary_matrix();
    void assemble_gradient_matrix();
    void assemble_indicator_fct();



    SparseDirectUMFPACK A_direct;

    BlockSparseMatrix<double> system_matrix;
    BlockSparseMatrix<double> system_matrix_check;

    BlockVector<double> solution;
    BlockVector<double> old_timestep_solution;
    BlockVector<double> newton_update;
    BlockVector<double> system_rhs;

    double       time;
    unsigned int timestep_number;

    std::string test_case;
    double newton_tol;
    double fluid_density;
    double viscosity;

    double coarse_timestep;
    double fine_timestep;
    double coarse_endtime;
    double fine_endtime;
    double POD_start_time;
    bool POD_offline;
    std::string intial_solution_snapshot;
    std::string fem_path;
  };


  // In this class, we define a function
  // that deals with the boundary values.
  // For our configuration,
  // we impose of parabolic inflow profile for the
  // velocity at the left hand side of the channel.
  template <int dim>
  class BoundaryParabel : public Function<dim>
  {
    public:
    BoundaryParabel (const double time)
      : Function<dim>(dim+1)
      {
        _time = time;
      }

    virtual double value (const Point<dim>   &p,
  			const unsigned int  component = 0) const;

    virtual void vector_value (const Point<dim> &p,
  			     Vector<double>   &value) const;

  private:
    double _time;

  };
} // namespace NSE

#endif
