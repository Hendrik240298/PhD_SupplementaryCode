#ifndef POD_GREEDY_H
#define POD_GREEDY_H

#include "../include/nse.h"
#include "../include/pod.h"
#include "../include/rom.h"

#include <experimental/filesystem>

namespace POD_GREEDY
{
  using namespace dealii;

  template<int dim>
  class PODGreedy
  {
  public:
	  PODGreedy();
	  PODGreedy(PDEInfo *pde_info);
	  void run(int refinements, bool output_files);
	  void run_vp(int refinements, bool output_files);

  private:
	  void generate_surrogate();
	  void setup(int refinements);
	  void setup_vp(int refinements);
	  void compute_POD();
	  void compute_POD_vp_velo();
	  void compute_POD_vp_supremizer();
	  void compute_POD_vp_press();
	  void assemble_velo_supr_modes();
	  void assemble_supremizer(double time);
	  void compute_supremizer();
	  void compute_supremizer_each(int i,Vector<double> &pressure_snapshot,Vector<double> &output);
	  void compute_POD_no_mean();
	  void compute_mean_vector();
	  void compute_reduced_matrices(double time_step);
	  void compute_reduced_matrices_vp();
	  void compute_reduced_matrices_no_mean(double time_step);
	  void apply_boundary_values_rhs(const std::map<types::global_dof_index, double> &boundary_values,
	  																SparseMatrix<double> &matrix,
	  																Vector<double> &right_hand_side);
	  std::vector<int> greedy_loop();
	  double compute_error();
	  FullMatrix<double> compute_reduced_matrix(SparseMatrix<double> &fem_matrix);
	  FullMatrix<double> compute_reduced_matrix_vp(BlockSparseMatrix<double> &fem_matrix);
	  Vector<double> compute_reduced_vector(Vector<double> &fem_vector);
	  Vector<double> compute_projected_vector(Vector<double> &rom_vector);
	  Vector<double> compute_reduced_vector_vp(BlockVector<double> &fem_vector);
	  NSE::NavierStokes<dim> navier_stokes_solver;
	  POD::ProperOrthogonalDecomposition<dim> pod_solver;
//	  ROM::ReducedOrderModel<dim> rom_solver;
      PDEInfo *pde_info;
      Vector<double> mean_vector;
      BlockVector<double> lifting_vector;
      std::vector<double> surrogate;
      std::vector<double> surrogate_sample;
      std::vector<double> surrogate_pod_size_per_parameter;
      std::vector<Vector<double>> surrogate_mean_vectors;

      std::vector<Vector<double>> pod_vectors;

      std::vector<Vector<double>> pod_vectors_all_mu;
      std::vector<BlockVector<double>> pod_vectors_velo;

      std::vector<Vector<double>> pod_vectors_all_mu_supremizer;
      std::vector<BlockVector<double>> pod_vectors_supremizer;

      std::vector<BlockVector<double>> pod_vectors_velo_supr;

      std::vector<Vector<double>> pod_vectors_all_mu_press;
      std::vector<BlockVector<double>> pod_vectors_press;

	  SparseMatrix<double> M_supremizer;
	  SparseMatrix<double> B_supremizer;
	  SparseDirectUMFPACK M_supremizer_direct;

	  std::vector<std::map<unsigned int, double>> boundary_values_vector;
      // greedy
      double max_error_mu;

      bool mean = true;

      int r=0;
      int r_p = 0;
      int r_v = 0;
      int r_s = 0;
      int r_vs = 0;
      int refinements;
      double viscosity_tmp;
      std::string filename_h5;

      int modes_of_each_para = 5;
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
} // namespace POD_GREEDY

#endif
