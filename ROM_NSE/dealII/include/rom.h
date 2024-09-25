#ifndef ROM_H
#define ROM_H

#include "../include/pod.h"
// #include "../include/nn.h" by HF
#include <deal.II/lac/solver_gmres.h>

#include <cmath>

namespace ROM
{
  using namespace dealii;

  template<int dim>
  class ReducedOrderModel
  {
  public:
	ReducedOrderModel();
    ReducedOrderModel(PDEInfo *pde_info);
    void run(int refinements, bool output_files, bool compute_error);
    void run_vp(int refinements, bool output_files, bool compute_error);
    void run_no_mean(int refinements, bool output_files, bool compute_error);

  private:
    void setup(int refinements, bool output_files);
    void setup_vp(int refinements, bool output_files);
    FullMatrix<double> nonlinearity_evaluated_at(int argument, Vector<double> &vector);
    void nonlinearity_evaluated_at_void(int argument, Vector<double> &vector, FullMatrix<double> &output_matrix);
    void nonlinearity_evaluated_inner_loop(int argument, FullMatrix<double> &output_matrix, Vector<double> &vector, int i);
    Vector<double> nonlinearity_twice_evaluated_at_pressure(Vector<double> &vector);
    FullMatrix<double> nonlinearity_first_matrix_pressure(Vector<double> &mean_vector);
    FullMatrix<double> nonlinearity_second_matrix_pressure(Vector<double> &mean_vector);
    Vector<double> nonlinearity_mean_contribution_pressure(Vector<double> &mean_vector);
    Vector<double> laplace_mean_contribution_pressure(Vector<double> &mean_vector);
    Vector<double> nonlinearity_boundary_twice_evaluated_at_pressure(Vector<double> &vector);
    FullMatrix<double> nonlinearity_first_matrix_boundary_pressure(Vector<double> &mean_vector);
    FullMatrix<double> nonlinearity_second_matrix_boundary_pressure(Vector<double> &mean_vector);
    Vector<double> nonlinearity_boundary_mean_contribution_pressure(Vector<double> &mean_vector);
    void compute_reduced_matrices(double time_step, double theta);
    void load_reduced_matrices(); // HF
    void compute_reduced_matrices_linearized(double time_step, double theta);
    void compute_reduced_matrices_pressure();
    void assemble_system_matrix(double time_step, double theta);
    void assemble_system_matrix_penalty(double time_step, double theta, double tau);
    void assemble_system_rhs(double time_step, double theta);
    void assemble_system_rhs_no_mean(double time_step, double theta);
    void vmult_for_pp(FullMatrix<double> &matrix, Vector<double> &result, Vector<double> &vector);
    void solve_reduced_order_model(double start_time, double end_time, double time_step, double theta, bool output_files, bool compute_error);
    void solve_reduced_order_model_vp(double start_time, double end_time, double time_step, double theta, bool output_files, bool compute_error);
    void solve_reduced_order_model_no_mean(double start_time, double end_time, double time_step, double theta, bool output_files, bool compute_error);
    void newton_iteration(double time, double time_step, double theta);
    void newton_iteration_no_mean(double time, double time_step, double theta);
    void compute_linearized_velocity(double time_step, double theta);
    void compute_projected_solution();
    void compute_projected_solution_no_mean();
    void compute_projected_solution_pressure();
    void output_results();
    void output_results_vp();
    void output_results_pressure();
    void output_error();
    void solve();
    void solve_linearized();
    void pressure_reconstruction();
//    void pressure_reconstruction_with_nn();
    void compute_functional_values();
    void compute_functional_values_vp();
    void compute_drag_lift_tensor();
    void compute_drag_lift_tensor_vp();
    double compute_pressure(Point<dim> p) const;
    double compute_pressure_vp(Point<dim> p) const;
//    void init_neural_network(int n_hidden_layers, int n_hidden_neurons);
//    void collect_training_data();
    void save_h5_reduced_matrix(const std::string &file_name, const dealii::FullMatrix<double> &matrix);
    void load_h5_reduced_matrix(const std::string &file_name, dealii::FullMatrix<double> &matrix);
    void load_reduced_matrices(double time_step, double theta);
    void load_reduced_matrices_greedy(double time_step, double theta);
    void load_reduced_matrices_greedy_vp(double time_step, double theta);
    void load_reduced_matrices_greedy_no_mean(double time_step, double theta);

    void save_h5_matrix(const std::string &file_name, const dealii::FullMatrix<double> &matrix);
    void load_h5_matrix(const std::string &file_name, dealii::FullMatrix<double> &matrix);


    POD::ProperOrthogonalDecomposition<dim> pod;

//    NN::FCNN neural_network;
    std::vector<std::vector<double>> training_data_velocity; // input
    std::vector<std::vector<double>> training_data_pressure; // output

    // VELOCITY ROM
    FullMatrix<double> reduced_linear_operator_theta;
    FullMatrix<double> reduced_linear_operator_one_minus_theta;
    FullMatrix<double> reduced_indicator_matrix;
    Vector<double> reduced_mean_vector_contribution_rhs;
    std::vector<FullMatrix<double>> reduced_nonlinearity_tensor;

    FullMatrix<double> reduced_system_matrix;
    FullMatrix<double> reduced_system_matrix_inverse;
    Vector<double> reduced_system_rhs;


    Vector<double> old_solution; // last Newton iterate
    Vector<double> old_timestep_solution; // last time step
    Vector<double> newton_update; // update in the Newton step
    Vector<double> newton_update_interim; // update in the Newton step
    Vector<double> solution; // current solution
    Vector<double> projected_solution; // current solution projected into the FEM space
    BlockVector<double> projected_solution_combined;

    // PRESSURE ROM
    std::vector<FullMatrix<double>> reduced_nonlinearity_tensor_pressure;
    FullMatrix<double> first_nonlinearity_matrix_pressure;
    FullMatrix<double> second_nonlinearity_matrix_pressure;
    Vector<double> nonlin_mean_contrib_rhs_pressure;
    Vector<double> laplace_mean_contrib_rhs_pressure;
    FullMatrix<double> boundary_integral_matrix_pressure;
    Vector<double> boundary_mean_contrib_rhs_pressure;
    std::vector<FullMatrix<double>> reduced_nonlinearity_tensor_boundary_pressure;
    FullMatrix<double> first_nonlinearity_matrix_boundary_pressure;
    FullMatrix<double> second_nonlinearity_matrix_boundary_pressure;
    Vector<double> nonlin_boundary_mean_contrib_rhs_pressure;

    FullMatrix<double> reduced_laplace_matrix_pressure;
    FullMatrix<double> reduced_system_matrix_inverse_pressure; //--> iterative solver instead ?!
    Vector<double> reduced_system_rhs_pressure;
    Vector<double> solution_pressure; // current solution
    Vector<double> projected_solution_pressure; // current solution projected into the FEM space


    PDEInfo *pde_info;

    double timestep_number = 0;
    double time;
    double 		end_time;


    double error = 0.0;

    double fluid_density;
    double viscosity;
    double newton_tol;
    std::string intial_solution_snapshot;
    std::string filename_h5;

    bool sfbm = true;
    int r_vs = 0;
    int r_p = 0;
    int r_additional_for_nonlinearity = 0;

    // measures for efficiency
    std::vector<std::vector<double>> computational_time_per_iteration;
  };

}
#endif
