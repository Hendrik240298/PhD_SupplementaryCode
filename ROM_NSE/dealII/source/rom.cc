#include "../include/rom.h"

namespace ROM
{
  using namespace dealii;

  template<int dim>
  ReducedOrderModel<dim>::ReducedOrderModel()
  {

  }

  template<int dim>
  ReducedOrderModel<dim>::ReducedOrderModel(PDEInfo *pde_info)
  : pde_info(pde_info)
  {
    fluid_density = pde_info->fluid_density;
    viscosity =  pde_info->viscosity;
    intial_solution_snapshot = pde_info->intial_solution_snapshot;
    newton_tol = pde_info->newton_tol;
  }

  template<int dim>
  void ReducedOrderModel<dim>::setup_vp(int refinements, bool output_files)
  {
    pod.setup_vp(pde_info, refinements); // decompose
    std::cout << "start" << std::endl;
    std::string pod_vectors_file_path = pde_info->pod_path + "pod_vectors/";

    pod.r = compute_number_snapshots(pod_vectors_file_path);
	r_vs = pod.r ;//pod_basis_size; //compute_number_snapshots(pod_vectors_file_path)-1;

	pod_vectors_file_path = pde_info->pod_path + "pod_vectors_press/";
	pod.r_p = compute_number_snapshots(pod_vectors_file_path);
	r_p = pod.r_p;

	pod_vectors_file_path = pde_info->pod_path + "pod_vectors_supremizer/";
	pod.r_s = compute_number_snapshots(pod_vectors_file_path);

	std::cout << "Number of POD modes: ((v+s) + p): ((" << r_vs - pod.r_s <<" + " << pod.r_s <<") +" << r_p<< ")"<< std::endl;

	pod.m = pod.navier_stokes_solver.dofs_velocity;
	pod.m_p = pod.navier_stokes_solver.dofs_pressure;
  }

  template<int dim>
  void ReducedOrderModel<dim>::setup(int refinements, bool output_files)
  {
    pod.setup(pde_info, refinements); // decompose
    std::cout << "start" << std::endl;
    std::string pod_vectors_file_path = pde_info->pod_path + "pod_vectors/";
	pod.r = compute_number_snapshots(pod_vectors_file_path);//pod_basis_size; //compute_number_snapshots(pod_vectors_file_path)-1;
	pod.m = pod.navier_stokes_solver.dofs_velocity;
//	pod.m = pod.navier_stokes_solver.dof_handler_velocity.n_dofs();
//    filename_h5 = "result/POD/mu=0.001000/space_weights.h5";
//    pod.load_h5_vector_HF(filename_h5,pod.space_weights);
	std::cout << "Number of POD modes: " << pod.r << std::endl;
  }

  template<int dim>
  void ReducedOrderModel<dim>::compute_reduced_matrices(double time_step, double theta)
  {
    // project matrices from FEM space to POD space
    FullMatrix<double> reduced_mass_matrix = pod.compute_reduced_matrix(pod.navier_stokes_solver.mass_matrix_velocity);

    FullMatrix<double> reduced_laplace_matrix = pod.compute_reduced_matrix(pod.navier_stokes_solver.laplace_matrix_velocity);
    FullMatrix<double> reduced_laplace_matrix_with_transposed_trial_function = pod.compute_reduced_matrix(pod.navier_stokes_solver.laplace_matrix_velocity_with_transposed_trial_function);

    // integral over gamma out
    FullMatrix<double> reduced_boundary_matrix = pod.compute_reduced_matrix(pod.navier_stokes_solver.boundary_matrix_velocity);

    // mean convection terms
    pod.navier_stokes_solver.assemble_first_convection_term_with_mean(pod.mean_vector);
    pod.navier_stokes_solver.assemble_second_convection_term_with_mean(pod.mean_vector);
    FullMatrix<double> reduced_first_convection_matrix_with_mean = pod.compute_reduced_matrix(pod.navier_stokes_solver.first_convection_matrix_velocity_with_mean);
    FullMatrix<double> reduced_second_convection_matrix_with_mean = pod.compute_reduced_matrix(pod.navier_stokes_solver.second_convection_matrix_velocity_with_mean);

    // tensor (list of matrices) for the nonlinearity
    pod.navier_stokes_solver.assemble_nonlinearity_tensor_velocity(pod.pod_vectors);
    reduced_nonlinearity_tensor.resize(pod.r);
    for (int i = 0; i < pod.r; ++i){
        reduced_nonlinearity_tensor[i] = pod.compute_reduced_matrix(pod.navier_stokes_solver.nonlinear_tensor_velocity[i]);
        filename_h5 = pde_info->rom_path + "matrices/nonlinearity-" + Utilities::int_to_string(i, 6) + ".h5";
        save_h5_matrix(filename_h5, reduced_nonlinearity_tensor[i]);
    }

    // compute the reduced system and right hand side matrices
    // from the reduced mass and laplace matrices

    // NOTATION:

    // reduced matrices:
    // M = (ψ_j, ψ_i)_i,j=1^r
    // L = (∇ψ_j, ∇ψ_i)_i,j=1^r
    // L_t = (∇ψ_j^T, ∇ψ_i)_i,j=1^r
    // N = ((∇ψ_j^T · n, ψ_i)_Γ_out)_i,j=1^r
    // K_1 = ((\bar{v} · ∇)ψ_j, ψ_i)_i,j=1^r
    // K_2 = ((ψ_j · ∇)\bar{v}, ψ_i)_i,j=1^r

    // reduced tensor:
    // C = ((ψ_j · ∇)ψ_k, ψ_i)_i,j,k=1^r

    // A(θ) := ...
    // parameter independent
    reduced_linear_operator_theta.reinit(pod.r, pod.r);
    reduced_linear_operator_theta.add(fluid_density, reduced_mass_matrix);  // ρ * M
    reduced_linear_operator_theta.add(time_step * theta * fluid_density, reduced_first_convection_matrix_with_mean); // kθρ * K_1
    reduced_linear_operator_theta.add(time_step * theta * fluid_density, reduced_second_convection_matrix_with_mean); // kθρ * K_2
    // parameter dependent
    reduced_linear_operator_theta.add(time_step * theta * fluid_density * viscosity, reduced_laplace_matrix); // kθμ * L
    reduced_linear_operator_theta.add(time_step * theta * fluid_density * viscosity, reduced_laplace_matrix_with_transposed_trial_function); // kθμ * L_t
    reduced_linear_operator_theta.add(- time_step * theta * fluid_density * viscosity, reduced_boundary_matrix); // - kθμ * N

    filename_h5 = pde_info->rom_path + "matrices/A.h5";
    save_h5_matrix(filename_h5, reduced_linear_operator_theta);

    // A(-(1-θ)) := ...
    // parameter independent
    reduced_linear_operator_one_minus_theta.reinit(pod.r, pod.r);
    reduced_linear_operator_one_minus_theta.add(fluid_density, reduced_mass_matrix);  // ρ * M
    reduced_linear_operator_one_minus_theta.add(- time_step * (1.0 - theta) * fluid_density, reduced_first_convection_matrix_with_mean); // -k(1-θ)ρ * K_1
    reduced_linear_operator_one_minus_theta.add(- time_step * (1.0 - theta) * fluid_density, reduced_second_convection_matrix_with_mean); // -k(1-θ)ρ * K_2
    // paramter dependent
    reduced_linear_operator_one_minus_theta.add(- time_step * (1.0 - theta) * fluid_density * viscosity, reduced_laplace_matrix); // -k(1-θ)μ * L
    reduced_linear_operator_one_minus_theta.add(- time_step * (1.0 - theta) * fluid_density * viscosity, reduced_laplace_matrix_with_transposed_trial_function); // -k(1-θ)μ * L_t
    reduced_linear_operator_one_minus_theta.add(time_step * (1.0 - theta) * fluid_density * viscosity, reduced_boundary_matrix); // k(1-θ)μ * N

    filename_h5 = pde_info->rom_path + "matrices/At.h5";
    save_h5_matrix(filename_h5, reduced_linear_operator_one_minus_theta);

    pod.navier_stokes_solver.assemble_mean_vector_contribution_rhs(pod.mean_vector, time_step);
    reduced_mean_vector_contribution_rhs = pod.compute_reduced_vector(pod.navier_stokes_solver.mean_vector_contribution_rhs);


    reduced_system_matrix.reinit(pod.r,pod.r);
    reduced_system_rhs.reinit(pod.r);
    reduced_system_matrix_inverse.reinit(pod.r,pod.r);
  }

  template<int dim>
    void ReducedOrderModel<dim>::load_reduced_matrices(double time_step, double theta)
    {

	  reduced_linear_operator_theta.reinit(pod.r, pod.r);
      filename_h5 = pde_info->rom_path + "matrices/A.h5";
      load_h5_matrix(filename_h5, reduced_linear_operator_theta);
      std::cout << "Loading A" << std::endl;

      reduced_linear_operator_one_minus_theta.reinit(pod.r, pod.r);
      filename_h5 = pde_info->rom_path + "matrices/At.h5";
      load_h5_matrix(filename_h5, reduced_linear_operator_one_minus_theta);
      std::cout << "Loading A'" << std::endl;

      reduced_nonlinearity_tensor.resize(pod.r);
      for (int i = 0; i < pod.r; ++i){
          filename_h5 = pde_info->rom_path + "matrices/nonlinearity-" + Utilities::int_to_string(i, 6) + ".h5";
          load_h5_matrix(filename_h5, reduced_nonlinearity_tensor[i]);
      }
      std::cout << "Loading nonlinearities" << std::endl;

      filename_h5 = "result/POD/mu=0.001000/mean_vector.h5";
      filename_h5 = pde_info->pod_path + "mean_vector.h5";
      pod.load_h5_vector_HF(filename_h5,pod.mean_vector);
      std::cout << "Loading mean_vector" << std::endl;

      pod.pod_vectors.resize(pod.r);
      for (int i=0; i<pod.r; i++)
      {
      	filename_h5 = "result/POD/mu=0.001000/pod_vectors/pod_vectors"+ Utilities::int_to_string(i,6) +".h5";
    	filename_h5 = pde_info->pod_path + "pod_vectors/pod_vectors"+ Utilities::int_to_string(i,6) +".h5";
        pod.load_h5_vector_HF(filename_h5, pod.pod_vectors[i]);
      }
      std::cout << "Loading pod_vectors" << std::endl;

      pod.navier_stokes_solver.assemble_mean_vector_contribution_rhs(pod.mean_vector, time_step);
      std::cout << "1" << std::endl;

      filename_h5 = pde_info->rom_path + "matrices/mean_vector_rhs.h5";
      pod.load_h5_vector_HF(filename_h5,reduced_mean_vector_contribution_rhs);
//      reduced_mean_vector_contribution_rhs = pod.compute_reduced_vector(pod.navier_stokes_solver.mean_vector_contribution_rhs);
      std::cout << "2" << std::endl;
      reduced_system_matrix.reinit(pod.r,pod.r);
      std::cout << "3" << std::endl;
      reduced_system_rhs.reinit(pod.r);
      std::cout << "4" << std::endl;
      reduced_system_matrix_inverse.reinit(pod.r,pod.r);
      std::cout << "End Loading" << std::endl;
    }

  template<int dim>
    void ReducedOrderModel<dim>::load_reduced_matrices_greedy(double time_step, double theta)
    {
	  reduced_linear_operator_theta.reinit(pod.r, pod.r);
	  reduced_linear_operator_one_minus_theta.reinit(pod.r, pod.r);
	  FullMatrix<double> temp_matrix;
	  temp_matrix.reinit(pod.r,pod.r);

	  filename_h5 = pde_info->rom_path + "matrices/M.h5";
	  load_h5_matrix(filename_h5, temp_matrix);
	  reduced_linear_operator_theta.add				(fluid_density,temp_matrix);
	  reduced_linear_operator_one_minus_theta.add	(fluid_density,temp_matrix);

	  filename_h5 = pde_info->rom_path + "matrices/K1.h5";
	  load_h5_matrix(filename_h5, temp_matrix);
	  reduced_linear_operator_theta.add				(  time_step * theta         * fluid_density,temp_matrix);
	  reduced_linear_operator_one_minus_theta.add	(- time_step * (1.0 - theta) * fluid_density,temp_matrix);

	  filename_h5 = pde_info->rom_path + "matrices/K2.h5";
	  load_h5_matrix(filename_h5, temp_matrix);
	  reduced_linear_operator_theta.add				(  time_step * theta         * fluid_density,temp_matrix);
	  reduced_linear_operator_one_minus_theta.add	(- time_step * (1.0 - theta) * fluid_density,temp_matrix);

	  filename_h5 = pde_info->rom_path + "matrices/L.h5";
	  load_h5_matrix(filename_h5, temp_matrix);
	  reduced_linear_operator_theta.add				(  time_step * theta         * fluid_density * viscosity,temp_matrix);
	  reduced_linear_operator_one_minus_theta.add	(- time_step * (1.0 - theta) * fluid_density * viscosity,temp_matrix);

	  filename_h5 = pde_info->rom_path + "matrices/LT.h5";
	  load_h5_matrix(filename_h5, temp_matrix);
	  reduced_linear_operator_theta.add				(  time_step * theta         * fluid_density * viscosity,temp_matrix);
	  reduced_linear_operator_one_minus_theta.add	(- time_step * (1.0 - theta) * fluid_density * viscosity,temp_matrix);

	  filename_h5 = pde_info->rom_path + "matrices/B.h5";
	  load_h5_matrix(filename_h5, temp_matrix);
	  reduced_linear_operator_theta.add				(- time_step * theta         * fluid_density * viscosity,temp_matrix);
	  reduced_linear_operator_one_minus_theta.add	(  time_step * (1.0 - theta) * fluid_density * viscosity,temp_matrix);

      reduced_nonlinearity_tensor.resize(pod.r);
      for (int i = 0; i < pod.r; ++i){
          filename_h5 = pde_info->rom_path + "matrices/nonlinearity-" + Utilities::int_to_string(i, 6) + ".h5";
          load_h5_matrix(filename_h5, reduced_nonlinearity_tensor[i]);
      }
//      std::cout << "Loading nonlinearities" << std::endl;

      filename_h5 = pde_info->pod_path + "mean_vector.h5";
      pod.load_h5_vector_HF(filename_h5,pod.mean_vector);
//      std::cout << "Loading mean_vector" << std::endl;

      std::cout << " mv2: " << pod.mean_vector.linfty_norm() << std::endl;

      pod.pod_vectors.resize(pod.r);
      for (int i=0; i<pod.r; i++)
      {
    	filename_h5 = pde_info->pod_path + "pod_vectors/pod_vectors"+ Utilities::int_to_string(i,6) +".h5";
        pod.load_h5_vector_HF(filename_h5, pod.pod_vectors[i]);
      }
//      std::cout << "Loading pod_vectors" << std::endl;

//      pod.navier_stokes_solver.assemble_mean_vector_contribution_rhs(pod.mean_vector, time_step);

      filename_h5 = pde_info->rom_path + "matrices/mean_vector_rhs.h5";
      pod.load_h5_vector_HF(filename_h5,reduced_mean_vector_contribution_rhs);
//
//      std::cout <<reduced_mean_vector_contribution_rhs.norm_sqr() << std::endl;

      std::vector<Vector<double>> reduced_mean_vector_contribution_rhs_vector(3);

	  filename_h5 = pde_info->rom_path + "matrices/mean_vector_rhs_vector_0.h5";
	  pod.load_h5_vector_HF(filename_h5,reduced_mean_vector_contribution_rhs_vector[0]);
	  filename_h5 = pde_info->rom_path + "matrices/mean_vector_rhs_vector_1.h5";
	  pod.load_h5_vector_HF(filename_h5,reduced_mean_vector_contribution_rhs_vector[1]);
	  filename_h5 = pde_info->rom_path + "matrices/mean_vector_rhs_vector_2.h5";
	  pod.load_h5_vector_HF(filename_h5,reduced_mean_vector_contribution_rhs_vector[2]);

	  FullMatrix<double> L;
	  L.reinit(pod.r,pod.r);
	  FullMatrix<double> LT;
	  LT.reinit(pod.r,pod.r);
	  Vector<double> laplace_mean(pod.r);
	  filename_h5 = pde_info->rom_path + "matrices/L.h5";
	  load_h5_matrix(filename_h5, L);
	  filename_h5 = pde_info->rom_path + "matrices/LT.h5";
	  load_h5_matrix(filename_h5, LT);
	  L.add(1.,LT);
	  L.vmult(laplace_mean,pod.compute_reduced_vector(pod.mean_vector));
	  laplace_mean.add(-1.,reduced_mean_vector_contribution_rhs_vector[0]);
	  std::cout << "lapalce error: " << laplace_mean.linfty_norm() << std::endl;

	  reduced_mean_vector_contribution_rhs = 0;
	  reduced_mean_vector_contribution_rhs.add(time_step*pde_info->fluid_density*pde_info->viscosity, \
			  	  	  	  	  	  	  	  	  	  reduced_mean_vector_contribution_rhs_vector[0]);
	  reduced_mean_vector_contribution_rhs.add(time_step*pde_info->fluid_density, \
			  	  	  	  	  	  	  	  	  	  reduced_mean_vector_contribution_rhs_vector[1]);
	  reduced_mean_vector_contribution_rhs.add(time_step*pde_info->fluid_density*pde_info->viscosity, \
			  	  	  	  	  	  	  	  	  	  reduced_mean_vector_contribution_rhs_vector[2]);

//      std::cout <<reduced_mean_vector_contribution_rhs.norm_sqr() << std::endl;

//    reduced_mean_vector_contribution_rhs = pod.compute_reduced_vector(pod.navier_stokes_solver.mean_vector_contribution_rhs);
      reduced_system_matrix.reinit(pod.r,pod.r);
      reduced_system_rhs.reinit(pod.r);
      reduced_system_matrix_inverse.reinit(pod.r,pod.r);
//      std::cout << "End Loading" << std::endl;
    }

  template<int dim>
      void ReducedOrderModel<dim>::load_reduced_matrices_greedy_vp(double time_step, double theta)
      {
  	  reduced_linear_operator_theta.reinit(r_vs + r_p, r_vs + r_p);
  	  reduced_linear_operator_one_minus_theta.reinit(r_vs + r_p, r_vs + r_p);
  	  FullMatrix<double> temp_matrix;
  	  temp_matrix.reinit(r_vs + r_p,r_vs + r_p);

  	  filename_h5 = pde_info->rom_path + "matrices/M.h5";
  	  load_h5_matrix(filename_h5, temp_matrix);
  	  reduced_linear_operator_theta.add				(fluid_density,temp_matrix);
  	  reduced_linear_operator_one_minus_theta.add	(fluid_density,temp_matrix);

  	  filename_h5 = pde_info->rom_path + "matrices/A.h5";
  	  load_h5_matrix(filename_h5, temp_matrix);
  	  reduced_linear_operator_theta.add				(  time_step * theta         * fluid_density * viscosity,temp_matrix);
  	  reduced_linear_operator_one_minus_theta.add	(- time_step * (1.0 - theta) * fluid_density * viscosity,temp_matrix);

  	  filename_h5 = pde_info->rom_path + "matrices/D.h5";
  	  load_h5_matrix(filename_h5, temp_matrix);
  	  reduced_linear_operator_theta.add				(- time_step * theta         * fluid_density * viscosity,temp_matrix);
  	  reduced_linear_operator_one_minus_theta.add	(  time_step * (1.0 - theta) * fluid_density * viscosity,temp_matrix);

  	  filename_h5 = pde_info->rom_path + "matrices/B.h5"; // pressure
  	  load_h5_matrix(filename_h5, temp_matrix);
//  	  reduced_linear_operator_theta.add				(- time_step * theta         ,temp_matrix);
//  	  reduced_linear_operator_one_minus_theta.add	(  time_step * (1.0 - theta) ,temp_matrix);
  	  reduced_linear_operator_theta.add				(- time_step         	,temp_matrix);
  	  reduced_linear_operator_one_minus_theta.add	(  0.0					,temp_matrix);

  	  filename_h5 = pde_info->rom_path + "matrices/BT.h5"; // div
  	  load_h5_matrix(filename_h5, temp_matrix);
  	  reduced_linear_operator_theta.add				(  1   ,temp_matrix);
  	  reduced_linear_operator_one_minus_theta.add	(  0.0 ,temp_matrix);

  	  filename_h5 = pde_info->rom_path + "matrices/C.h5";
  	  load_h5_matrix(filename_h5, temp_matrix);
  	  reduced_linear_operator_theta.add				(  time_step * theta         * fluid_density,temp_matrix);
  	  reduced_linear_operator_one_minus_theta.add	(- time_step * (1.0 - theta) * fluid_density,temp_matrix);

  	  filename_h5 = pde_info->rom_path + "matrices/C1.h5";
  	  load_h5_matrix(filename_h5, temp_matrix);
  	  reduced_linear_operator_theta.add				(  time_step * theta         * fluid_density,temp_matrix);
  	  reduced_linear_operator_one_minus_theta.add	(- time_step * (1.0 - theta) * fluid_density,temp_matrix);

        reduced_nonlinearity_tensor.resize(r_vs);
        for (int i = 0; i < r_vs; ++i){
            filename_h5 = pde_info->rom_path + "matrices/nonlinearity-" + Utilities::int_to_string(i, 6) + ".h5";
            load_h5_matrix(filename_h5, reduced_nonlinearity_tensor[i]);
        }
  //      std::cout << "Loading nonlinearities" << std::endl;

        filename_h5 = pde_info->pod_path + "mean_vector.h5";
        pod.load_h5_vector_HF(filename_h5,pod.mean_vector);
  //      std::cout << "Loading mean_vector" << std::endl;

        std::cout << " mv2: " << pod.mean_vector.linfty_norm() << std::endl;


        pod.pod_vectors.resize(r_vs);
        for (int i=0; i<r_vs; i++)
        {
      	filename_h5 = pde_info->pod_path + "pod_vectors/pod_vectors"+ Utilities::int_to_string(i,6) +".h5";
          pod.load_h5_vector_HF(filename_h5, pod.pod_vectors[i]);
        }

        pod.pod_vectors_p.resize(r_p);
        for (int i=0; i<r_p; i++)
        {
      	filename_h5 = pde_info->pod_path + "pod_vectors_press/pod_vectors_press"+ Utilities::int_to_string(i,6) +".h5";
          pod.load_h5_vector_HF(filename_h5, pod.pod_vectors_p[i]);
        }

        std::vector<Vector<double>> reduced_mean_vector_contribution_rhs_vector(4);

  	  filename_h5 = pde_info->rom_path + "matrices/lifting_rhs_vector_laplace.h5";
  	  pod.load_h5_vector_HF(filename_h5,reduced_mean_vector_contribution_rhs_vector[0]);
  	  filename_h5 = pde_info->rom_path + "matrices/lifting_rhs_vector_convection.h5";
  	  pod.load_h5_vector_HF(filename_h5,reduced_mean_vector_contribution_rhs_vector[1]);
  	  filename_h5 = pde_info->rom_path + "matrices/lifting_rhs_vector_boundary.h5";
  	  pod.load_h5_vector_HF(filename_h5,reduced_mean_vector_contribution_rhs_vector[2]);
  	  filename_h5 = pde_info->rom_path + "matrices/lifting_rhs_vector_incompressibility.h5";
  	  pod.load_h5_vector_HF(filename_h5,reduced_mean_vector_contribution_rhs_vector[3]);

  	  reduced_mean_vector_contribution_rhs.reinit(r_vs+r_p);
  	  reduced_mean_vector_contribution_rhs.add(- time_step*pde_info->fluid_density*pde_info->viscosity, \
  			  	  	  	  	  	  	  	  	  	  reduced_mean_vector_contribution_rhs_vector[0]);
  	  reduced_mean_vector_contribution_rhs.add(- time_step*pde_info->fluid_density, \
  			  	  	  	  	  	  	  	  	  	  reduced_mean_vector_contribution_rhs_vector[1]);
  	  reduced_mean_vector_contribution_rhs.add(  time_step*pde_info->fluid_density*pde_info->viscosity, \
  			  	  	  	  	  	  	  	  	  	  reduced_mean_vector_contribution_rhs_vector[2]);
  	  reduced_mean_vector_contribution_rhs.add(  -1.0 , \
  			  	  	  	  	  	  	  	  	  	  reduced_mean_vector_contribution_rhs_vector[3]);


  //      std::cout <<reduced_mean_vector_contribution_rhs.norm_sqr() << std::endl;

  //    reduced_mean_vector_contribution_rhs = pod.compute_reduced_vector(pod.navier_stokes_solver.mean_vector_contribution_rhs);
        reduced_system_matrix.reinit(r_vs+r_p,r_vs+r_p);
        reduced_system_rhs.reinit(r_vs+r_p);
        reduced_system_matrix_inverse.reinit(r_vs+r_p,r_vs+r_p);


        if (sfbm)
        {
            // test reduced reduced matrx
//            std::cout << "build reduced reduced system ..." <<std::endl;
//            std::cout << "from :" << r_vs << " + " << r_p <<std::endl;
//            std::cout << "to   :" << pod.r-pod.r_s << " + " << pod.r_p << " with " << pod.r << " and " <<pod.r_s << std::endl;
            FullMatrix<double> reduced_linear_operator_theta_temp(r_vs+r_p,r_vs+r_p);
            FullMatrix<double> reduced_linear_operator_one_minus_theta_temp(r_vs+r_p,r_vs+r_p);
            Vector<double> reduced_mean_vector_contribution_rhs_temp(r_vs+r_p);

//            std::cout << "copy ..." <<std::endl;
            reduced_linear_operator_theta_temp.copy_from(reduced_linear_operator_theta);
            reduced_linear_operator_one_minus_theta_temp.copy_from(reduced_linear_operator_one_minus_theta);
            reduced_mean_vector_contribution_rhs_temp = reduced_mean_vector_contribution_rhs;

//            std::cout << "reinit ..." <<std::endl;
            reduced_linear_operator_theta.reinit(pod.r,pod.r);
    		reduced_linear_operator_one_minus_theta.reinit(pod.r,pod.r);
    		reduced_mean_vector_contribution_rhs.reinit(pod.r);

//    		std::cout << "1. block ..." <<std::endl;
    		for (int i = 0; i< pod.r; i++){
    			for (int j= 0; j<pod.r - pod.r_s;j++) {
    				reduced_linear_operator_theta[i][j] = reduced_linear_operator_theta_temp[i][j];
    				reduced_linear_operator_one_minus_theta[i][j] = reduced_linear_operator_one_minus_theta_temp[i][j];
    			}
    		}
//    		std::cout << "2. block ..." <<std::endl;
    		for (int i = 0; i< pod.r; i++){
    			for (int j= pod.r - pod.r_s; j<pod.r;j++) {
    //				std::cout << j << ", " << pod.r_s+j <<std::endl;
    				reduced_linear_operator_theta[i][j] = reduced_linear_operator_theta_temp[i][pod.r_s+j];
    				reduced_linear_operator_one_minus_theta[i][j] = reduced_linear_operator_one_minus_theta_temp[i][pod.r_s+j];
    			}
    		}
//    		std::cout << "rhs ..." <<std::endl;
    		for (int i = 0; i< pod.r; i++){
    			reduced_mean_vector_contribution_rhs[i] = reduced_mean_vector_contribution_rhs_temp[i];
    		}
    		for (int i = pod.r- pod.r_s; i< pod.r; i++){
    			reduced_mean_vector_contribution_rhs[i] = reduced_mean_vector_contribution_rhs_temp[i];
    		}
    		pod.r -= pod.r_s;
    		pod.r_p = pod.r_s;
            r_additional_for_nonlinearity = pod.r_s;
    		pod.r_s = 0;

//            std::cout << "reinit ..." <<std::endl;
            reduced_system_matrix.reinit(pod.r+pod.r_p  ,pod.r+pod.r_p );
            reduced_system_rhs.reinit(pod.r+pod.r_p );
            reduced_system_matrix_inverse.reinit(pod.r+pod.r_p,pod.r+pod.r_p );
            std::cout << "finished reduced reduced system ..." <<std::endl;
        }


        std::cout << "End Loading" << std::endl;
      }
  template<int dim>
      void ReducedOrderModel<dim>::load_reduced_matrices_greedy_no_mean(double time_step, double theta)
      {
  	  reduced_linear_operator_theta.reinit(pod.r, pod.r);
  	  reduced_linear_operator_one_minus_theta.reinit(pod.r, pod.r);
  	  reduced_indicator_matrix.reinit(pod.r,pod.r);
  	  FullMatrix<double> temp_matrix;
  	  temp_matrix.reinit(pod.r,pod.r);

  	  filename_h5 = pde_info->rom_path + "matrices/M.h5";
  	  load_h5_matrix(filename_h5, temp_matrix);
  	  reduced_linear_operator_theta.add				(fluid_density,temp_matrix);
  	  reduced_linear_operator_one_minus_theta.add	(fluid_density,temp_matrix);

  	  filename_h5 = pde_info->rom_path + "matrices/L.h5";
  	  load_h5_matrix(filename_h5, temp_matrix);
  	  reduced_linear_operator_theta.add				(  time_step * theta         * fluid_density * viscosity,temp_matrix);
  	  reduced_linear_operator_one_minus_theta.add	(- time_step * (1.0 - theta) * fluid_density * viscosity,temp_matrix);

  	  filename_h5 = pde_info->rom_path + "matrices/LT.h5";
  	  load_h5_matrix(filename_h5, temp_matrix);
  	  reduced_linear_operator_theta.add				(  time_step * theta         * fluid_density * viscosity,temp_matrix);
  	  reduced_linear_operator_one_minus_theta.add	(- time_step * (1.0 - theta) * fluid_density * viscosity,temp_matrix);

  	  filename_h5 = pde_info->rom_path + "matrices/B.h5";
  	  load_h5_matrix(filename_h5, temp_matrix);
  	  reduced_linear_operator_theta.add				(- time_step * theta         * fluid_density * viscosity,temp_matrix);
  	  reduced_linear_operator_one_minus_theta.add	(  time_step * (1.0 - theta) * fluid_density * viscosity,temp_matrix);

  	  filename_h5 = pde_info->rom_path + "matrices/Xi.h5";
  	  load_h5_matrix(filename_h5, reduced_indicator_matrix);

        reduced_nonlinearity_tensor.resize(pod.r);
        for (int i = 0; i < pod.r; ++i){
            filename_h5 = pde_info->rom_path + "matrices/nonlinearity-" + Utilities::int_to_string(i, 6) + ".h5";
            load_h5_matrix(filename_h5, reduced_nonlinearity_tensor[i]);
        }
  //      std::cout << "Loading nonlinearities" << std::endl;

        pod.pod_vectors.resize(pod.r);
        for (int i=0; i<pod.r; i++)
        {
      	filename_h5 = pde_info->pod_path + "pod_vectors/pod_vectors"+ Utilities::int_to_string(i,6) +".h5";
          pod.load_h5_vector_HF(filename_h5, pod.pod_vectors[i]);
        };
  //    reduced_mean_vector_contribution_rhs = pod.compute_reduced_vector(pod.navier_stokes_solver.mean_vector_contribution_rhs);
        reduced_system_matrix.reinit(pod.r,pod.r);
        reduced_system_matrix_inverse.reinit(pod.r,pod.r);
  //      std::cout << "End Loading" << std::endl;

      }

  template<int dim>
  void ReducedOrderModel<dim>::compute_reduced_matrices_linearized(double time_step, double theta)
  {
    // project matrices from FEM space to POD space
    FullMatrix<double> reduced_mass_matrix = pod.compute_reduced_matrix(pod.navier_stokes_solver.mass_matrix_velocity);

    FullMatrix<double> reduced_laplace_matrix = pod.compute_reduced_matrix(pod.navier_stokes_solver.laplace_matrix_velocity);
    FullMatrix<double> reduced_laplace_matrix_with_transposed_trial_function = pod.compute_reduced_matrix(pod.navier_stokes_solver.laplace_matrix_velocity_with_transposed_trial_function);

    // integral over gamma out
    FullMatrix<double> reduced_boundary_matrix = pod.compute_reduced_matrix(pod.navier_stokes_solver.boundary_matrix_velocity);

    // tensor (list of matrices) for the nonlinearity
    pod.navier_stokes_solver.assemble_nonlinearity_tensor_velocity(pod.pod_vectors);
    reduced_nonlinearity_tensor.resize(pod.r);
    for (int i = 0; i < pod.r; ++i)
      reduced_nonlinearity_tensor[i] = pod.compute_reduced_matrix(pod.navier_stokes_solver.nonlinear_tensor_velocity[i]);

    // mean convection terms
    pod.navier_stokes_solver.assemble_first_convection_term_with_mean(pod.mean_vector);
    pod.navier_stokes_solver.assemble_second_convection_term_with_mean(pod.mean_vector);
    FullMatrix<double> reduced_first_convection_matrix_with_mean = pod.compute_reduced_matrix(pod.navier_stokes_solver.first_convection_matrix_velocity_with_mean);
    FullMatrix<double> reduced_second_convection_matrix_with_mean = pod.compute_reduced_matrix(pod.navier_stokes_solver.second_convection_matrix_velocity_with_mean);

    /*
    // code for debugging reduced_first_convection_matrix_with_mean
    FullMatrix<double> first_matrix(pod.r, pod.r);
    Vector<double> tmpp(pod.m);
    Vector<double> restricted_tmpp(pod.r);
    for (int i = 0; i < pod.r; i++)
    {
      pod.navier_stokes_solver.nonlinear_tensor_velocity[i].Tvmult(tmpp,pod.mean_vector);
      restricted_tmpp = pod.compute_reduced_vector(tmpp);
      for (int j = 0; j < pod.r; j++)
      {
        first_matrix[i][j] = restricted_tmpp[j];
      }
    }
    std::cout << "reduced_first_convection_matrix_with_mean: " << std::endl;
    reduced_first_convection_matrix_with_mean.print_formatted(std::cout);

    std::cout << "first matrix: " << std::endl;
    first_matrix.print_formatted(std::cout);

    first_matrix.add(-1.0, reduced_first_convection_matrix_with_mean);
    std::cout << "difference matrix: " << std::endl;
    first_matrix.print_formatted(std::cout);
    std::cout << "infty norm: " << first_matrix.linfty_norm() << std::endl     -------> RESULT: looks good! (error in linfty_norm is 9.5e-14)
    exit(8);
    */

    // compute the reduced system and right hand side matrices
    // from the reduced mass and laplace matrices

    // NOTATION:

    // reduced matrices:
    // M = (ψ_j, ψ_i)_i,j=1^r
    // L = (∇ψ_j, ∇ψ_i)_i,j=1^r
    // L_t = (∇ψ_j^T, ∇ψ_i)_i,j=1^r
    // N = ((∇ψ_j^T · n, ψ_i)_Γ_out)_i,j=1^r
    // K_1 = ((\bar{v} · ∇)ψ_j, ψ_i)_i,j=1^r
    // K_2 = ((ψ_j · ∇)\bar{v}, ψ_i)_i,j=1^r

    // reduced tensor:
    // C = ((ψ_j · ∇)ψ_k, ψ_i)_i,j,k=1^r

    // A(θ) := ...
    reduced_linear_operator_theta.reinit(pod.r, pod.r);
    reduced_linear_operator_theta.add(fluid_density, reduced_mass_matrix);  // ρ * M
    reduced_linear_operator_theta.add(time_step * theta * fluid_density * viscosity, reduced_laplace_matrix); // kθμ * L
    reduced_linear_operator_theta.add(time_step * theta * fluid_density * viscosity, reduced_laplace_matrix_with_transposed_trial_function); // kθμ * L_t
    reduced_linear_operator_theta.add(- time_step * theta * fluid_density * viscosity, reduced_boundary_matrix); // - kθμ * N
    reduced_linear_operator_theta.add(time_step * theta * fluid_density, reduced_first_convection_matrix_with_mean); // kθρ * K_1

    // A(-(1-θ)) := ...
    reduced_linear_operator_one_minus_theta.reinit(pod.r, pod.r);
    reduced_linear_operator_one_minus_theta.add(fluid_density, reduced_mass_matrix);  // ρ * M
    reduced_linear_operator_one_minus_theta.add(- time_step * (1.0 - theta) * fluid_density * viscosity, reduced_laplace_matrix); // -k(1-θ)μ * L
    reduced_linear_operator_one_minus_theta.add(- time_step * (1.0 - theta) * fluid_density * viscosity, reduced_laplace_matrix_with_transposed_trial_function); // -k(1-θ)μ * L_t
    reduced_linear_operator_one_minus_theta.add(time_step * (1.0 - theta) * fluid_density * viscosity, reduced_boundary_matrix); // k(1-θ)μ * N
    reduced_linear_operator_one_minus_theta.add(- time_step * (1.0 - theta) * fluid_density, reduced_first_convection_matrix_with_mean); // -k(1-θ)ρ * K_1
    reduced_linear_operator_one_minus_theta.add(- time_step * fluid_density, reduced_second_convection_matrix_with_mean); // -kρ * K_2

    pod.navier_stokes_solver.assemble_mean_vector_contribution_rhs(pod.mean_vector, time_step);
    reduced_mean_vector_contribution_rhs = pod.compute_reduced_vector(pod.navier_stokes_solver.mean_vector_contribution_rhs);

    reduced_system_matrix.reinit(pod.r,pod.r);
    reduced_system_rhs.reinit(pod.r);
    reduced_system_matrix_inverse.reinit(pod.r,pod.r);
  }


  template<int dim>
  FullMatrix<double> ReducedOrderModel<dim>::nonlinearity_evaluated_at(int argument, Vector<double> &vector)
  {
    // nonlinearity: C = ((ψ_j · ∇)ψ_k, φ_i)_i,j,k=1^r
    // argument == 1: evaluate at j component
    // argument == 2: evaluate at k component
//	std::cout << reduced_nonlinearity_tensor[0].m() << ", " << reduced_nonlinearity_tensor[0].n() << std::endl;
//	std::cout << pod.r+pod.r_p << ", " << pod.r+pod.r_p << std::endl;

    FullMatrix<double> output_matrix(pod.r + pod.r_p, pod.r + pod.r_p);


    Threads::TaskGroup<void> task_group;
//    for (int i = 0; i < pod.r+r_additional_for_nonlinearity; i++)
//    {
    for (int j = 0; j < pod.r; j++)
    {
//    	task_group += Threads::new_task (
//    				&ROM::ReducedOrderModel<dim>::nonlinearity_evaluated_inner_loop, *this, argument, output_matrix, vector, i);
    	task_group += Threads::new_task (
    				&ROM::ReducedOrderModel<dim>::nonlinearity_evaluated_inner_loop, *this, argument, output_matrix, vector, j);
//    	nonlinearity_evaluated_inner_loop(argument, output_matrix, vector, i);
    }
    task_group.join_all ();


//    		for (int l = 0; l < pod.r; l++)
//    			if (argument == 1)
//    				output_matrix[i][j] += reduced_nonlinearity_tensor[i][l][j] * vector[l];
//    			else // argument == 2
//    				output_matrix[i][j] += reduced_nonlinearity_tensor[i][j][l] * vector[l];
    return output_matrix;
  }

  template<int dim>
  void ReducedOrderModel<dim>::nonlinearity_evaluated_at_void(int argument, Vector<double> &vector, FullMatrix<double> &output_matrix)
  {
    // nonlinearity: C = ((ψ_j · ∇)ψ_k, φ_i)_i,j,k=1^r
    // argument == 1: evaluate at j component
    // argument == 2: evaluate at k component
//	std::cout << reduced_nonlinearity_tensor[0].m() << ", " << reduced_nonlinearity_tensor[0].n() << std::endl;
//	std::cout << pod.r+pod.r_p << ", " << pod.r+pod.r_p << std::endl;

    Threads::TaskGroup<void> task_group;
//    for (int i = 0; i < pod.r+r_additional_for_nonlinearity; i++)
//    {
    for (int j = 0; j < pod.r; j++)
    {
//    	task_group += Threads::new_task (
//    				&ROM::ReducedOrderModel<dim>::nonlinearity_evaluated_inner_loop, *this, argument, output_matrix, vector, i);
    	task_group += Threads::new_task (
    				&ROM::ReducedOrderModel<dim>::nonlinearity_evaluated_inner_loop, *this, argument, output_matrix, vector, j);
//    	nonlinearity_evaluated_inner_loop(argument, output_matrix, vector, i);
    }
    task_group.join_all ();


//    		for (int l = 0; l < pod.r; l++)
//    			if (argument == 1)
//    				output_matrix[i][j] += reduced_nonlinearity_tensor[i][l][j] * vector[l];
//    			else // argument == 2
//    				output_matrix[i][j] += reduced_nonlinearity_tensor[i][j][l] * vector[l];
  }

  template<int dim>
//  void ReducedOrderModel<dim>::nonlinearity_evaluated_inner_loop(int argument, FullMatrix<double> &output_matrix, Vector<double> &vector, int i)
  void ReducedOrderModel<dim>::nonlinearity_evaluated_inner_loop(int argument, FullMatrix<double> &output_matrix, Vector<double> &vector, int j)
  {
	for (int i = 0; i < pod.r+r_additional_for_nonlinearity; i++)
	{
//  	for (int j = 0; j < pod.r; j++)
//  	{
  		for (int l = 0; l < pod.r; l++){
  			if (argument == 1)
  				output_matrix[i][j] += reduced_nonlinearity_tensor[i][l][j] * vector[l];
  			else // argument == 2
  				output_matrix[i][j] += reduced_nonlinearity_tensor[i][j][l] * vector[l];
  		}
  	}
  }

  template<int dim>
  Vector<double> ReducedOrderModel<dim>::nonlinearity_twice_evaluated_at_pressure(Vector<double> &vector)
  {
    // nonlinearity: C = (∇·[(ψ_j · ∇)ψ_k], ψ_i)_i,j,k=1^r_p,r,r
    Vector<double> out_vector(pod.r_p);

    /*
    FullMatrix<double> output_matrix(pod.r_p, pod.r);
    for (int l = 0; l < pod.r; l++)
      for (int i = 0; i < pod.r_p; i++)
        for (int j = 0; j < pod.r; j++)
          output_matrix[i][j] += reduced_nonlinearity_tensor_pressure[i][l][j] * vector[l];

    output_matrix.vmult(out_vector, vector);

    return out_vector;
    */
    Vector<double> tmp_vector(pod.r_p);
    for (int i = 0; i < pod.r_p; i++)
    {
      reduced_nonlinearity_tensor_pressure[i].vmult(tmp_vector, vector);
      out_vector[i] = tmp_vector * vector;
    }
    return out_vector;

  }

  template<int dim>
  FullMatrix<double> ReducedOrderModel<dim>::nonlinearity_first_matrix_pressure(Vector<double> &mean_vector)
  {
    // nonlinearity: C = (∇·[(ψ_j · ∇)ψ_k], ψ_i)_i,j,k=1^r_p,r,r

    FullMatrix<double> first_matrix(pod.r_p, pod.r);
    Vector<double> tmpp(pod.m);
    Vector<double> restricted_tmpp(pod.r);
    for (int i = 0; i < pod.r_p; i++)
    {
      pod.navier_stokes_solver.nonlinear_tensor_pressure[i].Tvmult(tmpp,mean_vector);
      restricted_tmpp = pod.compute_reduced_vector(tmpp);
      for (int j = 0; j < pod.r; j++)
      {
        first_matrix[i][j] = restricted_tmpp[j];
      }
    }

    return first_matrix;
  }

  template<int dim>
  FullMatrix<double> ReducedOrderModel<dim>::nonlinearity_second_matrix_pressure(Vector<double> &mean_vector)
  {
    // nonlinearity: C = (∇·[(ψ_j · ∇)ψ_k], ψ_i)_i,j,k=1^r_p,r,r

    FullMatrix<double> second_matrix(pod.r_p, pod.r);
    Vector<double> tmpp(pod.m);
    Vector<double> restricted_tmpp(pod.r);
    for (int i = 0; i < pod.r_p; i++)
    {
      pod.navier_stokes_solver.nonlinear_tensor_pressure[i].vmult(tmpp, mean_vector);
      restricted_tmpp = pod.compute_reduced_vector(tmpp);
      for (int j = 0; j < pod.r; j++)
      {
        second_matrix[i][j] = restricted_tmpp[j];
      }
    }

    return second_matrix;
  }

  template<int dim>
  Vector<double> ReducedOrderModel<dim>::nonlinearity_mean_contribution_pressure(Vector<double> &mean_vector)
  {
    // nonlinearity: C = (∇·[(ψ_j · ∇)ψ_k], ψ_i)_i,j,k=1^r_p,r,r
    Vector<double> out_vector(pod.r_p);
    Vector<double> tmp(pod.m);
    for (int i = 0; i < pod.r_p; i++)
    {
      pod.navier_stokes_solver.nonlinear_tensor_pressure[i].vmult(tmp, mean_vector);
      out_vector[i] = tmp * mean_vector;
    }

    return out_vector;
  }

  template<int dim>
  Vector<double> ReducedOrderModel<dim>::nonlinearity_boundary_twice_evaluated_at_pressure(Vector<double> &vector)
  {
    Vector<double> out_vector(pod.r_p);
    Vector<double> tmp_vector(pod.r_p);
    for (int i = 0; i < pod.r_p; i++)
    {
      reduced_nonlinearity_tensor_boundary_pressure[i].vmult(tmp_vector, vector);
      out_vector[i] = tmp_vector * vector;
    }
    return out_vector;
  }

  template<int dim>
  FullMatrix<double> ReducedOrderModel<dim>::nonlinearity_first_matrix_boundary_pressure(Vector<double> &mean_vector)
  {
    FullMatrix<double> first_matrix(pod.r_p, pod.r);
    Vector<double> tmpp(pod.m);
    Vector<double> restricted_tmpp(pod.r);
    for (int i = 0; i < pod.r_p; i++)
    {
      pod.navier_stokes_solver.nonlinear_tensor_boundary_pressure[i].Tvmult(tmpp,mean_vector);
      restricted_tmpp = pod.compute_reduced_vector(tmpp);
      for (int j = 0; j < pod.r; j++)
      {
        first_matrix[i][j] = restricted_tmpp[j];
      }
    }
    return first_matrix;
  }

  template<int dim>
  FullMatrix<double> ReducedOrderModel<dim>::nonlinearity_second_matrix_boundary_pressure(Vector<double> &mean_vector)
  {
    FullMatrix<double> second_matrix(pod.r_p, pod.r);
    Vector<double> tmpp(pod.m);
    Vector<double> restricted_tmpp(pod.r);
    for (int i = 0; i < pod.r_p; i++)
    {
      pod.navier_stokes_solver.nonlinear_tensor_boundary_pressure[i].vmult(tmpp, mean_vector);
      restricted_tmpp = pod.compute_reduced_vector(tmpp);
      for (int j = 0; j < pod.r; j++)
      {
        second_matrix[i][j] = restricted_tmpp[j];
      }
    }
    return second_matrix;
  }

  template<int dim>
  Vector<double> ReducedOrderModel<dim>::nonlinearity_boundary_mean_contribution_pressure(Vector<double> &mean_vector)
  {
    Vector<double> out_vector(pod.r_p);
    Vector<double> tmp(pod.m);
    for (int i = 0; i < pod.r_p; i++)
    {
      pod.navier_stokes_solver.nonlinear_tensor_boundary_pressure[i].vmult(tmp, mean_vector);
      out_vector[i] = tmp * mean_vector;
    }
    return out_vector;
  }

  template<int dim>
  Vector<double> ReducedOrderModel<dim>::laplace_mean_contribution_pressure(Vector<double> &mean_vector)
  {
    Vector<double> out_vector(pod.r_p);
    Vector<double> tmp(pod.m_p);
    pod.navier_stokes_solver.laplace_matrix_pressure.vmult(tmp, mean_vector);
    return pod.compute_reduced_vector_pressure(tmp);

    /*
    Vector<double> tmp2(pod.m_p);
    pod.navier_stokes_solver.mass_matrix_pressure.vmult(tmp2, tmp);
    return pod.compute_reduced_vector_pressure(tmp2);
    */
  }

  template<int dim>
  void ReducedOrderModel<dim>::assemble_system_matrix(double time_step, double theta)
  {
	FullMatrix<double> tmp_matrix_sol_1(pod.r+pod.r_p,pod.r+pod.r_p);
	FullMatrix<double> tmp_matrix_sol_2(pod.r+pod.r_p,pod.r+pod.r_p);

	Threads::TaskGroup<void> task_group;
	task_group += Threads::new_task (&ReducedOrderModel<dim>::nonlinearity_evaluated_at_void, *this, 1, solution, tmp_matrix_sol_1);
	task_group += Threads::new_task (&ReducedOrderModel<dim>::nonlinearity_evaluated_at_void, *this, 2, solution, tmp_matrix_sol_2);
	task_group.join_all ();

    reduced_system_matrix.copy_from(reduced_linear_operator_theta);
    reduced_system_matrix.add(theta * time_step * fluid_density, tmp_matrix_sol_1);
    reduced_system_matrix.add(theta * time_step * fluid_density, tmp_matrix_sol_2);

    reduced_system_matrix_inverse = 0.0;
    reduced_system_matrix_inverse.invert(reduced_system_matrix);




//    reduced_system_matrix.copy_from(reduced_linear_operator_theta);
//    reduced_system_matrix.add(theta * time_step * fluid_density, nonlinearity_evaluated_at(1, solution));
//    reduced_system_matrix.add(theta * time_step * fluid_density, nonlinearity_evaluated_at(2, solution));
//
//    // std::cout << "REDUCED SYSTEM MATRIX: " << std::endl;
//    // reduced_system_matrix.print(std::cout);
//
//    reduced_system_matrix_inverse = 0.0;
//    reduced_system_matrix_inverse.invert(reduced_system_matrix);

    //FullMatrix<double> test_matrix(pod.r,pod.r);
    //reduced_system_matrix.mmult(test_matrix, reduced_system_matrix_inverse);
    //test_matrix.print(std::cout);

    // std::cout << "REDUCED SYSTEM MATRIX INVERSE: " << std::endl;
    // reduced_system_matrix_inverse.print(std::cout);
  }

  template<int dim>
  void ReducedOrderModel<dim>::assemble_system_matrix_penalty(double time_step, double theta, double tau)
  {
    reduced_system_matrix.copy_from(reduced_linear_operator_theta);
    reduced_system_matrix.add(theta * time_step * fluid_density, nonlinearity_evaluated_at(1, solution));
    reduced_system_matrix.add(theta * time_step * fluid_density, nonlinearity_evaluated_at(2, solution));

    // std::cout << "REDUCED SYSTEM MATRIX: " << std::endl;
    // reduced_system_matrix.print(std::cout);


    // add penalty matrix
    reduced_system_matrix.add(tau,reduced_indicator_matrix);

    reduced_system_matrix_inverse = 0.0;
    reduced_system_matrix_inverse.invert(reduced_system_matrix);

    //FullMatrix<double> test_matrix(pod.r,pod.r);
    //reduced_system_matrix.mmult(test_matrix, reduced_system_matrix_inverse);
    //test_matrix.print(std::cout);

    // std::cout << "REDUCED SYSTEM MATRIX INVERSE: " << std::endl;
    // reduced_system_matrix_inverse.print(std::cout);
  }

  template<int dim>
  void ReducedOrderModel<dim>::assemble_system_rhs(double time_step, double theta)
  {
	 reduced_system_rhs = 0.0;
	 Vector<double> tmp_newton(pod.r+pod.r_p);
	 Vector<double> tmp_timestep(pod.r+pod.r_p);
	 Vector<double> tmp_newton_nonlinearity(pod.r+pod.r_p);
	 Vector<double> tmp_timestep_nonlinearity(pod.r+pod.r_p);
	 FullMatrix<double> tmp_matrix_sol(pod.r+pod.r_p,pod.r+pod.r_p);
	 FullMatrix<double> tmp_matrix_sol_old(pod.r+pod.r_p,pod.r+pod.r_p);

	 Threads::TaskGroup<void> task_group;
	 task_group += Threads::new_task (&ReducedOrderModel<dim>::nonlinearity_evaluated_at_void, *this, 1, solution, tmp_matrix_sol);
	 task_group += Threads::new_task (&ReducedOrderModel<dim>::nonlinearity_evaluated_at_void, *this, 1, old_timestep_solution, tmp_matrix_sol_old);
	 task_group.join_all ();


	 Threads::TaskGroup<void> task_vmult;
	 task_vmult += Threads::new_task (&ReducedOrderModel<dim>::vmult_for_pp, *this, tmp_matrix_sol, tmp_newton_nonlinearity, solution);
	 task_vmult += Threads::new_task (&ReducedOrderModel<dim>::vmult_for_pp, *this, tmp_matrix_sol_old, tmp_timestep_nonlinearity, old_timestep_solution);
	 task_vmult += Threads::new_task (&ReducedOrderModel<dim>::vmult_for_pp, *this, reduced_linear_operator_theta, tmp_newton, solution);
	 task_vmult += Threads::new_task (&ReducedOrderModel<dim>::vmult_for_pp, *this, reduced_linear_operator_one_minus_theta, tmp_timestep, old_timestep_solution);
	 task_vmult.join_all ();

	 reduced_system_rhs.add(-1.0, tmp_newton);
	 reduced_system_rhs.add(-time_step * theta * fluid_density, tmp_newton_nonlinearity);
	 reduced_system_rhs.add(1.0, tmp_timestep);
	 reduced_system_rhs.add(-time_step * (1.0 - theta) * fluid_density, tmp_timestep_nonlinearity);
	 reduced_system_rhs.add(1.0, reduced_mean_vector_contribution_rhs);


//    reduced_system_rhs = 0.0;
//    Vector<double> tmp(pod.r+pod.r_p);
//
//    // adding terms from last Newton iterate
//    reduced_linear_operator_theta.vmult(tmp, solution);
//    reduced_system_rhs.add(-1.0, tmp);
//    nonlinearity_evaluated_at(1, solution).vmult(tmp, solution);
//    reduced_system_rhs.add(-time_step * theta * fluid_density, tmp);
//
//    // adding terms from last time step
//    reduced_linear_operator_one_minus_theta.vmult(tmp, old_timestep_solution);
//    reduced_system_rhs.add(1.0, tmp);
//    nonlinearity_evaluated_at(1, old_timestep_solution).vmult(tmp, old_timestep_solution);
//    reduced_system_rhs.add(-time_step * (1.0 - theta) * fluid_density, tmp);
//    // adding term from mean vector
//    reduced_system_rhs.add(1.0, reduced_mean_vector_contribution_rhs);
  }
  template<int dim>
  void ReducedOrderModel<dim>::vmult_for_pp(FullMatrix<double> &matrix, Vector<double> &result, Vector<double> &vector)
  {
	  matrix.vmult(result, vector);
  }

  template<int dim>
  void ReducedOrderModel<dim>::assemble_system_rhs_no_mean(double time_step, double theta)
  {
    reduced_system_rhs = 0.0;
    Vector<double> tmp(pod.r);

    // adding terms from last Newton iterate
    reduced_linear_operator_theta.vmult(tmp, solution);
    reduced_system_rhs.add(-1.0, tmp);

    nonlinearity_evaluated_at(1, solution).vmult(tmp, solution);
    reduced_system_rhs.add(-time_step * theta * fluid_density, tmp);

    // adding terms from last time step
    reduced_linear_operator_one_minus_theta.vmult(tmp, old_timestep_solution);
    reduced_system_rhs.add(1.0, tmp);

    nonlinearity_evaluated_at(1, old_timestep_solution).vmult(tmp, old_timestep_solution);
    reduced_system_rhs.add(-time_step * (1.0 - theta) * fluid_density, tmp);
    // std::cout << "REDUCED SYSTEM RHS: " << std::endl;
    // reduced_system_rhs.print(std::cout);
  }

  template<int dim>
  void ReducedOrderModel<dim>::solve_reduced_order_model(double start_time, double end_time, double time_step, double theta, bool output_files, bool compute_error)
  {
//    int m = pod.navier_stokes_solver.dof_handler_velocity.n_dofs(); // number of FEM DoFs
    // pod.r = number of POD DoFs
    old_timestep_solution.reinit(pod.r);
    solution.reinit(pod.r);
    reduced_system_rhs.reinit(pod.r);
    newton_update.reinit(pod.r);
    old_timestep_solution = 0.0;
    solution = 0.0;
    reduced_system_rhs = 0.0;

    reduced_system_rhs_pressure.reinit(pod.r_p);
    solution_pressure.reinit(pod.r_p);

    time = start_time;

    // if (time == 0)
    //  start with a zero solution
    if (time == 0)
    {
      // project initial condition into POD space
      intial_solution_snapshot = pde_info->fem_path + "mu="+ std::to_string(viscosity) +"/snapshots/snapshot_" \
	  	  	  + Utilities::int_to_string(0, 6) + ".h5"; //fast
//      std::cout << "initial condition" << std::endl;
//      std::cout << intial_solution_snapshot << std::endl;
      Vector<double> tmp = load_h5_vector(intial_solution_snapshot);

      tmp.add(-1.0, pod.mean_vector);
      Vector<double> tmp2(pod.m);

      pod.navier_stokes_solver.mass_matrix_velocity.vmult(tmp2, tmp);
      tmp2=tmp;
      std::cout << "number pod modes: " << pod.r << std::endl;
      solution = pod.compute_reduced_vector(tmp2);

//      pod.navier_stokes_solver.mass_matrix_velocity.vmult(tmp2, tmp);
//      tmp.scale(pod.space_weights);
//      for (int i = 0; i < pod.m; i++) {
//    	  tmp(i) +
//      }
//      solution = pod.compute_reduced_vector(tmp);
    }
    compute_projected_solution();
//    pressure_reconstruction_with_nn();
//    compute_projected_solution_pressure();

    std::ofstream  fw;
	Vector<double> bc_eval(dim);
    fw.open((pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/boudary_check.txt"), std::ofstream::out);
    if (output_files)
    {
    	mkdir((pde_info->rom_path + "mu=" + std::to_string(viscosity)).c_str() , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    	mkdir((pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/solution").c_str() , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    	mkdir((pde_info->rom_path + "mu=" + std::to_string(viscosity)+ "/error").c_str() , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    	mkdir((pde_info->rom_path + "mu=" + std::to_string(viscosity)+ "/h5").c_str() , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    	output_results();
  	VectorTools::point_value( pod.navier_stokes_solver.dof_handler_velocity,
  								 projected_solution,
									 Point<2>(0.0, 0.2),
									 bc_eval);
    	fw << time << ", " << bc_eval(0) << std::endl;
    }
    std::ofstream  fw2;
    fw2.open((pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/exectution_time_" + std::to_string(pod.r) + ".txt"), std::ofstream::out);
    auto start_time_rom = std::chrono::high_resolution_clock::now();
    while (time +time_step <= end_time- 0.01 * time_step)
      {
//    	std::cout << time << std::endl;
//    	std::cout << end_time << std::endl;
//    	std::cout << timestep_number << std::endl;

        time += time_step;
        ++timestep_number;

        if (int(timestep_number) % 50 == 0) {
        	std::cout << "\tTime step " << int(timestep_number) << " at t=" << time
                      << std::endl;
        }

        // Compute the next time step
        old_timestep_solution = solution;
        newton_iteration(time, time_step, theta);
        //compute_linearized_velocity(time_step, theta);
        compute_projected_solution();
        // for evulation of functional values is pressure needed: use FOM pressure
        compute_projected_solution_pressure();

        if (output_files)
        {
          output_results();
          VectorTools::point_value( pod.navier_stokes_solver.dof_handler_velocity,
                  								 projected_solution,
              									 Point<2>(0.0, 0.2),
              									 bc_eval);
                    	fw << time << ", " << bc_eval(0) << std::endl;
//          output_results_pressure();
          if (compute_error) {
        	  output_error();
          }
        }

//        compute_functional_values();
      }
    auto end_time_rom = std::chrono::high_resolution_clock::now();
    auto duration_rom = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_rom - start_time_rom);
  	fw2 			<< std::floor(duration_rom.count() / 1000) << "s " << duration_rom.count() % 1000 << " ms" << std::endl;
    std::cout 		<< std::floor(duration_rom.count() / 1000) << "s " << duration_rom.count() % 1000 << " ms" << std::endl;
    fw2.close();
    fw.close();
  }

  template<int dim>
  void ReducedOrderModel<dim>::solve_reduced_order_model_vp(double start_time, double end_time, double time_step, double theta, bool output_files, bool compute_error)
  {
//    int m = pod.navier_stokes_solver.dof_handler_velocity.n_dofs(); // number of FEM DoFs
    // pod.r = number of POD DoFs
    old_timestep_solution.reinit(pod.r+pod.r_p);
    solution.reinit(pod.r+pod.r_p);
    reduced_system_rhs.reinit(pod.r+pod.r_p);
    newton_update.reinit(pod.r+pod.r_p);
    old_timestep_solution = 0.0;
    solution = 0.0;
    reduced_system_rhs = 0.0;

    projected_solution_combined.reinit(2);
    projected_solution_combined.block(0).reinit(pod.navier_stokes_solver.dofs_velocity);
    projected_solution_combined.block(1).reinit(pod.navier_stokes_solver.dofs_pressure);
    projected_solution_combined.collect_sizes();
    time = start_time;

    //  start with a zero solution
    if (time == 0)
    {
      // project initial condition into POD space
      intial_solution_snapshot = pde_info->fem_path + "mu="+ std::to_string(viscosity) +"/snapshots/snapshot_" \
	  	  	  + Utilities::int_to_string(0, 6) + ".h5"; //fast
//      std::cout << "initial condition" << std::endl;
//      std::cout << intial_solution_snapshot << std::endl;
      Vector<double> tmp_velocity = load_h5_vector(intial_solution_snapshot);
      Vector<double> tmp_pressure = load_h5_vector_pressure(intial_solution_snapshot);



      tmp_velocity.add(-1.0, pod.mean_vector);
      Vector<double> tmp2_velocity(pod.navier_stokes_solver.dofs_velocity);

//      pod.navier_stokes_solver.mass_matrix_vp.block(0,0).vmult(tmp2_velocity, tmp_velocity);
      tmp2_velocity=tmp_velocity;



      std::cout << "pod.r   = " << pod.r   << std::endl;
      std::cout << "pod.r_s = " << pod.r_s << std::endl;
      std::cout << "pod.r_p = " << pod.r_p << std::endl;
      std::cout << "r_vs    = " << r_vs    << std::endl;
      std::cout << "r_p     = " << r_p     << std::endl;
      std::cout << "r_afn   = " << r_additional_for_nonlinearity << std::endl;

      system(("rm -r " + pde_info->rom_path + "time_counter.txt").c_str());
      if (sfbm)
      {
          Vector<double> rom_vector(pod.r + r_additional_for_nonlinearity);
          std::cout << "number modes: " << pod.pod_vectors.size() << std::endl;
          std::cout << "number supr : " << r_additional_for_nonlinearity << std::endl;
          rom_vector = 0.0;
          for (int i=0; i<pod.r +r_additional_for_nonlinearity; i++)
          {
//        	  std::cout << i << std::endl;
            rom_vector[i] = pod.pod_vectors[i] * tmp2_velocity;
          }
          std::cout << std::endl;
          for (int i=pod.r; i< pod.r +r_additional_for_nonlinearity; i++)
          {
//        	std::cout << i << ", " << rom_vector[i] << std::endl;
            rom_vector[i] = 0;
          }
          Vector<double> fom_vector(pod.navier_stokes_solver.dofs_velocity);
          for (int i = 0; i<pod.r+r_additional_for_nonlinearity; i++){
        	  fom_vector.add(rom_vector[i],pod.pod_vectors[i]);
          }
          tmp2_velocity = fom_vector;

          fom_vector.add(-1.,tmp_velocity);
//          std::cout << "diff: " << fom_vector.norm_sqr() << std::endl;
      }

      std::cout << "number pod modes: " << pod.r << " + " << pod.r_p << std::endl;
      Vector<double> tmp_velocity_red(pod.r);
      std::cout << "reducing velocity" << std::endl;
      tmp_velocity_red= pod.compute_reduced_vector(tmp2_velocity);
      Vector<double> tmp_pressure_red(pod.r_p);
      std::cout << "reducing pressure" << std::endl;
      tmp_pressure_red= pod.compute_reduced_vector_pressure(tmp_pressure);
      std::cout << "starting for loop" << std::endl;
      for (int i = 0; i<pod.r; i++){
    	  solution[i] = tmp_velocity_red[i];
      }
      for (int i = pod.r-pod.r_s; i<pod.r; i++){
//    	  std::cout << tmp_velocity_red[i] << std::endl;
//    	  solution[i] = tmp_velocity_red[i];
          solution[i] = 0;
      }
      for (int i = pod.r; i<pod.r+pod.r_p; i++){
    	  solution[i] = tmp_pressure_red[i-pod.r];
//    	  solution[i] = 0.0;
      }
    }


    std::ofstream p_out;
    std::ofstream v_out;
    if (output_files)
    {
        system(("rm -r " + pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/boudary_check.txt").c_str());
        system(("rm -r " + pde_info->rom_path + "reduced_pressure_modes.txt").c_str());
        system(("rm -r " + pde_info->rom_path + "reduced_velocity_modes.txt").c_str());

        p_out.open(pde_info->rom_path + "reduced_pressure_modes.txt", std::ios_base::app); // append instead of overwrite
        for (int ttt = pod.r ; ttt < pod.r + pod.r_p; ttt++)
        {
        	p_out << solution[ttt] << ", ";
        }
        p_out << std::endl;
        p_out.close();


        v_out.open(pde_info->rom_path + "reduced_velocity_modes.txt", std::ios_base::app); // append instead of overwrite
        for (int ttt = 0 ; ttt < pod.r; ttt++)
        {
        	v_out << solution[ttt] << ", ";
        }
        v_out << std::endl;
        v_out.close();

        compute_projected_solution();
        compute_projected_solution_pressure();

    	projected_solution_combined.block(0)= projected_solution;
        projected_solution_combined.block(1)= projected_solution_pressure;
        system(("rm -r " + pde_info->rom_path + "mu=" + std::to_string(viscosity)).c_str());
    	mkdir((pde_info->rom_path + "mu=" + std::to_string(viscosity)).c_str() , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    	mkdir((pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/solution").c_str() , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    	mkdir((pde_info->rom_path + "mu=" + std::to_string(viscosity)+ "/error").c_str() , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    	mkdir((pde_info->rom_path + "mu=" + std::to_string(viscosity)+ "/h5").c_str() , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    	output_results_vp();
    	compute_functional_values_vp();
    }

    auto start_time_rom = std::chrono::high_resolution_clock::now();
    while (time +time_step <= end_time- 0.01 * time_step)
      {

        time += time_step;
        ++timestep_number;

        if (int(timestep_number) % 50 == 0) {
        	std::cout << "\tTime step " << int(timestep_number) << " at t=" << time
                      << std::endl;
        }

        // Compute the next time step
        old_timestep_solution = solution;
//        for (int ttt = pod.r; ttt<pod.r+pod.r_p; ttt++){
//        	old_timestep_solution[ttt] = 0.0;
//  //    	  solution[i] = 0.0;
//        }
        newton_iteration(time, time_step, theta);
        //compute_linearized_velocity(time_step, theta);



        if (output_files)
        {
            p_out.open(pde_info->rom_path + "reduced_pressure_modes.txt", std::ios_base::app); // append instead of overwrite
            for (int ttt = pod.r ; ttt < pod.r + pod.r_p; ttt++)
            {
            	p_out << solution[ttt] << ", ";
            }
            p_out << std::endl;
            p_out.close();

            v_out.open(pde_info->rom_path + "reduced_velocity_modes.txt", std::ios_base::app); // append instead of overwrite
            for (int ttt = 0 ; ttt < pod.r; ttt++)
            {
               	v_out << solution[ttt] << ", ";
            }
            v_out << std::endl;
            v_out.close();

            Threads::Thread<void> thread_proj = Threads::new_thread (&ReducedOrderModel<dim>::compute_projected_solution_pressure,*this);
            compute_projected_solution();
            thread_proj.join();

    //        compute_projected_solution();
    //        compute_projected_solution_pressure();

            projected_solution_combined.block(0)= projected_solution;
            projected_solution_combined.block(1)= projected_solution_pressure;
            Threads::Thread<void> thread = Threads::new_thread (&ReducedOrderModel<dim>::output_results_vp,*this);
//        	output_results_vp();
        	if (compute_error) {
//        	  output_error();
          }
        	compute_functional_values_vp();
        	thread.join ();
        }


    }
    auto end_time_rom = std::chrono::high_resolution_clock::now();
    auto duration_rom = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_rom - start_time_rom);

    std::ofstream  fw2;
    fw2.open((pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/exectution_time_" + std::to_string(pod.r) + ".txt"), std::ofstream::out);
  	fw2 			<< std::floor(duration_rom.count() / 1000) << "s " << duration_rom.count() % 1000 << " ms" << std::endl;
    fw2.close();
    std::cout 		<< std::floor(duration_rom.count() / 1000) << "s " << duration_rom.count() % 1000 << " ms" << std::endl;

    std::ofstream timer_print;
    timer_print.open(pde_info->rom_path + "time_counter.txt", std::ios_base::app); // append instead of overwrite
    for (int i = 0; i<computational_time_per_iteration.size(); i++)
    {
    	timer_print << computational_time_per_iteration[i][0] << ", "
    	        	<< computational_time_per_iteration[i][1] << ", "
    	        	<< computational_time_per_iteration[i][2] << ", "
    	    		<< computational_time_per_iteration[i][3] << std::endl;
    }
    timer_print.close();
  }

  template<int dim>
    void ReducedOrderModel<dim>::solve_reduced_order_model_no_mean(double start_time, double end_time, double time_step, double theta, bool output_files, bool compute_error)
    {
  //    int m = pod.navier_stokes_solver.dof_handler_velocity.n_dofs(); // number of FEM DoFs
      // pod.r = number of POD DoFs
      old_timestep_solution.reinit(pod.r);
      solution.reinit(pod.r);
      reduced_system_rhs.reinit(pod.r);
      newton_update.reinit(pod.r);
      old_timestep_solution = 0.0;
      solution = 0.0;
      reduced_system_rhs = 0.0;

      reduced_system_rhs_pressure.reinit(pod.r_p);
      solution_pressure.reinit(pod.r_p);

      time = start_time;

      // if (time == 0)
      //  start with a zero solution
      if (time == 0)
      {
        // project initial condition into POD space
        intial_solution_snapshot = pde_info->fem_path + "mu="+ std::to_string(viscosity) +"/snapshots/snapshot_" \
  	  	  	  + Utilities::int_to_string(0, 6) + ".h5"; //fast
  //      std::cout << "initial condition" << std::endl;
  //      std::cout << intial_solution_snapshot << std::endl;

        Vector<double> tmp = load_h5_vector(intial_solution_snapshot);
        Vector<double> tmp2(pod.m);
        pod.navier_stokes_solver.mass_matrix_velocity.vmult(tmp2, tmp);
//        tmp2=tmp;
//        pod.navier_stokes_solver.mass_matrix_velocity.vmult(tmp, tmp2);
        solution = pod.compute_reduced_vector(tmp2);

  //      pod.navier_stokes_solver.mass_matrix_velocity.vmult(tmp2, tmp);
  //      tmp.scale(pod.space_weights);
  //      for (int i = 0; i < pod.m; i++) {
  //    	  tmp(i) +
  //      }
  //      solution = pod.compute_reduced_vector(tmp);
      }
      compute_projected_solution_no_mean();
  //    pressure_reconstruction_with_nn();
  //    compute_projected_solution_pressure();
      std::ofstream  fw;
  	Vector<double> bc_eval(dim);
      fw.open((pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/boudary_check.txt"), std::ofstream::out);
      newton_update_interim.reinit(pod.r);
      newton_update_interim = 0.0;
      if (output_files)
      {
      	mkdir((pde_info->rom_path + "mu=" + std::to_string(viscosity)).c_str() , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      	mkdir((pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/solution").c_str() , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      	mkdir((pde_info->rom_path + "mu=" + std::to_string(viscosity)+ "/error").c_str() , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      	mkdir((pde_info->rom_path + "mu=" + std::to_string(viscosity)+ "/h5").c_str() , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      	output_results();
    	VectorTools::point_value( pod.navier_stokes_solver.dof_handler_velocity,
    								 projected_solution,
									 Point<2>(0.0, 0.2),
									 bc_eval);
      	fw << time << ", " << bc_eval(0) << std::endl;
      	std::cout << time << ", " << bc_eval(0) << std::endl;
  //      output_results_pressure();
      }
      while (time +time_step <= end_time- 0.01 * time_step)
        {
  //    	std::cout << time << std::endl;
  //    	std::cout << end_time << std::endl;
  //    	std::cout << timestep_number << std::endl;

          time += time_step;
          ++timestep_number;

          if (int(timestep_number) % 50 == 0) {
          	std::cout << "\tTime step " << int(timestep_number) << " at t=" << time
                        << std::endl;
          }

          // Compute the next time step
          old_timestep_solution = solution;
          newton_iteration_no_mean(time, time_step, theta);
          //compute_linearized_velocity(time_step, theta);
          compute_projected_solution_no_mean();
          // for evulation of functional values is pressure needed: use FOM pressure
          compute_projected_solution_pressure();

          if (output_files)
          {
            output_results();
        	VectorTools::point_value( pod.navier_stokes_solver.dof_handler_velocity,
        								 projected_solution,
    									 Point<2>(0.0, 0.2),
    									 bc_eval);
          	fw << time << ", " << bc_eval(0) << std::endl;
          	std::cout << time << ", " << bc_eval(0) << std::endl;
  //          output_results_pressure();
            if (compute_error) {
          	  output_error();
            }
          }
          compute_functional_values();
        }
      fw.close();
    }

  template<int dim>
  void ReducedOrderModel<dim>::newton_iteration(double time, double time_step, double theta)
  {
    const unsigned int max_no_newton_steps = 100;

    // Decision whether the system matrix should be build
    // at each Newton step
    const double nonlinear_rho = 0.1;

    // Line search parameters
    unsigned int line_search_step;
    const unsigned int max_no_line_search_steps = 10;
    const double line_search_damping = 0.6;
    double new_newton_residual;

    // Application of the initial boundary conditions to the
    // variational equations:
    assemble_system_rhs(time_step, theta);
    double newton_residual = reduced_system_rhs.linfty_norm();

    double old_newton_residual = newton_residual;
    unsigned int newton_step = 1;


    auto start_time_matrix = std::chrono::high_resolution_clock::now();
	auto end_time_matrix = std::chrono::high_resolution_clock::now();
	auto duration_matrix = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_matrix - start_time_matrix);

    auto start_time_rhs= std::chrono::high_resolution_clock::now();
    auto end_time_rhs = std::chrono::high_resolution_clock::now();
    auto duration_rhs = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_rhs - start_time_rhs);

    auto start_time_solve = std::chrono::high_resolution_clock::now();
	auto end_time_solve = std::chrono::high_resolution_clock::now();
    auto duration_solve = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_solve - start_time_solve);

    auto start_time_newton = std::chrono::high_resolution_clock::now();

    while (newton_residual > newton_tol &&
           newton_step < max_no_newton_steps)
    {
      old_newton_residual = newton_residual;

      start_time_rhs= std::chrono::high_resolution_clock::now();
      assemble_system_rhs(time_step, theta);
      end_time_rhs = std::chrono::high_resolution_clock::now();
      duration_rhs += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_rhs - start_time_rhs);
      newton_residual = reduced_system_rhs.linfty_norm();

      // std::cout << "NEWTON RESIDUAL: " << newton_residual << std::endl;

      if (newton_residual < newton_tol)
      {
        break;
      }

      if (newton_residual / old_newton_residual > nonlinear_rho)
      {
//    	std::cout << " TIME STEP: " << timestep_number << std::endl;
    	start_time_matrix = std::chrono::high_resolution_clock::now();
    	assemble_system_matrix(time_step, theta);
    	end_time_matrix = std::chrono::high_resolution_clock::now();
    	duration_matrix += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_matrix - start_time_matrix);
      }

      // Solve Ax = b
      start_time_solve= std::chrono::high_resolution_clock::now();
      solve();
      end_time_solve = std::chrono::high_resolution_clock::now();
      duration_solve += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_solve - start_time_solve);

      line_search_step = 0;
      for (;
           line_search_step < max_no_line_search_steps;
           ++line_search_step)
      {
//    	for (int i = 20; i<40; i++){
//    		newton_update[i] = 0;
//    	      }
        solution += newton_update;
//        std::cout << "LINE SEARCH STEP: " << line_search_step << std::endl;
        start_time_rhs= std::chrono::high_resolution_clock::now();
        assemble_system_rhs(time_step, theta);
        end_time_rhs = std::chrono::high_resolution_clock::now();
        duration_rhs += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_rhs - start_time_rhs);
        new_newton_residual = reduced_system_rhs.linfty_norm();


        if (new_newton_residual < newton_residual)
          break;
        else
          solution -= newton_update;

        newton_update *= line_search_damping;
      }
      // Updates
      newton_step++;
    }
//    std::cout << reduced_system_rhs(20) <<", "<< reduced_system_rhs(25) <<", "<< reduced_system_rhs(30) <<", " << reduced_system_rhs(39) << std::endl;

//     std::cout << "NEW NEWTON RESIDUAL: " << new_newton_residual << ", NEWTON STEPS: " << newton_step<< std::endl;
    if (newton_step == max_no_newton_steps)
    {
      std::cout << "Newton method diverged" << std::endl;
      exit(2);
    }
	auto end_time_newton= std::chrono::high_resolution_clock::now();
    auto duration_newton = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_newton - start_time_newton);
//    std::vector<double> temp_vec = {duration_newton.count(), duration_solve.count(), duration_matrix.count(), duration_rhs.count()};
    computational_time_per_iteration.push_back({duration_newton.count(), duration_solve.count(), duration_matrix.count(), duration_rhs.count()});
//    std::ofstream timer_print;
//    timer_print.open(pde_info->rom_path + "time_counter.txt", std::ios_base::app); // append instead of overwrite
//    timer_print << duration_newton.count() << ", "
//    			<< duration_solve.count() << ", "
//    			<< duration_matrix.count() << ", "
//				<< duration_rhs.count() << std::endl;
//    timer_print.close();
  }

  template<int dim>
  void ReducedOrderModel<dim>::newton_iteration_no_mean(double time, double time_step, double theta)
  {
    const unsigned int max_no_newton_steps = 10000;

    // Decision whether the system matrix should be build
    // at each Newton step
    const double nonlinear_rho = 0.1;

    // Line search parameters
    unsigned int line_search_step;
    const unsigned int max_no_line_search_steps = 10;
    const double line_search_damping = 0.6;
    double new_newton_residual;
    double tau = 1;
    // Application of the initial boundary conditions to the
    // variational equations:
    assemble_system_rhs_no_mean(time_step, theta);
    double newton_residual = reduced_system_rhs.linfty_norm();

    double old_newton_residual = newton_residual;
    unsigned int newton_step = 1;

//    if (newton_residual < newton_tol)
//    {
//      std::cout << '\t'
//                << std::scientific
//                << newton_residual
//                << std::endl;
//    }

    while (newton_residual > newton_tol &&
           newton_step < max_no_newton_steps)
    {
      old_newton_residual = newton_residual;

      assemble_system_rhs_no_mean(time_step, theta);
      newton_residual = reduced_system_rhs.linfty_norm();

      // std::cout << "NEWTON RESIDUAL: " << newton_residual << std::endl;

      if (newton_residual < newton_tol)
      {
//        std::cout << '\t'
//                  << std::scientific
//                  << newton_residual << std::endl;
        break;
      }

      if (newton_residual / old_newton_residual > nonlinear_rho)
      {
        assemble_system_matrix_penalty(time_step, theta,tau);
      }

      // Solve Ax = b
      solve();

      // std::cout << "SOLUTION: " << std::endl;
      // solution.print(std::cout);

      // std::cout << "NEWTON_UPDATE: " << std::endl;
      // newton_update.print(std::cout);

      // std::cout << "Starting line search" << std::endl;
      line_search_step = 0;
      for (;
           line_search_step < max_no_line_search_steps;
           ++line_search_step)
      {
        solution += newton_update;

        // std::cout << "NEW SOLUTION: " << std::endl;
        // solution.print(std::cout);

        assemble_system_rhs_no_mean(time_step, theta);
        new_newton_residual = reduced_system_rhs.linfty_norm();

        // std::cout << "NEW NEWTON RESIDUAL: " << new_newton_residual << std::endl;

        if (new_newton_residual < newton_residual)
          break;
        else
          solution -= newton_update;

        newton_update *= line_search_damping;
      }

//      std::cout << std::setprecision(5) << newton_step << '\t'
//                << std::scientific << newton_residual << '\t'
//                << std::scientific << newton_residual / old_newton_residual << '\t';
//      if (newton_residual / old_newton_residual > nonlinear_rho)
//        std::cout << "r" << '\t';
//      else
//        std::cout << " " << '\t';
//      std::cout << line_search_step << std::endl;

      //<< '\t' << std::scientific << timer_newton.cpu_time ()

      newton_step++;
    }


    std::cout << std::endl;
    if (newton_step == max_no_newton_steps)
    {
      std::cout << "Newton method diverged" << std::endl;
      exit(2);
    }

    Vector <double> fom_res;
   	fom_res.reinit(pod.navier_stokes_solver.dof_handler_velocity.n_dofs());
   	for (int i = 0; i<pod.r; i++)
   		fom_res.add(reduced_system_rhs[i],pod.pod_vectors[i]);

    std::ofstream fom_res_out;
    fom_res_out.open(pde_info->rom_path +"residual_test.txt", std::ios_base::app); // append instead of overwrite
    fom_res_out << std::setprecision(16) << fom_res.norm_sqr() << ", ";
    if (timestep_number == 199) {
//    	std::cout << newton_residual << ", " << fom_res.norm_sqr() << std::endl;
    	fom_res_out << std::endl;
    }
    fom_res_out.close();



    QGauss<dim> quadrature_formula(pod.navier_stokes_solver.fe.degree + 2);
    FEValues<dim> fe_values(pod.navier_stokes_solver.fe, quadrature_formula, update_values | update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell = pod.navier_stokes_solver.fe.dofs_per_cell;

  }

  template<int dim>
  void ReducedOrderModel<dim>::solve()
  {
    // SolverControl            solver_control(1000, 1e-8 * reduced_system_rhs.l2_norm());
    // SolverGMRES<Vector<double>> gmres(solver_control);

    // gmres.solve(reduced_system_matrix, newton_update, reduced_system_rhs, PreconditionIdentity());

    // std::cout << "     " << solver_control.last_step() << " GMRES iterations."
    //           << std::endl;
    reduced_system_matrix_inverse.vmult(newton_update, reduced_system_rhs);
  }

  template<int dim>
  void ReducedOrderModel<dim>::solve_linearized()
  {
    reduced_system_matrix_inverse.vmult(solution, reduced_system_rhs);
  }

  template<int dim>
  void ReducedOrderModel<dim>::compute_linearized_velocity(double time_step, double theta)
  {
    reduced_system_matrix.copy_from(reduced_linear_operator_theta);
    reduced_system_matrix.add(theta * time_step * fluid_density, nonlinearity_evaluated_at(1, old_timestep_solution));

    reduced_system_matrix_inverse = 0.0;
    reduced_system_matrix_inverse.invert(reduced_system_matrix);

    reduced_system_rhs = 0.0;
    Vector<double> tmp(pod.r);

    // adding terms from last time step
    reduced_linear_operator_one_minus_theta.vmult(tmp, old_timestep_solution);
    reduced_system_rhs.add(1.0, tmp);

    nonlinearity_evaluated_at(1, old_timestep_solution).vmult(tmp, old_timestep_solution);
    reduced_system_rhs.add(-time_step * (1.0 - theta) * fluid_density, tmp);

    // adding term from mean vector
    reduced_system_rhs.add(1.0, reduced_mean_vector_contribution_rhs);

    solve_linearized();
  }

  template<int dim>
  void ReducedOrderModel<dim>::pressure_reconstruction()
  {
    // ASSEMBLE THE RHS OF THE POISSON EQUATION
    reduced_system_rhs_pressure = 0.0;
    reduced_system_rhs_pressure.add(fluid_density, nonlinearity_twice_evaluated_at_pressure(solution)); // solution is the velocity ROM solution

    Vector<double> tmp(pod.r_p);
    first_nonlinearity_matrix_pressure.vmult(tmp, solution);
    reduced_system_rhs_pressure.add(fluid_density, tmp); // --> is this incorrect ???

    /*
    // debug code for first nonlinearity_matrix_pressure
    Vector<double> tmpp(pod.m);
    Vector<double> tmp_clone(pod.r_p);
    for (int i = 0; i < pod.r_p; i++)
    {
      pod.navier_stokes_solver.nonlinear_tensor_pressure[i].Tvmult(tmpp,pod.mean_vector);
      tmp_clone[i] = tmpp * projected_solution;
      tmp_clone[i] -= tmpp * pod.mean_vector;

      double first = tmpp * projected_solution;
      first -= tmpp * pod.mean_vector;
      Vector<double> reduced_tmpp = pod.compute_reduced_vector(tmpp);
      double second = reduced_tmpp * solution;
      std::cout << "first - second = " << first - second << " (first: " << first << " , second: " << second << " )" << std::endl;
    }
    // reduced_system_rhs_pressure.add(fluid_density, tmp_clone);

    std::cout << "tmp:" << std::endl;
    tmp.print(std::cout);

    std::cout << "tmp_clone:" << std::endl;
    tmp_clone.print(std::cout);

    std::cout << "difference: ";
    tmp_clone.add(-1.0, tmp);
    std::cout << tmp_clone.linfty_norm() << std::endl;
    tmp_clone.print(std::cout);
    */

    Vector<double> tmp2(pod.r_p);
    second_nonlinearity_matrix_pressure.vmult(tmp2, solution);
    reduced_system_rhs_pressure.add(fluid_density, tmp2);

    /*
    // debug code for second_snonlinearity_matrix_pressure
    Vector<double> tmpp2(pod.m);
    Vector<double> tmp_clone2(pod.r_p);
    for (int i = 0; i < pod.r_p; i++)
    {
      pod.navier_stokes_solver.nonlinear_tensor_pressure[i].vmult(tmpp2,pod.mean_vector);
      tmp_clone2[i] = tmpp2 * projected_solution;
      tmp_clone2[i] -= tmpp2 * pod.mean_vector;
    }
    // reduced_system_rhs_pressure.add(fluid_density, tmp_clone2);

    std::cout << "tmp2:" << std::endl;
    tmp2.print(std::cout);

    std::cout << "tmp_clone2:" << std::endl;
    tmp_clone2.print(std::cout);

    std::cout << "difference: ";
    tmp_clone2.add(-1.0, tmp2);
    std::cout << tmp_clone2.linfty_norm() << std::endl; // seems a bit big... up to 0.001 for r = 25, up to 0.4 for r = 10
    tmp_clone2.print(std::cout);
    */

    reduced_system_rhs_pressure.add(fluid_density, nonlin_mean_contrib_rhs_pressure);

    /*
    Vector<double> out_vector(pod.r_p);
    Vector<double> tmppp(pod.m);
    for (int i = 0; i < pod.r_p; i++)
    {
      pod.navier_stokes_solver.nonlinear_tensor_pressure[i].vmult(tmppp, projected_solution);
      out_vector[i] = tmppp * projected_solution;
      out_vector[i] *= fluid_density;
    }
    */

    /*
    std::cout << "reduced_system_rhs_pressure:" << std::endl;
    reduced_system_rhs_pressure.print(std::cout);

    std::cout << "out_vector:" << std::endl;
    out_vector.print(std::cout);

    std::cout << "difference: ";
    out_vector.add(-1.0, reduced_system_rhs_pressure);
    std::cout << out_vector.linfty_norm() << std::endl;
    out_vector.print(std::cout);
    */

    reduced_system_rhs_pressure.add(-1.0, laplace_mean_contrib_rhs_pressure);
    // out_vector.add(-1.0, laplace_mean_contrib_rhs_pressure);

    // only for debugging:
    Vector<double> old_reduced_system_rhs_pressure = reduced_system_rhs_pressure;

    // add terms from boundary conditions:

    //  μ([∇·(∇v + ∇v^T)]·n, φ)_{∂Ω \ Γ_{out}}
    Vector<double> tmp3(pod.r_p);
    boundary_integral_matrix_pressure.vmult(tmp3, solution);
    reduced_system_rhs_pressure.add(1.0, tmp3);

    reduced_system_rhs_pressure.add(1.0, boundary_mean_contrib_rhs_pressure);

    Vector<double> old_reduced_system_rhs_pressure2 = reduced_system_rhs_pressure;
    //  -ϱ([(v·∇)v]·n, φ)_{∂Ω \ Γ_{out}}
    reduced_system_rhs_pressure.add(-fluid_density, nonlinearity_boundary_twice_evaluated_at_pressure(solution)); // solution is the velocity ROM solution

    Vector<double> tmp_bc(pod.r_p);
    first_nonlinearity_matrix_boundary_pressure.vmult(tmp_bc, solution);
    reduced_system_rhs_pressure.add(-fluid_density, tmp_bc);

    Vector<double> tmp_bc2(pod.r_p);
    second_nonlinearity_matrix_boundary_pressure.vmult(tmp_bc2, solution);
    reduced_system_rhs_pressure.add(-fluid_density, tmp_bc2);

    reduced_system_rhs_pressure.add(-fluid_density, nonlin_boundary_mean_contrib_rhs_pressure);

    // the next two lines are only for testing / debugging
    old_reduced_system_rhs_pressure2.add(-1.0, reduced_system_rhs_pressure);
    std::cout << "Boundary contribution (CONVECTION): " << old_reduced_system_rhs_pressure2.linfty_norm() << std::endl;
    old_reduced_system_rhs_pressure2.print(std::cout);

    // the next two lines are only for testing / debugging
    old_reduced_system_rhs_pressure.add(-1.0, reduced_system_rhs_pressure);
    std::cout << "Boundary contribution: " << old_reduced_system_rhs_pressure.linfty_norm() << std::endl;
    old_reduced_system_rhs_pressure.print(std::cout);

    // the next line is only for testing / debugging
    std::cout << "Linfty norm of RHS: " << reduced_system_rhs_pressure.linfty_norm() << std::endl;

    // the next lines are only for testing / debugging
    std::cout << "RHS: " << std::endl;
    reduced_system_rhs_pressure.print(std::cout);

    // SOLVE THE LINEAR EQUATION SYSTEM
    reduced_system_matrix_inverse_pressure.vmult(solution_pressure, reduced_system_rhs_pressure);
    //reduced_system_matrix_inverse_pressure.vmult(solution_pressure, out_vector);

    /*
    // iterative solver:
    SolverControl            solver_control(300, 1e-10 * reduced_system_rhs_pressure.l2_norm());
    SolverCG<Vector<double>> gmres(solver_control);
    gmres.solve(reduced_laplace_matrix_pressure, solution_pressure, reduced_system_rhs_pressure, PreconditionIdentity());
    */
  }

//  template<int dim>
//  void ReducedOrderModel<dim>::pressure_reconstruction_with_nn()
//  {
//    // convert velocity ROM solution to std::vector
//    std::vector<double> nn_input(pod.r);
//    for (int i = 0; i < pod.r; ++i)
//      nn_input[i] = solution[i];
//
//    // inference with neural network
//    std::vector<double> nn_output = neural_network.forward(nn_input);
//
//    // save neural network output to pressure ROM solution
//    for (int i = 0; i < pod.r_p; ++i)
//      solution_pressure[i] = nn_output[i];
//  }

  template<int dim>
  void ReducedOrderModel<dim>::compute_projected_solution()
  {
    // project the solution into the FEM space
    projected_solution.reinit(pod.navier_stokes_solver.dofs_velocity);
    for (int i = 0; i<pod.r-pod.r_s; i++){
//    for (int i = 0; i<pod.r; i++){
    	projected_solution.add(solution[i],pod.pod_vectors[i]);
    }
    // add mean vector to the projection
    projected_solution.add(1.0, pod.mean_vector);
  }

  template<int dim>
   void ReducedOrderModel<dim>::compute_projected_solution_no_mean()
   {
     // project the solution into the FEM space
     projected_solution.reinit(pod.navier_stokes_solver.dof_handler_velocity.n_dofs());
     for (int i = 0; i<pod.r; i++)
       projected_solution.add(solution[i],pod.pod_vectors[i]);
   }

  template<int dim>
  void ReducedOrderModel<dim>::compute_projected_solution_pressure()
  {
	 bool use_fem = false;
	  if (!use_fem) {

		  // project the solution into the FEM space
		  projected_solution_pressure.reinit(pod.navier_stokes_solver.dofs_pressure);
		  for (int i = 0; i<pod.r_p; i++)
		  {
			  projected_solution_pressure.add(solution[i+pod.r],pod.pod_vectors_p[i]);
		  }

	  }
	  else {
		  projected_solution_pressure.reinit(pod.navier_stokes_solver.dof_handler_pressure.n_dofs());
		  std::string filename_h5 = pde_info->fem_path + "mu="+ std::to_string(viscosity) +"/snapshots/snapshot_" \
				  	  	  + Utilities::int_to_string(timestep_number, 6) + ".h5";
		  projected_solution_pressure.add(1.0,load_h5_vector_pressure(filename_h5));
	  }
  }

  template<int dim>
  void ReducedOrderModel<dim>::save_h5_matrix(const std::string &file_name, const dealii::FullMatrix<double> &matrix)
  {
	  hid_t file_id = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                                H5P_DEFAULT);
      hsize_t dims[2];

      dims[0] = matrix.m();
      dims[1] = matrix.n();
      hid_t dataspace_id = H5Screate_simple(2, dims, nullptr);
      std::string dataset_name = "/A";
      hid_t dataset_id = H5Dcreate2(file_id, dataset_name.c_str(),
                                    H5T_NATIVE_DOUBLE, dataspace_id,
                                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
               static_cast<const void *>(&matrix(0, 0)));
      H5Dclose(dataset_id);
      H5Sclose(dataspace_id);
      H5Fclose(file_id);
  }

  template<int dim>
  void ReducedOrderModel<dim>::load_h5_matrix(const std::string &file_name, dealii::FullMatrix<double> &matrix)
  {
	  hid_t file_id = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

	        std::string dataset_name = "/A";
	        hid_t dataset = H5Dopen1(file_id, dataset_name.c_str());
	        hid_t datatype = H5Dget_type(dataset);
	        hid_t dataspace = H5Dget_space(dataset);
	        int rank = H5Sget_simple_extent_ndims(dataspace);
	        Assert(rank == 2, StandardExceptions::ExcInternalError());

	        std::vector<hsize_t> dims(rank);
	        std::vector<hsize_t> max_dims(rank);
	        H5Sget_simple_extent_dims(dataspace, dims.data(), max_dims.data());

	        matrix.reinit(dims[0], dims[1]);
	        H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT,
	                static_cast<void *>(&(matrix(0, 0))));

	        H5Sclose(dataspace);
	        H5Tclose(datatype);
	        H5Dclose(dataset);
	        H5Fclose(file_id);
  }

  template<int dim>
  void ReducedOrderModel<dim>::output_results()
  {
    Vector<double> tmp = load_h5_vector(intial_solution_snapshot);
    Vector<double> tmp2= load_h5_vector(intial_solution_snapshot);
    DataOut<dim> data_out;
    data_out.attach_dof_handler(pod.navier_stokes_solver.dof_handler_velocity);

    data_out.add_data_vector(projected_solution, "velocity");
    data_out.add_data_vector(pod.pod_vectors[0], "pod_vector");

    data_out.add_data_vector(pod.mean_vector, "velocity_mean");

    data_out.add_data_vector(tmp, "full");
    pod.navier_stokes_solver.mass_matrix_velocity.vmult(tmp2, tmp);
    data_out.add_data_vector(tmp2, "weights_MassMat");

//    tmp = load_h5_vector(intial_solution_snapshot);
//    tmp2= load_h5_vector(intial_solution_snapshot);
//    tmp2 = pod.space_weights;
//    tmp2.scale(tmp);
//    data_out.add_data_vector(tmp2, "space_weights");
    data_out.build_patches();
    data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));

    const std::string filename =
    		pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/solution/solution-" + Utilities::int_to_string(timestep_number, 6) + ".vtk";
    std::ofstream output(filename);
    data_out.write_vtk(output);
    const std::string filename_h5 =  pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/h5/solution-" + Utilities::int_to_string(timestep_number, 6) + ".h5";
    pod.save_h5_vector(filename_h5, projected_solution);
  }

  template<int dim>
  void ReducedOrderModel<dim>::output_results_vp()
  {
    std::vector<std::string> solution_names;
	solution_names.push_back("x_velo");
	solution_names.push_back("y_velo");
	solution_names.push_back("p_fluid");


    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    	data_component_interpretation(dim + 1, DataComponentInterpretation::component_is_scalar);
    DataOut<dim> data_out;
    data_out.attach_dof_handler(pod.navier_stokes_solver.dof_handler);
    data_out.add_data_vector(projected_solution_combined, solution_names,
                            DataOut<dim>::type_dof_data,
                            data_component_interpretation);
    data_out.build_patches();
    data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));

    const std::string filename =
    		pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/solution/solution-" + Utilities::int_to_string(timestep_number, 6) + ".vtk";
    std::ofstream output(filename);
    data_out.write_vtk(output);
    const std::string filename_h5 =  pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/h5/solution-" + Utilities::int_to_string(timestep_number, 6) + ".h5";
    pod.save_h5_vector(filename_h5, projected_solution);
  }

  template<int dim>
  void ReducedOrderModel<dim>::output_results_pressure()
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(pod.navier_stokes_solver.dof_handler_pressure);
    data_out.add_data_vector(projected_solution_pressure, "pressure");
    data_out.build_patches();
    data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));

    const std::string filename =
    		pde_info->rom_path + "solution_pressure/solution-" + Utilities::int_to_string(timestep_number, 6) + ".vtk";
    std::ofstream output(filename);
    data_out.write_vtk(output);
  }

  template<int dim>
  void ReducedOrderModel<dim>::output_error()
  {
	if (time>1.9) { //(pde_info->fine_endtime-pde_info->fine_timestep)) {
		return;
	}
    Vector<double> fem_solution(pod.navier_stokes_solver.dof_handler_velocity.n_dofs());
    Vector<double> fem_solution_pressure(pod.navier_stokes_solver.dof_handler_pressure.n_dofs());
    fem_solution = load_h5_vector(pde_info->fem_path + "/snapshots/snapshot_" + Utilities::int_to_string(timestep_number, 6) + ".h5");
//    fem_solution_pressure = load_h5_vector_pressure("result/FEM/snapshots/snapshot_" + Utilities::int_to_string(timestep_number, 6) + ".h5");
    fem_solution.add(-1.0, projected_solution);
//    fem_solution_pressure.add(-1.0, projected_solution_pressure);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(pod.navier_stokes_solver.dof_handler_velocity);
    data_out.add_data_vector(fem_solution, "velocity");
    data_out.build_patches();
    data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));

    const std::string filename =
    		pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/error/solution-" + Utilities::int_to_string(timestep_number, 6) + ".vtk";
    std::ofstream output(filename);
    data_out.write_vtk(output);
//
//    DataOut<dim> data_out2;
//    data_out2.attach_dof_handler(pod.navier_stokes_solver.dof_handler_pressure);
//    data_out2.add_data_vector(fem_solution_pressure, "pressure");
//    data_out2.build_patches();
//    data_out2.set_flags(DataOutBase::VtkFlags(time, timestep_number));
//
//    const std::string filename2 =
//      pde_info->rom_path + "error_pressure/solution-" + Utilities::int_to_string(timestep_number, 6) + ".vtk";
//    std::ofstream output2(filename2);
//    data_out2.write_vtk(output2);
  }

  // Compute the pressure at a certain point.
  template <int dim>
  double ReducedOrderModel<dim>::compute_pressure(Point<dim> p) const
  {
    Vector<double> tmp(1);
    VectorTools::point_value(pod.navier_stokes_solver.dof_handler_pressure,
                             projected_solution_pressure,
                             p,
                             tmp);
    return tmp(0);
  }

  template <int dim>
  double ReducedOrderModel<dim>::compute_pressure_vp(Point<dim> p) const
  {
    Vector<double> tmp(dim + 1);
    VectorTools::point_value(pod.navier_stokes_solver.dof_handler,
    							projected_solution_combined,
                             p,
                             tmp);
    return tmp(dim);
  }

  // Now, we arrive at the function that is responsible
  // to compute the line integrals for the drag and the lift. Note, that
  // by a proper transformation via the Gauss theorem, the both
  // quantities could also be achieved by domain integral computation.
  // Nevertheless, we choose the line integration because deal.II provides
  // all routines for face value evaluation.
  template <int dim>
  void ReducedOrderModel<dim>::compute_drag_lift_tensor()
  {
    const QGauss<dim - 1> face_quadrature_formula(3);
    FEFaceValues<dim> fe_face_values(pod.navier_stokes_solver.fe_velocity, face_quadrature_formula,
                                     update_values | update_gradients | update_normal_vectors |
                                         update_JxW_values);

     FEFaceValues<dim> fe_face_values_pressure(pod.navier_stokes_solver.fe_pressure, face_quadrature_formula,
                                      update_values | update_gradients | update_normal_vectors |
                                          update_JxW_values);

    const unsigned int dofs_per_cell = pod.navier_stokes_solver.fe_velocity.dofs_per_cell;
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    std::vector<unsigned int> local_dof_indices(dofs_per_cell);
    std::vector<double> face_solution_values(n_face_q_points);

    std::vector<std::vector<Tensor<1, dim>>>
        face_solution_grads(n_face_q_points, std::vector<Tensor<1, dim>>(dim));

    Tensor<1, dim> drag_lift_value;

    typename DoFHandler<dim>::active_cell_iterator
        cell = pod.navier_stokes_solver.dof_handler_velocity.begin_active(),
        endc = pod.navier_stokes_solver.dof_handler_velocity.end();

    typename DoFHandler<dim>::active_cell_iterator
        cell_p = pod.navier_stokes_solver.dof_handler_pressure.begin_active(),
        endc_p = pod.navier_stokes_solver.dof_handler_pressure.end();

    for (; cell != endc; ++cell, ++cell_p)
    {
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
        if (cell->face(face)->at_boundary() &&
            cell->face(face)->boundary_id() == 80)
        {
          fe_face_values.reinit(cell, face);
          fe_face_values_pressure.reinit(cell_p, face);
          fe_face_values_pressure.get_function_values(projected_solution_pressure, face_solution_values);
          fe_face_values.get_function_gradients(projected_solution, face_solution_grads);

          for (unsigned int q = 0; q < n_face_q_points; ++q)
          {
            Tensor<2, dim> pI;
            pI.clear(); // reset all values to zero
            for (unsigned int l = 0; l < dim; l++)
              pI[l][l] = face_solution_values[q];

            Tensor<2, dim> grad_v;
            for (unsigned int l = 0; l < dim; l++)
              for (unsigned int m = 0; m < dim; m++)
                grad_v[l][m] = face_solution_grads[q][l][m];

            Tensor<2, dim> sigma_fluid = -pI + fluid_density * viscosity * (grad_v + transpose(grad_v));

            drag_lift_value -= sigma_fluid * fe_face_values.normal_vector(q) * fe_face_values.JxW(q);
          }
        } // end boundary 80 for fluid

    } // end cell

    drag_lift_value *= 20.0;

//    std::cout << "Face drag:   "
//              << "   " << std::setprecision(16) << drag_lift_value[0] << std::endl;
//    std::cout << "Face lift:   "
//              << "   " << std::setprecision(16) << drag_lift_value[1] << std::endl;

    // save drag and lift values to text files

    if (timestep_number == 1) {
    	std::remove((pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/drag.txt").c_str());
    	std::remove((pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/lift.txt").c_str());
//    	std::cout << "No prob I have killed old functional data for you, bro" << std::endl;
    }

    std::ofstream drag_out;
    drag_out.open(pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/drag.txt", std::ios_base::app); // append instead of overwrite
    drag_out << time << "," << std::setprecision(16) << drag_lift_value[0] << std::endl;
    drag_out.close();

    std::ofstream lift_out;
    lift_out.open(pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/lift.txt", std::ios_base::app); // append instead of overwrite
    lift_out << time << "," << std::setprecision(16) << drag_lift_value[1]  << std::endl;
    lift_out.close();
  }

  template <int dim>
   void ReducedOrderModel<dim>::compute_drag_lift_tensor_vp()
   {
	  const QGauss<dim - 1> face_quadrature_formula(3);
	      FEFaceValues<dim> fe_face_values(pod.navier_stokes_solver.fe, face_quadrature_formula,
	                                       update_values | update_gradients | update_normal_vectors |
	                                           update_JxW_values);

	      const unsigned int dofs_per_cell = pod.navier_stokes_solver.fe.dofs_per_cell;
	      const unsigned int n_face_q_points = face_quadrature_formula.size();

	      std::vector<unsigned int> local_dof_indices(dofs_per_cell);
	      std::vector<Vector<double>> face_solution_values(n_face_q_points,
	                                                       Vector<double>(dim + 1));

	      std::vector<std::vector<Tensor<1, dim>>>
	          face_solution_grads(n_face_q_points, std::vector<Tensor<1, dim>>(dim + 1));

	      Tensor<1, dim> drag_lift_value;

	      typename DoFHandler<dim>::active_cell_iterator
	          cell = pod.navier_stokes_solver.dof_handler.begin_active(),
	          endc = pod.navier_stokes_solver.dof_handler.end();

	      for (; cell != endc; ++cell)
	      {

	        // First, we are going to compute the forces that
	        // act on the cylinder. We notice that only the fluid
	        // equations are defined here.
	        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
	          if (cell->face(face)->at_boundary() &&
	              cell->face(face)->boundary_id() == 80)
	          {
	            fe_face_values.reinit(cell, face);
	            fe_face_values.get_function_values(projected_solution_combined, face_solution_values);
	            fe_face_values.get_function_gradients(projected_solution_combined, face_solution_grads);

	            for (unsigned int q = 0; q < n_face_q_points; ++q)
	            {

	              Tensor<2, dim> pI;
	              pI.clear(); // reset all values to zero
	              for (unsigned int l = 0; l < dim; l++)
	                pI[l][l] = face_solution_values[q](dim);

	              Tensor<2, dim> grad_v;
	              for (unsigned int l = 0; l < dim; l++)
	                for (unsigned int m = 0; m < dim; m++)
	                  grad_v[l][m] = face_solution_grads[q][l][m];

	              Tensor<2, dim> sigma_fluid = -pI + fluid_density * viscosity * (grad_v + transpose(grad_v));

	              drag_lift_value -= sigma_fluid * fe_face_values.normal_vector(q) * fe_face_values.JxW(q);
	            }
	          } // end boundary 80 for fluid

	      } // end cell

     drag_lift_value *= 20.0;

 //    std::cout << "Face drag:   "
 //              << "   " << std::setprecision(16) << drag_lift_value[0] << std::endl;
 //    std::cout << "Face lift:   "
 //              << "   " << std::setprecision(16) << drag_lift_value[1] << std::endl;

     // save drag and lift values to text files

//     if (timestep_number == 1) {
//     	std::remove((pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/drag.txt").c_str());
//     	std::remove((pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/lift.txt").c_str());
// //    	std::cout << "No prob I have killed old functional data for you, bro" << std::endl;
//     }

     std::ofstream drag_out;
     drag_out.open(pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/drag.txt", std::ios_base::app); // append instead of overwrite
     drag_out << time << "," << std::setprecision(16) << drag_lift_value[0] << std::endl;
     drag_out.close();

     std::ofstream lift_out;
     lift_out.open(pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/lift.txt", std::ios_base::app); // append instead of overwrite
     lift_out << time << "," << std::setprecision(16) << drag_lift_value[1]  << std::endl;
     lift_out.close();
   }

  // Here, we compute the four quantities of interest:
  // the drag, and the lift and a pressure difference
  template <int dim>
  void ReducedOrderModel<dim>::compute_functional_values()
  {
	  // negelct till 21.02.2022
    double p_front = compute_pressure(Point<dim>(0.15, 0.2)); // pressure - left  point on circle
    double p_back = compute_pressure(Point<dim>(0.25, 0.2));  // pressure - right point on circle

    double p_diff = p_front - p_back;

    // save pressure difference to text file
    if (timestep_number == 1) {
    	std::remove((pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/pressure.txt").c_str());
//    	std::cout << "No prob I have killed old pressure data for you, bro" << std::endl;
    }

    std::ofstream p_out;
    p_out.open(pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/pressure.txt", std::ios_base::app); // append instead of overwrite
    p_out << time << "," << std::setprecision(16) << p_diff << std::endl;
    p_out.close();

   double time_2 =0.0;
   double p_diff_2 =0.0;
   char ttttt;
	std::ifstream p_in;
	p_in.open(pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/pressure.txt", std::ios_base::app);
	while (p_in >> time_2 >> ttttt >> p_diff_2)
	{}
	p_in.close();
//	if ((std::abs(time_2-time)>1e-14) || (std::abs(p_diff_2-p_diff)>1e-14))
//		std::cout << "read: " << time_2-time << "," << std::setprecision(16) << p_diff_2-p_diff << std::endl;
//    std::cout << "------------------" << std::endl;
//    std::cout << "Pressure difference:  " << "   " << std::setprecision(16) << p_diff << std::endl;
//    //std::cout << "P-front: "  << "   " << std::setprecision(16) << p_front << std::endl;
//    // std::cout << "P-back:  "  << "   " << std::setprecision(16) << p_back << std::endl;
//    std::cout << "------------------" << std::endl;

    // Compute drag and lift via line integral

//    compute_drag_lift_tensor(); //important 16.09.2022

//    std::cout << "------------------" << std::endl;

//    std::cout << std::endl;
  }

  // Here, we compute the four quantities of interest:
   // the drag, and the lift and a pressure difference
   template <int dim>
   void ReducedOrderModel<dim>::compute_functional_values_vp()
   {
 	  // negelct till 21.02.2022
     double p_front = compute_pressure_vp(Point<dim>(0.15, 0.2)); // pressure - left  point on circle
     double p_back = compute_pressure_vp(Point<dim>(0.25, 0.2));  // pressure - right point on circle

     double p_diff = p_front - p_back;

     // save pressure difference to text file
//     if (timestep_number == 1) {
//     	std::remove((pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/pressure.txt").c_str());
// //    	std::cout << "No prob I have killed old pressure data for you, bro" << std::endl;
//     }

     std::ofstream p_out;
     p_out.open(pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/pressure.txt", std::ios_base::app); // append instead of overwrite
     p_out << time << "," << std::setprecision(16) << p_diff << std::endl;
     p_out.close();

    double time_2 =0.0;
    double p_diff_2 =0.0;
    char ttttt;
 	std::ifstream p_in;
 	p_in.open(pde_info->rom_path + "mu=" + std::to_string(viscosity) + "/pressure.txt", std::ios_base::app);
 	while (p_in >> time_2 >> ttttt >> p_diff_2)
 	{}
 	p_in.close();
// 	if ((std::abs(time_2-time)>1e-14) || (std::abs(p_diff_2-p_diff)>1e-14))
// 		std::cout << "read: " << time_2-time << "," << std::setprecision(16) << p_diff_2-p_diff << std::endl;
 //    std::cout << "------------------" << std::endl;
 //    std::cout << "Pressure difference:  " << "   " << std::setprecision(16) << p_diff << std::endl;
 //    //std::cout << "P-front: "  << "   " << std::setprecision(16) << p_front << std::endl;
 //    // std::cout << "P-back:  "  << "   " << std::setprecision(16) << p_back << std::endl;
 //    std::cout << "------------------" << std::endl;

     // Compute drag and lift via line integral

     compute_drag_lift_tensor_vp(); //important 16.09.2022

 //    std::cout << "------------------" << std::endl;

 //    std::cout << std::endl;
   }

  template<int dim>
  void ReducedOrderModel<dim>::run(int refinements, bool output_files, bool compute_error)
  {
//	  pde_info->viscosity = 0.001;
	    std::cout << "------------------------------------------------------ \n" \
	    		  << "Reynolds: " << 0.1/pde_info->viscosity << ",  mu: " << pde_info->viscosity << "\n" \
	    		  << "------------------------------------------------------ \n" << std::endl;


    std::cout << "ROM setup ..." << std::endl;
    setup(refinements, output_files); // compute the POD basis

    load_reduced_matrices_greedy(pde_info->fine_timestep, pod.navier_stokes_solver.theta); // use: compute_reduced_matrices_linearized(...) if using linearization of NSE

    std::cout << "Solving the reduced order model ..." << std::endl;

    solve_reduced_order_model(/*start_time*/ pde_info->start_time, \
    						/*end_time*/ pde_info->fine_endtime-pde_info->POD_start_time, \
							/*time step*/pde_info->fine_timestep, /*theta*/pod.navier_stokes_solver.theta, \
							output_files, compute_error);
    std::cout << "Done" << std::endl << std::endl;
  }

  template<int dim>
  void ReducedOrderModel<dim>::run_no_mean(int refinements, bool output_files, bool compute_error)
  {
//	  pde_info->viscosity = 0.001;
	    std::cout << "------------------------------------------------------ \n" \
	    		  << "Reynolds: " << 0.1/pde_info->viscosity << ",  mu: " << pde_info->viscosity << "\n" \
	    		  << "no mean"<< "\n" \
	    		  << "------------------------------------------------------ \n" << std::endl;


    std::cout << "ROM setup ..." << std::endl;
    setup(refinements, output_files); // compute the POD basis

    load_reduced_matrices_greedy_no_mean(pde_info->fine_timestep, pod.navier_stokes_solver.theta); // use: compute_reduced_matrices_linearized(...) if using linearization of NSE

    std::cout << "Solving the reduced order model ..." << std::endl;

    solve_reduced_order_model_no_mean(/*start_time*/ pde_info->start_time, \
    						/*end_time*/ pde_info->fine_endtime-pde_info->POD_start_time, \
							/*time step*/pde_info->fine_timestep, /*theta*/pod.navier_stokes_solver.theta, \
							output_files, compute_error);
    std::cout << "Done" << std::endl << std::endl;
  }

  template<int dim>
  void ReducedOrderModel<dim>::run_vp(int refinements, bool output_files, bool compute_error)
  {
//	  pde_info->viscosity = 0.001;
	    std::cout << "------------------------------------------------------ \n" \
	    		  << "Reynolds: " << 0.1/pde_info->viscosity << ",  mu: " << pde_info->viscosity << "\n" \
	    		  << "------------------------------------------------------ \n" << std::endl;


    std::cout << "ROM setup ..." << std::endl;
    setup_vp(refinements, output_files); // compute the POD basis

    load_reduced_matrices_greedy_vp(pde_info->fine_timestep, pod.navier_stokes_solver.theta);

    std::cout << "Solving the reduced order model ..." << std::endl;
    if (output_files)
    	std::cout << "do output" <<std::endl;
    solve_reduced_order_model_vp(/*start_time*/ pde_info->start_time, \
    						/*end_time*/ pde_info->fine_endtime-pde_info->POD_start_time, \
							/*time step*/pde_info->fine_timestep, /*theta*/pod.navier_stokes_solver.theta, \
							output_files, compute_error);
    std::cout << "Done" << std::endl << std::endl;

//    load_reduced_matrices_greedy(pde_info->fine_timestep, pod.navier_stokes_solver.theta); // use: compute_reduced_matrices_linearized(...) if using linearization of NSE
//
//    std::cout << "Solving the reduced order model ..." << std::endl;
//
//    solve_reduced_order_model(/*start_time*/ pde_info->start_time, \
//    						/*end_time*/ pde_info->fine_endtime-pde_info->POD_start_time, \
//							/*time step*/pde_info->fine_timestep, /*theta*/pod.navier_stokes_solver.theta, \
//							output_files, compute_error);
//    std::cout << "Done" << std::endl << std::endl;
  }
  template class ReducedOrderModel<2>;
}
