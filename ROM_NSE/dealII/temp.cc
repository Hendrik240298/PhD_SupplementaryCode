// additional assembly functions for reduced order modeling
  template <int dim>
  void NavierStokes<dim>::assemble_laplace_with_transposed_trial_function()
  {
    laplace_matrix_velocity_with_transposed_trial_function = 0;

    QGauss<dim> quadrature_formula(fe_velocity.degree + 2);

    FEValues<dim> fe_values(fe_velocity,
                        quadrature_formula,
                        update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe_velocity.n_dofs_per_cell();

    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<unsigned int> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector velocities(0);

    // test function
    std::vector<Tensor<2, dim>> phi_i_grads(dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_velocity.begin_active(),
      endc = dof_handler_velocity.end();

    // Next, we run over all cells
    for (; cell != endc; ++cell)
    {
      fe_values.reinit(cell);
      local_matrix = 0;

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          phi_i_grads[k] = fe_values[velocities].gradient(k, q);
        }

        // Outer loop for dofs
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          // Inner loop for dofs
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            local_matrix(j, i) += scalar_product(transpose(phi_i_grads[i]), phi_i_grads[j]) * fe_values.JxW(q);
          } // end j dofs
        } // end i dofs
      } // end n_q_points

      // This is the same as discussed in step-22:
      cell->get_dof_indices(local_dof_indices);
      no_constraints.distribute_local_to_global(local_matrix, local_dof_indices,
                                             laplace_matrix_velocity_with_transposed_trial_function);
      // end cell
    }
  }
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  template <int dim>
    void NavierStokes<dim>::assemble_system_matrix()
    {
      system_matrix = 0;

      QGauss<dim> quadrature_formula(fe.degree + 2);
      QGauss<dim - 1> face_quadrature_formula(fe.degree + 2);

      FEValues<dim> fe_values(fe, quadrature_formula,
                              update_values |
                                  update_quadrature_points |
                                  update_JxW_values |
                                  update_gradients);

      FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                       update_values | update_quadrature_points |
                                           update_normal_vectors | update_gradients |
                                           update_JxW_values);

      const unsigned int dofs_per_cell = fe.dofs_per_cell;

      const unsigned int n_q_points = quadrature_formula.size();
      const unsigned int n_face_q_points = face_quadrature_formula.size();

      FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);

      std::vector<unsigned int> local_dof_indices(dofs_per_cell);

      // Now, we are going to use the
      // FEValuesExtractors to determine
      // the principle variables
      const FEValuesExtractors::Vector velocities(0);
      const FEValuesExtractors::Scalar pressure(dim);

      // We declare Vectors and Tensors for
      // the solutions at the previous Newton iteration:
      std::vector<Vector<double>> old_solution_values(n_q_points,
                                                      Vector<double>(dim + 1));

      std::vector<std::vector<Tensor<1, dim>>> old_solution_grads(n_q_points,
                                                                  std::vector<Tensor<1, dim>>(dim + 1));

      std::vector<Vector<double>> old_solution_face_values(n_face_q_points,
                                                           Vector<double>(dim + 1));

      std::vector<std::vector<Tensor<1, dim>>> old_solution_face_grads(n_face_q_points,
                                                                       std::vector<Tensor<1, dim>>(dim + 1));

      // Declaring test functions:
      std::vector<Tensor<1, dim>> phi_i_v(dofs_per_cell);
      std::vector<Tensor<2, dim>> phi_i_grads_v(dofs_per_cell);
      std::vector<double> phi_i_p(dofs_per_cell);

      typename DoFHandler<dim>::active_cell_iterator
          cell = dof_handler.begin_active(),
          endc = dof_handler.end();

      // Next, we run over all cells
      for (; cell != endc; ++cell)
      {
        fe_values.reinit(cell);
        local_matrix = 0;

        // Old Newton iteration values
        fe_values.get_function_values(solution, old_solution_values);
        fe_values.get_function_gradients(solution, old_solution_grads);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
          for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            phi_i_v[k] = fe_values[velocities].value(k, q);
            phi_i_grads_v[k] = fe_values[velocities].gradient(k, q);
            phi_i_p[k] = fe_values[pressure].value(k, q);
          }

          // We build values, vectors, and tensors
          // from information of the previous Newton step
          // which are required for the convection term
          Tensor<1, dim> v;
          for (unsigned int l = 0; l < dim; l++)
            v[l] = old_solution_values[q][l];

          Tensor<2, dim> grad_v;
          for (unsigned int l = 0; l < dim; l++)
            for (unsigned int m = 0; m < dim; m++)
              grad_v[l][m] = old_solution_grads[q][l][m];

          // Outer loop for dofs
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            Tensor<2, dim> pI_LinP;
            pI_LinP.clear(); // reset all values to zero
            for (unsigned int l = 0; l < dim; l++)
              pI_LinP[l][l] = phi_i_p[i];

            Tensor<2, dim> grad_v_LinV;
            for (unsigned int l = 0; l < dim; l++)
              for (unsigned int m = 0; m < dim; m++)
                grad_v_LinV[l][m] = phi_i_grads_v[i][l][m];

            // stress tensor for incompressible Newtonian fluid
            // const Tensor<2, dim> stress_fluid_LinAll = -pI_LinP + fluid_density * viscosity * (phi_i_grads_v[i] + transpose(phi_i_grads_v[i]));
            const Tensor<2, dim> stress_fluid_LinAll_1st_term = -pI_LinP;
            const Tensor<2, dim> stress_fluid_LinAll_2nd_term = fluid_density * viscosity * (phi_i_grads_v[i] + transpose(phi_i_grads_v[i]));

            // derivative of convection term
            const Tensor<1, dim> convection_fluid_LinAll = fluid_density * (phi_i_grads_v[i] * v + grad_v * phi_i_v[i]);

            const double incompressibility_LinAll = trace(phi_i_grads_v[i]);

            // Inner loop for dofs
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              // Fluid , NSE
              const unsigned int comp_j = fe.system_to_component_index(j).first;
              if (comp_j == 0 || comp_j == 1)
              {
                local_matrix(j, i) += (fluid_density * phi_i_v[i] * phi_i_v[j] +
                                       time_step * theta * convection_fluid_LinAll * phi_i_v[j] +
                                       // time_step * scalar_product(stress_fluid_LinAll, phi_i_grads_v[j])) *
                                       time_step * scalar_product(stress_fluid_LinAll_1st_term, phi_i_grads_v[j]) +
                                       time_step * theta * scalar_product(stress_fluid_LinAll_2nd_term, phi_i_grads_v[j])) *
                                      fe_values.JxW(q);
              }
              else if (comp_j == 2)
              {
                // incompressibility condition
                local_matrix(j, i) += (incompressibility_LinAll * phi_i_p[j]) * fe_values.JxW(q);
              }
              // end j dofs
            }
            // end i dofs
          }
          // end n_q_points
        }
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        template <int dim>
        Vector<double> NavierStokes<dim>::reconstruct_pressure_FEM(Vector<double> &velocity_solution)
        {
          // create a vector for thre reconstructed pressure
          Vector<double> pressure_solution(dof_handler_pressure.n_dofs());

          // assemble the right hand side vector of the Poisson equation
          Vector<double> right_hand_side(dof_handler_pressure.n_dofs());

          // ASSEMBLY:
          QGauss<dim> quadrature_formula(2 * fe_pressure.degree + 2);

          FEValues<dim> fe_values(fe_velocity,
                              quadrature_formula,
                              update_values | update_gradients | update_quadrature_points | update_JxW_values);
          FEValues<dim> fe_values_pressure(fe_pressure,
                              quadrature_formula,
                              update_values | update_gradients | update_quadrature_points | update_JxW_values);

          const unsigned int dofs_per_cell = fe_pressure.dofs_per_cell;
          Vector<double> local_rhs(dofs_per_cell);
          std::vector<unsigned int> local_dof_indices(dofs_per_cell);

          const unsigned int n_q_points = quadrature_formula.size();
          const FEValuesExtractors::Vector velocities(0);

          std::vector<std::vector<Tensor<1, dim>>>
              velocity_grads(n_q_points, std::vector<Tensor<1, dim>>(dim));

          typename DoFHandler<dim>::active_cell_iterator
            cell = dof_handler_velocity.begin_active(),
            endc = dof_handler_velocity.end();
          typename DoFHandler<dim>::active_cell_iterator
            cell_p = dof_handler_pressure.begin_active(),
            endc_p = dof_handler_pressure.end();

          // Next, we run over all cells
          for (; cell != endc; ++cell, ++cell_p)
          {
            fe_values.reinit(cell);
            fe_values_pressure.reinit(cell_p);

            fe_values.get_function_gradients(velocity_solution, velocity_grads);

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
              Tensor<2, dim> grad_v;
              for (unsigned int l = 0; l < dim; l++)
                for (unsigned int m = 0; m < dim; m++)
                  grad_v[l][m] = velocity_grads[q][l][m];

              double tmp = 0.0;
              for (unsigned int l = 0; l < dim; l++)
                for (unsigned int m = 0; m < dim; m++)
                  tmp += grad_v[l][m] * grad_v[m][l];

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                local_rhs(i) += fluid_density * tmp * fe_values_pressure.shape_value(i, q) * fe_values_pressure.JxW(q);
              }
            } // end n_q_points

            cell_p->get_dof_indices(local_dof_indices);
            pressure_constraints.distribute_local_to_global(local_rhs, local_dof_indices,
                                                   right_hand_side);
            // end cell
          }
        