#include "../include/nse.h"

// source: https://github.com/drwells/dealii-pod/blob/master/include/deal.II-pod/h5/h5.templates.h
// load block vector from HDF5 file
dealii::BlockVector<double> load_h5_block_vector(const std::string &file_path)
{
  // create an empty block vector
  dealii::BlockVector<double> h5_vector(2);

  hid_t file = H5Fopen(file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  // velocity block
  std::string dataset_name_v = "/velocity";
  hid_t dataset_v = H5Dopen1(file, dataset_name_v.c_str());
  hid_t datatype_v = H5Dget_type(dataset_v);
  hid_t dataspace_v = H5Dget_space(dataset_v);

  int n_v = H5Dget_storage_size(dataset_v)/sizeof(double);
  h5_vector.block(0).reinit(n_v);
  H5Dread(dataset_v, datatype_v, H5S_ALL, H5S_ALL, H5P_DEFAULT,
          static_cast<void *>(&(h5_vector.block(0)[0])));
  H5Sclose(dataspace_v);
  H5Tclose(datatype_v);
  H5Dclose(dataset_v);

  // pressure block
  std::string dataset_name_p = "/pressure";
  hid_t dataset_p = H5Dopen1(file, dataset_name_p.c_str());
  hid_t datatype_p = H5Dget_type(dataset_p);
  hid_t dataspace_p = H5Dget_space(dataset_p);

  int n_p = H5Dget_storage_size(dataset_p)/sizeof(double);
  h5_vector.block(1).reinit(n_p);
  H5Dread(dataset_p, datatype_p, H5S_ALL, H5S_ALL, H5P_DEFAULT,
          static_cast<void *>(&(h5_vector.block(1)[0])));
  H5Sclose(dataspace_p);
  H5Tclose(datatype_p);
  H5Dclose(dataset_p);

  h5_vector.collect_sizes();

  return h5_vector;
}

double load_h5_time(const std::string &file_path)
{
  // create an empty block vector
  double time;
  hid_t file = H5Fopen(file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  // time stamp
  std::string dataset_name_n_t = "/time";
  hid_t dataset_n_t = H5Dopen1(file, dataset_name_n_t.c_str());
  hid_t datatype_n_t = H5Dget_type(dataset_n_t);
  hid_t dataspace_n_t = H5Dget_space(dataset_n_t);

  //int n_t = H5Dget_storage_size(dataset_n_t)/sizeof(double);
  H5Dread(dataset_n_t, datatype_n_t, H5S_ALL, H5S_ALL, H5P_DEFAULT,
          static_cast<void *>(&time));
  H5Sclose(dataspace_n_t);
  H5Tclose(datatype_n_t);
  H5Dclose(dataset_n_t);

  return time;
}

// source: https://github.com/drwells/dealii-pod/blob/master/include/deal.II-pod/h5/h5.templates.h
// save block vector to HDF5 file
void save_h5_block_vector(const std::string &file_path, const dealii::BlockVector<double> &block_vector, const double time)
{
  hid_t file = H5Fcreate(file_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                                H5P_DEFAULT);

  // velocity block
  hsize_t n_v[1];
  n_v[0] = block_vector.block(0).size();
  hid_t dataspace_id_v = H5Screate_simple(1, n_v, nullptr);
  std::string dataset_name_v = "/velocity";
  hid_t dataset_id_v = H5Dcreate2 (file, dataset_name_v.c_str (),
                                 H5T_NATIVE_DOUBLE, dataspace_id_v,
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset_id_v, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
           static_cast<const void *>(block_vector.block(0).begin()));
  H5Dclose(dataset_id_v);
  H5Sclose(dataspace_id_v);

  // pressure block
  hsize_t n_p[1];
  n_p[0] = block_vector.block(1).size();
  hid_t dataspace_id_p = H5Screate_simple(1, n_p, nullptr);
  std::string dataset_name_p = "/pressure";
  hid_t dataset_id_p = H5Dcreate2 (file, dataset_name_p.c_str (),
                                 H5T_NATIVE_DOUBLE, dataspace_id_p,
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset_id_p, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
           static_cast<const void *>(block_vector.block(1).begin()));
  H5Dclose(dataset_id_p);
  H5Sclose(dataspace_id_p);

  // time stamp
  hsize_t n_t[1];
  n_t[0] = 1;
  hid_t dataspace_id_n_t = H5Screate_simple(1, n_t, nullptr);
  std::string dataset_name_n_t = "/time";
  hid_t dataset_id_n_t = H5Dcreate2 (file, dataset_name_n_t.c_str (),
                                 H5T_NATIVE_DOUBLE, dataspace_id_n_t,
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset_id_n_t, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
           static_cast<const void *>(&time));
  H5Dclose(dataset_id_n_t);
  H5Sclose(dataspace_id_n_t);

//  // time step size
//  hsize_t n_t[1];
//  n_t[0] = 1;
//  hid_t dataspace_id_n_t = H5Screate_simple(1, n_t, nullptr);
//  std::string dataset_name_n_t = "/time";
//  hid_t dataset_id_n_t = H5Dcreate2 (file, dataset_name_n_t.c_str (),
//                                 H5T_NATIVE_DOUBLE, dataspace_id_n_t,
//                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//  H5Dwrite(dataset_id_n_t, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
//           static_cast<const void *>(time));
//  H5Dclose(dataset_id_n_t);
//  H5Sclose(dataspace_id_n_t);


  H5Fclose(file);

}


namespace NSE
{
  using namespace dealii;

  // The boundary values are given to component
  // with number 0 (namely the x-velocity)
  template <int dim>
  double
  BoundaryParabel<dim>::value(const Point<dim> &p,
                              const unsigned int component) const
  {
    Assert(component < this->n_components,
           ExcIndexRange(component, 0, this->n_components));

    if (component == 0)
    {
      // BFAC 2D-1, 2D-2
      {
        return ((p(0) == 0) && (p(1) <= 0.41) ? -1.5 * (4.0 / 0.1681) *
                                                    (std::pow(p(1), 2) - 0.41 * std::pow(p(1), 1))
                                              : 0);
      }
    }

    return 0;
  }

  template <int dim>
  void
  BoundaryParabel<dim>::vector_value(const Point<dim> &p,
                                     Vector<double> &values) const
  {
    for (unsigned int c = 0; c < this->n_components; ++c)
      values(c) = BoundaryParabel<dim>::value(p, c);
  }

  template <int dim>
  NavierStokes<dim>::NavierStokes()
      : fe(FE_Q<dim>(2), dim, // velocities
           FE_Q<dim>(1), 1),  // pressure
        fe_velocity(FE_Q<dim>(2), dim),
        fe_pressure(FE_Q<dim>(1), 1),
        dof_handler(triangulation),
        dof_handler_velocity(triangulation),
        dof_handler_pressure(triangulation)
  {
    PDEInfo pde_info;
    init(&pde_info);
  }

  template <int dim>
  NavierStokes<dim>::NavierStokes(PDEInfo *pde_info)
      : fe(FE_Q<dim>(2), dim,                        // velocities
           FE_Q<dim>(1), 1),                         // pressure
       fe_velocity(FE_Q<dim>(2), dim),
       fe_pressure(FE_Q<dim>(1), 1),
       dof_handler(triangulation),
       dof_handler_velocity(triangulation),
       dof_handler_pressure(triangulation)
  {
    init(pde_info);
  }

  template <int dim>
  void NavierStokes<dim>::init(PDEInfo *pde_info)
  {
    time = pde_info->start_time;
    timestep_number = 0;
    theta = 0.5; // Cranck-Nicholson
    test_case = pde_info->test_case;
    newton_tol = pde_info->newton_tol;
    fluid_density = pde_info->fluid_density;
    viscosity = pde_info->viscosity;
    intial_solution_snapshot = pde_info->intial_solution_snapshot;
    coarse_timestep = pde_info->coarse_timestep;
    fine_timestep = pde_info->fine_timestep;
    coarse_endtime = pde_info->coarse_endtime;
    fine_endtime = pde_info->fine_endtime;
    POD_start_time = pde_info->POD_start_time;
    fem_path = pde_info->fem_path;
    if (coarse_endtime > time)
      time_step = coarse_timestep;
    else
      time_step = fine_timestep;
  }


  template <int dim>
    void NavierStokes<dim>::setup_system(int refinements)
    {
      const unsigned int initial_global_refinement = refinements;

      // In the following, we read a *.inp grid from a file.
      // The geometry information is based on the
      // 2D Schaefer/Turek 1996 benchmarks
      std::string grid_name;
      if (test_case == "2D-1" || test_case == "2D-2" || test_case == "2D-3")
        grid_name = "nsbench4_original.inp";
      AssertThrow(test_case == "2D-2", ExcMessage("Currently only the 2D-2 benchmark has been implemented."));

      GridIn<dim> grid_in;
      grid_in.attach_triangulation(triangulation);
      std::ifstream input_file(grid_name.c_str());
      Assert(dim == 2, ExcInternalError());
      grid_in.read_ucd(input_file);

      Point<dim> p(0.2, 0.2);

      static const SphericalManifold<dim> boundary(p);
      triangulation.set_all_manifold_ids_on_boundary(80, 80);
      triangulation.set_manifold(80, boundary);

      triangulation.refine_global(initial_global_refinement);

      std::ofstream out("grid.svg");
      GridOut grid_out;
      grid_out.write_svg(triangulation, out);

      dof_handler.distribute_dofs(fe);
      // DoFRenumbering::Cuthill_McKee(dof_handler);

      // We are dealing with 3 components for this
      // two-dimensional NSE problem:
      // Precisely, we use:
      // velocity in x and y:                0
      // scalar pressure field:              2
      std::vector<unsigned int> block_component(3, 0);
      block_component[dim] = 1;

      DoFRenumbering::component_wise(dof_handler, block_component);

      {
        constraints.clear();
        set_newton_bc();
      }
      constraints.close();

      // Two blocks: velocity and pressure
      std::vector<unsigned int> dofs_per_block(2);
      dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
      const unsigned int n_v = dofs_per_block[0],
                         n_p = dofs_per_block[1];

      std::cout << "\tNumber of active cells: " << triangulation.n_active_cells()
                << std::endl
                << "\tNumber of degrees of freedom: " << dof_handler.n_dofs()
                << " (" << n_v << '+' << n_p << ')'
                << std::endl
                << std::endl;

      system_matrix.clear();
      system_matrix_check.clear();
      {
        BlockDynamicSparsityPattern csp(2, 2);

        csp.block(0, 0).reinit(n_v, n_v);
        csp.block(0, 1).reinit(n_v, n_p);

        csp.block(1, 0).reinit(n_p, n_v);
        csp.block(1, 1).reinit(n_p, n_p);

        csp.collect_sizes();

        DoFTools::make_sparsity_pattern(dof_handler, csp, constraints, false);

        sparsity_pattern.copy_from(csp);
      }

      system_matrix.reinit(sparsity_pattern);
//      system_matrix_check.reinit(sparsity_pattern);



  //    std::ofstream outt("sparsity_pattern1.svg");
  //    sparsity_pattern.print_svg(outt);
  //    std::ofstream outtt("sparsity_pattern2.csv");
  //    sparsity_pattern.print(outtt);
      // resize the block vectors needed for the Newton solver
      for (auto vec : {&solution, &old_timestep_solution, &newton_update, &system_rhs})
      {
        vec->reinit(2);
        vec->block(0).reinit(n_v);
        vec->block(1).reinit(n_p);
        vec->collect_sizes();
      }
    }

  template <int dim>
    void NavierStokes<dim>::setup_system_ROM(int refinements)
    {
      const unsigned int initial_global_refinement = refinements;

      // In the following, we read a *.inp grid from a file.
      // The geometry information is based on the
      // 2D Schaefer/Turek 1996 benchmarks
      std::string grid_name;
      if (test_case == "2D-1" || test_case == "2D-2" || test_case == "2D-3")
        grid_name = "nsbench4_original.inp";
      AssertThrow(test_case == "2D-2", ExcMessage("Currently only the 2D-2 benchmark has been implemented."));

      GridIn<dim> grid_in;
      grid_in.attach_triangulation(triangulation);
      std::ifstream input_file(grid_name.c_str());
      Assert(dim == 2, ExcInternalError());
      grid_in.read_ucd(input_file);

      Point<dim> p(0.2, 0.2);

      static const SphericalManifold<dim> boundary(p);
      triangulation.set_all_manifold_ids_on_boundary(80, 80);
      triangulation.set_manifold(80, boundary);

      triangulation.refine_global(initial_global_refinement);

//      std::ofstream out("grid.svg");
//      GridOut grid_out;
//      grid_out.write_svg(triangulation, out);

      dof_handler.distribute_dofs(fe);
      // DoFRenumbering::Cuthill_McKee(dof_handler);

      // We are dealing with 3 components for this
      // two-dimensional NSE problem:
      // Precisely, we use:
      // velocity in x and y:                0
      // scalar pressure field:              2
      std::vector<unsigned int> block_component(3, 0);
      block_component[dim] = 1;

      DoFRenumbering::component_wise(dof_handler, block_component);

      {
        constraints.clear();
//        set_newton_bc();
      }
      constraints.close();

      // Two blocks: velocity and pressure
      std::vector<unsigned int> dofs_per_block(2);
      dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
      const unsigned int n_v = dofs_per_block[0],
                         n_p = dofs_per_block[1];

      dofs_velocity = n_v;
      dofs_pressure = n_p;

      system_matrix.clear();
      system_matrix_check.clear();
      {
        BlockDynamicSparsityPattern csp(2, 2);

        csp.block(0, 0).reinit(n_v, n_v);
        csp.block(0, 1).reinit(n_v, n_p);

        csp.block(1, 0).reinit(n_p, n_v);
        csp.block(1, 1).reinit(n_p, n_p);

        csp.collect_sizes();

        DoFTools::make_sparsity_pattern(dof_handler, csp, constraints, false);

        sparsity_pattern.copy_from(csp);
      }

  //    system_matrix.reinit(sparsity_pattern);
//      system_matrix_check.reinit(sparsity_pattern);
      mass_matrix_vp.reinit(sparsity_pattern);
      laplace_matrix_vp.reinit(sparsity_pattern);
      boundary_matrix_vp.reinit(sparsity_pattern);
      pressure_matrix_vp.reinit(sparsity_pattern);
      incompressibilty_matrix_vp.reinit(sparsity_pattern);
      first_convection_matrix_velocity_with_mean_vp.reinit(sparsity_pattern);
	  second_convection_matrix_velocity_with_mean_vp.reinit(sparsity_pattern);
      system_matrix.reinit(sparsity_pattern);


//      assemble_FOM_matrices_aff_decomp();

//      std::ofstream outt("pressure.txt");
//      	      std::cout << "start" << std::endl;
//      	    incompressibilty_matrix_vp.block(0,1).print_as_numpy_arrays(outt,40);
//      	      std::cout << "end: " << std::endl;
  //    std::ofstream outt("sparsity_pattern1.svg");
  //    sparsity_pattern.print_svg(outt);
  //    std::ofstream outtt("sparsity_pattern2.csv");
  //    sparsity_pattern.print(outtt);
      // resize the block vectors needed for the Newton solver
      for (auto vec : {&solution, &old_timestep_solution, &newton_update, &system_rhs})
      {
        vec->reinit(2);
        vec->block(0).reinit(n_v);
        vec->block(1).reinit(n_p);
        vec->collect_sizes();
      }

//      if (time == 0.0)
//            VectorTools::interpolate(dof_handler,
//                                     Functions::ZeroFunction<dim>(dim + 1),
//                                     old_timestep_solution);
//          else
//            old_timestep_solution = load_h5_block_vector(intial_solution_snapshot);
//          solution = old_timestep_solution;
//      assemble_system_matrix();
//
//      std::ofstream outtt("pressure_system.txt");
//            	      std::cout << "start" << std::endl;
//            	      system_matrix.block(0,1).print_as_numpy_arrays(outtt,40);
//            	      std::cout << "end: " << std::endl;
    }


  template <int dim>
  void NavierStokes<dim>::setup_system_only_velocity(int refinements)
  {
    const unsigned int initial_global_refinement = refinements;

    // In the following, we read a *.inp grid from a file.
    // The geometry information is based on the
    // 2D Schaefer/Turek 1996 benchmarks


    std::string grid_name;
    if (test_case == "2D-1" || test_case == "2D-2" || test_case == "2D-3")
      grid_name = "nsbench4_original.inp";
    AssertThrow(test_case == "2D-2", ExcMessage("Currently only the 2D-2 benchmark has been implemented."));

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation);
    std::ifstream input_file(grid_name.c_str());
    Assert(dim == 2, ExcInternalError());
    grid_in.read_ucd(input_file);

    Point<dim> p(0.2, 0.2);

    static const SphericalManifold<dim> boundary(p);
    triangulation.set_all_manifold_ids_on_boundary(80, 80);
    triangulation.set_manifold(80, boundary);

    triangulation.refine_global(initial_global_refinement);

    dof_handler_velocity.distribute_dofs(fe_velocity);
    // DoFRenumbering::Cuthill_McKee(dof_handler_velocity);

    DynamicSparsityPattern dsp(dof_handler_velocity.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_velocity, dsp);
    sparsity_pattern_velocity.copy_from(dsp);

    mass_matrix_velocity.reinit(sparsity_pattern_velocity);
    laplace_matrix_velocity.reinit(sparsity_pattern_velocity);
    laplace_matrix_velocity_with_transposed_trial_function.reinit(sparsity_pattern_velocity);
    boundary_matrix_velocity.reinit(sparsity_pattern_velocity);
    first_convection_matrix_velocity_with_mean.reinit(sparsity_pattern_velocity);
    second_convection_matrix_velocity_with_mean.reinit(sparsity_pattern_velocity);

    indicator_matrix_velocity.reinit(sparsity_pattern_velocity);

    mean_vector_contribution_rhs.reinit(dof_handler_velocity.n_dofs());

    MatrixCreator::create_mass_matrix(dof_handler_velocity,
                                      QGauss<dim>(fe_velocity.degree + 2),
                                      mass_matrix_velocity);

    MatrixCreator::create_laplace_matrix(dof_handler_velocity,
                                         QGauss<dim>(fe_velocity.degree + 2),
                                         laplace_matrix_velocity);

    assemble_laplace_with_transposed_trial_function();

    assemble_boundary_matrix();

    dofs_velocity = dof_handler_velocity.n_dofs();
//    assemble_indicator_fct();

//    gradient_matrix.reinit(sparsity_pattern_pressure);
//    assemble_gradient_matrix();
  }

  template <int dim>
  void NavierStokes<dim>::setup_system_only_pressure(int refinements)
  {
    const unsigned int initial_global_refinement = refinements;

    dof_handler_pressure.distribute_dofs(fe_pressure);

    /*
    // renumber the pressure DoFs the same way that they have been ordered in the FEM simulation
    IndexSet locally_relevant_dofs;

    // Ignore velocity
    std::vector<bool> component_mask(dim + 1, false);
    component_mask[dim] = true;

    std::vector<IndexSet> locally_owned_dofs = DoFTools::locally_owned_dofs_per_component(dof_handler, component_mask);
    std::cout << locally_owned_dofs.size() << std::endl;
    exit(1);
    */
    // DoFRenumbering::Cuthill_McKee(dof_handler_pressure);


    pressure_constraints.clear();
    // set homogeneous Dirichlet BC on the outflow boundary
    VectorTools::interpolate_boundary_values(dof_handler_pressure,
                                             1,
                                             ZeroFunction<dim>(1),
                                             pressure_constraints);
    pressure_constraints.close();

//    std::cout << "Number of pressure constraints: " << pressure_constraints.n_constraints() << std::endl;
    /*
    for (int i = 0; i < dof_handler_pressure.n_dofs(); i++)
      if (pressure_constraints.is_constrained(i))
        std::cout << i << ",";
    std::cout << std::endl;
    exit(99);
    */

    DynamicSparsityPattern dsp(dof_handler_pressure.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_pressure, dsp);
    sparsity_pattern_pressure.copy_from(dsp);

    mass_matrix_pressure.reinit(sparsity_pattern_pressure);
    laplace_matrix_pressure.reinit(sparsity_pattern_pressure);
    laplace_matrix_pressure_with_bc.reinit(sparsity_pattern_pressure);

    MatrixCreator::create_mass_matrix(dof_handler_pressure,
                                      QGauss<dim>(fe_pressure.degree + 1),
                                      mass_matrix_pressure);

    MatrixCreator::create_laplace_matrix(dof_handler_pressure,
                                         QGauss<dim>(fe_pressure.degree + 1),
                                         laplace_matrix_pressure);

    MatrixCreator::create_laplace_matrix(dof_handler_pressure,
                                         QGauss<dim>(fe_pressure.degree + 1),
                                         laplace_matrix_pressure_with_bc, // <-- !!!
                                         (const Function<dim> *const)nullptr,
                                         pressure_constraints);

    // for the pressure reconstruction invert the laplace matrix:
    laplace_matrix_pressure_inverse.factorize(laplace_matrix_pressure_with_bc);

  }

  // Here, we impose boundary conditions
  // for the whole system. The fluid inflow
  // is prescribed by a parabolic profile.
  // The pressure variable is not subjected to any
  // Dirichlet boundary conditions and is left free
  // in this method.
  template <int dim>
  void
  NavierStokes<dim>::set_initial_bc(const double time)
  {
    std::map<unsigned int, double> boundary_values;
    std::vector<bool> component_mask(dim + 1, true);
    // Ignore pressure
    component_mask[dim] = false;

    // parabolic inflow
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             BoundaryParabel<dim>(time),
                                             boundary_values,
                                             component_mask);

    // apply homogeneous Dirichlet BC to velocity
    for (int boundary_id : {2, 3, 80})
      VectorTools::interpolate_boundary_values(dof_handler,
                                               boundary_id,
                                               ZeroFunction<dim>(dim + 1),
                                               boundary_values,
                                               component_mask);

    // apply boundary_values to the solution
    for (typename std::map<unsigned int, double>::const_iterator
             i = boundary_values.begin();
         i != boundary_values.end();
         ++i)
      solution(i->first) = i->second;
  }

  // This function applies boundary conditions
  // to the Newton iteration steps. For all variables that
  // have Dirichlet conditions on some (or all) parts
  // of the outer boundary, we apply zero-Dirichlet
  // conditions, now.
  template <int dim>
  void NavierStokes<dim>::set_newton_bc()
  {
    std::vector<bool> component_mask(dim + 1, true);
    // Ignore pressure
    component_mask[dim] = false;

    // apply homogeneous Dirichlet BC to velocity
    for (int boundary_id : {0, 2, 3, 80})
      VectorTools::interpolate_boundary_values(dof_handler,
                                               boundary_id,
                                               ZeroFunction<dim>(dim + 1),
                                               constraints,
                                               component_mask);
  }

  // In this function, we assemble the Jacobian matrix
  // for the Newton iteration.
  //
  // To compensate the well-known problem in fluid
  // dynamics on the outflow boundary, we also
  // add some correction term on the outflow boundary.
  // This relation is known as `do-nothing' condition.
  // In the inner loops of the local_cell_matrix.
  //
  // Assembling of the inner most loop is treated with help of
  // the fe.system_to_component_index(j).first function from
  // the library.
  // Using this function makes the assembling process much faster
  // than running over all local degrees of freedom.
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

      // We compute in the following
      // one term on the outflow boundary.
      // This relation is well-know in the literature
      // as "do-nothing" condition (Heywood/Rannacher/Turek, 1996). Therefore, we only
      // ask for the corresponding color at the outflow
      // boundary that is 1 in our case.
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
      {
        if (cell->face(face)->at_boundary() &&
            (cell->face(face)->boundary_id() == 1))
        {

          fe_face_values.reinit(cell, face);

          fe_face_values.get_function_values(solution, old_solution_face_values);
          fe_face_values.get_function_gradients(solution, old_solution_face_grads);

          for (unsigned int q = 0; q < n_face_q_points; ++q)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
              phi_i_v[k] = fe_face_values[velocities].value(k, q);
              phi_i_grads_v[k] = fe_face_values[velocities].gradient(k, q);
            }

            Tensor<2, dim> grad_v;
            for (unsigned int l = 0; l < dim; l++)
              for (unsigned int m = 0; m < dim; m++)
                grad_v[l][m] = old_solution_face_grads[q][l][m];

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              Tensor<2, dim> grad_v_LinV;
              for (unsigned int l = 0; l < dim; l++)
                for (unsigned int m = 0; m < dim; m++)
                  grad_v_LinV[l][m] = phi_i_grads_v[i][l][m];

              const Tensor<2, dim> stress_fluid_3rd_term_LinAll = fluid_density * viscosity * transpose(phi_i_grads_v[i]);

              // Here, we multiply the symmetric part of fluid's stress tensor
              // with the normal direction.
              const Tensor<1, dim> neumann_value = (stress_fluid_3rd_term_LinAll * fe_face_values.normal_vector(q));

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                const unsigned int comp_j = fe.system_to_component_index(j).first;
                if (comp_j == 0 || comp_j == 1)
                {
                  local_matrix(j, i) -= 1.0 * (time_step * theta * neumann_value * phi_i_v[j]) * fe_face_values.JxW(q);
                }
                // end j
              }
              // end i
            }
            // end q_face_points
          }
          // end if-routine face integrals
        }
        // end face integrals do-nothing
      }

      // This is the same as discussed in step-22:
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(local_matrix, local_dof_indices,
                                             system_matrix);
      // end cell
    }
  }

  // In this function we assemble the semi-linear
  // of the right hand side of Newton's method (its residual).
  // The framework is in principal the same as for the
  // system matrix.
  template <int dim>
  void NavierStokes<dim>::assemble_system_rhs()
  {
    system_rhs = 0;

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

    Vector<double> local_rhs(dofs_per_cell);

    std::vector<unsigned int> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    std::vector<Vector<double>>
        old_solution_values(n_q_points, Vector<double>(dim + 1));

    std::vector<std::vector<Tensor<1, dim>>>
        old_solution_grads(n_q_points, std::vector<Tensor<1, dim>>(dim + 1));

    std::vector<Vector<double>>
        old_solution_face_values(n_face_q_points, Vector<double>(dim + 1));

    std::vector<std::vector<Tensor<1, dim>>>
        old_solution_face_grads(n_face_q_points, std::vector<Tensor<1, dim>>(dim + 1));

    std::vector<Vector<double>>
        old_timestep_solution_values(n_q_points, Vector<double>(dim + 1));

    std::vector<std::vector<Tensor<1, dim>>>
        old_timestep_solution_grads(n_q_points, std::vector<Tensor<1, dim>>(dim + 1));

    std::vector<Vector<double>>
        old_timestep_solution_face_values(n_face_q_points, Vector<double>(dim + 1));

    std::vector<std::vector<Tensor<1, dim>>>
        old_timestep_solution_face_grads(n_face_q_points, std::vector<Tensor<1, dim>>(dim + 1));

    typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();

    for (; cell != endc; ++cell)
    {
      fe_values.reinit(cell);
      local_rhs = 0;

      // old Newton iteration
      fe_values.get_function_values(solution, old_solution_values);
      fe_values.get_function_gradients(solution, old_solution_grads);

      // old timestep iteration
      fe_values.get_function_values(old_timestep_solution, old_timestep_solution_values);
      fe_values.get_function_gradients(old_timestep_solution, old_timestep_solution_grads);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        Tensor<2, dim> pI;
        pI.clear(); // reset all values to zero
        for (unsigned int l = 0; l < dim; l++)
          pI[l][l] = old_solution_values[q](dim);

        Tensor<1, dim> v;
        for (unsigned int l = 0; l < dim; l++)
          v[l] = old_solution_values[q](l);

        Tensor<2, dim> grad_v;
        for (unsigned int l = 0; l < dim; l++)
          for (unsigned int m = 0; m < dim; m++)
            grad_v[l][m] = old_solution_grads[q][l][m];

        // fluid stress tensor (without pressure)
        Tensor<2, dim> sigma_fluid;
        sigma_fluid = fluid_density * viscosity * (grad_v + transpose(grad_v));

        // Divergence of the fluid
        const double incompressiblity_fluid = trace(grad_v);

        // Convection term of the fluid
        Tensor<1, dim> convection_fluid = fluid_density * grad_v * v;

        // Computing tensors and values for OLD TIMESTEP:
        Tensor<1, dim> old_timestep_v;
        for (unsigned int l = 0; l < dim; l++)
          old_timestep_v[l] = old_timestep_solution_values[q](l);

        Tensor<2, dim> old_timestep_grad_v;
        for (unsigned int l = 0; l < dim; l++)
          for (unsigned int m = 0; m < dim; m++)
            old_timestep_grad_v[l][m] = old_timestep_solution_grads[q][l][m];

        // old fluid stress tensor (without pressure)
        Tensor<2, dim> old_timestep_sigma_fluid;
        old_timestep_sigma_fluid = fluid_density * viscosity * (old_timestep_grad_v + transpose(old_timestep_grad_v));

        // Convection term of the fluid
        Tensor<1, dim> old_timestep_convection_fluid = fluid_density * old_timestep_grad_v * old_timestep_v;

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const unsigned int comp_i = fe.system_to_component_index(i).first;
          if (comp_i == 0 || comp_i == 1)
          {
            const Tensor<1, dim> phi_i_v = fe_values[velocities].value(i, q);
            const Tensor<2, dim> phi_i_grads_v = fe_values[velocities].gradient(i, q);

            local_rhs(i) -= (fluid_density *
                                 (v - old_timestep_v) * phi_i_v +
                             time_step * theta * convection_fluid * phi_i_v +
                             time_step * (1.0 - theta) *
                                 old_timestep_convection_fluid * phi_i_v -
                             time_step * scalar_product(pI, phi_i_grads_v) +
                             time_step * theta * scalar_product(sigma_fluid, phi_i_grads_v) +
                             time_step * (1.0 - theta) *
                                 scalar_product(old_timestep_sigma_fluid, phi_i_grads_v)) *
                            fe_values.JxW(q);
          }
          else if (comp_i == 2)
          {
            const double phi_i_p = fe_values[pressure].value(i, q);
            local_rhs(i) -= (incompressiblity_fluid * phi_i_p) * fe_values.JxW(q);
          }
          // end i dofs
        }
        // close n_q_points
      }

      // As already discussed in the assembly method for the matrix,
      // we have to integrate some terms on the outflow boundary:
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
      {
        if (cell->face(face)->at_boundary() &&
            (cell->face(face)->boundary_id() == 1))
        {
          fe_face_values.reinit(cell, face);

          fe_face_values.get_function_values(solution, old_solution_face_values);
          fe_face_values.get_function_gradients(solution, old_solution_face_grads);

          fe_face_values.get_function_values(old_timestep_solution, old_timestep_solution_face_values);
          fe_face_values.get_function_gradients(old_timestep_solution, old_timestep_solution_face_grads);

          for (unsigned int q = 0; q < n_face_q_points; ++q)
          {
            Tensor<2, dim> grad_v;
            for (unsigned int l = 0; l < dim; l++)
              for (unsigned int m = 0; m < dim; m++)
                grad_v[l][m] = old_solution_face_grads[q][l][m];

            Tensor<2, dim> old_timestep_grad_v;
            for (unsigned int l = 0; l < dim; l++)
              for (unsigned int m = 0; m < dim; m++)
                old_timestep_grad_v[l][m] = old_timestep_solution_face_grads[q][l][m];

            // Neumann boundary integral
            Tensor<2, dim> stress_fluid_transposed_part;
            stress_fluid_transposed_part.clear();
            stress_fluid_transposed_part = fluid_density * viscosity * transpose(grad_v);

            const Tensor<1, dim> neumann_value = (stress_fluid_transposed_part * fe_face_values.normal_vector(q));

            Tensor<2, dim> old_timestep_stress_fluid_transposed_part;
            old_timestep_stress_fluid_transposed_part.clear();
            old_timestep_stress_fluid_transposed_part = fluid_density * viscosity * transpose(old_timestep_grad_v);

            const Tensor<1, dim> old_timestep_neumann_value = (old_timestep_stress_fluid_transposed_part * fe_face_values.normal_vector(q));

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const unsigned int comp_i = fe.system_to_component_index(i).first;
              if (comp_i == 0 || comp_i == 1)
              {
                local_rhs(i) += 1.0 * (time_step * theta * neumann_value * fe_face_values[velocities].value(i, q) + time_step * (1.0 - theta) * old_timestep_neumann_value * fe_face_values[velocities].value(i, q)) * fe_face_values.JxW(q);
              }
              // end i
            }
            // end face_n_q_points
          }
        }
      } // end face integrals do-nothing condition

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(local_rhs, local_dof_indices,
                                             system_rhs);

    } // end cell
  }

  template <int dim>
  void NavierStokes<dim>::solve()
  {
    Vector<double> sol, rhs;
    sol = newton_update;
    rhs = system_rhs;

    A_direct.vmult(sol, rhs);
    newton_update = sol;

    constraints.distribute(newton_update);
  }

  // This is the Newton iteration with simple linesearch backtracking
  // to solve the
  // non-linear system of equations. First, we declare some
  // standard parameters of the solution method. Addionally,
  // we also implement an easy line search algorithm.
  template <int dim>
  void NavierStokes<dim>::newton_iteration(const double time)

  {
    const unsigned int max_no_newton_steps = 20;

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
    set_initial_bc(time);
    assemble_system_rhs();

    double newton_residual = system_rhs.linfty_norm();
    double old_newton_residual = newton_residual;
    unsigned int newton_step = 1;

    if (newton_residual < newton_tol)
    {
      std::cout << '\t'
                << std::scientific
                << newton_residual
                << std::endl;
    }

    while (newton_residual > newton_tol &&
           newton_step < max_no_newton_steps)
    {
      old_newton_residual = newton_residual;

      assemble_system_rhs();
      newton_residual = system_rhs.linfty_norm();

      if (newton_residual < newton_tol)
      {
        std::cout << '\t'
                  << std::scientific
                  << newton_residual << std::endl;
        break;
      }

      if (newton_residual / old_newton_residual > nonlinear_rho)
      {
        assemble_system_matrix();
//        assemble_gradient_matrix();
        // Only factorize when matrix is re-built
        A_direct.factorize(system_matrix);
      }

      // Solve Ax = b
      solve();

      line_search_step = 0;
      for (;
           line_search_step < max_no_line_search_steps;
           ++line_search_step)
      {
        solution += newton_update;

        assemble_system_rhs();
        new_newton_residual = system_rhs.linfty_norm();

        if (new_newton_residual < newton_residual)
          break;
        else
          solution -= newton_update;

        newton_update *= line_search_damping;
      }

      std::cout << std::setprecision(5) << newton_step << '\t'
                << std::scientific << newton_residual << '\t'
                << std::scientific << newton_residual / old_newton_residual << '\t';
      if (newton_residual / old_newton_residual > nonlinear_rho)
        std::cout << "r" << '\t';
      else
        std::cout << " " << '\t';
      std::cout << line_search_step << std::endl;

      //<< '\t' << std::scientific << timer_newton.cpu_time ()

      // Updates
      newton_step++;
    }
  }

  template <int dim>
  void NavierStokes<dim>::output_results() const
  {
	if (time<POD_start_time){
		return;
	}
	int output_time_step = timestep_number- \
		(round(coarse_endtime/coarse_timestep)+ ((POD_start_time-coarse_endtime)!=0) +round((POD_start_time-coarse_endtime)/fine_timestep));
	std::string path = fem_path + "/mu="+ std::to_string(viscosity) +"/snapshots/";
	int time_counter = 0;

	struct dirent *entry = nullptr;
	  DIR *dp = nullptr;

	  dp = opendir(path.c_str());
	  if (dp != nullptr) {
	      while ((entry = readdir(dp)))
	      {
	        //std::cout << "entry: " << entry->d_name << std::endl;
	        if (!(strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0))
	        	time_counter++; // this is a valid file and not "." or ".."
	      }
	  }
	  closedir(dp);
	  output_time_step = time_counter;
//	  std::cout << time_counter << std::endl;
//	for (const auto & entry : std::filesystem::directory_iterator(path))
//	        std::cout << entry.path() << std::endl;

	std::cout << "ots: " << output_time_step << std::endl;
	std::vector<std::string> solution_names;
    solution_names.push_back("x_velo");
    solution_names.push_back("y_velo");
    solution_names.push_back("p_fluid");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(dim + 1, DataComponentInterpretation::component_is_scalar);
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches();
    data_out.set_flags(DataOutBase::VtkFlags(time-POD_start_time, output_time_step));
    std::cout << timestep_number << std::endl;
    std::cout << output_time_step << std::endl;
    // save VTK files
    const std::string filename = fem_path + "/mu="+ std::to_string(viscosity) +"/solution/solution-" + Utilities::int_to_string(output_time_step, 6) + ".vtk";
    std::ofstream output(filename);
    data_out.write_vtk(output);
    // save HDF5 files
    const std::string filename_h5 = fem_path + "/mu="+ std::to_string(viscosity) +"/snapshots/snapshot_" + Utilities::int_to_string(output_time_step, 6) + ".h5";
    //DataOutBase::DataOutFilter data_filter(DataOutBase::DataOutFilterFlags(/*filter_duplicate_vertices=*/true, /*xdmf_hdf5_output*/ true));
    //data_out.write_filtered_data(data_filter);
    //data_out.write_hdf5_parallel(data_filter, filename_h5, MPI_COMM_WORLD);
    save_h5_block_vector(filename_h5, solution,(time-POD_start_time));
    // DataOut<dim> data_out;

    // data_out.attach_dof_handler(dof_handler);
    // data_out.add_data_vector(solution, "U");

    // data_out.build_patches();

    // data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));

    // const std::string filename =
    //   "result/FEM/solution/solution-" + Utilities::int_to_string(timestep_number, 6) + ".vtk";
    // std::ofstream output(filename);
    // data_out.write_vtk(output);

    // DataOut<dim> data_out2;
    // data_out2.attach_dof_handler(dof_handler);
    // data_out2.add_data_vector(forcing_term, "U");
    // data_out2.build_patches();
    // data_out2.set_flags(DataOutBase::VtkFlags(time, timestep_number));

    // const std::string filename2 =
    //   "result/FEM/forcing_term/force_vector-" + Utilities::int_to_string(timestep_number, 6) + ".h5";
    // DataOutBase::DataOutFilter data_filter2(DataOutBase::DataOutFilterFlags(/*filter_duplicate_vertices=*/true, /*xdmf_hdf5_output*/true));
    // data_out2.write_filtered_data(data_filter2);
    // data_out2.write_hdf5_parallel(data_filter2, filename2, MPI_COMM_WORLD);
  }

  // Compute the pressure at a certain point.
  template <int dim>
  double NavierStokes<dim>::compute_pressure(Point<dim> p) const
  {
    Vector<double> tmp(dim + 1);
    VectorTools::point_value(dof_handler,
                             solution,
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
  void NavierStokes<dim>::compute_drag_lift_tensor()
  {

    const QGauss<dim - 1> face_quadrature_formula(3);
    FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                     update_values | update_gradients | update_normal_vectors |
                                         update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    std::vector<unsigned int> local_dof_indices(dofs_per_cell);
    std::vector<Vector<double>> face_solution_values(n_face_q_points,
                                                     Vector<double>(dim + 1));

    std::vector<std::vector<Tensor<1, dim>>>
        face_solution_grads(n_face_q_points, std::vector<Tensor<1, dim>>(dim + 1));

    Tensor<1, dim> drag_lift_value;

    typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();

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
          fe_face_values.get_function_values(solution, face_solution_values);
          fe_face_values.get_function_gradients(solution, face_solution_grads);

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

    // 2D-1: 500; 2D-2 and 2D-3: 20 (see Schaefer/Turek 1996)
    if (test_case == "2D-1")
      drag_lift_value *= 500.0;
    else if (test_case == "2D-2" || test_case == "2D-3")
      drag_lift_value *= 20.0;

    std::cout << "Face drag:   "
              << "   " << std::setprecision(16) << drag_lift_value[0] << std::endl;
    std::cout << "Face lift:   "
              << "   " << std::setprecision(16) << drag_lift_value[1] << std::endl;

    // save drag and lift values to text files
    std::ofstream drag_out;
    drag_out.open(fem_path + "mu="+ std::to_string(viscosity) +"/drag.txt", std::ios_base::app); // append instead of overwrite
    drag_out << time << "," << std::setprecision(16) << drag_lift_value[0] << std::endl;
    drag_out.close();

    std::ofstream lift_out;
    lift_out.open(fem_path + "mu="+ std::to_string(viscosity) +"/lift.txt", std::ios_base::app); // append instead of overwrite
    lift_out << time << "," << std::setprecision(16) << drag_lift_value[1]  << std::endl;
    lift_out.close();
  }

  // Here, we compute the four quantities of interest:
  // the drag, and the lift and a pressure difference
  template <int dim>
  void NavierStokes<dim>::compute_functional_values()
  {
    double p_front = compute_pressure(Point<dim>(0.15, 0.2)); // pressure - left  point on circle
    double p_back = compute_pressure(Point<dim>(0.25, 0.2));  // pressure - right point on circle

    double p_diff = p_front - p_back;

    // save pressure difference to text file
    std::ofstream p_out;
    p_out.open(fem_path + "/mu="+ std::to_string(viscosity) +"/pressure.txt", std::ios_base::app); // append instead of overwrite
    p_out << time << "," << std::setprecision(16) << p_diff << std::endl;
    p_out.close();

    std::cout << "------------------" << std::endl;
    std::cout << "Pressure difference:  " << "   " << std::setprecision(16) << p_diff << std::endl;
    //std::cout << "P-front: "  << "   " << std::setprecision(16) << p_front << std::endl;
    // std::cout << "P-back:  "  << "   " << std::setprecision(16) << p_back << std::endl;
    std::cout << "------------------" << std::endl;

    // Compute drag and lift via line integral
    compute_drag_lift_tensor();

    std::cout << "------------------" << std::endl;

    std::cout << std::endl;
  }

//  template <int dim>
//  void NavierStokes<dim>::set_viscosity(double viscosity)
//  {
//	  this->viscosity = viscosity;
//  }


  template <int dim>
  void NavierStokes<dim>::run(int refinements, bool output_files)
  {
     setup_system(refinements);
    Vector<double> tmp(solution.size());
    // forcing_term.reinit(solution.size());
    if (time == 0.0)
      VectorTools::interpolate(dof_handler,
                               Functions::ZeroFunction<dim>(dim + 1),
                               old_timestep_solution);
    else
      old_timestep_solution = load_h5_block_vector(intial_solution_snapshot);
    solution = old_timestep_solution;

    if (output_files)
    {
    	mkdir((fem_path + "mu="+ std::to_string(viscosity)).c_str() , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    	mkdir((fem_path + "mu="+ std::to_string(viscosity)+"/solution").c_str() , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    	mkdir((fem_path + "mu="+ std::to_string(viscosity)+"/snapshots").c_str() , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    	output_results();
    }
    std::ofstream  fw;
    fw.open((fem_path + "mu=" + std::to_string(viscosity) + "/exectution_time.txt"), std::ofstream::out);

    std::vector<std::vector<double>> computational_time_per_iteration;
    auto start_time_solve= std::chrono::high_resolution_clock::now();
    auto end_time_solve = std::chrono::high_resolution_clock::now();
    auto duration_solve = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_solve - start_time_solve);

    auto start_time = std::chrono::high_resolution_clock::now();
    while (time <= fine_endtime - 0.01 * time_step)
    {
      time += time_step;
      ++timestep_number;

      std::cout << "Time step " << timestep_number << " at t=" << std::setprecision(5) << time
                << std::endl;

      // Compute the next time step
      start_time_solve= std::chrono::high_resolution_clock::now();
      old_timestep_solution = solution;
      newton_iteration(time);
      end_time_solve = std::chrono::high_resolution_clock::now();
      duration_solve = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_solve - start_time_solve);
      computational_time_per_iteration.push_back({time, duration_solve.count()});

      compute_functional_values();

      if (output_files)
        output_results();

      // decrease time step size, after coarse_endtime has passed
      if (time_step == coarse_timestep && time >= coarse_endtime - 1e-7)
        time_step = fine_timestep;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  	fw 			<< std::floor(duration.count() / 1000) << "s " << duration.count() % 1000 << " ms" << std::endl;
    std::cout 	<< std::floor(duration.count() / 1000) << "s " << duration.count() % 1000 << " ms" << std::endl;
    fw.close();

    std::ofstream timer_print;
    timer_print.open(fem_path + "mu=" + std::to_string(viscosity) + "/time_counter.txt", std::ios_base::app); // append instead of overwrite
    for (int i = 0; i<computational_time_per_iteration.size(); i++)
    {
    	timer_print << computational_time_per_iteration[i][0] << ", "
    	        	<< computational_time_per_iteration[i][1] << std::endl;
    }
    timer_print.close();
  }

  template <int dim>
  void NavierStokes<dim>::run_ROM(int refinements, bool output_files)
  {
     setup_system_ROM(refinements);
    Vector<double> tmp(solution.size());
    // forcing_term.reinit(solution.size());
    if (time == 0.0)
      VectorTools::interpolate(dof_handler,
                               Functions::ZeroFunction<dim>(dim + 1),
                               old_timestep_solution);
    else
      old_timestep_solution = load_h5_block_vector(intial_solution_snapshot);
    solution = old_timestep_solution;

    assemble_gradient_matrix();
  }


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
  void NavierStokes<dim>::assemble_boundary_matrix()
  {
    boundary_matrix_velocity = 0;

    QGauss<dim - 1> face_quadrature_formula(fe_velocity.degree + 2);

    FEFaceValues<dim> fe_face_values(fe_velocity, face_quadrature_formula,
                                      update_values | update_quadrature_points |
                                          update_normal_vectors | update_gradients |
                                          update_JxW_values);

    const unsigned int dofs_per_cell = fe_velocity.n_dofs_per_cell();

    const unsigned int n_face_q_points = face_quadrature_formula.size();

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<unsigned int> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector velocities(0);

    // test function
    std::vector<Tensor<1, dim>> phi_i(dofs_per_cell);
    std::vector<Tensor<2, dim>> phi_i_grads(dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_velocity.begin_active(),
      endc = dof_handler_velocity.end();

    // Next, we run over all cells
    for (; cell != endc; ++cell)
    {
      local_matrix = 0;
      // if we are at the outflow boundary
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
      {
        if (cell->face(face)->at_boundary() &&
            (cell->face(face)->boundary_id() == 1))
        {
          fe_face_values.reinit(cell, face);

          for (unsigned int q = 0; q < n_face_q_points; ++q)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
              phi_i[k] = fe_face_values[velocities].value(k, q);
              phi_i_grads[k] = fe_face_values[velocities].gradient(k, q);
            }
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                local_matrix(j,i) += transpose(phi_i_grads[i]) * fe_face_values.normal_vector(q) * phi_i[j] * fe_face_values.JxW(q);
              } // end j
            } // end i
          } // end q-points
        } // end if at ouflow boundary
      } // end faces

      // This is the same as discussed in step-22:
      cell->get_dof_indices(local_dof_indices);
      no_constraints.distribute_local_to_global(local_matrix, local_dof_indices,
                                             boundary_matrix_velocity);
    }  // end cell
  }


//
//  template <int dim>
//  void NavierStokes<dim>::assemble_indicator_fct()
//  {
//	indicator_matrix_velocity = 0;
//	std::vector<bool> indicator;
//	std::vector<bool> component_mask(dim, true);
//	 // Ignore u_y
//	component_mask[dim-1] = false;
//	const std::set< types::boundary_id >  boundary_ids = {0};
////	DoFTools::extract_boundary_dofs(dof_handler_velocity),//, velocity_x_mask, indicator, 0);
//
//	for (types::global_dof_index i :
//			DoFTools::extract_boundary_dofs(dof_handler_velocity,component_mask,boundary_ids))
//	{
//		indicator_matrix_velocity.set(i,i,1);
//		std::cout << i << std::endl;
//	}


//	IndexSet ind = DoFTools::extract_boundary_dofs(dof_handler_velocity,component_mask,boundary_ids);
//	std::cout << ind << std::endl;
//  }


//  template <int dim>
//    void NavierStokes<dim>::assemble_

  template <int dim>
  void NavierStokes<dim>::assemble_gradient_matrix()
  {
	  system_matrix_check = 0;

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
	        for (unsigned int q = 0; q < n_q_points; ++q)
	        {

	          for (unsigned int k = 0; k < dofs_per_cell; ++k)
	          {
	            phi_i_v[k] = fe_values[velocities].value(k, q);
	            phi_i_grads_v[k] = fe_values[velocities].gradient(k, q);
	            phi_i_p[k] = fe_values[pressure].value(k, q);
	          }

	          // Outer loop for dofs
	          for (unsigned int i = 0; i < dofs_per_cell; ++i)
	          {
	            // Inner loop for dofs
	            for (unsigned int j = 0; j < dofs_per_cell; ++j)
	            {
	              // Fluid , NSE
	              const unsigned int comp_j = fe.system_to_component_index(j).first;
	              if (comp_j == 0 || comp_j == 1)
	              {
	                local_matrix(j, i) += (
	                						scalar_product(phi_i_grads_v[i],phi_i_grads_v[j])
//	                						phi_i_v[i] * phi_i_v[j]
	                						) *  fe_values.JxW(q);
	              }
	              else if (comp_j == 2)
	              {
	                // incompressibility condition
	                local_matrix(j, i) += (trace(phi_i_grads_v[i]) * phi_i_p[j]) * fe_values.JxW(q);
	              }
	              // end j dofs
	            }
	            // end i dofs
	          }
	          // end n_q_points
	        }

	        // We compute in the following
	        // one term on the outflow boundary.
	        // This relation is well-know in the literature
	        // as "do-nothing" condition (Heywood/Rannacher/Turek, 1996). Therefore, we only
	        // ask for the corresponding color at the outflow
	        // boundary that is 1 in our case

	        // This is the same as discussed in step-22:
	        cell->get_dof_indices(local_dof_indices);
	        constraints.distribute_local_to_global(local_matrix, local_dof_indices,
	                                               system_matrix_check);
	        // end cell
	      }


	      dof_handler_velocity.distribute_dofs(fe_velocity);
	      // DoFRenumbering::Cuthill_McKee(dof_handler_velocity);

	      DynamicSparsityPattern dsp(dof_handler_velocity.n_dofs());
	      DoFTools::make_sparsity_pattern(dof_handler_velocity, dsp);
	      sparsity_pattern_velocity.copy_from(dsp);

	      mass_matrix_velocity.reinit(sparsity_pattern_velocity);
	      laplace_matrix_velocity.reinit(sparsity_pattern_velocity);

	      MatrixCreator::create_mass_matrix(dof_handler_velocity,
	                                        QGauss<dim>(fe_velocity.degree + 1),
	                                        mass_matrix_velocity);
	      MatrixCreator::create_laplace_matrix(dof_handler_velocity,
	                                        QGauss<dim>(fe_velocity.degree + 2),
	                                        laplace_matrix_velocity);
	      std::cout << "dof: " << fe.degree+2 << ", dof_vel: " << fe_velocity.degree+1 << std::endl;
	      std::ofstream outt("M_block.txt");
	      std::cout << "start" << std::endl;
	      system_matrix_check.block(0,0).print_as_numpy_arrays(outt);
	      std::cout << "end: " << std::endl;

	      std::ofstream outtt("M_full.txt");
	      std::cout << "start" << std::endl;
//	      mass_matrix_velocity.print_as_numpy_arrays(outtt);
	      laplace_matrix_velocity.print_as_numpy_arrays(outtt);
	      std::cout << "end: " << std::endl;

	      std::ofstream outttt("B.txt");
	      std::cout << "start" << std::endl;
	      //	      mass_matrix_velocity.print_as_numpy_arrays(outtt);
	      system_matrix_check.block(0,1).print_as_numpy_arrays(outttt);
	      std::cout << "end: " << std::endl;

//	      SparseMatrix<double> MM;
//	      MM.reinit(system_matrix_check.block(0,1).m(),system_matrix_check.block(0,1).n());
//	      MM.copy_from(system_matrix_check.block(0,1));
//	      system_matrix_check.block(0,0).add(-1.0, mass_matrix_velocity);
//	      std::cout << "checker: " << system_matrix_check.block(0,0).l1_norm() << std::endl;
	      std::exit(0);
  }

  template <int dim>
  void NavierStokes<dim>::assemble_first_convection_term_with_mean(Vector<double> &mean_vector)
  {
    first_convection_matrix_velocity_with_mean = 0;

    QGauss<dim> quadrature_formula(fe_velocity.degree + 2);

    FEValues<dim> fe_values(fe_velocity,
                        quadrature_formula,
                        update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe_velocity.n_dofs_per_cell();

    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<unsigned int> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector velocities(0);

    // contribution of the mean vector
    std::vector<Vector<double>> mean_solution_values(n_q_points,
                                                    Vector<double>(dim));

    // test functions
    std::vector<Tensor<1, dim>> phi_i(dofs_per_cell);
    std::vector<Tensor<2, dim>> phi_i_grads(dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_velocity.begin_active(),
      endc = dof_handler_velocity.end();

    // Next, we run over all cells
    for (; cell != endc; ++cell)
    {
      fe_values.reinit(cell);
      local_matrix = 0;

      fe_values.get_function_values(mean_vector, mean_solution_values);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          phi_i[k] = fe_values[velocities].value(k, q);
          phi_i_grads[k] = fe_values[velocities].gradient(k, q);
        }

        Tensor<1, dim> v_mean;
        for (unsigned int l = 0; l < dim; l++)
          v_mean[l] = mean_solution_values[q][l];

        // Outer loop for dofs
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          // Inner loop for dofs
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            local_matrix(j, i) += (phi_i_grads[i] * v_mean) * phi_i[j] * fe_values.JxW(q);
          } // end j dofs
        } // end i dofs
      } // end n_q_points

      // This is the same as discussed in step-22:
      cell->get_dof_indices(local_dof_indices);
      no_constraints.distribute_local_to_global(local_matrix, local_dof_indices,
                                             first_convection_matrix_velocity_with_mean);
      // end cell
    }
  }

  template <int dim>
  void NavierStokes<dim>::assemble_second_convection_term_with_mean(Vector<double> &mean_vector)
  {
    second_convection_matrix_velocity_with_mean = 0;

    QGauss<dim> quadrature_formula(fe_velocity.degree + 2);

    FEValues<dim> fe_values(fe_velocity,
                        quadrature_formula,
                        update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe_velocity.n_dofs_per_cell();

    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<unsigned int> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector velocities(0);

    // contribution of the mean vector
    std::vector<std::vector<Tensor<1, dim>>> mean_solution_grads(n_q_points,
                                                                std::vector<Tensor<1, dim>>(dim));

    // test functions
    std::vector<Tensor<1, dim>> phi_i(dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_velocity.begin_active(),
      endc = dof_handler_velocity.end();

    // Next, we run over all cells
    for (; cell != endc; ++cell)
    {
      fe_values.reinit(cell);
      local_matrix = 0;

      fe_values.get_function_gradients(mean_vector, mean_solution_grads);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          phi_i[k] = fe_values[velocities].value(k, q);
        }

        Tensor<2, dim> grad_v_mean;
        for (unsigned int l = 0; l < dim; l++)
          for (unsigned int m = 0; m < dim; m++)
            grad_v_mean[l][m] = mean_solution_grads[q][l][m];

        // Outer loop for dofs
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          // Inner loop for dofs
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            local_matrix(j, i) += (grad_v_mean * phi_i[i]) * phi_i[j] * fe_values.JxW(q);
          } // end j dofs
        } // end i dofs
      } // end n_q_points

      // This is the same as discussed in step-22:
      cell->get_dof_indices(local_dof_indices);
      no_constraints.distribute_local_to_global(local_matrix, local_dof_indices,
                                             second_convection_matrix_velocity_with_mean);
      // end cell
    }
  }

  template <int dim>
  void NavierStokes<dim>::assemble_nonlinearity_tensor_velocity(std::vector<Vector<double>> &pod_vectors)
  {
    // reinit the tensor
    int r = pod_vectors.size();
    nonlinear_tensor_velocity.resize(r);
    for (int i = 0; i < r; i++)
      nonlinear_tensor_velocity[i].reinit(sparsity_pattern_velocity);

    QGauss<dim> quadrature_formula(fe_velocity.degree + 2);

    FEValues<dim> fe_values(fe_velocity,
                        quadrature_formula,
                        update_values | update_gradients | update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe_velocity.n_dofs_per_cell();

    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<unsigned int> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector velocities(0);

    for (int i = 0; i < r; i++)
    {
      // contribution of the i.th POD vector
      std::vector<Vector<double>> pod_vector_values(n_q_points,
                                                      Vector<double>(dim));

      // test functions
      std::vector<Tensor<1, dim>> phi_j(dofs_per_cell);
      std::vector<Tensor<2, dim>> phi_j_grads(dofs_per_cell);

      typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler_velocity.begin_active(),
        endc = dof_handler_velocity.end();

      // Next, we run over all cells
      for (; cell != endc; ++cell)
      {
        fe_values.reinit(cell);
        local_matrix = 0;

        fe_values.get_function_values(pod_vectors[i], pod_vector_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
          for (unsigned int l = 0; l < dofs_per_cell; ++l)
          {
            phi_j[l] = fe_values[velocities].value(l, q);
            phi_j_grads[l] = fe_values[velocities].gradient(l, q);
          }

          Tensor<1, dim> psi;
          for (unsigned int l = 0; l < dim; l++)
            psi[l] = pod_vector_values[q][l];

          // Outer loop for dofs
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            // Inner loop for dofs
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
              local_matrix(j, k) += (phi_j_grads[k] *  phi_j[j]) * psi * fe_values.JxW(q);
            } // end k dofs
          } // end j dofs
        } // end n_q_points

        // This is the same as discussed in step-22:
        cell->get_dof_indices(local_dof_indices);
        no_constraints.distribute_local_to_global(local_matrix, local_dof_indices,
                                               nonlinear_tensor_velocity[i]);
        // end cell
      }
    } // end iteration over POD vectors

  }

  template <int dim>
  void NavierStokes<dim>::assemble_mean_vector_contribution_rhs(Vector<double> &mean_vector, double rom_time_step)
  {
    mean_vector_contribution_rhs = 0;

    QGauss<dim> quadrature_formula(fe_velocity.degree + 2);
    QGauss<dim - 1> face_quadrature_formula(fe_velocity.degree + 2);

    FEValues<dim> fe_values(fe_velocity, quadrature_formula,
                            update_values |
                                update_quadrature_points |
                                update_JxW_values |
                                update_gradients);

    FEFaceValues<dim> fe_face_values(fe_velocity, face_quadrature_formula,
                                     update_values | update_quadrature_points |
                                         update_normal_vectors | update_gradients |
                                         update_JxW_values);

    const unsigned int dofs_per_cell = fe_velocity.dofs_per_cell;

    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    Vector<double> local_rhs(dofs_per_cell);

    std::vector<unsigned int> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector velocities(0);

    // contribution of the mean vector
    std::vector<Vector<double>> mean_solution_values(n_q_points,
                                                    Vector<double>(dim));

    std::vector<std::vector<Tensor<1, dim>>> mean_solution_grads(n_q_points,
                                                                std::vector<Tensor<1, dim>>(dim));

    std::vector<std::vector<Tensor<1, dim>>>
        mean_solution_face_grads(n_face_q_points, std::vector<Tensor<1, dim>>(dim));

    typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler_velocity.begin_active(),
        endc = dof_handler_velocity.end();

    for (; cell != endc; ++cell)
    {
      fe_values.reinit(cell);
      local_rhs = 0;

      fe_values.get_function_values(mean_vector, mean_solution_values);
      fe_values.get_function_gradients(mean_vector, mean_solution_grads);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        Tensor<1, dim> v_mean;
        for (unsigned int l = 0; l < dim; l++)
          v_mean[l] = mean_solution_values[q](l);

        Tensor<2, dim> grad_v_mean;
        for (unsigned int l = 0; l < dim; l++)
          for (unsigned int m = 0; m < dim; m++)
            grad_v_mean[l][m] = mean_solution_grads[q][l][m];

        // fluid stress tensor (without pressure)
        Tensor<2, dim> sigma_fluid;
        sigma_fluid = fluid_density * viscosity * (grad_v_mean + transpose(grad_v_mean));

        // Convection term of the fluid
        Tensor<1, dim> convection_fluid = grad_v_mean * v_mean;

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const Tensor<1, dim> phi_i = fe_values[velocities].value(i, q);
          const Tensor<2, dim> phi_i_grads = fe_values[velocities].gradient(i, q);

          local_rhs(i) -= (
                            rom_time_step * scalar_product(sigma_fluid, phi_i_grads) +
                            rom_time_step * fluid_density * convection_fluid * phi_i
                          ) * fe_values.JxW(q);
        } // end i dofs
      } // end n_q_points

      // integrate some terms on the outflow boundary:
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
      {
        if (cell->face(face)->at_boundary() &&
            (cell->face(face)->boundary_id() == 1))
        {
          fe_face_values.reinit(cell, face);

          fe_face_values.get_function_gradients(mean_vector, mean_solution_face_grads);

          for (unsigned int q = 0; q < n_face_q_points; ++q)
          {
            Tensor<2, dim> grad_v_mean;
            for (unsigned int l = 0; l < dim; l++)
              for (unsigned int m = 0; m < dim; m++)
                grad_v_mean[l][m] = mean_solution_face_grads[q][l][m];

            // Neumann boundary integral
            Tensor<2, dim> stress_fluid_transposed_part;
            stress_fluid_transposed_part.clear();
            stress_fluid_transposed_part = fluid_density * viscosity * transpose(grad_v_mean);

            const Tensor<1, dim> neumann_value = (stress_fluid_transposed_part * fe_face_values.normal_vector(q));

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              local_rhs(i) += rom_time_step * neumann_value * fe_face_values[velocities].value(i, q) * fe_face_values.JxW(q);
          } // end face_n_q_points
        }
      } // end face integrals do-nothing condition

      cell->get_dof_indices(local_dof_indices);
      no_constraints.distribute_local_to_global(local_rhs, local_dof_indices,
                                             mean_vector_contribution_rhs);
    } // end cell
  }


  template <int dim>
  void NavierStokes<dim>::assemble_mean_vector_contribution_rhs_greedy(Vector<double> &mean_vector, std::vector<Vector<double>> &mean_vector_contribution_rhs_vector)
  {
//    mean_vector_contribution_rhs = 0;

    mean_vector_contribution_rhs_vector.resize(3);
    mean_vector_contribution_rhs_vector[0].reinit(dof_handler_velocity.n_dofs());// = mean_vector_contribution_rhs;
    mean_vector_contribution_rhs_vector[1].reinit(dof_handler_velocity.n_dofs());// = mean_vector_contribution_rhs;
    mean_vector_contribution_rhs_vector[2].reinit(dof_handler_velocity.n_dofs());// = mean_vector_contribution_rhs;

    QGauss<dim> quadrature_formula(fe_velocity.degree + 2);
    QGauss<dim - 1> face_quadrature_formula(fe_velocity.degree + 2);

    FEValues<dim> fe_values(fe_velocity, quadrature_formula,
                            update_values |
                                update_quadrature_points |
                                update_JxW_values |
                                update_gradients);

    FEFaceValues<dim> fe_face_values(fe_velocity, face_quadrature_formula,
                                     update_values | update_quadrature_points |
                                         update_normal_vectors | update_gradients |
                                         update_JxW_values);

    const unsigned int dofs_per_cell = fe_velocity.dofs_per_cell;

    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

//    Vector<double> local_rhs(dofs_per_cell);

    std::vector<Vector<double>> local_rhs_vector(3);
    local_rhs_vector[0].reinit(dofs_per_cell);
    local_rhs_vector[1].reinit(dofs_per_cell);
    local_rhs_vector[2].reinit(dofs_per_cell);

    std::vector<unsigned int> local_dof_indices(dofs_per_cell);

    const FEValuesExtractors::Vector velocities(0);

    // contribution of the mean vector
    std::vector<Vector<double>> mean_solution_values(n_q_points,
                                                    Vector<double>(dim));

    std::vector<std::vector<Tensor<1, dim>>> mean_solution_grads(n_q_points,
                                                                std::vector<Tensor<1, dim>>(dim));

    std::vector<std::vector<Tensor<1, dim>>>
        mean_solution_face_grads(n_face_q_points, std::vector<Tensor<1, dim>>(dim));

    typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler_velocity.begin_active(),
        endc = dof_handler_velocity.end();

    for (; cell != endc; ++cell)
    {
      fe_values.reinit(cell);
//      local_rhs = 0;
      local_rhs_vector[0] = (0);
      local_rhs_vector[1] = (0);
      local_rhs_vector[2] = (0);


      fe_values.get_function_values(mean_vector, mean_solution_values);
      fe_values.get_function_gradients(mean_vector, mean_solution_grads);

      local_rhs_vector[0] = 0.;
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        Tensor<1, dim> v_mean;
        for (unsigned int l = 0; l < dim; l++)
          v_mean[l] = mean_solution_values[q](l);

        Tensor<2, dim> grad_v_mean;
        for (unsigned int l = 0; l < dim; l++)
          for (unsigned int m = 0; m < dim; m++)
            grad_v_mean[l][m] = mean_solution_grads[q][l][m];

        // fluid stress tensor (without pressure)
        Tensor<2, dim> sigma_fluid;
        sigma_fluid = (grad_v_mean + transpose(grad_v_mean));

        // Convection term of the fluid
        Tensor<1, dim> convection_fluid = grad_v_mean * v_mean;

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const Tensor<1, dim> phi_i = fe_values[velocities].value(i, q);
          const Tensor<2, dim> phi_i_grads = fe_values[velocities].gradient(i, q);

          local_rhs_vector[0](i) -= (scalar_product(sigma_fluid, phi_i_grads)
								 )* fe_values.JxW(q);
          local_rhs_vector[1](i) -= (convection_fluid * phi_i
        		  	  	  	  	 )* fe_values.JxW(q);

        } // end i dofs
      } // end n_q_points


      // integrate some terms on the outflow boundary:
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
      {
        if (cell->face(face)->at_boundary() &&
            (cell->face(face)->boundary_id() == 1))
        {
          fe_face_values.reinit(cell, face);

          fe_face_values.get_function_gradients(mean_vector, mean_solution_face_grads);

          for (unsigned int q = 0; q < n_face_q_points; ++q)
          {
            Tensor<2, dim> grad_v_mean;
            for (unsigned int l = 0; l < dim; l++)
              for (unsigned int m = 0; m < dim; m++)
                grad_v_mean[l][m] = mean_solution_face_grads[q][l][m];

            // Neumann boundary integral
            Tensor<2, dim> stress_fluid_transposed_part;
            stress_fluid_transposed_part.clear();
//            stress_fluid_transposed_part = fluid_density * viscosity * transpose(grad_v_mean);
            stress_fluid_transposed_part = transpose(grad_v_mean);

            const Tensor<1, dim> neumann_value = (stress_fluid_transposed_part * fe_face_values.normal_vector(q));

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              local_rhs_vector[2](i) += neumann_value * fe_face_values[velocities].value(i, q) * fe_face_values.JxW(q);
            }
          } // end face_n_q_points
        }
      } // end face integrals do-nothing condition

      cell->get_dof_indices(local_dof_indices);
//      no_constraints.distribute_local_to_global(local_rhs, local_dof_indices,
//                                                   mean_vector_contribution_rhs);
      no_constraints.distribute_local_to_global(local_rhs_vector[0], local_dof_indices,
                                                   mean_vector_contribution_rhs_vector[0]);
      no_constraints.distribute_local_to_global(local_rhs_vector[1], local_dof_indices,
    		  	  	  	  	  	  	  	  	  	   mean_vector_contribution_rhs_vector[1]);
      no_constraints.distribute_local_to_global(local_rhs_vector[2], local_dof_indices,
    		  	  	  	  	  	  	  	  	  	   mean_vector_contribution_rhs_vector[2]);
    } // end cell
  }

  template <int dim>
    void NavierStokes<dim>::assemble_lifting_contribution_rhs_greedy_vp(BlockVector<double> &lifting_vector, std::vector<BlockVector<double>> &lifting_vector_contribution_rhs_vector)
    {

      // [0]: A or L+L^T
      // [1]: C or convection
      // [2]: D or B (boundary)
      // [3]: M
      // [4]: B^T or incompress.

      lifting_vector_contribution_rhs_vector.resize(5);
      for (int i = 0; i<5; i++){
    	  lifting_vector_contribution_rhs_vector[i].reinit(2);
    	  lifting_vector_contribution_rhs_vector[i].block(0).reinit(mass_matrix_vp.block(0,0).m());
    	  lifting_vector_contribution_rhs_vector[i].block(1).reinit(mass_matrix_vp.block(1,0).m());
    	  lifting_vector_contribution_rhs_vector[i].collect_sizes();
      }

//      std::cout << lifting_vector_contribution_rhs_vector[0].block(0).size() << std::endl;

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

  //    Vector<double> local_rhs(dofs_per_cell);

      std::vector<Vector<double>> local_rhs_vector(5);
      local_rhs_vector[0].reinit(dofs_per_cell);
      local_rhs_vector[1].reinit(dofs_per_cell);
      local_rhs_vector[2].reinit(dofs_per_cell);
      local_rhs_vector[3].reinit(dofs_per_cell);
      local_rhs_vector[4].reinit(dofs_per_cell);

      std::vector<unsigned int> local_dof_indices(dofs_per_cell);

      const FEValuesExtractors::Vector velocities(0);
      const FEValuesExtractors::Scalar pressure(dim);

      // contribution of the mean vector
      std::vector<Vector<double>>
	  	  lifting_solution_values(n_q_points, Vector<double>(dim+1));

      std::vector<std::vector<Tensor<1, dim>>>
	  	  lifting_solution_grads(n_q_points,std::vector<Tensor<1, dim>>(dim+1));

      std::vector<std::vector<Tensor<1, dim>>>
          lifting_solution_face_grads(n_face_q_points, std::vector<Tensor<1, dim>>(dim+1));

      typename DoFHandler<dim>::active_cell_iterator
          cell = dof_handler.begin_active(),
          endc = dof_handler.end();

      for (; cell != endc; ++cell)
      {
        fe_values.reinit(cell);
  //      local_rhs = 0;
        local_rhs_vector[0] = (0);
        local_rhs_vector[1] = (0);
        local_rhs_vector[2] = (0);
        local_rhs_vector[3] = (0);
        local_rhs_vector[4] = (0);

        fe_values.get_function_values(lifting_vector, lifting_solution_values);
        fe_values.get_function_gradients(lifting_vector, lifting_solution_grads);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
          Tensor<1, dim> v_lifting;
          for (unsigned int l = 0; l < dim; l++)
            v_lifting[l] = lifting_solution_values[q](l);

          Tensor<2, dim> grad_v_lifting;
          for (unsigned int l = 0; l < dim; l++)
            for (unsigned int m = 0; m < dim; m++)
              grad_v_lifting[l][m] = lifting_solution_grads[q][l][m];

          // fluid stress tensor (without pressure)
          Tensor<2, dim> sigma_fluid;
          sigma_fluid = (grad_v_lifting + transpose(grad_v_lifting));

          const double incompressiblity_fluid = trace(grad_v_lifting);

          // Convection term of the fluid
          Tensor<1, dim> convection_fluid = grad_v_lifting * v_lifting;

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
        	  const unsigned int comp_i = fe.system_to_component_index(i).first;
        	  if (comp_i == 0 || comp_i == 1)
        	  {
        		  const Tensor<1, dim> phi_i = fe_values[velocities].value(i, q);
        		  const Tensor<2, dim> phi_i_grads = fe_values[velocities].gradient(i, q);
        		  // [0]: A or L+L^T
        		  local_rhs_vector[0](i) += (scalar_product(sigma_fluid, phi_i_grads)
  								 	 	 	 )* fe_values.JxW(q);
        		  // [1]: C or convection
        		  local_rhs_vector[1](i) += (convection_fluid * phi_i
          		  	  	  	  	 	 	 	 )* fe_values.JxW(q);
        		  // [3]: M
        		  local_rhs_vector[3](i) += (v_lifting *phi_i
        				  	  	  	  	  	 )* fe_values.JxW(q);
        	  }
        	  else if (comp_i == 2)
        	  {
        		  const double phi_i_p = fe_values[pressure].value(i, q);
        		  // [4]: B^T or incompress.
        		  local_rhs_vector[4](i) += (incompressiblity_fluid * phi_i_p) * fe_values.JxW(q);
        	  }
          } // end i dofs
        } // end n_q_points


        // integrate some terms on the outflow boundary:
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
        {
          if (cell->face(face)->at_boundary() &&
              (cell->face(face)->boundary_id() == 1))
          {
            fe_face_values.reinit(cell, face);

            fe_face_values.get_function_gradients(lifting_vector, lifting_solution_face_grads);

            for (unsigned int q = 0; q < n_face_q_points; ++q)
            {
              Tensor<2, dim> grad_v_lifting;
              for (unsigned int l = 0; l < dim; l++)
                for (unsigned int m = 0; m < dim; m++)
                  grad_v_lifting[l][m] = lifting_solution_face_grads[q][l][m];

              // Neumann boundary integral
              Tensor<2, dim> stress_fluid_transposed_part;
              stress_fluid_transposed_part.clear();
  //            stress_fluid_transposed_part = fluid_density * viscosity * transpose(grad_v_mean);
              stress_fluid_transposed_part = transpose(grad_v_lifting);

              const Tensor<1, dim> neumann_value = (stress_fluid_transposed_part * fe_face_values.normal_vector(q));

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
            	  const unsigned int comp_i = fe.system_to_component_index(i).first;
            	  if (comp_i == 0 || comp_i == 1)
            	  {
            		  // [2]: D or B (boundary)
            		  local_rhs_vector[2](i) += neumann_value * fe_face_values[velocities].value(i, q) * fe_face_values.JxW(q);
            	  }
              }
            } // end face_n_q_points
          }
        } // end face integrals do-nothing condition

        cell->get_dof_indices(local_dof_indices);
  //      no_constraints.distribute_local_to_global(local_rhs, local_dof_indices,
  //                                                   mean_vector_contribution_rhs);

        for (int i = 0; i<5; i++)
        {
        	no_constraints.distribute_local_to_global(local_rhs_vector[i], local_dof_indices,
        	        		lifting_vector_contribution_rhs_vector[i]);
        }
      } // end cell
    }


  template <int dim>
    void NavierStokes<dim>::assemble_FOM_matrices_aff_decomp()
    {
      mass_matrix_vp = 0;
      laplace_matrix_vp = 0;
      boundary_matrix_vp = 0;
      pressure_matrix_vp = 0;
      incompressibilty_matrix_vp = 0;

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

      FullMatrix<double> local_matrix_mass(dofs_per_cell, dofs_per_cell);
      FullMatrix<double> local_matrix_laplace(dofs_per_cell, dofs_per_cell);
      FullMatrix<double> local_matrix_boundary(dofs_per_cell, dofs_per_cell);
      FullMatrix<double> local_matrix_pressure(dofs_per_cell, dofs_per_cell);
      FullMatrix<double> local_matrix_incompressibility(dofs_per_cell, dofs_per_cell);

      std::vector<unsigned int> local_dof_indices(dofs_per_cell);

      // Now, we are going to use the
      // FEValuesExtractors to determine
      // the principle variables
      const FEValuesExtractors::Vector velocities(0);
      const FEValuesExtractors::Scalar pressure(dim);

      // We declare Vectors and Tensors for
      // the solutions at the previous Newton iteration:
//      std::vector<Vector<double>> old_solution_values(n_q_points,
//                                                      Vector<double>(dim + 1));
//
//      std::vector<std::vector<Tensor<1, dim>>> old_solution_grads(n_q_points,
//                                                                  std::vector<Tensor<1, dim>>(dim + 1));
//
//      std::vector<Vector<double>> old_solution_face_values(n_face_q_points,
//                                                           Vector<double>(dim + 1));
//
//      std::vector<std::vector<Tensor<1, dim>>> old_solution_face_grads(n_face_q_points,
//                                                                       std::vector<Tensor<1, dim>>(dim + 1));

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
        local_matrix_mass = 0;
        local_matrix_laplace = 0;
        local_matrix_boundary = 0;
        local_matrix_pressure = 0;
        local_matrix_incompressibility = 0;

        // Old Newton iteration values
//        fe_values.get_function_values(solution, old_solution_values);
//        fe_values.get_function_gradients(solution, old_solution_grads);

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
//          Tensor<1, dim> v;
//          for (unsigned int l = 0; l < dim; l++)
//            v[l] = old_solution_values[q][l];
//
//          Tensor<2, dim> grad_v;
//          for (unsigned int l = 0; l < dim; l++)
//            for (unsigned int m = 0; m < dim; m++)
//              grad_v[l][m] = old_solution_grads[q][l][m];

          // Outer loop for dofs
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            Tensor<2, dim> pI_LinP;
            pI_LinP.clear(); // reset all values to zero
            for (unsigned int l = 0; l < dim; l++)
              pI_LinP[l][l] = phi_i_p[i];

//            Tensor<2, dim> grad_v_LinV;
//            for (unsigned int l = 0; l < dim; l++)
//              for (unsigned int m = 0; m < dim; m++)
//                grad_v_LinV[l][m] = phi_i_grads_v[i][l][m];

            // stress tensor for incompressible Newtonian fluid
            // const Tensor<2, dim> stress_fluid_LinAll = -pI_LinP + fluid_density * viscosity * (phi_i_grads_v[i] + transpose(phi_i_grads_v[i]));
            const Tensor<2, dim> stress_fluid_LinAll_1st_term = pI_LinP;
            const Tensor<2, dim> stress_fluid_LinAll_2nd_term = (phi_i_grads_v[i] + transpose(phi_i_grads_v[i]));

            // derivative of convection term
//            const Tensor<1, dim> convection_fluid_LinAll = fluid_density * (phi_i_grads_v[i] * v + grad_v * phi_i_v[i]);

            const double incompressibility_LinAll = trace(phi_i_grads_v[i]);

            // Inner loop for dofs
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              // Fluid , NSE
              const unsigned int comp_j = fe.system_to_component_index(j).first;
              if (comp_j == 0 || comp_j == 1)
              {
                local_matrix_mass(j, i) 	+= (phi_i_v[i] * phi_i_v[j]) * fe_values.JxW(q);
                local_matrix_laplace(j, i) 	+= (scalar_product(stress_fluid_LinAll_2nd_term, phi_i_grads_v[j])) * fe_values.JxW(q);
                local_matrix_pressure(j, i) += (scalar_product(stress_fluid_LinAll_1st_term, phi_i_grads_v[j])) * fe_values.JxW(q);
              }
              else if (comp_j == 2)
              {
                // incompressibility condition
                local_matrix_incompressibility(j , i) += (incompressibility_LinAll * phi_i_p[j]) * fe_values.JxW(q);
//                std:: cout <<local_matrix_incompressibility(j,i) << ", " << incompressibility_LinAll << ", " <<  phi_i_p[j]<< ", " << fe_values.JxW(q) << std::endl;
              }
              // end j dofs
            }
            // end i dofs
          }
          // end n_q_points
        }

        // We compute in the following
        // one term on the outflow boundary.
        // This relation is well-know in the literature
        // as "do-nothing" condition (Heywood/Rannacher/Turek, 1996). Therefore, we only
        // ask for the corresponding color at the outflow
        // boundary that is 1 in our case.
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
        {
          if (cell->face(face)->at_boundary() &&
              (cell->face(face)->boundary_id() == 1))
          {

            fe_face_values.reinit(cell, face);

//            fe_face_values.get_function_values(solution, old_solution_face_values);
//            fe_face_values.get_function_gradients(solution, old_solution_face_grads);

            for (unsigned int q = 0; q < n_face_q_points; ++q)
            {
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                phi_i_v[k] = fe_face_values[velocities].value(k, q);
                phi_i_grads_v[k] = fe_face_values[velocities].gradient(k, q);
              }

//              Tensor<2, dim> grad_v;
//              for (unsigned int l = 0; l < dim; l++)
//                for (unsigned int m = 0; m < dim; m++)
//                  grad_v[l][m] = old_solution_face_grads[q][l][m];
//
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
//                Tensor<2, dim> grad_v_LinV;
//                for (unsigned int l = 0; l < dim; l++)
//                  for (unsigned int m = 0; m < dim; m++)
//                    grad_v_LinV[l][m] = phi_i_grads_v[i][l][m];
//
//                const Tensor<2, dim> stress_fluid_3rd_term_LinAll = fluid_density * viscosity * transpose(phi_i_grads_v[i]);
//
//                // Here, we multiply the symmetric part of fluid's stress tensor
//                // with the normal direction.
//                const Tensor<1, dim> neumann_value = (stress_fluid_3rd_term_LinAll * fe_face_values.normal_vector(q));

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const unsigned int comp_j = fe.system_to_component_index(j).first;
                  if (comp_j == 0 || comp_j == 1)
                  {
//                    local_matrix(j, i) -= 1.0 * (time_step * theta * neumann_value * phi_i_v[j]) * fe_face_values.JxW(q);
                	  local_matrix_boundary(j, i) += transpose(phi_i_grads_v[i]) * fe_face_values.normal_vector(q) * phi_i_v[j] * fe_face_values.JxW(q);
                  }
                  // end j
                }
                // end i
              }
              // end q_face_points
            }
            // end if-routine face integrals
          }
          // end face integrals do-nothing
        }

        // This is the same as discussed in step-22:

//        for(int iii = 0; iii < dofs_per_cell; iii++)
//        				{
//        					for (int jjj = 0; jjj < dofs_per_cell; jjj++)
//        					{
//        						std::cout << local_matrix_mass(iii,jjj) << " ";
//        					}
//        					std::cout << std::endl;
//        				}
//        std::cout << std::endl;
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(local_matrix_mass, local_dof_indices,
        				mass_matrix_vp);
        constraints.distribute_local_to_global(local_matrix_laplace, local_dof_indices,
        				laplace_matrix_vp);
        constraints.distribute_local_to_global(local_matrix_boundary, local_dof_indices,
        				boundary_matrix_vp);
        constraints.distribute_local_to_global(local_matrix_pressure, local_dof_indices,
        				pressure_matrix_vp);
        constraints.distribute_local_to_global(local_matrix_incompressibility, local_dof_indices,
        				incompressibilty_matrix_vp);
        // end cell
      }
    }

  template <int dim>
      void NavierStokes<dim>::assemble_convection_term_with_lifting(BlockVector<double> &mean_vector)
      {
  	  first_convection_matrix_velocity_with_mean_vp = 0;
  	  second_convection_matrix_velocity_with_mean_vp = 0;

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

        FullMatrix<double> local_first_convection_matrix_velocity_with_mean_vp(dofs_per_cell, dofs_per_cell);
        FullMatrix<double> local_second_convection_matrix_velocity_with_mean_vp(dofs_per_cell, dofs_per_cell);

        std::vector<unsigned int> local_dof_indices(dofs_per_cell);

        // Now, we are going to use the
        // FEValuesExtractors to determine
        // the principle variables
        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);

        // contribution of the lifting vector
        std::vector<Vector<double>>
  	  		mean_solution_values(n_q_points, Vector<double>(dim+1));
        std::vector<std::vector<Tensor<1, dim>>>
  	  		mean_solution_grads(n_q_points, std::vector<Tensor<1, dim>>(dim+1));
        // We declare Vectors and Tensors for
        // the solutions at the previous Newton iteration:
  //      std::vector<Vector<double>> old_solution_values(n_q_points,
  //                                                      Vector<double>(dim + 1));
  //
  //      std::vector<std::vector<Tensor<1, dim>>> old_solution_grads(n_q_points,
  //                                                                  std::vector<Tensor<1, dim>>(dim + 1));
  //
  //      std::vector<Vector<double>> old_solution_face_values(n_face_q_points,
  //                                                           Vector<double>(dim + 1));
  //
  //      std::vector<std::vector<Tensor<1, dim>>> old_solution_face_grads(n_face_q_points,
  //                                                                       std::vector<Tensor<1, dim>>(dim + 1));

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
          local_first_convection_matrix_velocity_with_mean_vp = 0;
          local_second_convection_matrix_velocity_with_mean_vp = 0;

          fe_values.get_function_values(mean_vector, mean_solution_values);
          fe_values.get_function_gradients(mean_vector, mean_solution_grads);
          // Old Newton iteration values
  //        fe_values.get_function_values(solution, old_solution_values);
  //        fe_values.get_function_gradients(solution, old_solution_grads);

          for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
              phi_i_v[k] = fe_values[velocities].value(k, q);
              phi_i_grads_v[k] = fe_values[velocities].gradient(k, q);
            }

            Tensor<2, dim> grad_v_mean;
            for (unsigned int l = 0; l < dim; l++)
              for (unsigned int m = 0; m < dim; m++)
                grad_v_mean[l][m] = mean_solution_grads[q][l][m];

            Tensor<1, dim> v_mean;
            for (unsigned int l = 0; l < dim; l++)
            v_mean[l] = mean_solution_values[q][l];

            // Outer loop for dofs
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              // Inner loop for dofs
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                // Fluid , NSE
                const unsigned int comp_j = fe.system_to_component_index(j).first;
                if (comp_j == 0 || comp_j == 1)
                {
              	local_first_convection_matrix_velocity_with_mean_vp(j, i) 	+= (phi_i_grads_v[i] * v_mean) * phi_i_v[j] * fe_values.JxW(q);
              	local_second_convection_matrix_velocity_with_mean_vp(j, i) 	+= (grad_v_mean * phi_i_v[i]) * phi_i_v[j] * fe_values.JxW(q);
                }
                // end j dofs
              }
              // end i dofs
            }
            // end n_q_points
          }

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(local_first_convection_matrix_velocity_with_mean_vp, local_dof_indices,
          		first_convection_matrix_velocity_with_mean_vp);
          constraints.distribute_local_to_global(local_second_convection_matrix_velocity_with_mean_vp, local_dof_indices,
          		second_convection_matrix_velocity_with_mean_vp);
          // end cell
        }
      }

  template <int dim>
    void NavierStokes<dim>::assemble_nonlinearity_tensor_velocity_vp(std::vector<BlockVector<double>> &pod_vectors)
  {
      // reinit the tensor
      int r = pod_vectors.size();
      nonlinear_tensor_velocity_vp = std::vector<BlockSparseMatrix<double>>(r); // .resize(r);
      for (int i = 0; i < r; i++)
        nonlinear_tensor_velocity_vp[i].reinit(sparsity_pattern);

      Threads::TaskGroup<void> task_group;
      for (int i = 0; i < r; i++)
      {
    	  task_group += Threads::new_task (&NSE::NavierStokes<2>::assemble_nonlinearity_tensor_velocity_vp_i, *this, pod_vectors[i], i);
//    	  assemble_nonlinearity_tensor_velocity_vp_i(pod_vectors[i], i);
      }
      task_group.join_all ();
    }

  template <int dim>
      void NavierStokes<dim>::assemble_nonlinearity_tensor_velocity_vp_i(BlockVector<double> &pod_vector, int i)
    {

        QGauss<dim> quadrature_formula(fe.degree + 2);
        QGauss<dim - 1> face_quadrature_formula(fe.degree + 2);

        FEValues<dim> fe_values(fe, quadrature_formula,
                                update_values |
                                    update_quadrature_points |
                                    update_JxW_values |
                                    update_gradients);

        const unsigned int dofs_per_cell = fe.dofs_per_cell;

        const unsigned int n_q_points = quadrature_formula.size();
        const unsigned int n_face_q_points = face_quadrature_formula.size();

        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);

        std::vector<unsigned int> local_dof_indices(dofs_per_cell);

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);

        // contribution of the i.th POD vector
        std::vector<Vector<double>> pod_vector_values(n_q_points,
                                                            Vector<double>(dim+1));
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

          fe_values.get_function_values(pod_vector, pod_vector_values);

          for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
              phi_i_v[k] = fe_values[velocities].value(k, q);
              phi_i_grads_v[k] = fe_values[velocities].gradient(k, q);
            }

            Tensor<1, dim> psi;
            for (unsigned int l = 0; l < dim; l++)
                    psi[l] = pod_vector_values[q][l];

            // Outer loop for dofs
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
             // Inner loop for dofs
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                // Fluid , NSE
                const unsigned int comp_j = fe.system_to_component_index(j).first;
                if (comp_j == 0 || comp_j == 1)
                {
                  local_matrix (j,k)	+= (phi_i_grads_v[k] * phi_i_v[j]) *psi* fe_values.JxW(q);
                }
                // end j dofs
              }
              // end i dofs
            }
            // end n_q_points
          }

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(local_matrix, local_dof_indices,
          		nonlinear_tensor_velocity_vp[i]);
          // end cell
        }
      }

  template class NavierStokes<2>;
} // namespace NSE
