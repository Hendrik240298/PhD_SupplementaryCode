#include "../include/pod.h"
#include "../include/RedSVD.h"

// count number of files in directory 'path'
int compute_number_snapshots(std::string path)
{
  int snapshot_counter = 0;

  struct dirent *entry = nullptr;
  DIR *dp = nullptr;

  dp = opendir(path.c_str());
  if (dp != nullptr) {
      while ((entry = readdir(dp)))
      {
        //std::cout << "entry: " << entry->d_name << std::endl;
        if (!(strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0))
          snapshot_counter++; // this is a valid file and not "." or ".."
      }
  }
  closedir(dp);

  /*
  for (const auto & file : std::filesystem::directory_iterator(path))
    snapshot_counter++; // std::cout << file.path() << std::endl;
  */

  return snapshot_counter;
}

// load velocity vector from HDF5 file
dealii::Vector<double> load_h5_vector(std::string file_path)
{
  hid_t file = H5Fopen(file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  // velocity block
  std::string dataset_name_v = "/velocity";
  hid_t dataset_v = H5Dopen1(file, dataset_name_v.c_str());
  hid_t datatype_v = H5Dget_type(dataset_v);
  hid_t dataspace_v = H5Dget_space(dataset_v);

  int n_v = H5Dget_storage_size(dataset_v)/sizeof(double);
  dealii::Vector<double> h5_vector(n_v);

  H5Dread(dataset_v, datatype_v, H5S_ALL, H5S_ALL, H5P_DEFAULT,
          static_cast<void *>(&(h5_vector[0])));
  H5Sclose(dataspace_v);
  H5Tclose(datatype_v);
  H5Dclose(dataset_v);
  H5Fclose(file);

  return h5_vector;
}

// load pressure vector from HDF5 file
dealii::Vector<double> load_h5_vector_pressure(std::string file_path)
{
  hid_t file = H5Fopen(file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  // pressure block
  std::string dataset_name_p = "/pressure";
  hid_t dataset_p = H5Dopen1(file, dataset_name_p.c_str());
  hid_t datatype_p = H5Dget_type(dataset_p);
  hid_t dataspace_p = H5Dget_space(dataset_p);

  int n_p = H5Dget_storage_size(dataset_p)/sizeof(double);
  dealii::Vector<double> h5_vector(n_p);

  H5Dread(dataset_p, datatype_p, H5S_ALL, H5S_ALL, H5P_DEFAULT,
          static_cast<void *>(&(h5_vector[0])));
  H5Sclose(dataspace_p);
  H5Tclose(datatype_p);
  H5Dclose(dataset_p);
  H5Fclose(file);

  return h5_vector;
}

dealii::Vector<double> get_column(dealii::LAPACKFullMatrix<double> &matrix, int column)
{
	dealii::Vector<double> vector(matrix.m());
	for (unsigned int i=0; i < matrix.m(); i++) {
		vector(i) = matrix(i,column);
	}
	return vector;
}

void get_column(dealii::LAPACKFullMatrix<double> &matrix, dealii::Vector<double> &vector, int column)
{
	for (unsigned int i=0; i < matrix.m(); i++) {
		vector(i) = matrix(i,column);
	}
}


dealii::Vector<double> get_row(Eigen::MatrixXd &matrix, int row) {
	dealii::Vector<double> vector(matrix.cols());
	for (int i = 0; i<matrix.cols(); i++) {
		vector(i) = matrix(row,i);
	}
	return vector;
}

void get_row(Eigen::MatrixXd &matrix, int row, dealii::Vector<double> &vector) {
//	vector.reinit(matrix.cols());
	for (int i = 0; i<matrix.cols(); i++) {
		vector(i) = matrix(row,i);
	}
}


void set_column(dealii::LAPACKFullMatrix<double> &matrix, dealii::Vector<double> vector,  int column)
{
	for (unsigned int i=0; i < matrix.m(); i++) {
		matrix(i,column) = vector(i);
	}
}

void add_2_column(dealii::LAPACKFullMatrix<double> &matrix, dealii::Vector<double> vector,  double scale, int column)
{
	for (unsigned int i=0; i < matrix.m(); i++) {
		matrix(i,column) += scale*vector(i);
	}
}

void scale_column(dealii::LAPACKFullMatrix<double> &matrix, double scale,  int column)
{
	for (unsigned int i=0; i < matrix.m(); i++) {
		matrix(i,column) *= scale;
	}
}


namespace POD
{
  using namespace dealii;

  template<int dim>
  ProperOrthogonalDecomposition<dim>::ProperOrthogonalDecomposition()  {

  }

  template<int dim>
  ProperOrthogonalDecomposition<dim>::ProperOrthogonalDecomposition(PDEInfo *pde_info)
  : pde_info(pde_info)
  {
	  fluid_density = pde_info->fluid_density;
	  viscosity =  pde_info->viscosity;
	  information_content = pde_info->information_content;
	  pod_basis_size = pde_info->pod_basis_size;
	  timestep = pde_info->fine_timestep;
  }

  template<int dim>
  void ProperOrthogonalDecomposition<dim>::setup(PDEInfo *pde_info, int refinements)
  {
    navier_stokes_solver.init(pde_info);

    // we will need the DoFHandler and the MassMatrix from the Navier Stokes assembly
    navier_stokes_solver.setup_system_only_velocity(refinements);
//    navier_stokes_solver.setup_system_only_pressure(refinements);
    // one could have also solved this with saving/loading of the mass matrix
  }

  template<int dim>
  void ProperOrthogonalDecomposition<dim>::setup_vp(PDEInfo *pde_info, int refinements)
  {
    navier_stokes_solver.init(pde_info);

    // we will need the DoFHandler and the MassMatrix from the Navier Stokes assembly
    navier_stokes_solver.setup_system_ROM(refinements);
    navier_stokes_solver.assemble_FOM_matrices_aff_decomp();
//    navier_stokes_solver.setup_system_only_pressure(refinements);
    // one could have also solved this with saving/loading of the mass matrix
  }

  // quadrature weigths of trapezoidal rule
  template<int dim>
  double ProperOrthogonalDecomposition<dim>::quad_weight(int i)
  {
    double weight = navier_stokes_solver.time_step;
    if (i == 0 || i == n)
      weight *= 0.5;
    return weight;
  }

  template<int dim>
  void ProperOrthogonalDecomposition<dim>::compute_pod_basis(double information_content, int pod_basis_size)
  {
	std::cout << "start v-POD ...  (eig+mean)" << std::endl;
    std::string snapshot_file_path = pde_info->fem_path + "mu="+ std::to_string(viscosity) +"/snapshots/";
    n = compute_number_snapshots(snapshot_file_path);
//    n_start = 0;
    m = navier_stokes_solver.dofs_velocity;
    std::cout << "\tNumber of snapshots: " << n << std::endl
              << "\tNumber of DoFs: " << m << std::endl;
    // the snapshot matrix has the shape: (n x m) where usually n << m
    // we thus prefer to work with the correlation matrix which has the shape: (n x n)

    /* -----------------------
    *   METHOD OF SNAPSHOTS  |
    * --------------------- */

    // ---------------------
    // 0. Load the snapshots
    // ---------------------

    std::vector<Vector<double>> snapshots;
    for (int i = 0; i < n; i++)
    {
      // load i.th snapshot
      snapshots.push_back(load_h5_vector(snapshot_file_path + "snapshot_" + Utilities::int_to_string(i,6) + ".h5"));
    }

    // ----------------------
    // 1. Compute lifting function which is denoted as mean vector
    // ----------------------

    mean_vector.reinit(m);
    compute_lifting_function();
//    for (Vector<double> &snapshot : snapshots)
//      mean_vector.add(1.0 / n, snapshot);


    // output mean vector to vtk / h5
    //std::cout << (pde_info->pod_path).c_str() << std::endl;
    mkdir((pde_info->pod_path).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    mkdir((pde_info->pod_path + "mu="+ std::to_string(viscosity)).c_str() , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    mkdir((pde_info->pod_path + "mu="+ std::to_string(viscosity)+ "/pod_vectors").c_str()  , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
//    mkdir("result/POD/pod_vectors_"+boost::lexical_cast<std::string>(fluid_density) , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
//    output_mean_vector();

    Vector<double> mean_value_mu;
    mean_value_mu.reinit(m);
    for (Vector<double> &snapshot : snapshots)
    	mean_value_mu.add(1.0 / n, snapshot);
    filename_h5 = pde_info->pod_path + "mu="+ std::to_string(viscosity) +"/mean_vector_mu.h5";
    save_h5_vector(filename_h5, mean_value_mu);

    filename_h5 = pde_info->pod_path + "mu="+ std::to_string(viscosity) +"/mean_vector.h5";
    save_h5_vector(filename_h5, mean_vector);
    // ---------------------------------------------------
    // 2. Center snapshots (i.e. subtract the mean vector)
    // ---------------------------------------------------
    for (Vector<double> &snapshot : snapshots)
      snapshot.add(-1.0, mean_vector);

    // -----------------------------
    // 2. Compute correlation matrix
    // -----------------------------


    LAPACKFullMatrix<double> correlation_matrix(n);
    LAPACKFullMatrix<double> identity_matrix(n); // only needed to compute eigenvalue of correlation_matrix
    identity_matrix = 0.0;
    Vector<double> temp(m);

    for (int i = 0; i < n; i++)
    {
//     navier_stokes_solver.mass_matrix_vp.block(0,0).vmult(temp, snapshots[i]);
//     std::cout << "check_size: " << temp.size() << std::endl;
//     std::cout << "check_sqrt: " << temp.norm_sqr() << std::endl;
      temp = snapshots[i];
      for (int j = i; j < n; j++)
      {
        double val = snapshots[j] * temp;
        val *= std::sqrt(quad_weight(i) * quad_weight(j));
        correlation_matrix(i, j) = correlation_matrix(j, i) = val;
      }
      identity_matrix(i, i) = 1.0;
    }

    // --------------------------------------------
    // 3. Compute eigenvalues of correlation matrix
    // --------------------------------------------
    std::vector<Vector<double>> eigenvectors(n);
    eigenvalues.resize(n);
    correlation_matrix.compute_generalized_eigenvalues_symmetric(identity_matrix, eigenvectors);

    // we only need enough POD vectors such that:
    // total_information * information_content < partial_information
    double total_information = 0.0;
    double partial_information = 0.0;

    // store all eigenvalues in array
    for (int i = 0; i < n; i++)
    {
      const std::complex<double> eigenvalue = correlation_matrix.eigenvalue(i);
      Assert(eigenvalue.imag() == 0.0, ExcInternalError()); // correlation matrix is symmetric positive definite
      total_information += eigenvalues[i] = eigenvalue.real();
    }

    //  sort eigenvalues and eigenvectors in descending order
    std::reverse(eigenvectors.begin(), eigenvectors.end());
    std::reverse(eigenvalues.begin(), eigenvalues.end());


//    filename_h5 = "result/POD/mu="+ std::to_string(viscosity)+"/eigenvalues.h5";
//    save_h5_vector(filename_h5, eigenvalues);

    // r = size of the POD basis
    r = 0;
    while (r < n)
    {
      partial_information += eigenvalues[r];
      r++;
      // Is POD basis big enough?
      if (partial_information > information_content * total_information || r == pod_basis_size)
        break;
    }
    std::cout << "\tApproximation error: " << total_information - partial_information << " of " << total_information << std::endl;
    std::cout << std::setprecision(10) << "\tSize od POD basis: " << r
              << " (information content: " << partial_information / total_information
              << " [goal: " << information_content << "]) \n" << std::endl;

    temp.reinit(r);
     for (int i = 0; i < r; i++) {
   	  temp(i) = eigenvalues[i];
     }

     filename_h5 = pde_info->pod_path + "mu="+ std::to_string(viscosity)+"/eigenvalues.h5";
     save_h5_vector(filename_h5, temp);
    // ----------------------------
    // 4. Compute POD basis vectors
    // ----------------------------
    pod_vectors.resize(r);
    for (int i=0; i<r; i++)
    {
      Vector<double> basis_vector(m);
      for (int j=0;j<n;j++)
      {
        basis_vector.add(std::sqrt(quad_weight(j)) * eigenvectors[i][j],snapshots[j]);
      }
      basis_vector *= 1.0/std::sqrt(eigenvalues[i]);
      pod_vectors[i] = std::move(basis_vector);
      filename_h5 = pde_info->pod_path + "mu="+ std::to_string(viscosity)+"/pod_vectors/pod_vectors"+ Utilities::int_to_string(i,6) +".h5";
      save_h5_vector(filename_h5, pod_vectors[i]);
    }
  }

  template<int dim>
    void ProperOrthogonalDecomposition<dim>::compute_pod_basis_supremizer(double information_content, int pod_basis_size)
    {
  	std::cout << "start s-POD ...  (eig+mean)" << std::endl;
      std::string snapshot_file_path = pde_info->fem_path + "mu="+ std::to_string(viscosity)+ "/snapshot_supremizer/";
      n = compute_number_snapshots(snapshot_file_path);
  //    n_start = 0;
      m = navier_stokes_solver.dofs_velocity;
      std::cout << "\tNumber of snapshots: " << n << std::endl
                << "\tNumber of DoFs: " << m << std::endl;


      std::vector<Vector<double>> snapshots;
      Vector<double> tmp_load(m);
      for (int i = 0; i < n; i++)
      {
        // load i.th snapshot
    	load_h5_vector_HF(snapshot_file_path + "supremizer_" + Utilities::int_to_string(i,6) + ".h5",tmp_load );
        snapshots.push_back(tmp_load);
      }

      // ----------------------
      // 1. Compute lifting function which is denoted as mean vector
      // ----------------------

      mean_vector.reinit(m);
      compute_lifting_function();



      // output mean vector to vtk / h5
      //std::cout << (pde_info->pod_path).c_str() << std::endl;
      mkdir((pde_info->pod_path).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      mkdir((pde_info->pod_path + "mu="+ std::to_string(viscosity)).c_str() , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      mkdir((pde_info->pod_path + "mu="+ std::to_string(viscosity)+ "/pod_vectors_supremizer").c_str()  , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

      // ---------------------------------------------------
      // 2. Center snapshots (i.e. subtract the mean vector)
      // ---------------------------------------------------

      for (Vector<double> &snapshot : snapshots)
        snapshot.add(-1.0, mean_vector);

      // -----------------------------
      // 2. Compute correlation matrix
      // -----------------------------


      LAPACKFullMatrix<double> correlation_matrix(n);
      LAPACKFullMatrix<double> identity_matrix(n); // only needed to compute eigenvalue of correlation_matrix
      identity_matrix = 0.0;
      Vector<double> temp(m);

      for (int i = 0; i < n; i++)
      {
       navier_stokes_solver.mass_matrix_vp.block(0,0).vmult(temp, snapshots[i]);
        temp = snapshots[i];
        for (int j = i; j < n; j++)
        {
          double val = snapshots[j] * temp;
          val *= std::sqrt(quad_weight(i) * quad_weight(j));
          correlation_matrix(i, j) = correlation_matrix(j, i) = val;
        }
        identity_matrix(i, i) = 1.0;
      }

      // --------------------------------------------
      // 3. Compute eigenvalues of correlation matrix
      // --------------------------------------------
      std::vector<Vector<double>> eigenvectors(n);
      eigenvalues.resize(n);
      correlation_matrix.compute_generalized_eigenvalues_symmetric(identity_matrix, eigenvectors);

      // we only need enough POD vectors such that:
      // total_information * information_content < partial_information
      double total_information = 0.0;
      double partial_information = 0.0;

      // store all eigenvalues in array
      for (int i = 0; i < n; i++)
      {
        const std::complex<double> eigenvalue = correlation_matrix.eigenvalue(i);
        Assert(eigenvalue.imag() == 0.0, ExcInternalError()); // correlation matrix is symmetric positive definite
        total_information += eigenvalues[i] = eigenvalue.real();
      }

      //  sort eigenvalues and eigenvectors in descending order
      std::reverse(eigenvectors.begin(), eigenvectors.end());
      std::reverse(eigenvalues.begin(), eigenvalues.end());


  //    filename_h5 = "result/POD/mu="+ std::to_string(viscosity)+"/eigenvalues.h5";
  //    save_h5_vector(filename_h5, eigenvalues);

      // r = size of the POD basis
      r_s = 0;
      while (r_s < n)
      {
        partial_information += eigenvalues[r_s];
        r_s++;
        // Is POD basis big enough?
        if (partial_information > information_content * total_information || r_s == pod_basis_size)
          break;
      }
      std::cout << "\tApproximation error: " << total_information - partial_information << " of " << total_information << std::endl;
      std::cout << std::setprecision(10) << "\tSize of POD basis: " << r
                << " (information content: " << partial_information / total_information
                << " [goal: " << information_content << "]) \n" << std::endl;

      temp.reinit(r_s);
       for (int i = 0; i < r_s; i++) {
     	  temp(i) = eigenvalues[i];
       }

       filename_h5 = pde_info->pod_path + "mu="+ std::to_string(viscosity)+"/eigenvalues_supremizer.h5";
       save_h5_vector(filename_h5, temp);
      // ----------------------------
      // 4. Compute POD basis vectors
      // ----------------------------
      pod_vectors.resize(r);
      for (int i=0; i<r_s; i++)
      {
        Vector<double> basis_vector(m);
        for (int j=0;j<n;j++)
        {
          basis_vector.add(std::sqrt(quad_weight(j)) * eigenvectors[i][j],snapshots[j]);
        }
        basis_vector *= 1.0/std::sqrt(eigenvalues[i]);
        pod_vectors[i] = std::move(basis_vector);
        filename_h5 = pde_info->pod_path + "mu="+ std::to_string(viscosity)+"/pod_vectors_supremizer/pod_vectors_supremizer_"+ Utilities::int_to_string(i,6) +".h5";
        save_h5_vector(filename_h5, pod_vectors[i]);
      }
    }

  template<int dim>
      void ProperOrthogonalDecomposition<dim>::compute_pod_basis_pressure(double information_content, int pod_basis_size)
      {
        std::string snapshot_file_path = pde_info->fem_path + "snapshots/";
        n = compute_number_snapshots(snapshot_file_path);
    //    n = n - n_start;
        m_p = navier_stokes_solver.dof_handler_pressure.n_dofs();
        std::cout << "\tNumber of snapshots: " << n << std::endl
                  << "\tNumber of DoFs: " << m_p << std::endl;
        // the snapshot matrix has the shape: (n x m) where usually n << m
        // we thus prefer to work with the correlation matrix which has the shape: (n x n)

        /* -----------------------
        *   METHOD OF SNAPSHOTS  |
        * --------------------- */

        // ---------------------
        // 0. Load the snapshots
        // ---------------------
        std::vector<Vector<double>> snapshots;
        for (int i = 0; i < n; i++)
        {
          // load i.th snapshot
          snapshots.push_back(load_h5_vector_pressure(snapshot_file_path + "snapshot_" + Utilities::int_to_string(i,6) + ".h5"));
        }

        /*
        // TODO: Is this correct?
        int iii = 0;
        for (Vector<double> &snapshot : snapshots)
        {
          DataOut<dim> data_out;

          data_out.attach_dof_handler(navier_stokes_solver.dof_handler_pressure);
          data_out.add_data_vector(snapshot, "pressure");

          data_out.build_patches();

          const std::string filename =
            "result/POD/debug/vector-" + Utilities::int_to_string(iii, 6) + ".vtk";
          std::ofstream output(filename);
          data_out.write_vtk(output);
          iii++;
        }
        */

        // ----------------------
        // 1. Compute mean vector
        // ----------------------
        mean_vector_p.reinit(m_p);
        for (Vector<double> &snapshot : snapshots)
          mean_vector_p.add(1.0, snapshot);
        mean_vector_p *= (1.0 / n);

        // output mean vector to vtk / h5
        output_mean_vector_pressure();

        // ---------------------------------------------------
        // 2. Center snapshots (i.e. subtract the mean vector)
        // ---------------------------------------------------
        for (Vector<double> &snapshot : snapshots)
          snapshot.add(-1.0, mean_vector_p);

        // -----------------------------
        // 2. Compute correlation matrix
        // -----------------------------
        LAPACKFullMatrix<double> correlation_matrix(n);
        LAPACKFullMatrix<double> identity_matrix(n); // only needed to compute eigenvalue of correlation_matrix
        identity_matrix = 0.0;
        Vector<double> temp(m_p);

        for (int i = 0; i < n; i++)
        {
          navier_stokes_solver.mass_matrix_pressure.vmult(temp, snapshots[i]);
          temp = snapshots[i];
          for (int j = i; j < n; j++)
          {
            double val = snapshots[j] * temp;
            val *= std::sqrt(quad_weight(i) * quad_weight(j));
            correlation_matrix(i, j) = correlation_matrix(j, i) = val;
          }
          identity_matrix(i, i) = 1.0;
        }

        // --------------------------------------------
        // 3. Compute eigenvalues of correlation matrix
        // --------------------------------------------
        std::vector<Vector<double>> eigenvectors(n);
        eigenvalues_p.resize(n);
        correlation_matrix.compute_generalized_eigenvalues_symmetric(identity_matrix, eigenvectors);

        // we only need enough POD vectors such that:
        // total_information * information_content < partial_information
        double total_information = 0.0;
        double partial_information = 0.0;

        // store all eigenvalues in array
        for (int i = 0; i < n; i++)
        {
          const std::complex<double> eigenvalue = correlation_matrix.eigenvalue(i);
          Assert(eigenvalue.imag() == 0.0, ExcInternalError()); // correlation matrix is symmetric positive definite
          total_information += eigenvalues_p[i] = eigenvalue.real();
        }

        //  sort eigenvalues and eigenvectors in descending order
        std::reverse(eigenvectors.begin(), eigenvectors.end());
        std::reverse(eigenvalues_p.begin(), eigenvalues_p.end());

        // r_p = size of the POD basis of the pressure
        while (r_p < n)
        {
          partial_information += eigenvalues_p[r_p];
          r_p++;
          // Is POD basis big enough?
          if (partial_information > information_content * total_information || r_p == pod_basis_size)
            break;
        }
        std::cout << "\tPRESSURE:\n\tApproximation error: " << total_information - partial_information << std::endl;
        std::cout << std::setprecision(10) << "\tSize od POD basis: " << r_p
                  << " (information content: " << partial_information / total_information
                  << " [goal: " << information_content << "])" << std::endl << std::endl;


        temp.reinit(r_p);
         for (int i = 0; i < r_p; i++) {
       	  temp(i) = eigenvalues[i];
         }

         filename_h5 = pde_info->pod_path + "mu="+ std::to_string(viscosity)+"/eigenvalues_supremizer.h5";
         save_h5_vector(filename_h5, temp);
        // ----------------------------
        // 4. Compute POD basis vectors
        // ----------------------------
        pod_vectors_p.resize(r_p);
        for (int i=0; i<r_p; i++)
        {
          Vector<double> basis_vector(m_p);
          for (int j=0;j<n;j++)
          {
            basis_vector.add(std::sqrt(quad_weight(j)) * eigenvectors[i][j],snapshots[j]);
          }
          basis_vector *= 1.0/std::sqrt(eigenvalues_p[i]);
          pod_vectors_p[i] = std::move(basis_vector);
        }
      }

  template<int dim>
       void ProperOrthogonalDecomposition<dim>::compute_pod_basis_no_mean_press(double information_content, int pod_basis_size)
       {
         std::string snapshot_file_path = pde_info->fem_path + "mu="+ std::to_string(viscosity) +"/snapshots/";
         n = compute_number_snapshots(snapshot_file_path);
     //    n_start = 0;
         m = navier_stokes_solver.dofs_pressure;
         std::cout << "Start p-POD ... " << std::endl
  	   	   	   	 << "\tNumber of snapshots: " << n << std::endl
                   << "\tNumber of DoFs: " << m << std::endl;
         // the snapshot matrix has the shape: (n x m) where usually n << m
         // we thus prefer to work with the correlation matrix which has the shape: (n x n)

         /* -----------------------
         *   METHOD OF SNAPSHOTS  |
         * --------------------- */

         // ---------------------
         // 0. Load the snapshots
         // ---------------------
         std::vector<Vector<double>> snapshots;
         for (int i = 0; i < n; i++)
         {
           // load i.th snapshot
           snapshots.push_back(load_h5_vector_pressure(snapshot_file_path + "snapshot_" + Utilities::int_to_string(i,6) + ".h5"));
         }

         mkdir((pde_info->pod_path).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
         mkdir((pde_info->pod_path + "mu="+ std::to_string(viscosity)).c_str() , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
         mkdir((pde_info->pod_path + "mu="+ std::to_string(viscosity)+ "/pod_vectors_press").c_str()  , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
     //    mkdir("result/POD/pod_vectors_"+boost::lexical_cast<std::string>(fluid_density) , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

         // -----------------------------
         // 1. Compute correlation matrix
         // -----------------------------


         LAPACKFullMatrix<double> correlation_matrix(n);
         LAPACKFullMatrix<double> identity_matrix(n); // only needed to compute eigenvalue of correlation_matrix
         identity_matrix = 0.0;
         Vector<double> temp(m);

         for (int i = 0; i < n; i++)
         {
//           navier_stokes_solver.mass_matrix_pressure.vmult(temp, snapshots[i]);
           temp = snapshots[i];
           for (int j = i; j < n; j++)
           {
             double val = snapshots[j] * temp;
             val *= std::sqrt(quad_weight(i) * quad_weight(j));
             correlation_matrix(i, j) = correlation_matrix(j, i) = val;
           }
           identity_matrix(i, i) = 1.0;
         }

         // --------------------------------------------
         // 2. Compute eigenvalues of correlation matrix
         // --------------------------------------------
         std::vector<Vector<double>> eigenvectors(n);
         eigenvalues.resize(n);
         correlation_matrix.compute_generalized_eigenvalues_symmetric(identity_matrix, eigenvectors);

         // we only need enough POD vectors such that:
         // total_information * information_content < partial_information
         double total_information = 0.0;
         double partial_information = 0.0;

         // store all eigenvalues in array
         for (int i = 0; i < n; i++)
         {
           const std::complex<double> eigenvalue = correlation_matrix.eigenvalue(i);
           Assert(eigenvalue.imag() == 0.0, ExcInternalError()); // correlation matrix is symmetric positive definite
           total_information += eigenvalues[i] = eigenvalue.real();
         }

         //  sort eigenvalues and eigenvectors in descending order
         std::reverse(eigenvectors.begin(), eigenvectors.end());
         std::reverse(eigenvalues.begin(), eigenvalues.end());

         // r = size of the POD basis
         r = 0;
         while (r < n)
         {
           partial_information += eigenvalues[r];
           r++;
           // Is POD basis big enough?
           if (partial_information > information_content * total_information || r == pod_basis_size)
             break;
         }
         std::cout << "\tApproximation error: " << total_information - partial_information  << " of " << total_information << std::endl;
         std::cout << std::setprecision(10) << "\tSize od POD basis: " << r
                   << " (information content: " << partial_information / total_information
                   << " [goal: " << information_content << "]) \n" << std::endl;

         temp.reinit(r);
          for (int i = 0; i < r; i++) {
        	  temp(i) = eigenvalues[i];
          }

          filename_h5 = pde_info->pod_path + "mu="+ std::to_string(viscosity)+"/eigenvalues_press.h5";
          save_h5_vector(filename_h5, temp);
         // ----------------------------
         // 3. Compute POD basis vectors
         // ----------------------------
         pod_vectors.resize(r);
         for (int i=0; i<r; i++)
         {
           Vector<double> basis_vector(m);
           for (int j=0;j<n;j++)
           {
             basis_vector.add(std::sqrt(quad_weight(j)) * eigenvectors[i][j],snapshots[j]);
           }
           basis_vector *= 1.0/std::sqrt(eigenvalues[i]);
           pod_vectors[i] = std::move(basis_vector);
           filename_h5 = pde_info->pod_path + "mu="+ std::to_string(viscosity)+"/pod_vectors_press/pod_vectors_press"+ Utilities::int_to_string(i,6) +".h5";
           save_h5_vector(filename_h5, pod_vectors[i]);
         }
       }


  template<int dim>
    void ProperOrthogonalDecomposition<dim>::compute_pod_basis_no_mean(double information_content, int pod_basis_size)
    {
      std::string snapshot_file_path = pde_info->fem_path + "mu="+ std::to_string(viscosity) +"/snapshots/";
      n = compute_number_snapshots(snapshot_file_path);
  //    n_start = 0;
      m = navier_stokes_solver.dof_handler_velocity.n_dofs();
      std::cout << "Start v-POD ... " << std::endl
    		    << "\tNumber of snapshots: " << n << std::endl
                << "\tNumber of DoFs: " << m << std::endl;
      // the snapshot matrix has the shape: (n x m) where usually n << m
      // we thus prefer to work with the correlation matrix which has the shape: (n x n)

      /* -----------------------
      *   METHOD OF SNAPSHOTS  |
      * --------------------- */

      // ---------------------
      // 0. Load the snapshots
      // ---------------------
      std::vector<Vector<double>> snapshots;
      for (int i = 0; i < n; i++)
      {
        // load i.th snapshot
        snapshots.push_back(load_h5_vector(snapshot_file_path + "snapshot_" + Utilities::int_to_string(i,6) + ".h5"));
      }

      mkdir((pde_info->pod_path).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      mkdir((pde_info->pod_path + "mu="+ std::to_string(viscosity)).c_str() , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      mkdir((pde_info->pod_path + "mu="+ std::to_string(viscosity)+ "/pod_vectors").c_str()  , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  //    mkdir("result/POD/pod_vectors_"+boost::lexical_cast<std::string>(fluid_density) , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

      // -----------------------------
      // 1. Compute correlation matrix
      // -----------------------------


      LAPACKFullMatrix<double> correlation_matrix(n);
      LAPACKFullMatrix<double> identity_matrix(n); // only needed to compute eigenvalue of correlation_matrix
      identity_matrix = 0.0;
      Vector<double> temp(m);

      for (int i = 0; i < n; i++)
      {
        navier_stokes_solver.mass_matrix_velocity.vmult(temp, snapshots[i]);
//        temp = snapshots[i];
        for (int j = i; j < n; j++)
        {
          double val = snapshots[j] * temp;
          val *= std::sqrt(quad_weight(i) * quad_weight(j));
          correlation_matrix(i, j) = correlation_matrix(j, i) = val;
        }
        identity_matrix(i, i) = 1.0;
      }

      // --------------------------------------------
      // 2. Compute eigenvalues of correlation matrix
      // --------------------------------------------
      std::vector<Vector<double>> eigenvectors(n);
      eigenvalues.resize(n);
      correlation_matrix.compute_generalized_eigenvalues_symmetric(identity_matrix, eigenvectors);

      // we only need enough POD vectors such that:
      // total_information * information_content < partial_information
      double total_information = 0.0;
      double partial_information = 0.0;

      // store all eigenvalues in array
      for (int i = 0; i < n; i++)
      {
        const std::complex<double> eigenvalue = correlation_matrix.eigenvalue(i);
        Assert(eigenvalue.imag() == 0.0, ExcInternalError()); // correlation matrix is symmetric positive definite
        total_information += eigenvalues[i] = eigenvalue.real();
      }

      //  sort eigenvalues and eigenvectors in descending order
      std::reverse(eigenvectors.begin(), eigenvectors.end());
      std::reverse(eigenvalues.begin(), eigenvalues.end());

      // r = size of the POD basis
      r=0;
      while (r < n)
      {
        partial_information += eigenvalues[r];
        r++;
        // Is POD basis big enough?
        if (partial_information > information_content * total_information || r == pod_basis_size)
          break;
      }
      std::cout << "\tApproximation error: " << total_information - partial_information  << " of " << total_information << std::endl;
      std::cout << std::setprecision(10) << "\tSize od POD basis: " << r
                << " (information content: " << partial_information / total_information
                << " [goal: " << information_content << "])" << std::endl;

      temp.reinit(r);
       for (int i = 0; i < r; i++) {
     	  temp(i) = eigenvalues[i];
       }

       filename_h5 = pde_info->pod_path + "mu="+ std::to_string(viscosity)+"/eigenvalues.h5";
       save_h5_vector(filename_h5, temp);
      // ----------------------------
      // 3. Compute POD basis vectors
      // ----------------------------
      pod_vectors.resize(r);
      for (int i=0; i<r; i++)
      {
        Vector<double> basis_vector(m);
        for (int j=0;j<n;j++)
        {
          basis_vector.add(std::sqrt(quad_weight(j)) * eigenvectors[i][j],snapshots[j]);
        }
        basis_vector *= 1.0/std::sqrt(eigenvalues[i]);
        pod_vectors[i] = std::move(basis_vector);
        filename_h5 = pde_info->pod_path + "mu="+ std::to_string(viscosity)+"/pod_vectors/pod_vectors"+ Utilities::int_to_string(i,6) +".h5";
        save_h5_vector(filename_h5, pod_vectors[i]);
      }
    }

  template<int dim>
    void ProperOrthogonalDecomposition<dim>::compute_pod_basis_SVD(double information_content, int pod_basis_size)
    {
	  std::string snapshot_file_path = pde_info->fem_path + "mu="+ std::to_string(viscosity) +"/snapshots/";
	     n = compute_number_snapshots(snapshot_file_path);
	 //    n_start = 0;
	     m = navier_stokes_solver.dof_handler_velocity.n_dofs();
	     std::cout << "\tNumber of snapshots: " << n << std::endl
	               << "\tNumber of DoFs: " << m << std::endl;
	     // the snapshot matrix has the shape: (n x m) where usually n << m
	     // we thus prefer to work with the correlation matrix which has the shape: (n x n)

	     /* -----------------------
	     *   METHOD OF SNAPSHOTS  |
	     * --------------------- */

	     // ---------------------
	     // 0. Load the snapshots
	     // ---------------------
	     std::vector<Vector<double>> snapshots;
	     LAPACKFullMatrix<double> snapshot_matrix(m,n);
	     for (int i = 0; i < n; i++)
	     {
	    	 set_column(snapshot_matrix,load_h5_vector(snapshot_file_path + "snapshot_" + Utilities::int_to_string(i,6) + ".h5"),i);
	     }

	     // ----------------------
	     // 1. Compute mean vector
	     // ----------------------
	     mean_vector.reinit(m);
	     for (int i = 0; i < n; i++)
	     {
	    	 mean_vector.add(1.0/n,get_column(snapshot_matrix,i));
	     }

	     // output mean vector to vtk / h5
	     output_mean_vector();
	     mkdir(("result/POD/mu="+ std::to_string(viscosity)).c_str() , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	     mkdir(("result/POD/mu="+ std::to_string(viscosity)+ "/pod_vectors").c_str()  , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	     filename_h5 = "result/POD/mu="+ std::to_string(viscosity) +"/mean_vector.h5";
	     save_h5_vector(filename_h5, mean_vector);

	     // ---------------------------------------------------
	     // 2. Center snapshots (i.e. subtract the mean vector)
	     // ---------------------------------------------------
	     for (int i = 0; i < n; i++)
	     {
	    	set_column(snapshot_matrix,get_column(snapshot_matrix,i)-=mean_vector,i);
	     }

	     space_weights.reinit(m);
	     space_weights = 0.0;
	     SparseMatrixIterators::Iterator< double, false > it = navier_stokes_solver.mass_matrix_velocity.begin();

//	     Eigen::SparseMatrix<double> mass_matrix(m,m);
//	     typedef Eigen::Triplet<double> T;
//	     std::vector<T> tripletList;
//	     tripletList.reserve(navier_stokes_solver.mass_matrix_velocity.n_nonzero_elements());
	     std::cout << "start weigthing space" << std::endl;


	     for (int i = 0; i<m; i++) {
	    	 for (it = navier_stokes_solver.mass_matrix_velocity.begin(i); it != navier_stokes_solver.mass_matrix_velocity.end(i); ++it) {
	    		 	 space_weights(i) += navier_stokes_solver.mass_matrix_velocity.el(i,it->column());
//	    		 	tripletList.push_back(T(i,it->column(),navier_stokes_solver.mass_matrix_velocity.el(i,it->column())));
	    	 }
//	    	 space_weights(i) = 1;
	     }
	     filename_h5 = "result/POD/mu="+ std::to_string(viscosity)+"/space_weights" +".h5";
	     save_h5_vector(filename_h5, space_weights);
//	     mass_matrix.setFromTriplets(tripletList.begin(), tripletList.end());
	     std::cout << "end" << std::endl;

	     //---------------------------------------------
	     // Compute sqrt(weight_matrix)
	     // W = mass_matrix
	     // W = U*S*V'
	     // W^1/2 = U*S^1/2*U'
	     //
	     // Y_bar = W^1/2*Y*D^1/2
	     //       = U*S^1/2*U'*Y*D^1/2
	     //		  = UU*(UU'*Y_D)
	     //---------------------------------------------

//	     int nodes_W = 20;
//	     std::cout << "start redsvd massmatrix" << std::endl;
//	     RedSVD::RedSVD<Eigen::SparseMatrix<double>> redsvd_mass(mass_matrix,nodes_W);
//	     Eigen::MatrixXd U_W_eig = redsvd_mass.matrixU();
//	     Eigen::VectorXd S_W_eig = redsvd_mass.singularValues();
//
//	     LAPACKFullMatrix<double> UU(m,nodes_W);
//
//	     LAPACKFullMatrix<double> snapshot_matrix_temp(m,n);
//	     for (int i = 0; i<m; i++) {
//	    	 for (int j = 0; j<n; j++) {
//	    		 snapshot_matrix_temp(i,j) = snapshot_matrix(i,j)* std::sqrt(quad_weight(j));
//	    	 }
//	     }
//
//	    std::cout << "start weighting dealii" << std::endl;
//
//	    LAPACKFullMatrix<double> snap_ttt(m,n);
//	    LAPACKFullMatrix<double> UtY(nodes_W,n);
//		for (int i = 0; i<m; i++) {
//		  	 for (int j = 0; j<nodes_W; j++) {
//	    		 UU(i,j) = U_W_eig(i,j)*std::sqrt(std::sqrt(S_W_eig(j)));
//	    	 }
//	     }
//	      UU.Tmmult(UtY,snapshot_matrix_temp,false);
//	      UU.mmult(snapshot_matrix_temp,UtY,false);
//
//	      Eigen::MatrixXd snapshot_matrix_eigen(m,n);
//	      for (int i = 0; i<m; i++) {
//	          	 for (int j = 0; j<n; j++) {
//	      	   		 snapshot_matrix_eigen(i,j) = snapshot_matrix_temp(i,j);
//	         	 }
//	      }
//	     std::cout << "end weighting dealii" << std::endl;
//
//
//	     std::cout << redsvd_mass.singularValues() << std::endl;
//	     filename_h5 = "result/POD/sing_red_mass.txt";
//	     std::ofstream myfile;
//	     myfile.open(filename_h5);
//	     myfile << redsvd_mass.singularValues();
//	     myfile.close();
//	     std::cout << "end" << std::endl;

//	     if (false) {
	     std::cout << "start weighting snapshotmatrix" << std::endl;
//	     snapshot_matrix_eigen(m,n);
	     Eigen::MatrixXd snapshot_matrix_eigen(m,n);
	     for (int i = 0; i<m; i++) {
	    	 for (int j = 0; j<n; j++) {
//	    		 snapshot_matrix(i,j) *= std::sqrt(space_weights(i)*quad_weight(j));
	    		 snapshot_matrix_eigen(i,j) = snapshot_matrix(i,j)* std::sqrt(space_weights(i)*quad_weight(j));
	    	 }
	     }
//	     }
//	     filename_h5 = "result/POD/W.h5";
//	     save_h5_vector(filename_h5, space_weights);
	     std::cout << "end" << std::endl;
//	     for (int i = 0; i<m; i++) {
//	    	 unit_vec(i) = 1.0;
//	    	 std::cout << i << std::endl;
//	    	 navier_stokes_solver.mass_matrix_velocity.vmult(temp,unit_vec);
////	    	 for (int j = 0; j<m; j++) {
////	    		 space_weights2(i) += navier_stokes_solver.mass_matrix_velocity.el(i,j);
////	    	 }
//	    	 space_weights2(i) = m*temp.mean_value();
//	    	 if (space_weights(i)-space_weights2(i) >= 1e-15)
//	    	 std::cout << space_weights(i)-space_weights2(i) << std::endl;
//	    	 unit_vec(i) = 0.0;
//	     }
//	     filename_h5 = "result/POD/W2.h5";
//	     save_h5_vector(filename_h5, space_weights2);
//	     std::cout << "end" << std::endl;
//	     sparsity_pattern_W.reinit(m,m,1);
//	     SparseMatrix<double> W(navier_stokes_solver.sparsity_pattern_velocity,IdentityMatrix(m));
//	     SparsityPattern sparsity_pattern_W;
//	     sparsity_pattern_W = W.get_sparsity_pattern();
//	     std::ofstream out("sparsity_pattern_W.svg");
//	     sparsity_pattern_W.print_svg(out);

//	     std::cout << W(0,0) << "  " << W(0,1) << "   " << W(0,m-1) << std::endl;
//	     W.reinit(sparsity_pattern_W);
//	     W = 0.0;
//	     W.copy_from(navier_stokes_solver.mass_matrix_velocity);
//	     W.compute_svd();

	     // -----------------------------
	     // 2. Compute SVD
	     // -----------------------------

	     std::cout << "start redsvd snapshots matrix" << std::endl;
	      auto start_time2 = std::chrono::high_resolution_clock::now();
	      RedSVD::RedSVD<Eigen::MatrixXd> redsvd(snapshot_matrix_eigen,pod_basis_size);
	      auto end_time2 = std::chrono::high_resolution_clock::now();
	      auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time2 - start_time2);
	      std::cout << "redsvd computation time:" << std::floor(duration2.count() / 1000) << " seconds "
	                      << duration2.count() % 1000 << " milliseconds" << std::endl;
//	     std::cout << redsvd.singularValues() << std::endl;
	     filename_h5 = "result/POD/sing_red.txt";
	     std::ofstream myfile;
	     myfile.open(filename_h5);
	     myfile << std::setprecision(16) << redsvd.singularValues();
	     myfile.close();
//	     RedSVD::RedSVD<LAPACKFullMatrix<double>> redsvd(snapshot_matrix,10);
	     std::cout << "end redsvd" << std::endl;

	     std::cout << "write pod_vec" << std::endl;
	      Eigen::MatrixXd singular_values(pod_basis_size,1);
	      singular_values = redsvd.singularValues();
	      Eigen::MatrixXd U(m,pod_basis_size);
	      U = redsvd.matrixU();
	      Vector<double> tmp(m);
	      pod_vectors.resize(pod_basis_size);
	      for (int i = 0; i < pod_basis_size; i++) {
	    	  tmp = 0.;
	    	  for (int j= 0; j < m; j++){
	    		  tmp(j)= U(j,i)/std::sqrt(space_weights(j));
	    	  }
//	    	  pod_vectors.push_back(tmp);
	    	  pod_vectors[i] = tmp;
	    	  filename_h5 = "result/POD/mu="+ std::to_string(viscosity)+"/pod_vectors/pod_vectors"+ Utilities::int_to_string(i,6) +".h5";
	    	  save_h5_vector(filename_h5, tmp);
	      }
		  std::cout << "end pod_vec" << std::endl;

	      r = pod_basis_size;

	      std::cout << "compute POD quality" << std::endl;
	      double total_information = 0.0;
	     double partial_information = 0.0;
	     eigenvalues.resize(r);
	     tmp.reinit(r);
	      for (int i = 0; i < pod_basis_size; i++) {
	    	  eigenvalues[i] = singular_values(i)*singular_values(i);
	    	  tmp(i) = eigenvalues[i];
	    	  partial_information += singular_values(i)*singular_values(i);
	      }

	      filename_h5 = "result/POD/mu="+ std::to_string(viscosity)+"/eigenvalues.h5";
	      save_h5_vector(filename_h5, tmp);

		  Vector<double> tmp_snap(m);
		  for (int i = 0; i < n; i++) {
		 	  tmp_snap = get_column(snapshot_matrix,i);
		  	  for (int j = 0; j < m; j++) {
		   		  tmp_snap(j) *= space_weights(j);
		   	  }
		   	  total_information += quad_weight(i)*tmp_snap.operator*(get_column(snapshot_matrix,i));
		   	  tmp_snap = 0;
	 	 }
		     std::cout << "write pod_sings" << std::endl;

	      double total_information_bound = partial_information + (n-pod_basis_size)*singular_values(pod_basis_size-1)*singular_values(pod_basis_size-1);

		     std::cout << "\tApproximation error: " << total_information - partial_information << std::endl;
		     std::cout << "\ttotal:      " << total_information << std::endl;
		     std::cout << "\ttotalbound: " << total_information_bound << std::endl;
		     std::cout << "\tpartial:    " << partial_information << std::endl;
		     std::cout << std::setprecision(10) << "\tSize od POD basis: " << r
		               << " (information content: " << partial_information / total_information
		               << " [goal: " << information_content << "])" << std::endl << std::endl;
//	     std::cout << "start svd" << std::endl;
//	      start_time2 = std::chrono::high_resolution_clock::now();
//	      snapshot_matrix.compute_svd();
//	      end_time2 = std::chrono::high_resolution_clock::now();
//	      duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time2 - start_time2);
//	      std::cout << "dealsvd computation time:" << std::floor(duration2.count() / 1000) << " seconds "
//	                      << duration2.count() % 1000 << " milliseconds" << std::endl;
//	     filename_h5 = "result/POD/sing.txt";
//	     myfile.open(filename_h5);
//	     for (int i=0; i<n; i++) {
//		     myfile << std::setprecision(16) << snapshot_matrix.singular_value(i) << "\n";
//	     }
//	     myfile.close();
////	     save_h5_vector(filename_h5, snapshot_matrix.singular_value(0));
//	     std::cout << "end" << std::endl;

	     // -----------------------------
	     // 2. Compute correlation matrix
	     // -----------------------------
//	     LAPACKFullMatrix<double> correlation_matrix(n);
//	     LAPACKFullMatrix<double> identity_matrix(n); // only needed to compute eigenvalue of correlation_matrix
//	     identity_matrix = 0.0;
//	     Vector<double> temp(m);
//	     std::cout << "start" << std::endl;
//	     for (int i = 0; i < n; i++)
//	     {
//	       navier_stokes_solver.mass_matrix_velocity.vmult(temp, snapshots[i]);
//	       for (int j = i; j < n; j++)
//	       {
//	         double val = snapshots[j] * temp;
//	         val *= std::sqrt(quad_weight(i) * quad_weight(j));
//	         correlation_matrix(i, j) = correlation_matrix(j, i) = val;
//	       }
//	       identity_matrix(i, i) = 1.0;
//	     }
//	     std::cout << "end" << std::endl;
//	     // --------------------------------------------
//	     // 3. Compute eigenvalues of correlation matrix
//	     // --------------------------------------------
//	     std::vector<Vector<double>> eigenvectors(n);
//	     eigenvalues.resize(n);
//	     correlation_matrix.compute_generalized_eigenvalues_symmetric(identity_matrix, eigenvectors);
//
//	     // we only need enough POD vectors such that:
//	     // total_information * information_content < partial_information
//	     total_information = 0.0;
//	     partial_information = 0.0;
//
//	     // store all eigenvalues in array
//	     for (int i = 0; i < n; i++)
//	     {
//	       const std::complex<double> eigenvalue = correlation_matrix.eigenvalue(i);
//	       Assert(eigenvalue.imag() == 0.0, ExcInternalError()); // correlation matrix is symmetric positive definite
//	       total_information += eigenvalues[i] = eigenvalue.real();
//	     }
//
//	     //  sort eigenvalues and eigenvectors in descending order
//	     std::reverse(eigenvectors.begin(), eigenvectors.end());
//	     std::reverse(eigenvalues.begin(), eigenvalues.end());
//
//	     // r = size of the POD basis
//	     while (r < n)
//	     {
//	       partial_information += eigenvalues[r];
//	       r++;
//	       // Is POD basis big enough?
//	       if (partial_information > information_content * total_information || r == pod_basis_size)
//	         break;
//	     }
//	     std::cout << "\tApproximation error: " << total_information - partial_information << std::endl;
//	     std::cout << std::setprecision(10) << "\tSize od POD basis: " << r
//	               << " (information content: " << partial_information / total_information
//	               << " [goal: " << information_content << "])" << std::endl << std::endl;
//
//	     // ----------------------------
//	     // 4. Compute POD basis vectors
//	     // ----------------------------
//	     pod_vectors.resize(r);
//	     for (int i=0; i<r; i++)
//	     {
//	       Vector<double> basis_vector(m);
//	       for (int j=0;j<n;j++)
//	       {
//	         basis_vector.add(std::sqrt(quad_weight(j)) * eigenvectors[i][j],snapshots[j]);
//	       }
//	       basis_vector *= 1.0/std::sqrt(eigenvalues[i]);
//	       pod_vectors[i] = std::move(basis_vector);
//	       filename_h5 = "result/POD/pod_vectors_mu="+ std::to_string(viscosity)+"/pod_vectors"+ Utilities::int_to_string(i,6) +".h5";
//	       save_h5_vector(filename_h5, pod_vectors[i]);
//	     }
    }


  template<int dim>
      void ProperOrthogonalDecomposition<dim>::compute_pod_basis_eigen(double information_content, int pod_basis_size)
      {
  	  std::string snapshot_file_path = pde_info->fem_path + "mu="+ std::to_string(viscosity) +"/snapshots/";
  	     n = compute_number_snapshots(snapshot_file_path);
  	 //    n_start = 0;
  	     m = navier_stokes_solver.dof_handler_velocity.n_dofs();
  	     std::cout << "Number of snapshots: " << n << std::endl
  	               << "Number of DoFs: " << m << std::endl;
  	     // the snapshot matrix has the shape: (n x m) where usually n << m
  	     // we thus prefer to work with the correlation matrix which has the shape: (n x n)

  	     /* -----------------------
  	     *   METHOD OF SNAPSHOTS  |
  	     * --------------------- */

  	     // ---------------------
  	     // 0. Load the snapshots
  	     // ---------------------
  	     std::vector<Vector<double>> snapshots;
  	   std::vector<Vector<double>> snapshots_vec;
  	   std::vector<Vector<double>> snapshots_vec2;
  	   LAPACKFullMatrix<double> snapshot_matrix(m,n);
  	    for (int i = 0; i < n; i++)
  	    {
  	      // load i.th snapshot
  	    }
  	     for (int i = 0; i < n; i++)
  	     {
  	    	 set_column(snapshot_matrix,load_h5_vector(snapshot_file_path + "snapshot_" + Utilities::int_to_string(i,6) + ".h5"),i);
  	     }

  	     // ----------------------
  	     // 1. Compute mean vector
  	     // ----------------------
  	     std::cout << "1. Compute mean vector" << std::endl;

  	     mean_vector.reinit(m);
  	     for (int i = 0; i < n; i++)
  	     {
  	    	 mean_vector.add(1.0/n,get_column(snapshot_matrix,i));
  	     }


  	     // output mean vector to vtk / h5
  	     output_mean_vector();
  	     mkdir(("result/POD/mu="+ std::to_string(viscosity)).c_str() , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  	     mkdir(("result/POD/mu="+ std::to_string(viscosity)+ "/pod_vectors").c_str()  , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  	     filename_h5 = "result/POD/mu="+ std::to_string(viscosity) +"/mean_vector.h5";
  	     save_h5_vector(filename_h5, mean_vector);

  	     // ---------------------------------------------------
  	     // 2. Center snapshots (i.e. subtract the mean vector)
  	     // ---------------------------------------------------
  	     std::cout << "2. Center snapshots" << std::endl;

  	     for (int i = 0; i < n; i++)
  	     {
//  	    	set_column(snapshot_matrix,get_column(snapshot_matrix,i)-=mean_vector,i);
  	    	add_2_column(snapshot_matrix,mean_vector,-1.0,i);
  	     }

//  	     for (int i = 0; i < n; i++)
//  	     {
//  	  	     snapshots_vec2.push_back(get_column(snapshot_matrix,i));
//  	     }

  	     for (int i = 0; i < n; i++)
  	     {
  	  	     snapshots_vec.push_back(get_column(snapshot_matrix,i));
  	     }
  	     // ---------------------------------------------------
  	     // 3. Time weight snapshots
  	     // Y_D = Y*D^1/2
  	     // ---------------------------------------------------
  	     std::cout << "3. Time weight snapshots" << std::endl;

  	     for (int i = 0; i < n; i++)
  	     {
  	    	scale_column(snapshot_matrix,std::sqrt(quad_weight(i)),i);
  	     }


//
// 	     for (int i = 0; i < n; i++)
//  	     {
//  	  	     snapshots_vec.push_back(get_column(snapshot_matrix,i));
//  	     }
  	     // ---------------------------------------------------
  	     // 4. Space weight snapshots --> Correlation matrix
  	     // C = Y_D^T*M*Y_D
  	     // ---------------------------------------------------


//  	  auto start_time = std::chrono::high_resolution_clock::now();

	     std::cout << "4. Space weight snapshots" << std::endl;

	   Eigen::MatrixXd correlation_matrix(n,n);
//	   Eigen::MatrixXd correlation2_matrix(n,n);
	    Vector<double> temp(m);

	Vector<double> tester(m);
	Vector<double> tester2(m);

	    for (int i = 0; i < n; i++)
	    {
	    navier_stokes_solver.mass_matrix_velocity.vmult(temp,snapshots_vec[i]);

	      for (int j = i; j < n; j++)
	      {
	        correlation_matrix(i, j) = correlation_matrix(j, i) = snapshots_vec[j]* temp*std::sqrt(quad_weight(i) * quad_weight(j));
	      }
	    }

//	     std::cout << "first" << std::endl;

//	    for (int i = 0; i < n; i++)
//	    {
//	      navier_stokes_solver.mass_matrix_velocity.vmult(temp, snapshots_vec2[i]);
//	      for (int j = i; j < n; j++)
//	      {
//	        double val = snapshots_vec[j] * temp;
//	        val *= std::sqrt(quad_weight(i) * quad_weight(j));
//	        correlation2_matrix(i, j) = correlation2_matrix(j, i) = val;
//	      }
//	    }
//	    std::cout << (correlation_matrix-correlation2_matrix).norm() << std::endl;

//  	      auto end_time = std::chrono::high_resolution_clock::now();
//  	      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
//  	      std::cout << "MM time:" << std::floor(duration.count() / 1000) << " seconds "
//  	     	                      << duration.count() % 1000 << " milliseconds" << std::endl;

  	     // -----------------------------
  	     // 5. Compute eigenvalues
  	     // -----------------------------
  	    std::cout << "5. Compute eigenvalues" << std::endl;

  	    RedSVD::RedSymEigen<Eigen::MatrixXd> ressymeigen(correlation_matrix,pod_basis_size);
//  	     std::cout << "end redeigen" << std::endl;

//  	     filename_h5 = "result/POD/eigs_red.txt";
//  	     std::ofstream myfile;
//  	     myfile.open(filename_h5);
//  	     myfile << std::setprecision(16) << ressymeigen.eigenvalues();
//  	     myfile.close();

//  	     std::cout << "write pod_vec" << std::endl;
  	     Eigen::VectorXd eigen_values = ressymeigen.eigenvalues();
  	     Eigen::MatrixXd eigen_vectors = ressymeigen.eigenvectors();

//  	     std::cout << ressymeigen.eigenvalues() << std::endl;

  	   // -----------------------------
  	   // 6. Compute eigenvectors
  	   // -----------------------------
  	   std::cout << "6. Compute eigenvectors" << std::endl;
  	     pod_vectors.resize(pod_basis_size);
  	     Vector<double> tmp_eig(n);
    	   Vector<double> tmp_snap(m);
//    	   std::cout << eigen_vectors.rows() << "   " << eigen_vectors.cols() << std::endl;
//    	   std::cout << n << "   " << pod_basis_size << std::endl;
    	     for (int l = 0; l < pod_basis_size; l++) {
    	    	tmp_eig = 0.0;
    	    	for (int i = 0; i< n; i++) {
    	    		tmp_eig(i) = eigen_vectors(i,l)/std::sqrt(eigen_values(l));
    	    	 }
    	    	snapshot_matrix.vmult(tmp_snap,tmp_eig);
    	    	// adapdt l, since ascending order of eigenvalues
    	    	pod_vectors[pod_basis_size-1-l] = tmp_snap;
    	    	filename_h5 = "result/POD/mu="+ std::to_string(viscosity)+"/pod_vectors/pod_vectors"+ Utilities::int_to_string(pod_basis_size-1-l,6) +".h5";
    	    	save_h5_vector(filename_h5, pod_vectors[pod_basis_size-1-l]);
    	     }
//    		  std::cout << "end pod_vec" << std::endl;


  	      r = pod_basis_size;

  	    // -----------------------------
  	    // 7. Compute POD quality
  	    // -----------------------------
  	    std::cout << "7. Compute POD Quality" << std::endl;
  	      double total_information = 0.0;
  	     double partial_information = 0.0;
  	     eigenvalues.resize(r);
  	     Vector<double> tmp(r);
  	      for (int i = 0; i < r; i++) {
  	    	  eigenvalues[i] = eigen_values(r-1-i);
  	    	  tmp(i) = eigenvalues[i];
  	    	  partial_information += eigenvalues[i];
  	      }
  	      filename_h5 = "result/POD/mu="+ std::to_string(viscosity)+"/eigenvalues.h5";
  	      save_h5_vector(filename_h5, tmp);

//  		  Vector<double> tmp_snap(m);

//  	    std::cout << "compute system energy" << std::endl;
  	  for (int i = 0; i < n; i++) {
  		  	  	  navier_stokes_solver.mass_matrix_velocity.vmult(tmp_snap,get_column(snapshot_matrix,i));
//  	 		 	  tmp_snap = get_column(snapshot_matrix,i);
  	 		   	  total_information += tmp_snap.operator*(get_column(snapshot_matrix,i));
  	 		   	  tmp_snap = 0;
  	 	 	 }


  	      double total_information_bound = partial_information + (n-pod_basis_size)*eigenvalues[pod_basis_size-1];

  	      std::cout << std::endl;
  		  std::cout << "Approximation error: " << total_information - partial_information << std::endl;
  		  std::cout << "total:      " << total_information << std::endl;
  		  std::cout << "totalbound: " << total_information_bound << std::endl;
  		  std::cout << "partial:    " << partial_information << std::endl;
  		  std::cout << std::setprecision(10) << "Size od POD basis: " << r
  		            << " (information content: " << partial_information / total_information
  		            << " [goal: " << information_content << "])" << std::endl << std::endl;
      }





  template <int dim>
  void ProperOrthogonalDecomposition<dim>::compute_lifting_function()
  {
//	  std::string snapshot_file_path = pde_info->fem_path + "mu="+ std::to_string(0.1/10) +"/snapshots/";
	  std::string snapshot_file_path = "/home/ifam/fischer/Code/rom-nse/lifting.h5";
//	  std::string snapshot_file_path = "/home/ifam/fischer/Code/rom-nse/mean_vector_mu.h5";
//	  std::string snapshot_file_path = "/media/hendrik/hard_disk/Nextcloud/Code/result/FEM/mu=0.001000/snapshots/snapshot_000000.h5";
//	  mean_vector = load_h5_vector(snapshot_file_path + "snapshot_" + Utilities::int_to_string(0,6) + ".h5");
	  load_h5_vector_HF(snapshot_file_path, mean_vector);
//	  mean_vector = 0.0;
  }


  template <int dim>
  void ProperOrthogonalDecomposition<dim>::output_mean_vector() const
  {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(navier_stokes_solver.dof_handler_velocity);
    data_out.add_data_vector(mean_vector, "velocity");

    data_out.build_patches();

    const std::string filename =
    		pde_info->pod_path +  "mu="+ std::to_string(viscosity) + "/mean_vector.vtk";
    std::ofstream output(filename);
    data_out.write_vtk(output);

    const std::string filename_h5 = pde_info->pod_path + "mu="+ std::to_string(viscosity) + "/mean_vector.h5";
    DataOutBase::DataOutFilter data_filter(DataOutBase::DataOutFilterFlags(/*filter_duplicate_vertices=*/true, /*xdmf_hdf5_output*/true));
    data_out.write_filtered_data(data_filter);
    data_out.write_hdf5_parallel(data_filter, filename_h5, MPI_COMM_WORLD);
  }

  template <int dim>
  void ProperOrthogonalDecomposition<dim>::save_h5_vector(const std::string &file_name, const Vector<double> &vector)
  {
    hid_t file_id = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                              H5P_DEFAULT);

    hsize_t n_dofs[1];
    n_dofs[0] = vector.size();
    hid_t dataspace_id = H5Screate_simple(1, n_dofs, nullptr);
    std::string dataset_name {"/mean_vector"};
    hid_t dataset_id = H5Dcreate2 (file_id, dataset_name.c_str(),
                                   H5T_NATIVE_DOUBLE, dataspace_id,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
             static_cast<const void *>(vector.begin()));
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);
  }

  template <int dim>
  void ProperOrthogonalDecomposition<dim>::load_h5_vector_HF(const std::string &file_name, Vector<double> &vector)
  {
      hid_t file_id = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

      std::string dataset_name = "/mean_vector";
      hid_t dataset = H5Dopen1(file_id, dataset_name.c_str());
      // TODO assert that this is double precision
      hid_t datatype = H5Dget_type(dataset);
      hid_t dataspace = H5Dget_space(dataset);
      int rank = H5Sget_simple_extent_ndims(dataspace);
      (void)rank;
      Assert(rank == 1, StandardExceptions::ExcInternalError());

      // Since rank must be 1...
      hsize_t dims[1];
      hsize_t max_dims[1];
      H5Sget_simple_extent_dims(dataspace, dims, max_dims);
      vector.reinit(dims[0]);
      H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              static_cast<void *>(&(vector[0])));

      H5Sclose(dataspace);
      H5Tclose(datatype);
      H5Dclose(dataset);
      H5Fclose(file_id);
  }

  template<int dim>
  void ProperOrthogonalDecomposition<dim>::save_h5_matrix(const std::string &file_name, const dealii::FullMatrix<double> &matrix)
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

  template <int dim>
  void ProperOrthogonalDecomposition<dim>::output_mean_vector_pressure() const
  {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(navier_stokes_solver.dof_handler_pressure);
    data_out.add_data_vector(mean_vector_p, "pressure");

    data_out.build_patches();

    const std::string filename =
    		pde_info->pod_path + "pod_vectors_pressure/mean_vector.vtk";
    std::ofstream output(filename);
    data_out.write_vtk(output);

    const std::string filename_h5 = pde_info->pod_path + "pod_vectors_pressure/mean_vector.h5";
    DataOutBase::DataOutFilter data_filter(DataOutBase::DataOutFilterFlags(/*filter_duplicate_vertices=*/true, /*xdmf_hdf5_output*/true));
    data_out.write_filtered_data(data_filter);
    data_out.write_hdf5_parallel(data_filter, filename_h5, MPI_COMM_WORLD);
  }

  template <int dim>
  void ProperOrthogonalDecomposition<dim>::output_pod_vectors() const
  {
    for (int i=0; i<r; i++)
    {
      DataOut<dim> data_out;

      data_out.attach_dof_handler(navier_stokes_solver.dof_handler_velocity);
      data_out.add_data_vector(pod_vectors[i], "velocity");

      data_out.build_patches();

      const std::string filename =
        "result/POD/pod_vectors/vtk/vector-" + Utilities::int_to_string(i, 6) + ".vtk";
      std::ofstream output(filename);
      data_out.write_vtk(output);

      const std::string filename_h5 = "result/POD/pod_vectors/h5/vector-" + Utilities::int_to_string(i, 6) + ".h5";
      DataOutBase::DataOutFilter data_filter(DataOutBase::DataOutFilterFlags(/*filter_duplicate_vertices=*/true, /*xdmf_hdf5_output*/true));
      data_out.write_filtered_data(data_filter);
      data_out.write_hdf5_parallel(data_filter, filename_h5, MPI_COMM_WORLD);
    }
  }

  template <int dim>
  void ProperOrthogonalDecomposition<dim>::output_pod_vectors_pressure() const
  {
    for (int i=0; i<r_p; i++)
    {
      DataOut<dim> data_out;

      data_out.attach_dof_handler(navier_stokes_solver.dof_handler_pressure);
      data_out.add_data_vector(pod_vectors_p[i], "pressure");

      data_out.build_patches();

      const std::string filename =
        "result/POD/pod_vectors_pressure/vtk/vector-" + Utilities::int_to_string(i, 6) + ".vtk";
      std::ofstream output(filename);
      data_out.write_vtk(output);

      const std::string filename_h5 = "result/POD/pod_vectors_pressure/h5/vector-" + Utilities::int_to_string(i, 6) + ".h5";
      DataOutBase::DataOutFilter data_filter(DataOutBase::DataOutFilterFlags(/*filter_duplicate_vertices=*/true, /*xdmf_hdf5_output*/true));
      data_out.write_filtered_data(data_filter);
      data_out.write_hdf5_parallel(data_filter, filename_h5, MPI_COMM_WORLD);
    }
  }

  template <int dim>
  void ProperOrthogonalDecomposition<dim>::output_eigenvalues() const
  {
    std::string snapshot_file_path = pde_info->fem_path + "snapshots/";
    // int n = compute_number_snapshots(snapshot_file_path);
    std::ofstream txt_file("result/POD/eigenvalues.txt");

    double total_information = 0.0;
    double partial_information = 0.0;

    for (int i = 0; i < r; i++)
      total_information += eigenvalues[i];
    txt_file << std::setprecision(12) << "Total information: " << total_information << std::endl;

    txt_file << "\n\nIndex,Eigenvalue,PartialEnergy,EnergyRatio" << std::endl;
    for (int i=0; i < r; i++)
    {
      partial_information += eigenvalues[i];
      txt_file << std::setprecision(12) << i+1 << "," << eigenvalues[i] << "," << partial_information << "," << partial_information / total_information << std::endl;
    }

    txt_file.close();
  }

  template <int dim>
  void ProperOrthogonalDecomposition<dim>::output_eigenvalues_pressure() const
  {
    std::string snapshot_file_path = pde_info->fem_path + "snapshots/";
    // int n = compute_number_snapshots(snapshot_file_path);
    std::ofstream txt_file("result/POD/eigenvalues_pressure.txt");

    double total_information = 0.0;
    double partial_information = 0.0;

    for (int i = 0; i < n; i++)
      total_information += eigenvalues_p[i];
    txt_file << std::setprecision(12) << "Total information: " << total_information << std::endl;

    txt_file << "\n\nIndex,Eigenvalue,PartialEnergy,EnergyRatio" << std::endl;
    for (int i=0; i < n; i++)
    {
      partial_information += eigenvalues_p[i];
      txt_file << std::setprecision(12) << i+1 << "," << eigenvalues_p[i] << "," << partial_information << "," << partial_information / total_information << std::endl;
    }

    txt_file.close();
  }

  template<int dim>
  void ProperOrthogonalDecomposition<dim>::compute_reduced_matrices(double time_step, double theta)
  {
    // project matrices from FEM space to POD space
    FullMatrix<double> reduced_mass_matrix = compute_reduced_matrix(navier_stokes_solver.mass_matrix_velocity);

    FullMatrix<double> reduced_laplace_matrix = compute_reduced_matrix(navier_stokes_solver.laplace_matrix_velocity);
    FullMatrix<double> reduced_laplace_matrix_with_transposed_trial_function = compute_reduced_matrix(navier_stokes_solver.laplace_matrix_velocity_with_transposed_trial_function);

    // integral over gamma out
    FullMatrix<double> reduced_boundary_matrix = compute_reduced_matrix(navier_stokes_solver.boundary_matrix_velocity);

    // mean convection terms
    navier_stokes_solver.assemble_first_convection_term_with_mean(mean_vector);
    navier_stokes_solver.assemble_second_convection_term_with_mean(mean_vector);
    FullMatrix<double> reduced_first_convection_matrix_with_mean = compute_reduced_matrix(navier_stokes_solver.first_convection_matrix_velocity_with_mean);
    FullMatrix<double> reduced_second_convection_matrix_with_mean = compute_reduced_matrix(navier_stokes_solver.second_convection_matrix_velocity_with_mean);

    // tensor (list of matrices) for the nonlinearity
    navier_stokes_solver.assemble_nonlinearity_tensor_velocity(pod_vectors);
    reduced_nonlinearity_tensor.resize(r);
    for (int i = 0; i < r; ++i){
        reduced_nonlinearity_tensor[i] = compute_reduced_matrix(navier_stokes_solver.nonlinear_tensor_velocity[i]);
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
    reduced_linear_operator_theta.reinit(r, r);
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
    reduced_linear_operator_one_minus_theta.reinit(r, r);
    reduced_linear_operator_one_minus_theta.add(fluid_density, reduced_mass_matrix);  // ρ * M
    reduced_linear_operator_one_minus_theta.add(- time_step * (1.0 - theta) * fluid_density, reduced_first_convection_matrix_with_mean); // -k(1-θ)ρ * K_1
    reduced_linear_operator_one_minus_theta.add(- time_step * (1.0 - theta) * fluid_density, reduced_second_convection_matrix_with_mean); // -k(1-θ)ρ * K_2
    // paramter dependent
    reduced_linear_operator_one_minus_theta.add(- time_step * (1.0 - theta) * fluid_density * viscosity, reduced_laplace_matrix); // -k(1-θ)μ * L
    reduced_linear_operator_one_minus_theta.add(- time_step * (1.0 - theta) * fluid_density * viscosity, reduced_laplace_matrix_with_transposed_trial_function); // -k(1-θ)μ * L_t
    reduced_linear_operator_one_minus_theta.add(time_step * (1.0 - theta) * fluid_density * viscosity, reduced_boundary_matrix); // k(1-θ)μ * N

    filename_h5 = pde_info->rom_path + "matrices/At.h5";
    save_h5_matrix(filename_h5, reduced_linear_operator_one_minus_theta);

    navier_stokes_solver.assemble_mean_vector_contribution_rhs(mean_vector, time_step);

    reduced_mean_vector_contribution_rhs = compute_reduced_vector(navier_stokes_solver.mean_vector_contribution_rhs);
    filename_h5 = pde_info->rom_path + "matrices/mean_vector_rhs.h5";
    save_h5_vector(filename_h5,reduced_mean_vector_contribution_rhs);

    reduced_system_matrix.reinit(r,r);
    reduced_system_rhs.reinit(r);
    reduced_system_matrix_inverse.reinit(r,r);
  }

  template <int dim>
  FullMatrix<double> ProperOrthogonalDecomposition<dim>::compute_reduced_matrix(SparseMatrix<double> &fem_matrix)
  {
    // int m = navier_stokes_solver.dof_handler.n_dofs(); // m: number of FEM DoFs; r: number of POD basis vectors / POD DoFs
    FullMatrix<double> rom_matrix(r, r);
    rom_matrix = 0.0;

    Vector<double> temp(m);
    for (int j=0; j<r; j++)
    {
      fem_matrix.vmult(temp, pod_vectors[j]);
      for (int i=0; i<r; i++)
      {
        rom_matrix(i, j) += pod_vectors[i] * temp;
      }
    }
    return rom_matrix;
  }

  template <int dim>
  FullMatrix<double> ProperOrthogonalDecomposition<dim>::compute_reduced_matrix_pressure(SparseMatrix<double> &fem_matrix)
  {
    FullMatrix<double> rom_matrix(r_p, r_p);
    rom_matrix = 0.0;

    Vector<double> temp(m_p);
    for (int j=0; j<r_p; j++)
    {
      fem_matrix.vmult(temp, pod_vectors_p[j]);
      for (int i=0; i<r_p; i++)
      {
        rom_matrix(i, j) += pod_vectors_p[i] * temp;
      }
    }
    return rom_matrix;
  }

  template <int dim>
  Vector<double> ProperOrthogonalDecomposition<dim>::compute_reduced_vector(Vector<double> &fem_vector)
  {
    // int m = navier_stokes_solver.dof_handler.n_dofs(); // m: number of FEM DoFs
    //  r = number of POD basis vectors / POD DoFs
    Vector<double> rom_vector(r);

    rom_vector = 0.0;
    std::cout <<"m: " << m << std::endl;
    Vector<double> temp(m);

    for (int i=0; i<r; i++)
    {
      rom_vector[i] = pod_vectors[i] * fem_vector;
    }
    return rom_vector;
  }

  template <int dim>
  Vector<double> ProperOrthogonalDecomposition<dim>::compute_reduced_vector_pressure(Vector<double> &fem_vector)
  {
    Vector<double> rom_vector(r_p);
    rom_vector = 0.0;

    Vector<double> temp(m_p);
    for (int i=0; i<r_p; i++)
    {
      rom_vector[i] = pod_vectors_p[i] * fem_vector;
    }
    return rom_vector;
  }


  template<int dim>
  void ProperOrthogonalDecomposition<dim>::run(int refinements, bool output_files)
  {
    std::cout << "start POD ..." << std::endl;
    setup(pde_info, refinements);

    auto start_time2 = std::chrono::high_resolution_clock::now();
    compute_pod_basis(information_content, pod_basis_size);
    auto end_time2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time2 - start_time2);
    std::cout << "POD time:" << std::floor(duration2.count() / 1000) << " seconds "
    	                      << duration2.count() % 1000 << " milliseconds" << std::endl;
    std::cout << "Done POD" << std::endl << std::endl;

    if (output_files) {
    	std::cout << "1" << std::endl;
        		output_pod_vectors();
    //      	pod.output_pod_vectors_pressure();
        		std::cout << "2" << std::endl;
        		output_eigenvalues();
    //      	pod.output_eigenvalues_pressure();
        	}

    std::cout << "Computing the reduced matrices ..." << std::endl;
    compute_reduced_matrices(pde_info->fine_timestep, navier_stokes_solver.theta); // use: compute_reduced_matrices_linearized(...) if using linearization of NSE

    std::cout << "Done" << std::endl << std::endl;
  }

  template<int dim>
  void ProperOrthogonalDecomposition<dim>::run_greedy(int refinements, bool output_files, bool mean)
  {

    setup(pde_info, refinements);

    auto start_time2 = std::chrono::high_resolution_clock::now();
    if (mean) {
    	compute_pod_basis(information_content, pod_basis_size);
    } else {
    	compute_pod_basis_no_mean(information_content, pod_basis_size);
    }
    auto end_time2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time2 - start_time2);
    std::cout << "POD time: " << std::floor(duration2.count() / 1000) << " seconds "
    	                      << duration2.count() % 1000 << " milliseconds \n" << std::endl;

    if (output_files) {
//        		output_pod_vectors();
    //      	pod.output_pod_vectors_pressure();
//        		output_eigenvalues();
    //      	pod.output_eigenvalues_pressure();
        	}
    std::cout << "Done" << std::endl << std::endl;
  }

  template<int dim>
  void ProperOrthogonalDecomposition<dim>::run_greedy_vp(int refinements, bool output_files)
  {
    std::cout << "start POD ..." << std::endl;
    setup_vp(pde_info, refinements);

    auto start_time2 = std::chrono::high_resolution_clock::now();
    	compute_pod_basis(information_content, pod_basis_size);
    	compute_pod_basis_supremizer(information_content, pod_basis_size);
    	compute_pod_basis_no_mean_press(information_content, pod_basis_size);

    auto end_time2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time2 - start_time2);
    std::cout << "POD time: " << std::floor(duration2.count() / 1000) << " seconds "
    	                      << duration2.count() % 1000 << " milliseconds \n" << std::endl;

    if (output_files) {
//        		output_pod_vectors();
    //      	pod.output_pod_vectors_pressure();
//        		output_eigenvalues();
    //      	pod.output_eigenvalues_pressure();
        	}
    std::cout << "Done" << std::endl << std::endl;
  }


  template class ProperOrthogonalDecomposition<2>;
} // namespace POD
