#include <../include/pod_greedy.h>
//#include <../include/nse.h>

namespace POD_GREEDY
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

	template<int dim>
	PODGreedy<dim>::PODGreedy()
	{

	}

	template<int dim>
	PODGreedy<dim>::PODGreedy(PDEInfo *pde_info)
	: pde_info(pde_info)
	{
		viscosity_tmp = pde_info->viscosity;
	}

	template<int dim>
	void PODGreedy<dim>::setup(int refinements)
	{
		this->refinements = refinements;
		generate_surrogate();

		navier_stokes_solver.init(pde_info);
		// we will need the DoFHandler and the MassMatrix from the Navier Stokes assembly
		navier_stokes_solver.setup_system_only_velocity(refinements);
//		navier_stokes_solver.setup_system_only_pressure(refinements);
		// one could have also solved this with saving/loading of the mass matrix
		std::cout << pde_info->rom_path << std::endl;
//		system(("rm -r " + (pde_info->rom_path + "matrices/")).c_str());
		system(("rm -r " + (pde_info->pod_path + "pod_vectors/")).c_str());
		system(("rm -r " + (pde_info->pod_path + "eigenvalues.h5")).c_str());
		system(("rm -r " + (pde_info->pod_path + "mean_vector.h5")).c_str());
		system(("rm -r " + (pde_info->pod_path + "error_*_greedy.txt")).c_str());
		system(("rm -r " + (pde_info->pod_path + "mu=*")).c_str());
//		std::experimental::filesystem::remove((pde_info->pod_path));
//		std::remove((pde_info->pod_path));
	}

	template<int dim>
	void PODGreedy<dim>::setup_vp(int refinements)
	{
		this->refinements = refinements;
		generate_surrogate();

		navier_stokes_solver.init(pde_info);
		// we will need the DoFHandler and the MassMatrix from the Navier Stokes assembly
		navier_stokes_solver.setup_system_ROM(refinements);
		navier_stokes_solver.assemble_FOM_matrices_aff_decomp();
		// one could have also solved this with saving/loading of the mass matrix
		std::cout << pde_info->rom_path << std::endl;
//		system(("rm -r " + (pde_info->rom_path + "matrices/")).c_str());
		system(("rm -r " + (pde_info->pod_path + "pod_vectors/")).c_str());
		system(("rm -r " + (pde_info->pod_path + "pod_vectors_press/")).c_str());
		system(("rm -r " + (pde_info->pod_path + "pod_vectors_supremizer/")).c_str());
		system(("rm -r " + (pde_info->pod_path + "eigenvalues.h5")).c_str());
		system(("rm -r " + (pde_info->pod_path + "eigenvalues_press.h5")).c_str());
		system(("rm -r " + (pde_info->pod_path + "mean_vector.h5")).c_str());
		system(("rm -r " + (pde_info->pod_path + "error_*_greedy.txt")).c_str());
		system(("rm -r " + (pde_info->pod_path + "mu=*")).c_str());
//		std::experimental::filesystem::remove((pde_info->pod_path));
//		std::remove((pde_info->pod_path));
	}

	template<int dim>
	void PODGreedy<dim>::compute_POD()
	{
		std::cout << "start greedy POD ... (with mean)" << std::endl;
	    std::string snapshot_file_path = pde_info->pod_path + "mu="+ std::to_string(pde_info->viscosity)+ "/pod_vectors/";
	    int n = compute_number_snapshots(snapshot_file_path);
//	    n = r + pde_info->pod_greedy_basis_size;
	    int m = navier_stokes_solver.dof_handler_velocity.n_dofs();
	    std::cout << "\tPresize of RB	: " << r << std::endl
	    		  << "\tNumber of new RB: " << n << std::endl
	              << "\tNumber of DoFs: " << m << std::endl;

//	    std::vector<Vector<double>> snapshots; //new pod-greedy
//	    for (int i = 0; i < int(pod_vectors.size()); i++)
//	    {
//	    	snapshots.push_back(pod_vectors[i]);
//	    }

	    Vector<double> eigenvalues_temp;
	    pod_solver.load_h5_vector_HF(snapshot_file_path + "../eigenvalues.h5",eigenvalues_temp);
	    if (eigenvalues_temp.size()!=n)
	    std::cout << "BULLSHIT WITH WEIGHTS OF TT POD MODES" << std::endl;

//	    n = modes_of_each_para;
	    std::cout << "modes used: " << n << std::endl;
	    Vector<double> temp_load;
	    for (int i = 0; i < n; i++)
	    {
	    	pod_solver.load_h5_vector_HF(snapshot_file_path + "pod_vectors" + Utilities::int_to_string(i,6) + ".h5",temp_load);
//	    	snapshots.push_back(temp_load); //new pod-greedy
//	    	temp_load.equ(eigenvalues_temp[i]/eigenvalues_temp[0],temp_load);
	    	temp_load.equ(eigenvalues_temp[i],temp_load);
	    	bool check = true;
	    	for (int j=0; j < pod_vectors_all_mu.size();j++) {
	    		Vector<double> diff = pod_vectors_all_mu[j];
	    		diff.add(-1.0,temp_load);
	    		if (diff.linfty_norm()==0)
	    			check=false;
	    	}
	    	if (check)
	    		pod_vectors_all_mu.push_back(temp_load);
	    }
//	    n += pod_vectors.size(); //new pod-greedy
	    n = pod_vectors_all_mu.size();

//	    modes_of_each_para += pde_info->pod_greedy_basis_size;

	    std::cout << "\tNumber of overall RB: " << n << std::endl;
	    // ------------------------------
	    // Collect surrogate mean vectors
	    // ------------------------------
//	    Vector<double> mean_vector;
	    pod_solver.load_h5_vector_HF(pde_info->pod_path + "mu=" + std::to_string(pde_info->viscosity) + "/mean_vector.h5",mean_vector);
	    surrogate_mean_vectors.push_back(mean_vector);

	    // ----------------------
	    // 1. Compute mean vector
	    // ----------------------
//	    mean_vector.reinit(m); //new pod-greedy
//	    for (Vector<double> &snapshot : snapshots)
//	      mean_vector.add(0.0, snapshot);

	    // output mean vector to vtk / h5
//	    pod_solver.output_mean_vector();
	    mkdir((pde_info->pod_path).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	    mkdir((pde_info->pod_path + "pod_vectors").c_str()  , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	    mkdir((pde_info->pod_path + "pod_vectors_all_mu").c_str()  , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
//	    filename_h5 = pde_info->pod_path + "/mean_vector.h5";
//	    pod_solver.save_h5_vector(filename_h5, mean_vector);


	    for (int i=0; i<n; i++)
	    	    {
	    	      filename_h5 = pde_info->pod_path + "pod_vectors_all_mu/pod_vectors_all_mu"+ Utilities::int_to_string(i,6) +".h5";
	    	      pod_solver.save_h5_vector(filename_h5, pod_vectors_all_mu[i]);
	    	    }

	    // -----------------------------
	    // 2. Compute correlation matrix
	    // -----------------------------

	    LAPACKFullMatrix<double> correlation_matrix(n);
	    LAPACKFullMatrix<double> identity_matrix(n); // only needed to compute eigenvalue of correlation_matrix
	    identity_matrix = 0.0;
	    Vector<double> temp(m);

	    for (int i = 0; i < n; i++)
	    {
//	      navier_stokes_solver.mass_matrix_velocity.vmult(temp, snapshots[i]); //new pod-greedy
//	      temp = snapshots[i];
	    	navier_stokes_solver.mass_matrix_velocity.vmult(temp, pod_vectors_all_mu[i]); //new pod-greedy
	      temp =  pod_vectors_all_mu[i]; //new pod-greedy
	      for (int j = i; j < n; j++)
	      {
//	        double val = snapshots[j] * temp; //new pod-greedy
	    	double val = pod_vectors_all_mu[j] * temp;
//	        val *= std::sqrt(quad_weight(i) * quad_weight(j));
	        correlation_matrix(i, j) = correlation_matrix(j, i) = val;
	      }
	      identity_matrix(i, i) = 1.0;
	    }

	    // --------------------------------------------
	    // 3. Compute eigenvalues of correlation matrix
	    // --------------------------------------------
	    std::vector<Vector<double>> eigenvectors(n);
	    std::vector<double> eigenvalues(n);
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

	    //  sort eigenvalues and eigenvectors in descending order11
	    std::reverse(eigenvectors.begin(), eigenvectors.end());
	    std::reverse(eigenvalues.begin(), eigenvalues.end());
	    // r = size of the POD basis

	    int r_old = r;
	    std::cout << "r_old = " << r_old << std::endl;
	    r = 0;
	    while (r < n)
	    {
	      partial_information += eigenvalues[r];
	      r++;
//
	      if (r>r_old+pde_info->pod_greedy_basis_size-1)
	    	  break;
	    }

	    std::cout << "\tApproximation error: " << total_information - partial_information << std::endl;
	    std::cout << std::setprecision(10) << "\tSize od POD basis: " << r
	              << " (information content: " << partial_information / total_information
	              << " [goal: " << pde_info->information_content_greedy << "]) \n" << std::endl;

	    temp.reinit(r);
	     for (int i = 0; i < r; i++) {
	   	  temp(i) = eigenvalues[i];
	     }

	     filename_h5 = pde_info->pod_path + "eigenvalues.h5";
	     pod_solver.save_h5_vector(filename_h5, temp);
	    // ----------------------------
	    // 4. Compute POD basis vectors
	    // ----------------------------
	     pod_vectors.resize(r);
	    for (int i=0; i<r; i++)
	    {
	      Vector<double> basis_vector(m);
	      for (int j=0;j<n;j++)
	      {
//	        basis_vector.add(std::sqrt(quad_weight(j)) * eigenvectors[i][j],snapshots[j]);
//	        basis_vector.add(eigenvectors[i][j],snapshots[j]); //new pod-greedy
	    	basis_vector.add(eigenvectors[i][j],pod_vectors_all_mu[j]);
	      }
	      basis_vector *= 1.0/std::sqrt(eigenvalues[i]);
	      pod_vectors[i] = std::move(basis_vector);
	      filename_h5 = pde_info->pod_path + "pod_vectors/pod_vectors"+ Utilities::int_to_string(i,6) +".h5";
	      pod_solver.save_h5_vector(filename_h5, pod_vectors[i]);
	    }
	}
	template<int dim>
		void PODGreedy<dim>::compute_POD_no_mean()
		{
			std::cout << "start greedy POD ..." << std::endl;
		    std::string snapshot_file_path = pde_info->pod_path + "mu="+ std::to_string(pde_info->viscosity)+ "/pod_vectors/";
		    int n = compute_number_snapshots(snapshot_file_path);
		    int m = navier_stokes_solver.dof_handler_velocity.n_dofs();
		    std::cout << "\tPresize of RB	: " << r << std::endl
		    		  << "\tNumber of new RB: " << n << std::endl
		              << "\tNumber of DoFs: " << m << std::endl;

//		    std::vector<Vector<double>> snapshots;
//		    for (int i = 0; i < int(pod_vectors.size()); i++)
//		    {
//		    	snapshots.push_back(pod_vectors[i]);
//		    }


//		    Vector<double> temp_load;
//		    for (int i = 0; i < n; i++)
//		    {
//		    	pod_solver.load_h5_vector_HF(snapshot_file_path + "pod_vectors" + Utilities::int_to_string(i,6) + ".h5",temp_load);
//		    	snapshots.push_back(temp_load);
//		    }
//		    n += pod_vectors.size();



		    Vector<double> temp_load;
		    for (int i = 0; i < n; i++)
		    {
		    	pod_solver.load_h5_vector_HF(snapshot_file_path + "pod_vectors" + Utilities::int_to_string(i,6) + ".h5",temp_load);
	//	    	snapshots.push_back(temp_load); //new pod-greedy
		    	bool check = true;
		    	for (int j=0; j < pod_vectors_all_mu.size();j++) {
		    		Vector<double> diff = pod_vectors_all_mu[j];
		    		diff.add(-1.0,temp_load);
		    		if (diff.linfty_norm()==0)
		    			check=false;
		    	}
		    	if (check)
		    		pod_vectors_all_mu.push_back(temp_load);
		    }
	//	    n += pod_vectors.size(); //new pod-greedy
		    n = pod_vectors_all_mu.size();

		    std::cout << "\tNumber of overall RB: " << n << std::endl;

		    // ----------------------
		    // 1. Compute mean vector
		    // ----------------------
		    mkdir((pde_info->pod_path).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		    mkdir((pde_info->pod_path + "pod_vectors").c_str()  , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

		    // -----------------------------
		    // 2. Compute correlation matrix
		    // -----------------------------

		    LAPACKFullMatrix<double> correlation_matrix(n);
		    LAPACKFullMatrix<double> identity_matrix(n); // only needed to compute eigenvalue of correlation_matrix
		    identity_matrix = 0.0;
		    Vector<double> temp(m);

		    for (int i = 0; i < n; i++)
		    {
//		      navier_stokes_solver.mass_matrix_velocity.vmult(temp, snapshots[i]);
//		      temp = snapshots[i];
		      navier_stokes_solver.mass_matrix_velocity.vmult(temp, pod_vectors_all_mu[i]); //new pod-greedy
		      temp =  pod_vectors_all_mu[i]; //new pod-greedy
		      for (int j = i; j < n; j++)
		      {
//		        double val = snapshots[j] * temp;
		    	double val = pod_vectors_all_mu[j] * temp;
	//	        val *= std::sqrt(quad_weight(i) * quad_weight(j));
		        correlation_matrix(i, j) = correlation_matrix(j, i) = val;
		      }
		      identity_matrix(i, i) = 1.0;
		    }

		    // --------------------------------------------
		    // 3. Compute eigenvalues of correlation matrix
		    // --------------------------------------------
		    std::vector<Vector<double>> eigenvectors(n);
		    std::vector<double> eigenvalues(n);
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

		    //  sort eigenvalues and eigenvectors in descending order11
		    std::reverse(eigenvectors.begin(), eigenvectors.end());
		    std::reverse(eigenvalues.begin(), eigenvalues.end());
		    // r = size of the POD basis

		    int r_old = r;
		    r = 0;
		    while (r < n)
		    {
		      partial_information += eigenvalues[r];
		      r++;
		      // Is POD basis big enough?
//		      if (r>pod_vectors.size() || eigenvalues[r] < 1e-12)
//		      {
//		    	if (partial_information > pde_info->information_content_greedy * total_information \
//		    	   || r == pde_info->pod_greedy_basis_size)
//		    		break;
//		      }
		      if (r>r_old+pde_info->pod_greedy_basis_size-1)
		      	    	  break;

		    }

		    std::cout << "\tApproximation error: " << total_information - partial_information << " of " << total_information << std::endl;
		    std::cout << std::setprecision(10) << "\tSize od POD basis: " << r
		              << " (information content: " << partial_information / total_information
		              << " [goal: " << pde_info->information_content_greedy << "]) \n" << std::endl;

		    temp.reinit(r);
		     for (int i = 0; i < r; i++) {
		   	  temp(i) = eigenvalues[i];
		     }

		     filename_h5 = pde_info->pod_path + "eigenvalues.h5";
		     pod_solver.save_h5_vector(filename_h5, temp);
		    // ----------------------------
		    // 4. Compute POD basis vectors
		    // ----------------------------
		     pod_vectors.resize(r);
		    for (int i=0; i<r; i++)
		    {
		      Vector<double> basis_vector(m);
		      for (int j=0;j<n;j++)
		      {
	//	        basis_vector.add(std::sqrt(quad_weight(j)) * eigenvectors[i][j],snapshots[j]);
//		        basis_vector.add(eigenvectors[i][j],snapshots[j]);
		        basis_vector.add(eigenvectors[i][j],pod_vectors_all_mu[j]);
		      }
		      basis_vector *= 1.0/std::sqrt(eigenvalues[i]);
		      pod_vectors[i] = std::move(basis_vector);
		      filename_h5 = pde_info->pod_path + "pod_vectors/pod_vectors"+ Utilities::int_to_string(i,6) +".h5";
		      pod_solver.save_h5_vector(filename_h5, pod_vectors[i]);
		    }
		}


	template<int dim>
			void PODGreedy<dim>::compute_POD_vp_velo()
			{
				std::cout << "start greedy POD - velocity ..." << std::endl;
			    std::string snapshot_file_path = pde_info->pod_path + "mu="+ std::to_string(pde_info->viscosity)+ "/pod_vectors/";

			    int n = compute_number_snapshots(snapshot_file_path);
//			    int m = navier_stokes_solver.dof_handler_velocity.n_dofs();
			    int m = navier_stokes_solver.mass_matrix_vp.block(0,0).m();
//			    int m_p = navier_stokes_solver.dof_handler_pressure.n_dofs();
			    int m_p = navier_stokes_solver.mass_matrix_vp.block(1,0).m();
			    std::cout << "\tPresize of RB	: " << r_s << std::endl
			    		  << "\tNumber of new RB: " << n << std::endl
			              << "\tNumber of DoFs - velo: " << m << std::endl
						  << "\tNumber of DoFs - pres: " << m_p << std::endl;



			    Vector<double> eigenvalues_temp;
			    pod_solver.load_h5_vector_HF(snapshot_file_path + "../eigenvalues.h5",eigenvalues_temp);
			    if (eigenvalues_temp.size()!=n)
			    	std::cout << "BULLSHIT WITH WEIGHTS OF TT POD MODES" << std::endl;

			    Vector<double> temp_load;
			    for (int i = 0; i < n; i++)
			    {
			    	pod_solver.load_h5_vector_HF(snapshot_file_path + "pod_vectors" + Utilities::int_to_string(i,6) + ".h5",temp_load);
			    	temp_load.equ(eigenvalues_temp[i]/eigenvalues_temp[0],temp_load);
			    	bool check = true;
			    	for (int j=0; j < pod_vectors_all_mu.size();j++) {
			    		Vector<double> diff = pod_vectors_all_mu[j];
			    		diff.add(-1.0,temp_load);
			    		if (diff.linfty_norm()==0)
			    			check=false;
			    	}
			    	if (check)
			    		pod_vectors_all_mu.push_back(temp_load);
			    }
		//	    n += pod_vectors.size(); //new pod-greedy
			    n = pod_vectors_all_mu.size();

			    std::cout << "\tNumber of overall RB: " << n << std::endl;


			    // ------------------------------
			    // Collect surrogate mean vectors
			    // ------------------------------
		//	    Vector<double> mean_vector;
			    pod_solver.load_h5_vector_HF(pde_info->pod_path + "mu=" + std::to_string(pde_info->viscosity) + "/mean_vector.h5",mean_vector);
			    surrogate_mean_vectors.push_back(mean_vector);

			    // ----------------------
			    // 1. Compute mean vector
			    // ----------------------
			    mkdir((pde_info->pod_path).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
			    mkdir((pde_info->pod_path + "pod_vectors").c_str()  , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

			    // -----------------------------
			    // 2. Compute correlation matrix
			    // -----------------------------

			    LAPACKFullMatrix<double> correlation_matrix(n);
			    LAPACKFullMatrix<double> identity_matrix(n); // only needed to compute eigenvalue of correlation_matrix
			    identity_matrix = 0.0;
			    Vector<double> temp(m);

			    for (int i = 0; i < n; i++)
			    {
	//		      navier_stokes_solver.mass_matrix_velocity.vmult(temp, snapshots[i]);
	//		      temp = snapshots[i];
//			      navier_stokes_solver.mass_matrix_velocity.vmult(temp, pod_vectors_all_mu[i]); //new pod-greedy
			      temp =  pod_vectors_all_mu[i]; //new pod-greedy
			      for (int j = i; j < n; j++)
			      {
	//		        double val = snapshots[j] * temp;
			    	double val = pod_vectors_all_mu[j] * temp;
		//	        val *= std::sqrt(quad_weight(i) * quad_weight(j));
			        correlation_matrix(i, j) = correlation_matrix(j, i) = val;
			      }
			      identity_matrix(i, i) = 1.0;
			    }

			    // --------------------------------------------
			    // 3. Compute eigenvalues of correlation matrix
			    // --------------------------------------------
			    std::vector<Vector<double>> eigenvectors(n);
			    std::vector<double> eigenvalues(n);
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

			    //  sort eigenvalues and eigenvectors in descending order11
			    std::reverse(eigenvectors.begin(), eigenvectors.end());
			    std::reverse(eigenvalues.begin(), eigenvalues.end());
			    // r = size of the POD basis

			    int r_old = r_v;
			    r_v = 0;
			    while (r_v < n)
			    {
			      partial_information += eigenvalues[r_v];
			      r_v++;
			      // Is POD basis big enough?
	//		      if (r>pod_vectors.size() || eigenvalues[r] < 1e-12)
	//		      {
	//		    	if (partial_information > pde_info->information_content_greedy * total_information \
	//		    	   || r == pde_info->pod_greedy_basis_size)
	//		    		break;
	//		      }
			      if (r_v>r_old+pde_info->pod_greedy_basis_size-1)
			      	    	  break;

			    }

			    std::cout << "\tApproximation error: " << total_information - partial_information << " of " << total_information << std::endl;
			    std::cout << std::setprecision(10) << "\tSize od POD basis: " << r_v
			              << " (information content: " << partial_information / total_information
			              << " [goal: " << pde_info->information_content_greedy << "]) \n" << std::endl;

			    temp.reinit(r_v);
			     for (int i = 0; i < r_v; i++) {
			   	  temp(i) = eigenvalues[i];
			     }

			     filename_h5 = pde_info->pod_path + "eigenvalues.h5";
			     pod_solver.save_h5_vector(filename_h5, temp);
//			     ----------------------------
//			     4. Compute POD basis vectors
//			     ----------------------------
			     pod_vectors.resize(r_v);
			     pod_vectors_velo.resize(r_v);
			    for (int i=0; i<r_v; i++)
			    {
			      Vector<double> basis_vector(m);
			      for (int j=0;j<n;j++)
			      {
		//	        basis_vector.add(std::sqrt(quad_weight(j)) * eigenvectors[i][j],snapshots[j]);
	//		        basis_vector.add(eigenvectors[i][j],snapshots[j]);
			        basis_vector.add(eigenvectors[i][j],pod_vectors_all_mu[j]);
			      }
			      basis_vector *= 1.0/std::sqrt(eigenvalues[i]);

			      pod_vectors_velo[i].reinit(2);
			      pod_vectors_velo[i].block(0).reinit(m);
			      pod_vectors_velo[i].block(1).reinit(m_p);
			      pod_vectors_velo[i].collect_sizes();
			      pod_vectors_velo[i].block(0) = std::move(basis_vector);
			      pod_vectors_velo[i].block(1) = 0;

//			      pod_vectors[i] = std::move(basis_vector);
			      filename_h5 = pde_info->pod_path + "pod_vectors/pod_vectors"+ Utilities::int_to_string(i,6) +".h5";
			      pod_solver.save_h5_vector(filename_h5, pod_vectors_velo[i].block(0));
			    }
			}

	template<int dim>
			void PODGreedy<dim>::compute_POD_vp_supremizer()
			{
				std::cout << "start greedy POD - supremizer ..." << std::endl;
			    std::string snapshot_file_path = pde_info->pod_path + "mu="+ std::to_string(pde_info->viscosity)+ "/pod_vectors_supremizer/";

			    int n = compute_number_snapshots(snapshot_file_path);
//			    int m = navier_stokes_solver.dof_handler_velocity.n_dofs();
			    int m = navier_stokes_solver.mass_matrix_vp.block(0,0).m();
//			    int m_p = navier_stokes_solver.dof_handler_pressure.n_dofs();
			    int m_p = navier_stokes_solver.mass_matrix_vp.block(1,0).m();

			    std::cout << "\tPresize of RB	: " << r_s << std::endl
			    		  << "\tNumber of new RB: " << n << std::endl
			              << "\tNumber of DoFs - velo: " << m << std::endl
						  << "\tNumber of DoFs - pres: " << m_p << std::endl;


			    Vector<double> eigenvalues_temp;
			    pod_solver.load_h5_vector_HF(snapshot_file_path + "../eigenvalues_supremizer.h5",eigenvalues_temp);
			    if (eigenvalues_temp.size()!=n)
			    	std::cout << "BULLSHIT WITH WEIGHTS OF TT sup POD MODES" << std::endl;

			    Vector<double> temp_load;
			    for (int i = 0; i < n; i++)
			    {
			    	pod_solver.load_h5_vector_HF(snapshot_file_path + "pod_vectors_supremizer_" + Utilities::int_to_string(i,6) + ".h5",temp_load);
			    	temp_load.equ(eigenvalues_temp[i]/eigenvalues_temp[0],temp_load);
			    	bool check = true;
			    	for (int j=0; j < pod_vectors_all_mu_supremizer.size();j++) {
			    		Vector<double> diff = pod_vectors_all_mu_supremizer[j];
			    		diff.add(-1.0,temp_load);
			    		if (diff.linfty_norm()==0)
			    			check=false;
			    	}
			    	if (check)
			    		pod_vectors_all_mu_supremizer.push_back(temp_load);
			    }
		//	    n += pod_vectors.size(); //new pod-greedy
			    n = pod_vectors_all_mu_supremizer.size();

			    std::cout << "\tNumber of overall RB: " << n << std::endl;


			    // ------------------------------
			    // Collect surrogate mean vectors
			    // ------------------------------
		//	    Vector<double> mean_vector;
//			    pod_solver.load_h5_vector_HF(pde_info->pod_path + "mu=" + std::to_string(pde_info->viscosity) + "/mean_vector.h5",mean_vector);
//			    surrogate_mean_vectors.push_back(mean_vector);

			    // ----------------------
			    // 1. Compute mean vector
			    // ----------------------
			    mkdir((pde_info->pod_path).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
			    mkdir((pde_info->pod_path + "pod_vectors_supremizer").c_str()  , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

			    // -----------------------------
			    // 2. Compute correlation matrix
			    // -----------------------------

			    LAPACKFullMatrix<double> correlation_matrix(n);
			    LAPACKFullMatrix<double> identity_matrix(n); // only needed to compute eigenvalue of correlation_matrix
			    identity_matrix = 0.0;
			    Vector<double> temp(m);

			    for (int i = 0; i < n; i++)
			    {
	//		      navier_stokes_solver.mass_matrix_velocity.vmult(temp, snapshots[i]);
	//		      temp = snapshots[i];
//			      navier_stokes_solver.mass_matrix_velocity.vmult(temp, pod_vectors_all_mu[i]); //new pod-greedy
			      temp =  pod_vectors_all_mu_supremizer[i]; //new pod-greedy
			      for (int j = i; j < n; j++)
			      {
	//		        double val = snapshots[j] * temp;
			    	double val = pod_vectors_all_mu_supremizer[j] * temp;
		//	        val *= std::sqrt(quad_weight(i) * quad_weight(j));
			        correlation_matrix(i, j) = correlation_matrix(j, i) = val;
			      }
			      identity_matrix(i, i) = 1.0;
			    }

			    // --------------------------------------------
			    // 3. Compute eigenvalues of correlation matrix
			    // --------------------------------------------
			    std::vector<Vector<double>> eigenvectors(n);
			    std::vector<double> eigenvalues(n);
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

			    //  sort eigenvalues and eigenvectors in descending order11
			    std::reverse(eigenvectors.begin(), eigenvectors.end());
			    std::reverse(eigenvalues.begin(), eigenvalues.end());
			    // r = size of the POD basis

			    int r_old = r_s;
			    r_s = 0;
			    while (r_s < n)
			    {
			      partial_information += eigenvalues[r_s];
			      r_s++;
			      // Is POD basis big enough?
	//		      if (r>pod_vectors.size() || eigenvalues[r] < 1e-12)
	//		      {
	//		    	if (partial_information > pde_info->information_content_greedy * total_information \
	//		    	   || r == pde_info->pod_greedy_basis_size)
	//		    		break;
	//		      }
			      if (r_s>r_old+pde_info->pod_greedy_basis_pressure_size-1)
			      	    	  break;
			    }

			    std::cout << "\tApproximation error: " << total_information - partial_information << " of " << total_information << std::endl;
			    std::cout << std::setprecision(10) << "\tSize od POD basis: " << r_s
			              << " (information content: " << partial_information / total_information
			              << " [goal: " << pde_info->information_content_greedy << "]) \n" << std::endl;

			    temp.reinit(r_s);
			     for (int i = 0; i < r_s; i++) {
			   	  temp(i) = eigenvalues[i];
			     }
			     filename_h5 = pde_info->pod_path + "eigenvalues_supremizer.h5";
			     pod_solver.save_h5_vector(filename_h5, temp);
//			     ----------------------------
//			     4. Compute POD basis vectors
//			     ----------------------------
//			     pod_vectors.resize(r_s);
			     pod_vectors_supremizer.resize(r_s);
			    for (int i=0; i<r_s; i++)
			    {
			      Vector<double> basis_vector(m);
			      for (int j=0;j<n;j++)
			      {
		//	        basis_vector.add(std::sqrt(quad_weight(j)) * eigenvectors[i][j],snapshots[j]);
	//		        basis_vector.add(eigenvectors[i][j],snapshots[j]);
			        basis_vector.add(eigenvectors[i][j],pod_vectors_all_mu_supremizer[j]);
			      }
			      basis_vector *= 1.0/std::sqrt(eigenvalues[i]);
			      pod_vectors_supremizer[i].reinit(2);
			      pod_vectors_supremizer[i].block(0).reinit(m);
			      pod_vectors_supremizer[i].block(1).reinit(m_p);
			      pod_vectors_supremizer[i].collect_sizes();
			      pod_vectors_supremizer[i].block(0) = std::move(basis_vector);
			      pod_vectors_supremizer[i].block(1) = 0;

//			      pod_vectors[i] = std::move(basis_vector);
			      filename_h5 = pde_info->pod_path + "pod_vectors_supremizer/pod_vectors_supremizer"+ Utilities::int_to_string(i,6) +".h5";
			      pod_solver.save_h5_vector(filename_h5, pod_vectors_supremizer[i].block(0));
			    }
			}

	template<int dim>
			void PODGreedy<dim>::compute_POD_vp_press()
			{
				std::cout << "start greedy POD - pressure ..." << std::endl;
			    std::string snapshot_file_path = pde_info->pod_path + "mu="+ std::to_string(pde_info->viscosity)+ "/pod_vectors_press/";

			    int n = compute_number_snapshots(snapshot_file_path);
			    //			    int m = navier_stokes_solver.dof_handler_velocity.n_dofs();
			    			    int m_v = navier_stokes_solver.mass_matrix_vp.block(0,0).m();
			    //			    int m_p = navier_stokes_solver.dof_handler_pressure.n_dofs();
			    			    int m = navier_stokes_solver.mass_matrix_vp.block(1,0).m();
			    std::cout << "\tPresize of RB	: " << r_p << std::endl
			    		  << "\tNumber of new RB: " << n << std::endl
			              << "\tNumber of DoFs - velo: " << m_v << std::endl
						  << "\tNumber of DoFs - pres: " << m << std::endl;

	//		    std::vector<Vector<double>> snapshots;
	//		    for (int i = 0; i < int(pod_vectors.size()); i++)
	//		    {
	//		    	snapshots.push_back(pod_vectors[i]);
	//		    }


	//		    Vector<double> temp_load;
	//		    for (int i = 0; i < n; i++)
	//		    {
	//		    	pod_solver.load_h5_vector_HF(snapshot_file_path + "pod_vectors" + Utilities::int_to_string(i,6) + ".h5",temp_load);
	//		    	snapshots.push_back(temp_load);
	//		    }
	//		    n += pod_vectors.size();


			    Vector<double> eigenvalues_temp;
			    pod_solver.load_h5_vector_HF(snapshot_file_path + "../eigenvalues_press.h5",eigenvalues_temp);
			    if (eigenvalues_temp.size()!=n)
			    	std::cout << "BULLSHIT WITH WEIGHTS OF TT press POD MODES" << std::endl;

			    Vector<double> temp_load;
			    for (int i = 0; i < n; i++)
			    {
			    	pod_solver.load_h5_vector_HF(snapshot_file_path + "pod_vectors_press" + Utilities::int_to_string(i,6) + ".h5",temp_load);
			    	temp_load.equ(eigenvalues_temp[i]/eigenvalues_temp[0],temp_load);
			    	bool check = true;
			    	for (int j=0; j < pod_vectors_all_mu_press.size();j++) {
			    		Vector<double> diff = pod_vectors_all_mu_press[j];
			    		diff.add(-1.0,temp_load);
			    		if (diff.linfty_norm()==0)
			    			check=false;
			    	}
			    	if (check)
			    		pod_vectors_all_mu_press.push_back(temp_load);
			    }
		//	    n += pod_vectors.size(); //new pod-greedy
			    n = pod_vectors_all_mu_press.size();

			    std::cout << "\tNumber of overall RB: " << n << std::endl;

			    // ----------------------
			    // 1. Compute mean vector
			    // ----------------------
			    mkdir((pde_info->pod_path).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
			    mkdir((pde_info->pod_path + "pod_vectors_press").c_str()  , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

			    // -----------------------------
			    // 2. Compute correlation matrix
			    // -----------------------------

			    LAPACKFullMatrix<double> correlation_matrix(n);
			    LAPACKFullMatrix<double> identity_matrix(n); // only needed to compute eigenvalue of correlation_matrix
			    identity_matrix = 0.0;
			    Vector<double> temp(m);

			    for (int i = 0; i < n; i++)
			    {
	//		      navier_stokes_solver.mass_matrix_velocity.vmult(temp, snapshots[i]);
	//		      temp = snapshots[i];
//			      navier_stokes_solver.mass_matrix_pressure.vmult(temp, pod_vectors_all_mu_press[i]); //new pod-greedy
			      temp =  pod_vectors_all_mu_press[i]; //new pod-greedy
			      for (int j = i; j < n; j++)
			      {
	//		        double val = snapshots[j] * temp;
			    	double val = pod_vectors_all_mu_press[j] * temp;
		//	        val *= std::sqrt(quad_weight(i) * quad_weight(j));
			        correlation_matrix(i, j) = correlation_matrix(j, i) = val;
			      }
			      identity_matrix(i, i) = 1.0;
			    }

			    // --------------------------------------------
			    // 3. Compute eigenvalues of correlation matrix
			    // --------------------------------------------
			    std::vector<Vector<double>> eigenvectors(n);
			    std::vector<double> eigenvalues(n);
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

			    //  sort eigenvalues and eigenvectors in descending order11
			    std::reverse(eigenvectors.begin(), eigenvectors.end());
			    std::reverse(eigenvalues.begin(), eigenvalues.end());
			    // r = size of the POD basis

			    int r_old = r_p;
			    r_p = 0;
			    while (r_p < n)
			    {
			      partial_information += eigenvalues[r_p];
			      r_p++;
			      // Is POD basis big enough?
	//		      if (r>pod_vectors.size() || eigenvalues[r] < 1e-12)
	//		      {
	//		    	if (partial_information > pde_info->information_content_greedy * total_information \
	//		    	   || r == pde_info->pod_greedy_basis_size)
	//		    		break;
	//		      }
			      if (r_p>r_old+pde_info->pod_greedy_basis_pressure_size-1)
			      	    	  break;

			    }

			    std::cout << "\tApproximation error: " << total_information - partial_information << " of " << total_information << std::endl;
			    std::cout << std::setprecision(10) << "\tSize od POD basis: " << r_p
			              << " (information content: " << partial_information / total_information
			              << " [goal: " << pde_info->information_content_greedy << "]) \n" << std::endl;

			    temp.reinit(r_p);
			     for (int i = 0; i < r_p; i++) {
			   	  temp(i) = eigenvalues[i];
			     }

			     filename_h5 = pde_info->pod_path + "eigenvalues_press.h5";
			     pod_solver.save_h5_vector(filename_h5, temp);
//			     ----------------------------
//			     4. Compute POD basis vectors
//			     ----------------------------
			     pod_vectors_press.resize(r_p);
			    for (int i=0; i<r_p; i++)
			    {
			      Vector<double> basis_vector(m);
			      for (int j=0;j<n;j++)
			      {
		//	        basis_vector.add(std::sqrt(quad_weight(j)) * eigenvectors[i][j],snapshots[j]);
	//		        basis_vector.add(eigenvectors[i][j],snapshots[j]);
			        basis_vector.add(eigenvectors[i][j],pod_vectors_all_mu_press[j]);
			      }
			      basis_vector *= 1.0/std::sqrt(eigenvalues[i]);

			      pod_vectors_press[i].reinit(2);
			      pod_vectors_press[i].block(0).reinit(m_v);
			      pod_vectors_press[i].block(1).reinit(m);
			      pod_vectors_press[i].collect_sizes();
			      pod_vectors_press[i].block(0) = 0;
			      pod_vectors_press[i].block(1) = std::move(basis_vector);

			      filename_h5 = pde_info->pod_path + "pod_vectors_press/pod_vectors_press"+ Utilities::int_to_string(i,6) +".h5";
			      pod_solver.save_h5_vector(filename_h5, pod_vectors_press[i].block(1));
			    }
			}

	template<int dim> void PODGreedy<dim>::assemble_velo_supr_modes()
	{
		r_s = r_p;
		r_vs = r_v+r_s;
		std::cout << "Build V^{u+s}  with r_v + r_s = " << r_v << " + " << r_s << std::endl;
		pod_vectors_velo_supr.resize(r_vs);

		for (int i = 0; i<r_v; i++)
		{
			pod_vectors_velo_supr[i] = pod_vectors_velo[i];
		}
		for (int i = r_v; i<r_v+r_s; i++)
		{
			pod_vectors_velo_supr[i] = pod_vectors_supremizer[i-r_v];
//			pod_vectors_velo_supr[i].block(0).equ(0.1,pod_vectors_velo_supr[i].block(0));
		    filename_h5 = pde_info->pod_path + "pod_vectors/pod_vectors"+ Utilities::int_to_string(i,6) +".h5";
		    pod_solver.save_h5_vector(filename_h5, pod_vectors_velo_supr[i].block(0));
		}
	}
	  // TODO:[WB] I don't think that the optimized storage of diagonals is needed
	  // (GK)
	template<int dim>
		void PODGreedy<dim>::apply_boundary_values_rhs(const std::map<types::global_dof_index, double> &boundary_values,
																SparseMatrix<double> &matrix,
																Vector<double> &right_hand_side)
	  {
	    Assert(matrix.n() == right_hand_side.size(),
	           ExcDimensionMismatch(matrix.n(), right_hand_side.size()));
	    Assert(matrix.n() == matrix.m(),
	           ExcDimensionMismatch(matrix.n(), matrix.m()));

	    // if no boundary values are to be applied
	    // simply return
	    if (boundary_values.size() == 0)
	      return;


	    const types::global_dof_index n_dofs = matrix.m();

	    // if a diagonal entry is zero
	    // later, then we use another
	    // number instead. take it to be
	    // the first nonzero diagonal
	    // element of the matrix, or 1 if
	    // there is no such thing
	    double first_nonzero_diagonal_entry = 1;
	    for (unsigned int i = 0; i < n_dofs; ++i)
	      if (matrix.diag_element(i) != double())
	        {
	          first_nonzero_diagonal_entry = matrix.diag_element(i);
	          break;
	        }


	    typename std::map<types::global_dof_index, double>::const_iterator
	      dof  = boundary_values.begin(),
	      endd = boundary_values.end();
	    for (; dof != endd; ++dof)
	      {
	        Assert(dof->first < n_dofs, ExcInternalError());

	        const types::global_dof_index dof_number = dof->first;
	        // for each boundary dof:


	        // set right hand side to
	        // wanted value: if main diagonal
	        // entry nonzero, don't touch it
	        // and scale rhs accordingly. If
	        // zero, take the first main
	        // diagonal entry we can find, or
	        // one if no nonzero main diagonal
	        // element exists. Normally, however,
	        // the main diagonal entry should
	        // not be zero.
	        //
	        // store the new rhs entry to make
	        // the gauss step more efficient
	        double new_rhs;
	        if (matrix.diag_element(dof_number) != double())
	          {
	            new_rhs = dof->second * matrix.diag_element(dof_number);
	            right_hand_side(dof_number) = new_rhs;
	          }
	        else
	          {
	            new_rhs = dof->second * first_nonzero_diagonal_entry;
	            right_hand_side(dof_number) = new_rhs;
	          }
	      }
	  }

	template<int dim>
	void PODGreedy<dim>::assemble_supremizer(double time)
	{
		mkdir((pde_info->fem_path + "mu="+ std::to_string(pde_info->viscosity)+"/solution_supremizer").c_str() , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		mkdir((pde_info->fem_path + "mu="+ std::to_string(pde_info->viscosity)+"/snapshot_supremizer").c_str() , S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	    Vector<double> supremizer_rhs;
	    supremizer_rhs.reinit(navier_stokes_solver.dofs_velocity);
	    Vector<double> supremizer;
	    supremizer.reinit(navier_stokes_solver.dofs_velocity);
	    M_supremizer.reinit(navier_stokes_solver.mass_matrix_vp.block(0,0).get_sparsity_pattern());
	    M_supremizer.copy_from(navier_stokes_solver.mass_matrix_vp.block(0,0));


	    B_supremizer.reinit(navier_stokes_solver.pressure_matrix_vp.block(0,1).get_sparsity_pattern());
		B_supremizer.copy_from(navier_stokes_solver.pressure_matrix_vp.block(0,1));

	    std::map<unsigned int, double> boundary_values;
	    boundary_values_vector.clear();
	    std::vector<bool> component_mask(dim + 1, true);
	    // Ignore pressure
	    component_mask[dim] = false;

	    // parabolic inflow
	    VectorTools::interpolate_boundary_values(navier_stokes_solver.dof_handler,
	                                             0,
												 BoundaryParabel<dim>(0),
	                                             boundary_values,
	                                             component_mask);
	    boundary_values_vector.push_back(boundary_values);
	    MatrixTools::apply_boundary_values(boundary_values,
	    									M_supremizer,
											supremizer,
											supremizer_rhs);
	    // apply homogeneous Dirichlet BC to velocity
	    for (int boundary_id : {2, 3, 80})
	    {
	      VectorTools::interpolate_boundary_values(navier_stokes_solver.dof_handler,
	                                               boundary_id,
	                                               ZeroFunction<dim>(dim + 1),
	                                               boundary_values,
	                                               component_mask);
	      boundary_values_vector.push_back(boundary_values);
		   MatrixTools::apply_boundary_values(boundary_values,
		    									M_supremizer,
												supremizer,
												supremizer_rhs);
	    }

	    M_supremizer_direct.factorize(M_supremizer);
	}

	template<int dim>
	void PODGreedy<dim>::compute_supremizer()
	{
		std::cout << "start supremizer ... " << std::endl;

		assemble_supremizer(0);

		std::string snapshot_file_path = pde_info->fem_path + "mu="+ std::to_string(pde_info->viscosity) +"/snapshots/";
		int n = compute_number_snapshots(snapshot_file_path);

	    Vector<double> pressure_snapshot;
	    std::vector<Vector<double>> pressure_snapshot_vec(n);
	    std::vector<Vector<double>> supremizer_vec(n);
	    for (int i = 0; i<n; i++)
	    {
	    	pressure_snapshot_vec[i] = load_h5_vector_pressure(snapshot_file_path + "snapshot_" + Utilities::int_to_string(i,6) + ".h5");
//	    	supremizer_vec[i].reinit(2);
//	    	supremizer_vec[i].block(0).reinit(navier_stokes_solver.mass_matrix_vp.block(0,0).m());
//	    	supremizer_vec[i].block(1).reinit(navier_stokes_solver.mass_matrix_vp.block(1,0).m());
//	    	supremizer_vec[i].collect_sizes();
	    }
		Threads::TaskGroup<void> task_group;
	    for (int i = 0; i<n; i++)
	    {
	    	task_group += Threads::new_task (&PODGreedy<dim>::compute_supremizer_each, *this, i,pressure_snapshot_vec[i], supremizer_vec[i]);
//	    	compute_supremizer_each(i);
	    }
		task_group.join_all ();

	    BlockVector<double> supremizer_full;
	    supremizer_full.reinit(2);
	    supremizer_full.block(0).reinit(navier_stokes_solver.mass_matrix_vp.block(0,0).m());
	    supremizer_full.block(1).reinit(navier_stokes_solver.mass_matrix_vp.block(1,0).m());
	    supremizer_full.collect_sizes();
	    for (int i = 0; i<n; i++)
	    {
	    			    filename_h5 = pde_info->fem_path + "mu="+ std::to_string(pde_info->viscosity)+"/snapshot_supremizer/supremizer_"+ Utilities::int_to_string(i, 6) +".h5";
	    			    pod_solver.save_h5_vector(filename_h5, supremizer_vec[i]);
	    			    supremizer_full.block(0)= supremizer_vec[i];
	    			    supremizer_full.block(1)= pressure_snapshot_vec[i];

	    			    std::vector<std::string> supremizer_names;
	    			    supremizer_names.push_back("supremizer_full_0");
	    			    supremizer_names.push_back("supremizer_full_1");
	    			    supremizer_names.push_back("pressure");
	    			    std::vector<DataComponentInterpretation::DataComponentInterpretation>
	    			    	data_component_interpretation(dim + 1, DataComponentInterpretation::component_is_scalar);
	    			    DataOut<dim> data_out;
	    			    data_out.attach_dof_handler(navier_stokes_solver.dof_handler);
	    			    data_out.add_data_vector(supremizer_full, supremizer_names,
	    			                            DataOut<dim>::type_dof_data,
	    			                            data_component_interpretation);
	    			    data_out.build_patches();

	    			    const std::string filename = pde_info->fem_path + "mu="+ std::to_string(pde_info->viscosity)+"/solution_supremizer/supremizer_"+ Utilities::int_to_string(i, 6)  +".vtk";
	    			    std::ofstream output(filename);
	    			    data_out.write_vtk(output);
	    }
	}

	template<int dim>
	void PODGreedy<dim>::compute_supremizer_each(int i, Vector<double> &pressure_snapshot, Vector<double> &output)
	{
//		std::cout << "start supremizer ... " << std::endl;

//		assemble_supremizer(0);
//		std::string snapshot_file_path = pde_info->fem_path + "mu="+ std::to_string(pde_info->viscosity) +"/snapshots/";
//		int n = compute_number_snapshots(snapshot_file_path);

		Vector<double> supremizer;
	    Vector<double> supremizer_rhs;


//	    for (int i = 0; i<n; i++)
//	    {
//	    	std::cout << "start supremizer: " << i << std::endl;
//	    	pressure_snapshot = load_h5_vector_pressure(snapshot_file_path + "snapshot_" + Utilities::int_to_string(i,6) + ".h5");
	    	supremizer.reinit(navier_stokes_solver.dofs_velocity);
			supremizer_rhs.reinit(navier_stokes_solver.dofs_velocity);
	    	B_supremizer.vmult(supremizer_rhs,pressure_snapshot);

	    	auto start_time = std::chrono::high_resolution_clock::now();

		    for (int boundary_number = 0; boundary_number<4;boundary_number++)
		    {
		    	apply_boundary_values_rhs(boundary_values_vector[boundary_number],
		    				    									M_supremizer,
		    														supremizer_rhs);
		    }


//		    auto end_time = std::chrono::high_resolution_clock::now();
//		    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
		    M_supremizer_direct.vmult(supremizer, supremizer_rhs);
//		    auto end_time2 = std::chrono::high_resolution_clock::now();
//		    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time2 - end_time);
//		    std::cout << "start supremizer: " << i << std::endl;
//		    std::cout<< std::floor(duration.count() / 1000) << "s " << duration.count() % 1000 << " ms" <<std::endl;
//		    std::cout<< std::floor(duration2.count() / 1000) << "s " << duration2.count() % 1000 << " ms" <<std::endl;


		    output = supremizer;


//		    filename_h5 = pde_info->fem_path + "mu="+ std::to_string(pde_info->viscosity)+"/snapshot_supremizer/supremizer_"+ Utilities::int_to_string(i, 6) +".h5";
//		    pod_solver.save_h5_vector(filename_h5, supremizer);
//		    supremizer_full.block(0)= supremizer;
//		    supremizer_full.block(1)= pressure_snapshot;
//
//		    std::vector<std::string> supremizer_names;
//		    supremizer_names.push_back("supremizer_full_0");
//		    supremizer_names.push_back("supremizer_full_1");
//		    supremizer_names.push_back("pressure");
//		    std::vector<DataComponentInterpretation::DataComponentInterpretation>
//		    	data_component_interpretation(dim + 1, DataComponentInterpretation::component_is_scalar);
//		    DataOut<dim> data_out;
//		    data_out.attach_dof_handler(navier_stokes_solver.dof_handler);
//		    data_out.add_data_vector(supremizer_full, supremizer_names,
//		                            DataOut<dim>::type_dof_data,
//		                            data_component_interpretation);
//		    data_out.build_patches();
//
//		    const std::string filename = pde_info->fem_path + "mu="+ std::to_string(pde_info->viscosity)+"/solution_supremizer/supremizer_"+ Utilities::int_to_string(i, 6)  +".vtk";
//		    std::ofstream output(filename);
//		    data_out.write_vtk(output);
//	    }
	}

	template<int dim>
	void PODGreedy<dim>::compute_mean_vector()
	{
		mean_vector.reinit(surrogate_mean_vectors[0].size());
		for (Vector<double> &vector : surrogate_mean_vectors)
			mean_vector.add(1.0, vector);
		mean_vector*= 1.0/surrogate_mean_vectors.size();   //.scale(1.0);

		int amount = 0;
		Vector<double> test;
		Vector<double> mean_vector_cleaned;
		mean_vector_cleaned.reinit(surrogate_mean_vectors[0].size());
		for (int i=0; i<surrogate_mean_vectors.size(); i++) {
			int j = 0;
			for (j = 0; j<i; j++) {
				if (surrogate_mean_vectors[i] == surrogate_mean_vectors[j]) {
					break;
				}
			}
			if (j==i) {
				mean_vector_cleaned.add(1.0,surrogate_mean_vectors[i]);
				amount += 1;
			}
		}
		mean_vector_cleaned*=1.0/amount;
		test = mean_vector_cleaned;
		test.add(-1.0,mean_vector);

		mean_vector = mean_vector_cleaned;

		lifting_vector.reinit(2);
		lifting_vector.block(0).reinit(navier_stokes_solver.dofs_velocity);
		lifting_vector.block(1).reinit(navier_stokes_solver.dofs_pressure);
		lifting_vector.collect_sizes();
		lifting_vector.block(0) = mean_vector;
		lifting_vector.block(1) = 0.0;
//		std::cout << navier_stokes_solver.dofs_velocity << " + " << navier_stokes_solver.dofs_pressure << std::endl;
//		std::cout << lifting_vector.size() << " + " << lifting_vector.block(0).size() << std::endl;
//
//		std::cout << "mean: " << amount << ", " << (test).norm_sqr() << std::endl;
	    std::string filename_h5 = pde_info->pod_path + "mean_vector.h5";
	    pod_solver.save_h5_vector(filename_h5, mean_vector);
	}

	template<int dim>
	void PODGreedy<dim>::compute_reduced_matrices(double time_step)
	  {
		system(("rm -r " + (pde_info->rom_path + "matrices/")).c_str());
		mkdir((pde_info->rom_path).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		mkdir((pde_info->rom_path + "matrices").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
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

	    std::vector<FullMatrix<double>> reduced_nonlinearity_tensor;
	    // tensor (list of matrices) for the nonlinearity
	    navier_stokes_solver.assemble_nonlinearity_tensor_velocity(pod_vectors);
	    reduced_nonlinearity_tensor.resize(r);
	    for (int i = 0; i < r; ++i){
	        reduced_nonlinearity_tensor[i] = compute_reduced_matrix(navier_stokes_solver.nonlinear_tensor_velocity[i]);
	        filename_h5 = pde_info->rom_path + "matrices/nonlinearity-" + Utilities::int_to_string(i, 6) + ".h5";
	        pod_solver.save_h5_matrix(filename_h5, reduced_nonlinearity_tensor[i]);
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

	    FullMatrix<double> reduced_linear_operator_theta;
	    FullMatrix<double> reduced_linear_operator_one_minus_theta;

	    // A(θ) := ...
	    // parameter independent
	    reduced_linear_operator_theta.reinit(r, r);
	    reduced_linear_operator_theta.add(1.0, reduced_mass_matrix);  // M
	    filename_h5 = pde_info->rom_path + "matrices/M.h5";
	    pod_solver.save_h5_matrix(filename_h5, reduced_linear_operator_theta);

	    reduced_linear_operator_theta.reinit(r, r);
	    reduced_linear_operator_theta.add(1.0, reduced_first_convection_matrix_with_mean); // K_1
	    filename_h5 = pde_info->rom_path + "matrices/K1.h5";
	    pod_solver.save_h5_matrix(filename_h5, reduced_linear_operator_theta);

	   	reduced_linear_operator_theta.reinit(r, r);
	    reduced_linear_operator_theta.add(1.0, reduced_second_convection_matrix_with_mean); // K_2
	    filename_h5 = pde_info->rom_path + "matrices/K2.h5";
	    pod_solver.save_h5_matrix(filename_h5, reduced_linear_operator_theta);

	    // parameter dependent
	    reduced_linear_operator_theta.reinit(r, r);
	    reduced_linear_operator_theta.add(1.0, reduced_laplace_matrix); // kθμ * L
	    filename_h5 = pde_info->rom_path + "matrices/L.h5";
	    pod_solver.save_h5_matrix(filename_h5, reduced_linear_operator_theta);

	    reduced_linear_operator_theta.reinit(r, r);
	    reduced_linear_operator_theta.add(1.0, reduced_laplace_matrix_with_transposed_trial_function); // kθμ * L_t
	    filename_h5 = pde_info->rom_path + "matrices/LT.h5";
	    pod_solver.save_h5_matrix(filename_h5, reduced_linear_operator_theta);

	    reduced_linear_operator_theta.reinit(r, r);
	    reduced_linear_operator_theta.add(1.0, reduced_boundary_matrix); // - kθμ * N
	    filename_h5 = pde_info->rom_path + "matrices/B.h5";
	    pod_solver.save_h5_matrix(filename_h5, reduced_linear_operator_theta);

	    navier_stokes_solver.assemble_mean_vector_contribution_rhs(mean_vector, time_step);
	    std::vector<Vector<double>> mean_vector_contribution_rhs_vector;
	    navier_stokes_solver.assemble_mean_vector_contribution_rhs_greedy(mean_vector,mean_vector_contribution_rhs_vector);

	    Vector<double> error = navier_stokes_solver.mean_vector_contribution_rhs;
	    error.add(-1*time_step*pde_info->fluid_density*pde_info->viscosity	,mean_vector_contribution_rhs_vector[0]);
	    error.add(-1*time_step*pde_info->fluid_density						,mean_vector_contribution_rhs_vector[1]);
	    error.add(-1*time_step*pde_info->fluid_density*pde_info->viscosity	,mean_vector_contribution_rhs_vector[2]);

//	    std::cout << "error: " << error.linfty_norm() << std::endl;

	    Vector<double> reduced_mean_vector_contribution_rhs;
	    std::vector<Vector<double>> reduced_mean_vector_contribution_rhs_vector(3);

	    reduced_mean_vector_contribution_rhs = compute_reduced_vector(navier_stokes_solver.mean_vector_contribution_rhs);

	    reduced_mean_vector_contribution_rhs_vector[0] = compute_reduced_vector(mean_vector_contribution_rhs_vector[0]);
	    reduced_mean_vector_contribution_rhs_vector[1] = compute_reduced_vector(mean_vector_contribution_rhs_vector[1]);
	    reduced_mean_vector_contribution_rhs_vector[2] = compute_reduced_vector(mean_vector_contribution_rhs_vector[2]);

	    filename_h5 = pde_info->rom_path + "matrices/mean_vector_rhs.h5";
	    pod_solver.save_h5_vector(filename_h5,reduced_mean_vector_contribution_rhs);
	    filename_h5 = pde_info->rom_path + "matrices/mean_vector_rhs_vector_0.h5";
	    pod_solver.save_h5_vector(filename_h5,reduced_mean_vector_contribution_rhs_vector[0]);
	    filename_h5 = pde_info->rom_path + "matrices/mean_vector_rhs_vector_1.h5";
	    pod_solver.save_h5_vector(filename_h5,reduced_mean_vector_contribution_rhs_vector[1]);
	    filename_h5 = pde_info->rom_path + "matrices/mean_vector_rhs_vector_2.h5";
	    pod_solver.save_h5_vector(filename_h5,reduced_mean_vector_contribution_rhs_vector[2]);




//	    Vector<double> temp_vector(21024);
//	    filename_h5 = "/home/ifam/fischer/Code/result/FEM/mu=0.001000/snapshots/snapshot_" + Utilities::int_to_string(0,6) + ".h5";
//	    temp_vector = load_h5_vector(filename_h5);
//	    temp_load.add(-1.,mean_vector);


	    //testing with u in reduced space instead of meaaaaaaaaaaaaaaaan_value
	    std::vector<Vector<double>> temp_vector_contribution_rhs_vector;
	    Vector<double> L_eval_at_temp_load(21024);
	    Vector<double> LT_eval_at_temp_load(21024);
//	    navier_stokes_solver.assemble_mean_vector_contribution_rhs_greedy(mean_vector,temp_vector_contribution_rhs_vector);

	    navier_stokes_solver.laplace_matrix_velocity.vmult(L_eval_at_temp_load,mean_vector);
	    navier_stokes_solver.laplace_matrix_velocity_with_transposed_trial_function.vmult(LT_eval_at_temp_load,mean_vector);
	    L_eval_at_temp_load.add(1., LT_eval_at_temp_load);

	    Vector<double> B_eval_at_temp_load(21024);
	    navier_stokes_solver.boundary_matrix_velocity.vmult(B_eval_at_temp_load,mean_vector);

	    {
	   			 DataOut<dim> data_out;
	   		     data_out.attach_dof_handler(navier_stokes_solver.dof_handler_velocity);
	   		     data_out.add_data_vector(mean_vector,"test");
	   		     data_out.add_data_vector(mean_vector_contribution_rhs_vector[0],"Loftest");
	   		     data_out.add_data_vector(L_eval_at_temp_load,"Ltimestest");
	   		     data_out.add_data_vector(mean_vector_contribution_rhs_vector[1],"Coftest");
	   		     data_out.add_data_vector(mean_vector_contribution_rhs_vector[2],"Boftest");
	   		  	 data_out.add_data_vector(B_eval_at_temp_load,"Btimestest");
	   		     data_out.build_patches();
	   		     const std::string filename ="test_mean.vtk";
	   		     std::ofstream output(filename);
	   		     data_out.write_vtk(output);
	   		  }

		  //testing how to process mean vector aka lifting
	    Vector<double> temp_load(21024);
	    temp_load = mean_vector;
		  Vector<double> test_vector(r);
		  Vector<double> test_vector_err(r);

		  // test_vector_N = B_N * baru_N
		  reduced_boundary_matrix.vmult(test_vector,compute_reduced_vector(mean_vector));

		  // test_vector_proj = test_vector_h = Z^T test_vector
		  // reduced_mean_vector_contribution_rhs_vector_proj =  Z^T B_N(baru_N)
		  Vector<double> test_vector_proj(21024);
		  Vector<double> reduced_mean_vector_contribution_rhs_vector_proj(21024);
		  reduced_mean_vector_contribution_rhs_vector_proj = 0;
		  for (int i = 0; i<r; i++){
			  test_vector_proj.add(test_vector[i],pod_vectors[i]);
			  reduced_mean_vector_contribution_rhs_vector_proj.add(reduced_mean_vector_contribution_rhs_vector[2][i],pod_vectors[i]);
		  }
//		      test_vector_proj.add(1.0, mean_vector);
//		      reduced_mean_vector_contribution_rhs_vector_proj.add(1.0, mean_vector);

		  // test_vector_full = B_h baru_h
		  Vector<double> test_vector_full(21024);
		  Vector<double> test_vector_full_err(21024);
		  Vector<double> test_vector_full_reduced(r);
		  navier_stokes_solver.boundary_matrix_velocity.vmult(test_vector_full,mean_vector);
		  test_vector_full_reduced = compute_reduced_vector(test_vector_full);
		  Vector<double> proj_red_test(21024);
		  Vector<double> proj_red_test_red(r);
		  proj_red_test_red= compute_reduced_vector(temp_load);
		  proj_red_test = compute_projected_vector(proj_red_test_red);
//		  // baru = B^-1 B(baru) ??
//
//		  SparseDirectUMFPACK boundary_matrix_velocity_direct;
//		  boundary_matrix_velocity_direct.factorize(navier_stokes_solver.boundary_matrix_velocity);
//		  Vector<double> test_mean(21024);
//		  boundary_matrix_velocity_direct.vmult(test_mean, reduced_mean_vector_contribution_rhs_vector[2]);

//		  {
//			 DataOut<dim> data_out;
//		     data_out.attach_dof_handler(navier_stokes_solver.dof_handler_velocity);
//
//		     // test_vector_full 									= B_h baru_h
//		     // test_vector_proj 									= Z^T B_N * baru_N
//		     // mean_vector_contribution_rhs_vector 				= B_h(baru_h)
//		     // reduced_mean_vector_contribution_rhs_vector_proj 	= Z^T B_N(baru_N)
//
//		     data_out.add_data_vector(temp_load_contribution_rhs_vector[2],"Boftest");
//		     data_out.add_data_vector(B_eval_at_temp_load,"Btimestest");
////		     data_out.add_data_vector(test_vector_full, "test_vector_full");
////		     data_out.add_data_vector(test_vector_proj, "test_vector_proj");
////		     data_out.add_data_vector(mean_vector_contribution_rhs_vector[2], "mean_vector_contribution_rhs_vector2");
////		     data_out.add_data_vector(reduced_mean_vector_contribution_rhs_vector_proj, "reduced_mean_vector_contribution_rhs_vector_proj");
////		     data_out.add_data_vector(proj_red_test, "mean_vector_contribution_rhs_vector2_tetser");
////		     data_out.add_data_vector(temp_load, "tmp_load");
//
//		     data_out.build_patches();
//		     const std::string filename ="test_mean.vtk";
//		     std::ofstream output(filename);
//		     data_out.write_vtk(output);
//		  }

		  test_vector_full_err = test_vector_full;
		  test_vector_full_err.add(-1.,mean_vector_contribution_rhs_vector[2]);
		  std::cout<< test_vector_full_err.norm_sqr() << std::endl;


//			for (int j = 0; j < (r); j++)
//			{
//				std::cout << test_vector(j) << " ";
//			}
//			std::cout << std::endl;
//			for (int j = 0; j < (r); j++)
//			{
//				std::cout << test_vector_full_reduced(j) << " ";
//			}
//			std::cout << std::endl;
//			for (int j = 0; j < (r); j++)
//			{
//				std::cout << reduced_mean_vector_contribution_rhs_vector[2][j] << " ";
//			}
//			std::cout << std::endl;
//
//			test_vector_err = test_vector;
//			test_vector_err.add(-1.0,reduced_mean_vector_contribution_rhs_vector[2]);
//
//			std::cout << "CHecki matrix lifting multi" << std::endl;
//			std::cout <<test_vector_err.norm_sqr() << ", " << test_vector_err.linfty_norm() << std::endl;
//
//		std::cout << std::endl;








	    Vector<double> error_red = reduced_mean_vector_contribution_rhs;
	    error_red.add(-1*time_step*pde_info->fluid_density*pde_info->viscosity	,reduced_mean_vector_contribution_rhs_vector[0]);
	    error_red.add(-1*time_step*pde_info->fluid_density						,reduced_mean_vector_contribution_rhs_vector[1]);
	    error_red.add(-1*time_step*pde_info->fluid_density*pde_info->viscosity	,reduced_mean_vector_contribution_rhs_vector[2]);

	    std::cout << "error_red: " << error_red.linfty_norm() << std::endl;
//
//	    std::cout << "end BC RHS" << std::endl;

	  }

	template<int dim>
		void PODGreedy<dim>::compute_reduced_matrices_no_mean(double time_step)
		  {
			mkdir((pde_info->rom_path).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
			mkdir((pde_info->rom_path + "matrices").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		    // project matrices from FEM space to POD space
		    FullMatrix<double> reduced_mass_matrix = compute_reduced_matrix(navier_stokes_solver.mass_matrix_velocity);

		    FullMatrix<double> reduced_laplace_matrix = compute_reduced_matrix(navier_stokes_solver.laplace_matrix_velocity);
		    FullMatrix<double> reduced_laplace_matrix_with_transposed_trial_function = compute_reduced_matrix(navier_stokes_solver.laplace_matrix_velocity_with_transposed_trial_function);

		    // integral over gamma out
		    FullMatrix<double> reduced_boundary_matrix = compute_reduced_matrix(navier_stokes_solver.boundary_matrix_velocity);

		    FullMatrix<double> reduced_indicator_matrix = compute_reduced_matrix(navier_stokes_solver.indicator_matrix_velocity);

		    std::vector<FullMatrix<double>> reduced_nonlinearity_tensor;
		    // tensor (list of matrices) for the nonlinearity
		    navier_stokes_solver.assemble_nonlinearity_tensor_velocity(pod_vectors);
		    reduced_nonlinearity_tensor.resize(r);
		    for (int i = 0; i < r; ++i){
		        reduced_nonlinearity_tensor[i] = compute_reduced_matrix(navier_stokes_solver.nonlinear_tensor_velocity[i]);
		        filename_h5 = pde_info->rom_path + "matrices/nonlinearity-" + Utilities::int_to_string(i, 6) + ".h5";
		        pod_solver.save_h5_matrix(filename_h5, reduced_nonlinearity_tensor[i]);
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

		    FullMatrix<double> reduced_linear_operator_theta;
		    FullMatrix<double> reduced_linear_operator_one_minus_theta;

		    // A(θ) := ...
		    // parameter independent
		    reduced_linear_operator_theta.reinit(r, r);
		    reduced_linear_operator_theta.add(1.0, reduced_mass_matrix);  // M
		    filename_h5 = pde_info->rom_path + "matrices/M.h5";
		    pod_solver.save_h5_matrix(filename_h5, reduced_linear_operator_theta);

		    // parameter dependent
		    reduced_linear_operator_theta.reinit(r, r);
		    reduced_linear_operator_theta.add(1.0, reduced_laplace_matrix); // kθμ * L
		    filename_h5 = pde_info->rom_path + "matrices/L.h5";
		    pod_solver.save_h5_matrix(filename_h5, reduced_linear_operator_theta);

		    reduced_linear_operator_theta.reinit(r, r);
		    reduced_linear_operator_theta.add(1.0, reduced_laplace_matrix_with_transposed_trial_function); // kθμ * L_t
		    filename_h5 = pde_info->rom_path + "matrices/LT.h5";
		    pod_solver.save_h5_matrix(filename_h5, reduced_linear_operator_theta);

		    reduced_linear_operator_theta.reinit(r, r);
		    reduced_linear_operator_theta.add(1.0, reduced_boundary_matrix); // - kθμ * N
		    filename_h5 = pde_info->rom_path + "matrices/B.h5";
		    pod_solver.save_h5_matrix(filename_h5, reduced_linear_operator_theta);


		    // indicator matrix
		    reduced_linear_operator_theta.reinit(r, r);
		    reduced_linear_operator_theta.add(1.0, reduced_indicator_matrix); // - kθμ * N
		    filename_h5 = pde_info->rom_path + "matrices/Xi.h5";
		    pod_solver.save_h5_matrix(filename_h5, reduced_linear_operator_theta);
		  }

	template<int dim>
			void PODGreedy<dim>::compute_reduced_matrices_vp()
			  {
				std::cout << "start matrix reduction ..." << std::endl;
	    		auto start_time = std::chrono::high_resolution_clock::now();
				mkdir((pde_info->rom_path).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
				mkdir((pde_info->rom_path + "matrices").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

				FullMatrix<double> reduced_mass_matrix = compute_reduced_matrix_vp(navier_stokes_solver.mass_matrix_vp);
				FullMatrix<double> reduced_diffusion_matrix = compute_reduced_matrix_vp(navier_stokes_solver.laplace_matrix_vp);
				FullMatrix<double> reduced_pressure_matrix = compute_reduced_matrix_vp(navier_stokes_solver.pressure_matrix_vp);
				FullMatrix<double> reduced_incompressible_matrix = compute_reduced_matrix_vp(navier_stokes_solver.incompressibilty_matrix_vp);
				FullMatrix<double> reduced_boundary_matrix = compute_reduced_matrix_vp(navier_stokes_solver.boundary_matrix_vp);

				navier_stokes_solver.assemble_convection_term_with_lifting(lifting_vector);
				FullMatrix<double> reduced_first_convection_matrix_with_lifting = compute_reduced_matrix_vp(navier_stokes_solver.first_convection_matrix_velocity_with_mean_vp);
				FullMatrix<double> reduced_second_convection_matrix_with_lifting = compute_reduced_matrix_vp(navier_stokes_solver.second_convection_matrix_velocity_with_mean_vp);


			    std::vector<FullMatrix<double>> reduced_nonlinearity_tensor_vp;
			    // tensor (list of matrices) for the nonlinearity
			    navier_stokes_solver.assemble_nonlinearity_tensor_velocity_vp(pod_vectors_velo_supr);
			    reduced_nonlinearity_tensor_vp.resize(r_vs);
			    Threads::TaskGroup<void> task_group;

			    for (int i = 0; i < r_vs; ++i){
			    	reduced_nonlinearity_tensor_vp[i] = compute_reduced_matrix_vp(navier_stokes_solver.nonlinear_tensor_velocity_vp[i]);
			        filename_h5 = pde_info->rom_path + "matrices/nonlinearity-" + Utilities::int_to_string(i, 6) + ".h5";
			        pod_solver.save_h5_matrix(filename_h5, reduced_nonlinearity_tensor_vp[i]);
			    }

			    filename_h5 = pde_info->rom_path + "matrices/M.h5";
			    pod_solver.save_h5_matrix(filename_h5, reduced_mass_matrix);

			    filename_h5 = pde_info->rom_path + "matrices/A.h5";
			    pod_solver.save_h5_matrix(filename_h5, reduced_diffusion_matrix);

			    filename_h5 = pde_info->rom_path + "matrices/B.h5";
			    pod_solver.save_h5_matrix(filename_h5, reduced_pressure_matrix);

			    filename_h5 = pde_info->rom_path + "matrices/BT.h5";
			    pod_solver.save_h5_matrix(filename_h5, reduced_incompressible_matrix);

			    filename_h5 = pde_info->rom_path + "matrices/D.h5";
			    pod_solver.save_h5_matrix(filename_h5, reduced_boundary_matrix);

			    filename_h5 = pde_info->rom_path + "matrices/C.h5";
			    pod_solver.save_h5_matrix(filename_h5, reduced_first_convection_matrix_with_lifting);

			    filename_h5 = pde_info->rom_path + "matrices/C1.h5";
			    pod_solver.save_h5_matrix(filename_h5, reduced_second_convection_matrix_with_lifting);

			    std::vector<BlockVector<double>> lifting_vector_contribution_rhs_vector(5);
			    for (int i = 0; i<5; i++){
			    	  lifting_vector_contribution_rhs_vector[i].reinit(2);
			    	  lifting_vector_contribution_rhs_vector[i].block(0).reinit(navier_stokes_solver.dofs_velocity);
			    	  lifting_vector_contribution_rhs_vector[i].block(1).reinit(navier_stokes_solver.dofs_pressure);
			    	  lifting_vector_contribution_rhs_vector[i].collect_sizes();
			    }
			    navier_stokes_solver.laplace_matrix_vp.vmult							(lifting_vector_contribution_rhs_vector[0],lifting_vector);
			    navier_stokes_solver.first_convection_matrix_velocity_with_mean_vp.vmult(lifting_vector_contribution_rhs_vector[1],lifting_vector);
			    navier_stokes_solver.boundary_matrix_vp.vmult							(lifting_vector_contribution_rhs_vector[2],lifting_vector);
			    navier_stokes_solver.mass_matrix_vp.vmult								(lifting_vector_contribution_rhs_vector[3],lifting_vector);
			    navier_stokes_solver.incompressibilty_matrix_vp.vmult					(lifting_vector_contribution_rhs_vector[4],lifting_vector);
			    std::vector<BlockVector<double>> lifting_vector_contribution_rhs_vector_test(5);
			    navier_stokes_solver.assemble_lifting_contribution_rhs_greedy_vp(lifting_vector, lifting_vector_contribution_rhs_vector_test);

			    std::vector<BlockVector<double>> error_test = lifting_vector_contribution_rhs_vector_test;

			    error_test[0].add(-1,lifting_vector_contribution_rhs_vector[0]);
			    error_test[1].add(-1,lifting_vector_contribution_rhs_vector[1]);
			    error_test[2].add(-1,lifting_vector_contribution_rhs_vector[2]);
			    error_test[3].add(-1,lifting_vector_contribution_rhs_vector[3]);
			    error_test[4].add(-1,lifting_vector_contribution_rhs_vector[4]);
			    std::cout << "error_lift_fct: " << error_test[0].linfty_norm() << ", " <<error_test[1].linfty_norm() << ", "
			    		<<error_test[2].linfty_norm() << ", " <<error_test[3].linfty_norm() << ", " <<error_test[4].linfty_norm() << ", " <<std::endl;

			    std::vector<Vector<double>> reduced_lifting_vector_contribution_rhs_vector(5);
			    for (int i = 0; i<5; i++){
			    	reduced_lifting_vector_contribution_rhs_vector[i] = compute_reduced_vector_vp(lifting_vector_contribution_rhs_vector[i]);
			    }


			    filename_h5 = pde_info->rom_path + "matrices/lifting_rhs_vector_laplace.h5";
			    pod_solver.save_h5_vector(filename_h5,reduced_lifting_vector_contribution_rhs_vector[0]);
			    filename_h5 = pde_info->rom_path + "matrices/lifting_rhs_vector_convection.h5";
			    pod_solver.save_h5_vector(filename_h5,reduced_lifting_vector_contribution_rhs_vector[1]);
			    filename_h5 = pde_info->rom_path + "matrices/lifting_rhs_vector_boundary.h5";
			    pod_solver.save_h5_vector(filename_h5,reduced_lifting_vector_contribution_rhs_vector[2]);
			    filename_h5 = pde_info->rom_path + "matrices/lifting_rhs_vector_mass.h5";
			    pod_solver.save_h5_vector(filename_h5,reduced_lifting_vector_contribution_rhs_vector[3]);
			    filename_h5 = pde_info->rom_path + "matrices/lifting_rhs_vector_incompressibility.h5";
			    pod_solver.save_h5_vector(filename_h5,reduced_lifting_vector_contribution_rhs_vector[4]);
	    		auto end_time = std::chrono::high_resolution_clock::now();
	    		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
	    		std::cout << "matrix reduction done in: " <<std::floor(duration.count() / 1000) << "s " << duration.count() % 1000 << " ms \n" << std::endl;
			  }

	template <int dim>
	FullMatrix<double> PODGreedy<dim>::compute_reduced_matrix(SparseMatrix<double> &fem_matrix)
	{
		// int m = navier_stokes_solver.dof_handler.n_dofs(); // m: number of FEM DoFs; r: number of POD basis vectors / POD DoFs
		FullMatrix<double> rom_matrix(r,r);
		rom_matrix = 0.0;

		Vector<double> temp(pod_vectors[0].size());
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
	FullMatrix<double> PODGreedy<dim>::compute_reduced_matrix_vp(BlockSparseMatrix<double> &fem_matrix)
	{
		// int m = navier_stokes_solver.dof_handler.n_dofs(); // m: number of FEM DoFs; r: number of POD basis vectors / POD DoFs
//		std::cout << "matrix size: " << r_vs << " + " << r_p << std::endl;
		FullMatrix<double> rom_matrix(r_vs+r_p,r_vs+r_p);
		rom_matrix = 0.0;

		// pod vectors are blockvectors enriched by zeors:  phi^u = [phi^u_1, ... , phi^u_Nu, 0, ...0], phi^p = [0, ..., 0, phi^p_1, ... , phi^p_Np]

		BlockVector<double> temp;
		BlockVector<double> pod_vector_right;
		BlockVector<double> pod_vector_left;
		temp = pod_vectors_velo_supr[0];


		// block (0,0)
		for (int j=0; j<r_vs; j++)
		{
			fem_matrix.vmult(temp, pod_vectors_velo_supr[j]);
			for (int i=0; i<r_vs; i++)
			{
				rom_matrix(i, j) += pod_vectors_velo_supr[i] * temp;
			}
		}
		// block (1,1)
		for (int j=0; j<r_p; j++)
		{
			fem_matrix.vmult(temp, pod_vectors_press[j]);
			for (int i=0; i<r_p; i++)
			{
				rom_matrix(i+r_vs, j+r_vs) += pod_vectors_press[i] * temp;
//				std::cout << pod_vectors_press[i] * temp << std::endl;
			}
		}

		Vector<double> temp_p( pod_vectors_velo_supr[0].block(0).size());
		// block(0,1)
		for (int j=0; j<r_p; j++)
		{
			fem_matrix.vmult(temp, pod_vectors_press[j]);
			for (int i=0; i<r_vs; i++)
			{
				rom_matrix(i, j+r_vs) += pod_vectors_velo_supr[i] * temp;
			}
		}

		// block(1,0)
		for (int j=0; j<r_vs; j++)
		{
			fem_matrix.vmult(temp, pod_vectors_velo_supr[j]);
			for (int i=0; i<r_p; i++)
			{
				rom_matrix(i+r_vs, j) += pod_vectors_press[i] * temp;
			}
		}

		return rom_matrix;
	}

	template <int dim>
	Vector<double> PODGreedy<dim>::compute_reduced_vector_vp(BlockVector<double> &fem_vector)
	{
	    Vector<double> rom_vector(r_vs+r_p);

	    rom_vector = 0.0;

	    for (int i=0; i<r_vs; i++)
	    {
	    	rom_vector[i] = pod_vectors_velo_supr[i] * fem_vector;
	    }
	    for (int i=0; i<r_p; i++)
	    {
	    	rom_vector[i+r_vs] = pod_vectors_press[i] * fem_vector;
	    }
	    return rom_vector;
	}

	template <int dim>
	Vector<double> PODGreedy<dim>::compute_reduced_vector(Vector<double> &fem_vector)
	{
		// int m = navier_stokes_solver.dof_handler.n_dofs(); // m: number of FEM DoFs
	    //  r = number of POD basis vectors / POD DoFs
	    Vector<double> rom_vector(r);

	    rom_vector = 0.0;

	    for (int i=0; i<r; i++)
	    {
	    	rom_vector[i] = pod_vectors[i] * fem_vector;
	    }
	    return rom_vector;
	}

	template <int dim>
	Vector<double> PODGreedy<dim>::compute_projected_vector(Vector<double> &rom_vector)
	{
		// int m = navier_stokes_solver.dof_handler.n_dofs(); // m: number of FEM DoFs
	    //  r = number of POD basis vectors / POD DoFs

	    Vector<double> fem_vector(pod_vectors[0].size());
	    fem_vector = 0.0;

	    for (int i = 0; i<r; i++){
	    	fem_vector.add(rom_vector[i],pod_vectors[i]);
	    }
	    return fem_vector;
	}

	template<int dim>
	void PODGreedy<dim>::generate_surrogate()
	{
		// Reynolds is parameter --> get it by changing viscosity
		// RE in [50, 150] --> RE(i) = start +i*dist/#paras
//		double RE;

//		for (int i=0; i<=10; i++)
//		{
//			double RE = 50+i*10;
//			// write viscosity = Lv/RE
//			surrogate.push_back(0.1/RE);
//		}

		std::string filename("/home/ifam/fischer/Code/rom-nse/surrogate.txt");
		int number;

		std:: ifstream input_file(filename);
	    if (!input_file.is_open()) {
	        std::cerr << "Could not open surrogate file." << std::endl;
	    }

	    while (input_file >> number) {
	    	surrogate.push_back(0.1/number);
	    	surrogate_pod_size_per_parameter.push_back(pde_info->pod_greedy_basis_size);
	    }
	    input_file.close();
	}

	template<int dim>
	double PODGreedy<dim>::compute_error()
	{
		std::string path_fom = pde_info->fem_path + "mu="+ std::to_string(pde_info->viscosity) +"/snapshots/";
		std::string path_rom = pde_info->rom_path + "mu="+ std::to_string(pde_info->viscosity) +"/h5/";
		Vector<double> sol_fom;
		Vector<double> sol_rom;
		Vector<double> tmp;
		Vector<double> tmp_rel;

		double error_M = 0.0;
		double error_M_rel =0.0;
		double error = 0.0;
		double error_rel = 0.0;
		double error_2 = 0.0;
		double error_2_over_time = 0.0;
		double error_2_rel = 0.0;
		int n = 200; //compute_number_snapshots(path_rom);
		for(int i = 0; i<n; i++) {
			sol_fom = load_h5_vector(path_fom + "snapshot_" + Utilities::int_to_string(i,6) + ".h5");
			pod_solver.load_h5_vector_HF(path_rom + "solution-" + Utilities::int_to_string(i,6) + ".h5",sol_rom);
			// sol_rom = differnence btw vectors
			sol_rom.add(-1.0,sol_fom);
			tmp.reinit(sol_rom.size());
			tmp_rel.reinit(sol_fom.size());
			navier_stokes_solver.mass_matrix_velocity.vmult(tmp,sol_rom);
			navier_stokes_solver.mass_matrix_velocity.vmult(tmp_rel,sol_fom);
			error_M = std::sqrt(sol_rom*tmp);
			error_M_rel = std::sqrt(sol_rom*tmp)/std::sqrt(sol_fom*tmp_rel);
			error = std::max(error, sol_rom.linfty_norm());
			error_rel = std::max(error_rel, sol_rom.linfty_norm()/sol_fom.linfty_norm());
			error_2 = std::max(error_2, sol_rom.l2_norm());
			error_2_rel = std::max(error_2_rel, sol_rom.l2_norm()/sol_fom.l2_norm());
			if (i<(n-1))
			{
				error_2_over_time += sol_rom.l2_norm()*sol_rom.l2_norm();
			}
		}
		error_2_over_time /= (n-1);
//
//		std::string path_functionals_rom = pde_info->rom_path + "mu="+ std::to_string(pde_info->viscosity) + "/";
//		std::string path_functionals_fom = pde_info->fem_path + "mu="+ std::to_string(pde_info->viscosity) + "/";
//
//		std::ifstream drag_rom_in;
//		std::ifstream drag_fom_in;
//
//		drag_rom_in.open(path_functionals_rom + "drag.txt", std::ios_base::app);
//		drag_fom_in.open(path_functionals_fom + "drag.txt", std::ios_base::app);
//
//		double drag_rom =0.0;
//		double drag_fom =0.0;
//
//
//		double placehol0;
//		char placehol1;
//		int lines_rom = 0;
//		int lines_fom = 0;
//		while (drag_rom_in >> placehol0 >> placehol1 >> drag_rom)
//		{
//			lines_rom++;
//		}
//		while (drag_fom_in >> placehol0 >> placehol1 >> drag_fom)
//		{
//			lines_fom++;
//		}
//
//		std::cout << "lines in drag: " << lines_rom << "   " << lines_fom << std::endl;
//
//		Vector<double> drag_rom_vec(lines_rom);
//		Vector<double> drag_fom_vec(lines_rom);
//
//		drag_rom_in.close();
//		drag_fom_in.close();
//		drag_rom_in.open(path_functionals_rom + "drag.txt", std::ios_base::app);
//		drag_fom_in.open(path_functionals_fom + "drag.txt", std::ios_base::app);
//
//		int i = 0;
//		while (drag_rom_in >> placehol0 >> placehol1 >> drag_rom)
//		{
//			drag_rom_vec(i)	= drag_rom;
//			i++;
//		}
//		i = 0;
//		while (drag_fom_in >> placehol0 >> placehol1 >> drag_fom)
//		{
//			drag_fom_vec(i)	= drag_fom;
//			i++;
//		}
//		drag_rom_vec.add(-1.0,drag_fom_vec);
//
////		std::cout << "norm in drag: " << drag_rom_vec.l1_norm() << "   " << drag_fom_vec.l1_norm() << std::endl;
//
//		double error_drag_rel = drag_rom_vec.l1_norm()/drag_fom_vec.l1_norm();

		double error_drag_rel = 0;

		error = error_2_over_time;
		std::ofstream p_out;
		p_out.open(pde_info->pod_path + "error_comparer_greedy.txt", std::ios_base::app); // append instead of overwrite
		p_out << 0.1/pde_info->viscosity << "," << std::setprecision(16) \
				<< error << "," << error_rel << "," << error_2 << "," <<  \
				error_2_rel << "," << error_M << "," << error_M_rel << "," << error_drag_rel << ", " <<error_2_over_time <<std::endl;
		p_out.close();

		error = error_2_rel;

		return error;
	}

	template<int dim>
	std::vector<int> PODGreedy<dim>::greedy_loop()
	{
		std::cout << "------------------------------------------------------ \n" \
				  << "Where is the black sheep? \n" \
				  << "------------------------------------------------------ \n" << std::endl;
//		double error = 0.0;

		max_error_mu = 0.0;
		int error_index =0;
		std::vector<int> error_index_vec= {0,0,0};
		std::vector<double> error_vec= {0,0,0};
		for (int i = 0; i<surrogate.size(); i++)
		{
			pde_info->viscosity = surrogate[i];
			ROM::ReducedOrderModel<2> rom_greedy(pde_info);

			if(mean){
				rom_greedy.run(refinements,true,false);
			} else {
				rom_greedy.run_no_mean(refinements,true,false);
			}

			double new_error = compute_error();
			std::cout << "old error: " << max_error_mu << ", new error: " << new_error << std::endl;
			std::ofstream p_out;
			p_out.open(pde_info->pod_path + "error_iteration_greedy.txt", std::ios_base::app); // append instead of overwrite
			p_out << i << "," << std::setprecision(16) << new_error << "," << max_error_mu << std::endl;
			p_out.close();
			if (new_error > error_vec[0]) {
				error_index_vec = {i, error_index_vec[0], error_index_vec[1]};
				error_vec = {new_error, error_vec[0], error_vec[1]};
				max_error_mu = new_error;
			}
			else if (new_error > error_vec[1]) {
				error_index_vec = {error_index_vec[0], i, error_index_vec[1]};
				error_vec = {error_vec[0], new_error, error_vec[1]};
			}
			else if (new_error > error_vec[2]) {
				error_index_vec = {error_index_vec[0], error_index_vec[1], i};
				error_vec = {error_vec[0], error_vec[1], new_error};
			}
			std::cout << "temp indicies: " << error_index_vec[0] << ", " << error_index_vec[1] << ", " << error_index_vec[2] << std::endl;
//			if (new_error>max_error_mu)
//			{
//				max_error_mu = new_error;
//				error_index = i;
////				error_index_vec = {error_index, error_index_vec[0], error_index_vec[1]};
//				error_index_vec[2] = error_index_vec[1];
//				error_index_vec[1] = error_index_vec[0];
//				error_index_vec[0] = error_index;
//			}
		}
		std::cout<< "largest error at: " << 0.1/surrogate[error_index_vec[0]] << ", with: " << max_error_mu << std::endl;
		return error_index_vec;
	}


	template<int dim>
	void PODGreedy<dim>::run(int refinements, bool output_files)
	{
		setup(refinements);

		max_error_mu = 0.0;
		int worst_index = 0;
		double max_error_mu_tmp = 0.0;
		std::vector<int> worst_index_vec = {0,0,0};
		std::vector<int> last_index_vec = {worst_index,-1,-1};

		do {
			pde_info->viscosity = surrogate[worst_index];
			std::cout << "------------------------------------------------------ \n" \
					  << "Enrich with worst parameter:  Reynolds: " << 0.1/pde_info->viscosity << ",  mu: " << pde_info->viscosity << "\n" \
					  << "------------------------------------------------------ \n" << std::endl;
			surrogate_sample.push_back(pde_info->viscosity);
//			NSE::NavierStokes<2> navier_stokes_solver_greedy(pde_info);
//			navier_stokes_solver_greedy.run(refinements,output_files);

			// needed for BC use liftung aka. some stupid solution fullwill BC // no mean is scam
			mean = true;

//			if((worst_index == last_index_vec[0]) && (last_index_vec[1] != -1))
//				{
//					std::cout << "pod_size: " << pde_info->information_content_greedy << std::endl;
//					pde_info->information_content 			+= (1-pde_info->information_content)		*0.9;
//					pde_info->information_content_greedy 	+= (1-pde_info->information_content_greedy)	*0.9;
//					std::cout << "pod_size: " << pde_info->information_content_greedy << std::endl;
//				}


			POD::ProperOrthogonalDecomposition<2> pod(pde_info);
			pod.run_greedy(refinements, output_files, mean);



			if (mean) {
				compute_POD();
				compute_mean_vector();
			} else {
				compute_POD_no_mean();
			}
			pde_info->viscosity = viscosity_tmp;
			std::cout << "------------------------------------------------------ \n" \
				      << "  Compute redMatrices   \n" \
					  << "------------------------------------------------------ \n" << std::endl;
			if (mean) {
				compute_reduced_matrices(pde_info->fine_timestep);
			} else {
				compute_reduced_matrices_no_mean(pde_info->fine_timestep);
			}
			max_error_mu_tmp = max_error_mu;
			worst_index_vec = greedy_loop();
//			if (max_error_mu_tmp == max_error_mu)
//			{
//				pde_info->pod_basis_size += 2;
//			}
//			pde_info->pod_basis_size += 5;
//			std::cout << "worst indicies: " << worst_index_vec[0] << ", " << worst_index_vec[1] << ", " << worst_index_vec[2] << std::endl;
//			std::cout << "last indicies: " << last_index_vec[0] << ", " << last_index_vec[1] << ", " << last_index_vec[2] << std::endl;
//			// jul 13
//			if (std::find(last_index_vec.begin(), last_index_vec.end()-1, worst_index_vec[0]) == last_index_vec.end()-1) {
//				worst_index =  worst_index_vec[0];
//			} else if (std::find(last_index_vec.begin(), last_index_vec.end()-1, worst_index_vec[1]) == last_index_vec.end()-1) {
//				worst_index =  worst_index_vec[1];
//			} else {
//				worst_index = worst_index_vec[2];
//			}

			worst_index =  worst_index_vec[0];

//			last_index_vec = {worst_index,last_index_vec[0],last_index_vec[1]};

//			last_index_vec[2] = last_index_vec[1];
//			last_index_vec[1] = last_index_vec[0];
//			last_index_vec[0] = worst_index;
			std::cout << "choose index: " << worst_index << std::endl;

//			max_error_mu = 0.2;
//			if (r==100)
//				max_error_mu=0;

		} while(max_error_mu>0.01);
	}

	template<int dim>
		void PODGreedy<dim>::run_vp(int refinements, bool output_files)
		{
			setup_vp(refinements);

			max_error_mu = 0.0;
			int worst_index = 0 ;

			do {
				pde_info->viscosity = surrogate[worst_index];
				std::cout << "------------------------------------------------------ \n" \
						  << "Enrich with worst parameter:  Reynolds: " << 0.1/pde_info->viscosity << ",  mu: " << pde_info->viscosity << "\n" \
						  << "------------------------------------------------------ \n" << std::endl;
				surrogate_sample.push_back(pde_info->viscosity);
	//			NSE::NavierStokes<2> navier_stokes_solver_greedy(pde_info);
	//			navier_stokes_solver_greedy.run(refinements,output_files);
//				compute_supremizer();
////
				POD::ProperOrthogonalDecomposition<2> pod(pde_info);
				pod.run_greedy_vp(refinements, output_files);
				compute_POD_vp_velo();
				compute_POD_vp_supremizer();
				compute_mean_vector();  // lifting fct
				compute_POD_vp_press();
				assemble_velo_supr_modes();
//
//				compute_reduced_matrices_vp();

				ROM::ReducedOrderModel<2> rom_greedy(pde_info);
				rom_greedy.run_vp(refinements,output_files,false);
//
//				worst_index_vec = greedy_loop();
//
//				std::cout << "worst indicies: " << worst_index_vec[0] << ", " << worst_index_vec[1] << ", " << worst_index_vec[2] << std::endl;
//				std::cout << "last indicies: " << last_index_vec[0] << ", " << last_index_vec[1] << ", " << last_index_vec[2] << std::endl;
//				// jul 13
//				if (std::find(last_index_vec.begin(), last_index_vec.end()-1, worst_index_vec[0]) == last_index_vec.end()-1) {
//					worst_index =  worst_index_vec[0];
//				} else if (std::find(last_index_vec.begin(), last_index_vec.end()-1, worst_index_vec[1]) == last_index_vec.end()-1) {
//					worst_index =  worst_index_vec[1];
//				} else {
//					worst_index = worst_index_vec[2];
//				}
//
//				worst_index =  worst_index_vec[0];
//
//	//			last_index_vec = {worst_index,last_index_vec[0],last_index_vec[1]};
//
//	//			last_index_vec[2] = last_index_vec[1];
//	//			last_index_vec[1] = last_index_vec[0];
//	//			last_index_vec[0] = worst_index;
//				std::cout << "choose index: " << worst_index << std::endl;

			} while(max_error_mu>0.01);
		}

	template class PODGreedy<2>;
}






//
//Vector<double> temp_lift;
//				std::string snapshot_file_path = "/home/ifam/fischer/Code/rom-nse/lifting.h5";
//				pod.load_h5_vector_HF(snapshot_file_path, temp_lift);
//				BlockVector<double> mean_vector_full;
//				mean_vector_full.reinit(2);
//				mean_vector_full.block(0).reinit(navier_stokes_solver.mass_matrix_vp.block(0,0).m());
//				mean_vector_full.block(1).reinit(navier_stokes_solver.mass_matrix_vp.block(1,0).m());
//				mean_vector_full.collect_sizes();
//
//				mean_vector_full.block(0) = temp_lift;
//				mean_vector_full.block(1) = 0.0;
//
//				BlockVector<double> convection_1_full;
//				convection_1_full.reinit(2);
//				convection_1_full.block(0).reinit(navier_stokes_solver.mass_matrix_vp.block(0,0).m());
//				convection_1_full.block(1).reinit(navier_stokes_solver.mass_matrix_vp.block(1,0).m());
//				convection_1_full.collect_sizes();
//				BlockVector<double> convection_2_full;
//				convection_2_full.reinit(2);
//				convection_2_full.block(0).reinit(navier_stokes_solver.mass_matrix_vp.block(0,0).m());
//				convection_2_full.block(1).reinit(navier_stokes_solver.mass_matrix_vp.block(1,0).m());
//				convection_2_full.collect_sizes();
////				std::cout << pod_vectors_velo[0].block(0).size() << " + " << pod_vectors_velo[0].block(1).size()<< std::endl;
////				mean_vector_full.block(0) = mean_vector;
//				std::cout << mean_vector_full.block(0).size() << " + " << mean_vector_full.block(1).size()<< std::endl;
//
//				// test lifting evals
//				std::vector<BlockVector<double>> lifting_vector_contribution_rhs_vector(5);
//
//				navier_stokes_solver.assemble_lifting_contribution_rhs_greedy_vp(mean_vector_full, lifting_vector_contribution_rhs_vector);
//				navier_stokes_solver.assemble_convection_term_with_lifting(mean_vector_full);
//				navier_stokes_solver.first_convection_matrix_velocity_with_mean_vp.vmult(convection_1_full,mean_vector_full);
//				navier_stokes_solver.second_convection_matrix_velocity_with_mean_vp.vmult(convection_2_full,mean_vector_full);
//
//				BlockVector<double> M_full;
//				M_full.reinit(2);
//				M_full.block(0).reinit(navier_stokes_solver.mass_matrix_vp.block(0,0).m());
//				M_full.block(1).reinit(navier_stokes_solver.mass_matrix_vp.block(1,0).m());
//				M_full.collect_sizes();
//				BlockVector<double> B_full;
//				B_full.reinit(2);
//				B_full.block(0).reinit(navier_stokes_solver.mass_matrix_vp.block(0,0).m());
//				B_full.block(1).reinit(navier_stokes_solver.mass_matrix_vp.block(1,0).m());
//				B_full.collect_sizes();
//
//				navier_stokes_solver.mass_matrix_vp.vmult(M_full,mean_vector_full);
//				navier_stokes_solver.boundary_matrix_vp.vmult(B_full,mean_vector_full);
//				{
//					std::vector<std::string> L_names;
//						L_names.push_back("L_0");
//					    L_names.push_back("L_1");
//					    L_names.push_back("L_p");
//					    std::vector<std::string> C_names;
//					    C_names.push_back("C_0");
//					    C_names.push_back("C_1");
//					    C_names.push_back("C_p");
//					    std::vector<std::string> B_names;
//					    B_names.push_back("B_0");
//					    B_names.push_back("B_1");
//					    B_names.push_back("B_p");
//					    std::vector<std::string> M_names;
//					    M_names.push_back("M_0");
//					    M_names.push_back("M_1");
//					    M_names.push_back("M_p");
//					    std::vector<std::string> BT_names;
//					    BT_names.push_back("BT_0");
//					    BT_names.push_back("BT_1");
//					    BT_names.push_back("BT_p");
//					    std::vector<std::string> CM1_names;
//					    CM1_names.push_back("CM1_0");
//					    CM1_names.push_back("CM1_1");
//					    CM1_names.push_back("CM1_p");
//					    std::vector<std::string> CM2_names;
//					    CM2_names.push_back("CM2_0");
//					    CM2_names.push_back("CM2_1");
//					    CM2_names.push_back("CM2_p");
//					    std::vector<std::string> M_full_names;
//					    M_full_names.push_back("M_full_0");
//					    M_full_names.push_back("M_full_1");
//					    M_full_names.push_back("M_full_p");
//					    std::vector<std::string> B_full_names;
//					    B_full_names.push_back("B_full_0");
//					    B_full_names.push_back("B_full_1");
//					    B_full_names.push_back("B_full_p");
//					    std::vector<DataComponentInterpretation::DataComponentInterpretation>
//					        data_component_interpretation(dim + 1, DataComponentInterpretation::component_is_scalar);
//					    DataOut<dim> data_out;
//					    data_out.attach_dof_handler(navier_stokes_solver.dof_handler);
//					    data_out.add_data_vector(lifting_vector_contribution_rhs_vector[0], L_names,
//					                             DataOut<dim>::type_dof_data,
//					                             data_component_interpretation);
//					    data_out.add_data_vector(lifting_vector_contribution_rhs_vector[1], C_names,
//					                             DataOut<dim>::type_dof_data,
//					                             data_component_interpretation);
//					    data_out.add_data_vector(lifting_vector_contribution_rhs_vector[2], B_names,
//					    					     DataOut<dim>::type_dof_data,
//					    					     data_component_interpretation);
//					    data_out.add_data_vector(lifting_vector_contribution_rhs_vector[3], M_names,
//					                             DataOut<dim>::type_dof_data,
//					                             data_component_interpretation);
//					    data_out.add_data_vector(lifting_vector_contribution_rhs_vector[4], BT_names,
//					                             DataOut<dim>::type_dof_data,
//					                             data_component_interpretation);
//					    data_out.add_data_vector(convection_1_full, CM1_names,
//					                             DataOut<dim>::type_dof_data,
//					                             data_component_interpretation);
//					    data_out.add_data_vector(convection_2_full, CM2_names,
//					                             DataOut<dim>::type_dof_data,
//					                             data_component_interpretation);
//					    data_out.add_data_vector(M_full, M_full_names,
//					                             DataOut<dim>::type_dof_data,
//					                             data_component_interpretation);
//					    data_out.add_data_vector(B_full, B_full_names,
//					                             DataOut<dim>::type_dof_data,
//					                             data_component_interpretation);
//					    data_out.build_patches();
//
//					    const std::string filename = "test_mean_vp.vtk";
//					    std::ofstream output(filename);
//					    data_out.write_vtk(output);
//					   }
