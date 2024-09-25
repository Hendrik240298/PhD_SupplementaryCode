// deal.II includes
#include <deal.II/base/utilities.h>
#include <deal.II/base/parameter_handler.h>

// includes from standard library
#include <chrono>

// ===  myincludes ===
#include <nse.h>
#include <pod.h>
#include <rom.h>
#include <pod_greedy.h>

#include <mpi.h>

using namespace dealii;


// === The Parameter Reader class (derived from ParameterHandler) ===

class ParameterReader : public Subscriptor
{
public:
  ParameterReader(ParameterHandler &);
  void read_parameters(const std::string &);

private:
  void declare_parameters();
  ParameterHandler &prm;
};

ParameterReader::ParameterReader(ParameterHandler &paramhandler) : prm(paramhandler)
{
}

//declare the parameters:
void ParameterReader::declare_parameters()
{
  //POD
  prm.enter_subsection("General");
  {
    prm.declare_entry("Refinements",
                      "5",
                      Patterns::Integer(0),
                      "Number of global refinements of mesh");
    prm.declare_entry("POD size",
                      "100",
                      Patterns::Integer(0),
                      "Maximum size of the POD basis.");
    prm.declare_entry("POD greedy size",
                      "100",
                      Patterns::Integer(0),
                      "Maximum size of the POD greedy basis.");
    prm.declare_entry("POD greedy pressure size",
                      "100",
                      Patterns::Integer(0),
                      "Maximum size of the POD greedy pressure basis.");
    prm.declare_entry("Information content",
                      "1.0",
                      Patterns::Double(0),
                      "Minimum information content of POD basis.");
    prm.declare_entry("Information content greedy",
                      "1.0",
                      Patterns::Double(0),
                      "Minimum information content of POD basis.");
    prm.declare_entry("FEM solve",
                      "false",
                      Patterns::Bool(),
                      "Solve the PDE with FEM");
    prm.declare_entry("POD solve",
                      "false",
                      Patterns::Bool(),
                      "Solve POD");
    prm.declare_entry("ROM solve",
                      "false",
                      Patterns::Bool(),
                      "Solve the PDE with POD-ROM");
    prm.declare_entry("POD GREEDY solve",
                      "false",
                      Patterns::Bool(),
                      "Solve POD Greedy");
    prm.declare_entry("POD vp-GREEDY solve",
                      "false",
                      Patterns::Bool(),
                      "Solve vp-POD Greedy");
    prm.declare_entry("Compute error",
                      "false",
                      Patterns::Bool(),
                      "Compute the error between the FEM and the POD solution");
    prm.declare_entry("Output files",
                      "false",
                      Patterns::Bool(),
                      "Output vtk and h5 files");
    prm.declare_entry("Test case",
                      "2D-1",
                      Patterns::Anything(),
                      "Name of Navier Stokes test case");
    prm.declare_entry("Newton tolerance",
                      "1.0e-8",
                      Patterns::Double(0),
                      "Tolerance for the Newton solver");
    prm.declare_entry("Density fluid",
                      "1.0",
                      Patterns::Double(0),
                      "Fluid density");
    prm.declare_entry("Viscosity",
                      "1.0e-3",
                      Patterns::Double(0),
                      "Fluid viscosity");
    prm.declare_entry("Start time",
                      "25.0",
                      Patterns::Double(0),
                      "When should the computations start? Should be > 0, if we have a snapshot from previous calculations as initial condition");
    prm.declare_entry("Initial solution snapshot",
                      "snapshot_initial_condition_t=25_ref=3.h5",
                      Patterns::Anything(),
                      "Snapshot of the initial condition (is being used if start_time > 0.0)");
    prm.declare_entry("FEM path",
                      "$RESULT/FEM/",
                      Patterns::Anything(),
                      "Path where safe the POD data");
    prm.declare_entry("POD path",
                      "$RESULT/POD/",
                      Patterns::Anything(),
                      "Path where safe the POD data");
    prm.declare_entry("ROM path",
                      "$RESULT/ROM/",
                      Patterns::Anything(),
                       "Path where safe the ROM data");
    prm.declare_entry("Coarse timestep",
                      "0.05",
                      Patterns::Double(0),
                      "Coarse timestep size to compute the initial solution");
    prm.declare_entry("Coarse endtime",
                      "3.5",
                      Patterns::Double(0),
                      "How long until initial solution has been computed?");
    prm.declare_entry("Fine timestep",
                      "0.0025",
                      Patterns::Double(0),
                      "Fine timestep size after initial solution");
    prm.declare_entry("Fine endtime",
                      "35",
                      Patterns::Double(0),
                      "End time of the FEM computations");
    prm.declare_entry("POD start time",
                   	   "3.5",
                       Patterns::Double(0),
                       "Initial time for POD");
    prm.declare_entry("POD offline",
                   	   "true",
                       Patterns::Bool(),
                       "Is offline is done in ROM");
    prm.declare_entry("Hidden layers",
                      "1",
                      Patterns::Integer(0),
                      "Number of hidden layers in the neural network");
    prm.declare_entry("Hidden neurons",
                      "30",
                      Patterns::Integer(0),
                      "Number of neurons in the hidden layers");

  }
  prm.leave_subsection();
}

// read parameters:
void ParameterReader::read_parameters(const std::string &parameter_file)
{
  declare_parameters();
  prm.parse_input(parameter_file);
}

int main(int argc, char **argv)
{
  try
  {
    // create a struct which holds all the information about the problem
    PDEInfo pde_info;

    //read the Parameters
    ParameterHandler prm;
    ParameterReader param(prm);
    param.read_parameters((argc==1 ? "options.prm" : argv[1]));

    prm.enter_subsection("General");
    int refinements = prm.get_integer("Refinements");
    int pod_size = prm.get_integer("POD size");
    double information_content = prm.get_double("Information content");
    bool solve_fem = prm.get_bool("FEM solve");
    bool solve_pod = prm.get_bool("POD solve");
    bool solve_rom = prm.get_bool("ROM solve");
    bool solve_pod_greedy = prm.get_bool("POD GREEDY solve");
    bool solve_pod_greedy_vp = prm.get_bool("POD vp-GREEDY solve");
    bool compute_error = prm.get_bool("Compute error");
    bool output_files = prm.get_bool("Output files");
    pde_info.test_case = prm.get("Test case");
    pde_info.newton_tol = prm.get_double("Newton tolerance");
    pde_info.fluid_density = prm.get_double("Density fluid");
    pde_info.viscosity = prm.get_double("Viscosity");
    pde_info.start_time = prm.get_double("Start time");
    pde_info.intial_solution_snapshot = prm.get("Initial solution snapshot");
    pde_info.fem_path = prm.get("FEM path");
    pde_info.pod_path = prm.get("POD path");
    pde_info.rom_path = prm.get("ROM path");
    pde_info.coarse_timestep = prm.get_double("Coarse timestep");
    pde_info.fine_timestep = prm.get_double("Fine timestep");
    pde_info.coarse_endtime = prm.get_double("Coarse endtime");
    pde_info.fine_endtime = prm.get_double("Fine endtime");
    pde_info.POD_start_time = prm.get_double("POD start time");
    pde_info.POD_offline = prm.get_bool("POD offline");
    pde_info.information_content = prm.get_double("Information content");
    pde_info.information_content_greedy = prm.get_double("Information content greedy");
    pde_info.pod_basis_size = prm.get_integer("POD size");
    pde_info.pod_greedy_basis_size = prm.get_integer("POD greedy size");
    pde_info.pod_greedy_basis_pressure_size = prm.get_integer("POD greedy pressure size");
    pde_info.hidden_layers = prm.get_integer("Hidden layers");
    pde_info.hidden_neurons = prm.get_integer("Hidden neurons");
    prm.leave_subsection();

    std::cout << "\n --------------------- PARAMETERS --------------------- " << std::endl;
    prm.print_parameters(std::cout, ParameterHandler::ShortText);
    std::cout << " ------------------------------------------------------ " << std::endl
              << std::endl;

    // initialize an MPI communicator with the maximum number of processors
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);
    std::cout << std::endl
              << std::endl; // make some space for program output
    // 1. SOLVING THE HEAT EQUATION WITH FINITE ELEMENTS.
    if (solve_pod_greedy){
    	using namespace POD_GREEDY;
    	PODGreedy<2> pod_greedy_solver(&pde_info);
    	if (solve_pod_greedy_vp)
    	{
    		pod_greedy_solver.run_vp(refinements, output_files);
    	}
    	else
    	{
    		pod_greedy_solver.run(refinements, output_files);
    	}
//    	pod_greedy_solver.run(refinements, output_files);
    	solve_fem = solve_pod = false;
    }

    if (solve_fem)
    {
      using namespace NSE;
      auto start_time = std::chrono::high_resolution_clock::now();
      NavierStokes<2> navier_stokes_solver(&pde_info);
      navier_stokes_solver.run(refinements, output_files);
//      navier_stokes_solver.run_ROM(refinements, output_files);
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

      std::cout << "FEM computation time:" << std::floor(duration.count() / 1000) << " seconds "
                << duration.count() % 1000 << " milliseconds" << std::endl;
    }
    // 2. POD THROUGH THE METHOD OF SNAPSHOTS.
    if (solve_pod)
    {
    	using namespace POD;
    	auto start_time2 = std::chrono::high_resolution_clock::now();
    	ProperOrthogonalDecomposition<2> pod(&pde_info);
    	pod.run_greedy(refinements, output_files, true);
    	auto end_time2 = std::chrono::high_resolution_clock::now();
    	auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time2 - start_time2);

    	std::cout << "ROM computation time:" << std::floor(duration2.count() / 1000) << " seconds "
    			                << duration2.count() % 1000 << " milliseconds" << std::endl;
    }
    // 3. REDUCED ORDER MODELING OF THE NAVIER STOKES EQUATIONs THROUGH THE METHOD OF SNAPSHOTS.
    if (solve_rom)
    {
      using namespace ROM;
      auto start_time2 = std::chrono::high_resolution_clock::now();
      ReducedOrderModel<2> rom(&pde_info);
      rom.run(refinements, output_files, compute_error);
      auto end_time2 = std::chrono::high_resolution_clock::now();
      auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time2 - start_time2);

      std::cout << "ROM computation time:" << std::floor(duration2.count() / 1000) << " seconds "
                << duration2.count() % 1000 << " milliseconds" << std::endl;
    }
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  return 0;
}
