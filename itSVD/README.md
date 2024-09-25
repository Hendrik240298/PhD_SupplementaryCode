# itSVD for the Navier-Stokes equations 

- mesh with about 290k velocity and 36k pressure DoFs
- many DoFs to show capability of itSVD 

## Run Code 

### Important functions

- Code is found in: **itSVD Branch of [[_MORe_DWR_FEniCS NSE]] Code** 
- `main.py` seems to make the main part 
	- performing FOM if needed 
	- itSVD execution for velocity and pressure
	- *Plotting Cost functional* 
- `itSVD_results_generation_timings_bunch_sizes.py` 
	- generates timings for different bunch sizes 
	- solves FOM in needed 
	- saves timings tp `itSVD_timings_bunch_sizes.pkl` 
- `itSVD_results_generation_energy.py` 
	- generates timings for different energy contents 
	- solves fom if needed
	- saves timings to `itSVD_timings_energy_content.pkl` 
	- saves cost fucntionals to `itSVD_cost_functionals_energy_content.pkl` 
- `plot_itSVD_timings.py` 
	- computes timing results
- `plots_itSVD.py`
	- plots cost functional results 

### Workflow 

1. Generate Data
```bash
python3 itSVD_results_generation_timings_bunch_sizes.py
python3 itSVD_results_generation_energy.py
```

2. Plot data 
```bash
python3 plot_itSVD_timings.py
python3 plots_itSVD.py 
```


## Sources
- `itSVD.py`: contains itSVD class fpr compression 
- `FOM.py`: FEM Code for NSE