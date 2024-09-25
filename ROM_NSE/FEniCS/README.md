# ROM NSE

In this branch no MORe DWR is used. Just plain rom-nse. Used for thesis.

## Parametrized results
1. `compute_FOM_parameter_space.py` to compute FOM space solutions
2. `main_paraemtrized.py` for evaluation or ROM


## Execute code 

```bash 
python3 main_parametrized.py --start 100 --end 200 --reynolds 100 --venergy 1e-6 --penergy 1e-6
```

Results by: 

```bash
python3 main_parametrized.py --start 100 --end 200 --reynolds 100 --venergy 1e-7 --penergy 1e-7
```

and 

```bash
python3 main_parametrized.py --start 100 --end 100 --reynolds 100 --venergy 1e-6 --penergy 1e-6
```