format:
	isort --profile black -l 100 *.py
	autopep8 --in-place --aggressive --aggressive *.py      
	black -l 100 *.py

run:
	python3 main.py

run1d: 
	python3 main_greedy.py config/result_generation_1D.yaml
	python3 plot_1D.py config/result_generation_1D.yaml

run4d:
	python3 main_greedy.py config/result_generation_4D.yaml
	python3 plot_4D.py config/result_generation_4D.yaml

run6d:
	python3 main_greedy.py config/result_generation_6D.yaml
	python3 plot_6D.py config/result_generation_6D.yaml

runall:
	python3 main_greedy.py config/result_generation_1D.yaml
	python3 main_greedy.py config/result_generation_4D.yaml
	python3 main_greedy.py config/result_generation_6D.yaml
	python3 main_greedy.py config/result_generation_1D_meyer.yaml
	python3 main_greedy.py config/result_generation_4D_meyer.yaml
	python3 main_greedy.py config/result_generation_6D_meyer.yaml

greedy:
	python3 main_greedy.py


condor:
	condor_submit cluster/condor.submit


uml:
	pyreverse -o png FOM.py
	mv classes.png FOM.png
	pyreverse -o png iROM.py
	mv classes.png iROM.png

plot_1D:
	python3 plot_1D.py config/result_generation_1D.yaml
