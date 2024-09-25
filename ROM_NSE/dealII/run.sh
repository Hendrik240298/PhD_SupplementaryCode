#!/bin/bash
## find word1 and replace with word2 using sed #
make 
sed -i "s/set POD GREEDY solve = true/set POD GREEDY solve = false/g" options.prm
#sed -i "s/set ROM solve = false/set ROM solve = true/g" options.prm

#for i in 0 5 10 #{0..10}
for i in {0..7}
do
    RE=$((65 + $i * 10))
   #RE=$((10 + $i * 10))
   #RE=$((50 + $i * 10))   
   echo $RE
   mu=`echo 14 k 0 0.1 $RE / + p | dc | bc -l`
   mu='set Viscosity = 0'${mu}
   mu_folder=`echo 6 k 0 0.1 $RE / + p | dc | bc -l`
   mu_folder2=`echo 7 k 0 0.1 $RE / + p | dc | bc -l`
   tester=`echo "($mu_folder2 - $mu_folder) * 10000000" | bc -l`
   tester=${tester%.*}
   if [ $tester -gt 4 ]
   then
       mu_folder=`echo "$mu_folder + 0.000001" | bc -l`
       echo result/ROM/mu=0${mu_folder2}
       echo "make to:"
   fi
   echo result/ROM/mu=0${mu_folder}
   echo " "
   rm result/ROM/mu=0${mu_folder}*/*txt
   mkdir result/FEM/mu=0${mu_folder}
   echo $mu
   sed -i "s/set Viscosity = 0.001/$mu/g" options.prm
   rm ${RE}.log
   make run  > logs/${RE}.log &	 
   sleep 10
   sed -i "s/${mu}/set Viscosity = 0.001/g" options.prm
done

wait
