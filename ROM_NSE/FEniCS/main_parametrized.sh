# to compute supremizer in parallel
# loop over all reynodls numbers in Re = np.arange(50, 200+1, 5)

for Re in $(seq 50 5 201)
do
    python3 main_parametrized.py --start $Re --end $Re &
    # wait 15 seconds 
    sleep 1
done
# wait for all background jobs to finish
wait
echo "All done"
