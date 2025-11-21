first_core=0        # first core
number_of_runs=80    # number of parallel runs
MEMORY_LIMIT="40G"   # memory limit

ulimit -v $((40 * 1024 * 1024))

for run_number in $(seq $first_core $((first_core + number_of_runs - 1))); do
    core_id=$run_number
    echo "run $run_number na core $core_id"
    nohup taskset -c $core_id python3 interstellar_impactors.py "$run_number" &
done
