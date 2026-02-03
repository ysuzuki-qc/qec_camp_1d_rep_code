# analyze_1d_rep_code

code for analyzing 1d repetition code

## Step1: obtain dataset

If you want to use experimental data, please create `NoiseModel` and `RecordDataset` from target devices or teams.

## Step2: estimate errors

- `run_qec_device_experiment.py`: Evaluate logical error rates with device noise model and experimental record data
- `run_qec_device_simulate.py`: Evaluate logical error rates with device noise model and simulated record data
- `run_qec_uniforme_experiment.py`: Evaluate logical error rates with uniform noise model and experimental record data
- `run_qec_uniform_simulate.py`: Evaluate logical error rates with uniform noise model and simulated record data
- `plot_result.py`: Plot all the above and save figure

## Step3: correlation matrix

- `analyze_correlation_experiment.py`: Evaluate correlation maps for experimental record data
- `analyze_correlation_simulation.py`: Evaluate correlation maps for simulated record data using device noise model
- `analyze_correlation_uniform.py`: Evaluate correlation maps for simulated record data using uniform noise model
- `plot_correlation.py`: Plot all the above and save figure



