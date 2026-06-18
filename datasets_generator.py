"""
This script generates the datasets for the F2G2 emulator.
It uses the JAX library to solve the ODEs and the Sobol sequence to sample the parameters.
It writes the results to a .npz file.
"""

import os

from numerical.ode import CONFIG, generate_samples, AandB_solver, write_results

### Main

N_train    = CONFIG["N_train"]
path_model = CONFIG["path_model"]

### Genera los samples del input
train_in, validation_in, test_in = generate_samples()
print(f"Sampled for train (~{N_train}), validation and test.")

### Resuelve el EDP para los output
train_out      = AandB_solver(train_in)
validation_out = AandB_solver(validation_in)
test_out       = AandB_solver(test_in)
print("EDP solver completed.")

### Escribe el resultado en un .npz
write_results(os.path.join(path_model, "train.npz"),      train_in,      train_out)
write_results(os.path.join(path_model, "validation.npz"), validation_in, validation_out)
write_results(os.path.join(path_model, "test.npz"),       test_in,       test_out)
print("Written.")
