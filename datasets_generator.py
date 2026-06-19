"""
This script generates the datasets for the F2G2 emulator.
It uses the JAX library to solve the ODEs and the Sobol sequence to sample the parameters.
It writes the results to a .npz file.
"""

import os

from common.progress import stage
from common.ode import CONFIG, generateSamples, solveAAndBBatch, writeResults, reportJaxBackend

path_datasets = CONFIG["path_datasets"]

reportJaxBackend()

with stage("Sampling cosmologies"):
    train_in, validation_in, test_in = generateSamples()

with stage(f"Solving train ({len(train_in)} rows)"):
    train_out = solveAAndBBatch(train_in)
with stage(f"Solving validation ({len(validation_in)} rows)"):
    validation_out = solveAAndBBatch(validation_in)
with stage(f"Solving test ({len(test_in)} rows)"):
    test_out = solveAAndBBatch(test_in)

with stage("Writing train/validation/test.npz"):
    writeResults(os.path.join(path_datasets, "train.npz"),      train_in,      train_out)
    writeResults(os.path.join(path_datasets, "validation.npz"), validation_in, validation_out)
    writeResults(os.path.join(path_datasets, "test.npz"),       test_in,       test_out)
