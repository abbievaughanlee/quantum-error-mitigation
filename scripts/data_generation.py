
from qiskit.circuit.random import random_circuit
from qiskit import transpile

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

import numpy as np
from tqdm import tqdm


# initial configuration of circuit
NUM_QUBITS = 2 
# number of layers of gates in each circuit
CIRCUIT_DEPTH = 10 # increased from five for round 2
NUM_SAMPLES = 1000
SHOTS = 1024
# probability of depolarizing error for each gate
ERROR_PROB = 0.1
OUTPUT_FILE = f'data/testing_dataset_{NUM_QUBITS}q_d{CIRCUIT_DEPTH}01.npz'


# define random circuit generation function
def create_random_circuit(num_qubits, depth):
    # create a random quantum circuit of given number of qubits and depth.
         # measure=True ensures that all qubits are measured at the end of the circuit
    qc = random_circuit(num_qubits, depth, measure=True)
    return qc

# convert qiskit dict to prob vector of length 2**num_qubits
def counts_to_vector(counts, num_qubits):
    num_outcomes = 2**num_qubits
    prob_vector = np.zeros(num_outcomes)
    total_shots = sum(counts.values())
    if total_shots == 0: return prob_vector

    for bitstring, count in counts.items():
        index = int(bitstring, 2)
        prob_vector[index] = count / total_shots
    return prob_vector



# create ideal and noisy simulators:
if __name__ == "__main__":
    print("setting up ideal and noisy simulators...")
    ideal_simulator = AerSimulator()

    # create depolarizing errors for 1-qubit and 2-qubit gates with specified error probability
    error_1 = depolarizing_error(ERROR_PROB, 1)
    error_2 = depolarizing_error(ERROR_PROB, 2)
    # create a noise model and add the errors to the appropriate gates
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3', 'u4', 'id', 'h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx', 'cz', 'swap', 'iswap'])

    noisy_simulator = AerSimulator(noise_model=noise_model)

    noisy_data = []
    ideal_data = []

    print(f"generating {NUM_SAMPLES} samples of data...")
    for _ in tqdm(range(NUM_SAMPLES)):
        qc = random_circuit(NUM_QUBITS, CIRCUIT_DEPTH, measure=True)
        # transpile for Aer backend
        transpiled_qc = transpile(qc, backend=ideal_simulator)
        # run on ideal simulator
        ideal_job = ideal_simulator.run(transpiled_qc, shots=SHOTS)
        ideal_result = ideal_job.result()
        ideal_counts = ideal_result.get_counts()
        # run on noisy simulator
        noisy_job = noisy_simulator.run(transpiled_qc, shots=SHOTS)
        noisy_result = noisy_job.result()
        noisy_counts = noisy_result.get_counts()
        # convert counts to probability vectors
        ideal_vector = counts_to_vector(ideal_counts, NUM_QUBITS)
        noisy_vector = counts_to_vector(noisy_counts, NUM_QUBITS)
        ideal_data.append(ideal_vector)
        noisy_data.append(noisy_vector)
    print("data generation complete.")

    x_data = np.array(noisy_data)
    y_data = np.array(ideal_data)   
    np.savez_compressed(OUTPUT_FILE, x=x_data, y=y_data)    
    print(f"data saved to {OUTPUT_FILE}")
    print(f"data shape: x={x_data.shape}, y={y_data.shape}")

