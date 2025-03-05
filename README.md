![pylint](https://img.shields.io/badge/PyLint-9.02-yellow?logo=python&logoColor=white)

# Quantum Computing Application Specifications

This project focuses on the documentation of applications for Quantum Computers. Quantum Computing Application (QCA) is a python library for constructing quantum circuits for a set of
kernels and applications from the corresponding paper, and generating a series of resource estimates from it. All resource estimates are in the Clifford + T gateset. 

QCA is structured the following:
- notebooks/
    - A series of jupyter notebooks corresponding to each chapter in the paper which describes on how the application's workflow would look like. The following is a mapping of the notebook to its corresponding chapter on the report
        - ExoticPhasesExample.ipynb -> Chapter 3. Exploring exotic phases of magnetic materials near instabilities
        - DickeModelExample.ipynb -> Chapter 4. Driven-dissipative Dicke model in the ultrastrong coupling regime
        - HighTemperatureSuperConductorExample.ipynb -> Chapter 5. High-temperature superconductivity and exotic properties of FermiHubbard models
        - ArtificialPhotosynthesisExample.ipynb -> Chapter 6. Computational catalysis for artificial photosynthesis
        - QuantumChromoDynamicsExample.ipynb -> Chapter 7. Simulations of quantum chromodynamics and nuclear astrophysics
- scripts/
    - Python scripts that allows an end user to run an application over the command line
- qca.utils
    - Module that contains utility functions regarding constructing circuits for a given application'
- tests/
    - Unit test cases to verify QCA's functionality

After running some program, you'd get the following output:
```json
{
    [(OPTIONAL) block of metadata information]
    [block of resource estimates]
}
```
where the block of metadata information is the following (note that any key below is optional):
```json
{
    "name": str, // Name of the experiment
    "category": str, // Type of experiment such as scientific or industrial
    "size": str, // Size of the Hamiltonian
    "task": str, // The computational task such as ground_state_energy_estimation and time_dependent_dynamics
    "implementation": str, // Description of the implementation such as block encoding used
    "value": float, // The utility estimate in US dollars for the problem
    "value_per_t_gate": float, // The utility estimate value per t gate,
    "repetitions_per_application": int, // Total number of times to repeat the circuit to realize its utility
    "is_extrapolated": bool, // Denoting if the circuit was extrapolated to save compute time
    "gate_synth_accuracy": float, // The approximation error to decompose the circuit
    "nsteps": int, // Total number of steps taken
    "evolution_time": float, // Total evolution time
    "energy_precision": float, // Acceptable shift in state energy
}
```

where the block of resource estimations is the following:
```json
{
    "num_qubits": int, // number of qubits in the system
    "t_count": int, // total number of T gates
    "clifford_count": int, // total number of clifford gates
    "gate_count": int, // total number of gates
    "gate_depth": int, // total gate depth
    "subcircuit_occurences": int, // total number of times we come across this subcircuit
    "t_depth": int, // total T depth
    "max_t_depth_wire": int, // maximum number of T-gates on a wire
}
```

unless stated otherwise, the output will be generated as "{application_name}_re.json"

# Citation

If qc-applications proves useful in your work, please consider citing it using the following BibTeX citation for the pre-print, available [here](https://arxiv.org/abs/2406.06625):

```
@misc{2406.06625,
    Title = {Potential Applications of Quantum Computing at Los Alamos National Laboratory},
    Author = {Andreas Bärtschi and Francesco Caravelli and Carleton Coffrin and Jonhas Colina and Stephan Eidenbenz and Abhijith Jayakumar and Scott Lawrence and Minseong Lee and Andrey Y. Lokhov and Avanish Mishra and Sidhant Misra and Zachary Morrell and Zain Mughal and Duff Neill and Andrei Piryatinski and Allen Scheie and Marc Vuffray and Yu Zhang},
    Year = {2024},
    Eprint = {2406.06625},
    archivePrefix={arXiv},
    primaryClass={quant-ph}
}
```

## License

This code is provided under a BSD license as part of the Quantum Application Specifications and Benchmarks project (O4626).
