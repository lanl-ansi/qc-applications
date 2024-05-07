import re
import sys
import time
from dataclasses import dataclass

import numpy as np

from openfermionpyscf import run_pyscf
from openfermion.chem import MolecularData
from openfermion.ops.representations import InteractionOperator

from pyLIQTR.PhaseEstimation.pe import PhaseEstimation

from qca.utils.utils import circuit_estimate, EstimateMetaData
import random
@dataclass
class molecular_info:
    """Class for keeping track of information for a given state in the molecular orbital basis"""
    basis: str
    occupied_qubits: int
    unoccupied_qubits: int
    initial_state: np.ndarray[int]
    hf_energy:float
    active_space_reduction:float
    molecular_hamiltonian: InteractionOperator

def grab_molecular_phase_offset(hf_energy: float):
    E_min = -abs(0.25 * hf_energy)
    E_max = 0
    omega = E_max - E_min
    t = 2*np.pi/omega
    return E_max * t

def extract_number(string):
    number = re.findall(r'\d+', string)
    return int(number[0]) if number else None

def grab_line_info(current_line:str):
    multiplicity = 0
    charge = 0
    multiplicity_match = re.search(r"multiplicity\s*=\s*(\d+)", current_line)
    if multiplicity_match:
        multiplicity = int(multiplicity_match.group(1))
    charge_match = re.search(r"charge\s*=\s*(\d+)", current_line)
    if charge_match:
        charge = int(charge_match.group(1))
    return multiplicity, charge

def grab_pathway_info(
        data: list[str],
        nat:int,
        current_line: str,
        coord_pathways:list,
        current_idx:int
    ):
    coords_list = []
    multiplicity, charge = grab_line_info(current_line)
    coords_list.append([nat, charge, multiplicity])
    for point in range(nat):
        data_point = data[current_idx+1+point].split()
        aty = data_point[0]
        xyz = [float(data_point[i]) for i in range(1,4)]
        coords_list.append([aty, xyz])
    coord_pathways.append(coords_list)

def load_pathway(fname:str, pathway:list[int]) -> list:
    '''
    Given some XYZ file and a pathway of interest, grab its charge,
    multiplicity, and cartesian coordinates. 
    '''
    with open(fname, 'r') as f:
        coordinates_pathway = []
        data = f.readlines()
        data_length = len(data)
        idx = 0
        while idx < data_length:
            line = data[idx]
            if 'charge' in line or 'multiplicity' in line:
                geo_name = ''
                if len(line.split(',')) > 2:
                    geo_name = line.split(',')[2]
                nat = int(data[idx-1].split()[0])
                if geo_name and pathway:
                    order = extract_number(geo_name)
                    if order and order in pathway:
                        grab_pathway_info(data, nat, line, coordinates_pathway, idx)
                else:
                    grab_pathway_info(data, nat, line, coordinates_pathway, idx)
                idx += nat + 2
            else:
                idx += 1
    return coordinates_pathway

def generate_electronic_hamiltonians(
        basis: str,
        active_space_frac: float,
        coordinates_pathway:list,
        run_scf:int,
        run_mp2:int=0,
        run_cisd:int=0,
        run_ccsd:int=0,
        run_fci:int=0
    ) -> list:
    molecular_hamiltonians = []
    for idx, coords in enumerate(coordinates_pathway):
        t_coord_start = time.perf_counter()
        _, charge, multi = [int(coords[0][j]) for j in range(3)]

        # set molecular geometry in pyscf format
        geometry = []
        for coord in coords[1:]:
            atom = (coord[0], tuple(coord[1]))
            geometry.append(atom)
        
        molecule = MolecularData(
            geometry=geometry,
            basis=basis,
            multiplicity=multi,
            charge=charge,
            description='catalyst'
        )
        t0 = time.perf_counter()
        molecule = run_pyscf(
            molecule,
            run_scf=run_scf,
            run_mp2=run_mp2,
            run_cisd=run_cisd,
            run_ccsd=run_ccsd,
            run_fci=run_fci
        )
        t1 = time.perf_counter()

        print(f'Time to perform a HF calculation on molecule {idx} : {t1-t0}')
        print(f'Number of orbitals          : {molecule.n_orbitals}')
        print(f'Number of electrons         : {molecule.n_electrons}')

        print(f'Number of qubits            : {molecule.n_qubits}')
        print(f'Hartree-Fock energy         : {molecule.hf_energy}')
        sys.stdout.flush()

        nocc = molecule.n_electrons // 2
        nvir = molecule.n_orbitals - nocc

        percent_occupied = nocc/molecule.n_orbitals
        percent_unoccupied = nvir/molecule.n_orbitals

        print(f'Number of unoccupied Molecular orbitals are: {nvir}')
        print(f'Number of occupied Molecular orbitals are: {nocc}')
        sys.stdout.flush()

        # get molecular Hamiltonian
        active_space_start =  nocc - nocc // active_space_frac # start index of active space
        active_space_stop = nocc + nvir // active_space_frac   # end index of active space

        print(f'active_space start : {active_space_start}')
        print(f'active_space stop  : {active_space_stop}')
        sys.stdout.flush()

        molecular_hamiltonian = molecule.get_molecular_hamiltonian(
            occupied_indices=range(active_space_start),
            active_indices=range(active_space_start, active_space_stop)
        )
        molecular_occupied = round(percent_occupied*molecular_hamiltonian.n_qubits)
        molecular_unoccupied = round(percent_unoccupied*molecular_hamiltonian.n_qubits)
        initial_state = [0]*molecular_unoccupied + [1]*molecular_occupied
       
        print(f'In the Molecular Orbital Basis: we have {molecular_hamiltonian.n_qubits} qubits')
        print(f'In the Molecular Orbital Basis: we have {molecular_occupied} qubits occupied')
        print(f'In the Molecular Orbital Basis: we have {molecular_unoccupied} qubits unoccupied')
        
        # shifted by HF energy
        molecular_hamiltonian -= molecule.hf_energy
        mi = molecular_info(
            basis=basis,
            occupied_qubits=molecular_occupied,
            unoccupied_qubits=molecular_unoccupied,
            initial_state=initial_state,
            hf_energy=molecule.hf_energy,
            active_space_reduction=active_space_frac,
            molecular_hamiltonian=molecular_hamiltonian
        )
        molecular_hamiltonians.append(mi)
        t_coord_end = time.perf_counter()
        print(f'Time to generate a molecular hamiltonian for molecule {idx} : {t_coord_end-t_coord_start}\n')
    return molecular_hamiltonians

def gsee_molecular_hamiltonian(
        catalyst_name:str,
        gse_args: dict,
        trotter_steps: int,
        bits_precision: int,
        molecular_hamiltonians: list[molecular_info]
    ) -> int:
    uid = int(random.random() % len(molecular_hamiltonians))*1000
    for idx, molecular_hamiltonian_info in enumerate(molecular_hamiltonians):
        molecular_hamiltonian = molecular_hamiltonian_info.molecular_hamiltonian
        molecular_hf_energy = molecular_hamiltonian_info.hf_energy
        active_space_frac = molecular_hamiltonian_info.active_space_reduction
        basis = molecular_hamiltonian_info.basis
        gse_args['mol_ham'] = molecular_hamiltonian
        phase_offset = grab_molecular_phase_offset(molecular_hf_energy)
        init_state = molecular_hamiltonian_info.initial_state
        molecular_metadata = EstimateMetaData(
            id = uid,
            name=f'{catalyst_name}_molecule({idx})',
            category='scientific',
            size=f'{molecular_hamiltonian.n_qubits}',
            task='Ground State Energy Estimation',
            implementations=f'trotterization subprocess, basis={basis}, active_space_reduction={active_space_frac}, bits_precision={bits_precision}',
            value_per_circuit=6000,
            repetitions_per_application=100
        )
        uid += 1

        t0 = time.perf_counter()
        gse_inst = PhaseEstimation(
            precision_order=1,
            init_state=init_state,
            phase_offset=phase_offset,
            include_classical_bits=False,
            kwargs=gse_args
        )
        gse_inst.generate_circuit()
        t1 = time.perf_counter()

        print(f'Time to generate a circuit for estimating the GSE for Co2O9H12 {idx}: {t1-t0}')
        gse_circuit = gse_inst.pe_circuit
        
        t0 = time.perf_counter()
        circuit_estimate(
            circuit=gse_circuit,
            metadata=molecular_metadata,
            outdir='GSE/Quantum_Chemistry/',
            numsteps=trotter_steps,
            circuit_name=f'{catalyst_name}_{idx}_active_space{active_space_frac}',
            bits_precision=bits_precision,
            write_circuits=False
        )
        t1 = time.perf_counter()
        print(f'Time to estimate Co2O9H12_step({idx}): {t1-t0}')
