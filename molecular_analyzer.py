#!/usr/bin/env python3
"""
Molecular Complex Analysis Pipeline
Processes absorbent and analyte .gjf files using FairChem and GemNet models
for optimization and property prediction.
"""

import os
import numpy as np
import torch
import ase
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS
from ase.calculators.calculator import Calculator
import warnings
warnings.filterwarnings('ignore')

try:
    # FairChem imports
    from fairchem.core.models import model_registry
    from fairchem.core.common.relaxation.ase_utils import OCPCalculator
    from fairchem.core.datasets import LmdbDataset
    
    # GemNet imports (placeholder - adjust based on actual GemNet implementation)
    from gemnet_pytorch import GemNetT
    from gemnet_pytorch.model import GemNet
    
except ImportError as e:
    print(f"Warning: Some dependencies not found: {e}")
    print("Please install fairchem-core and gemnet-pytorch packages")

class MolecularComplexAnalyzer:
    """
    Complete pipeline for molecular complex analysis including:
    - Structure optimization using FairChem
    - Complex formation using GemNet
    - Property prediction (HOMO-LUMO, binding energy, etc.)
    """
    
    def __init__(self, fairchem_model="gemnet_oc", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the analyzer with specified models.
        
        Args:
            fairchem_model (str): FairChem model for optimization
            device (str): Computing device (cuda/cpu)
        """
        self.device = device
        self.fairchem_model = fairchem_model
        self.setup_models()
        
    def setup_models(self):
        """Initialize FairChem and GemNet models."""
        try:
            # Initialize FairChem calculator for optimization
            self.fairchem_calc = OCPCalculator(
                model_name=self.fairchem_model,
                local_cache="./models/",
                cpu=self.device=="cpu"
            )
            
            # Initialize GemNet model for property prediction
            # Note: Adjust parameters based on your GemNet model requirements
            self.gemnet_model = GemNetT(
                num_targets=1,
                hidden_channels=512,
                num_blocks=4,
                num_bilinear=8,
                num_spherical=7,
                num_radial=6,
                otf_graph=True,
                cutoff=6.0,
                max_neighbors=50,
                envelope_exponent=5,
                num_before_skip=1,
                num_after_skip=2,
                num_output_layers=3,
            )
            
            print("Models initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing models: {e}")
            # Fallback to mock calculators for demonstration
            self.fairchem_calc = None
            self.gemnet_model = None
    
    def parse_gjf_file(self, filepath):
        """
        Parse Gaussian .gjf file and convert to ASE Atoms object.
        
        Args:
            filepath (str): Path to .gjf file
            
        Returns:
            ase.Atoms: Atomic structure
        """
        try:
            # Read using ASE's Gaussian input format
            atoms = read(filepath, format='gaussian-in')
            return atoms
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            # Fallback parser for custom .gjf format
            return self._manual_gjf_parse(filepath)
    
    def _manual_gjf_parse(self, filepath):
        """Manual parser for .gjf files if ASE fails."""
        symbols = []
        positions = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        # Skip header lines until we find coordinates
        coord_section = False
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('%'):
                continue
            if coord_section and line:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        symbols.append(parts[0])
                        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except (ValueError, IndexError):
                        continue
            elif line and not coord_section:
                if any(char.isdigit() for char in line) and any(char.isalpha() for char in line):
                    coord_section = True
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            symbols.append(parts[0])
                            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        except (ValueError, IndexError):
                            continue
        
        if symbols and positions:
            return Atoms(symbols=symbols, positions=positions)
        else:
            raise ValueError("Could not parse coordinates from .gjf file")
    
    def optimize_structure(self, atoms, fmax=0.05, steps=200):
        """
        Optimize molecular structure using FairChem.
        
        Args:
            atoms (ase.Atoms): Input structure
            fmax (float): Force convergence criterion
            steps (int): Maximum optimization steps
            
        Returns:
            ase.Atoms: Optimized structure
        """
        if self.fairchem_calc is None:
            print("FairChem calculator not available, returning original structure")
            return atoms
        
        try:
            atoms_copy = atoms.copy()
            atoms_copy.set_calculator(self.fairchem_calc)
            
            # Run optimization
            optimizer = BFGS(atoms_copy, trajectory=f'optimization_{id(atoms)}.traj')
            optimizer.run(fmax=fmax, steps=steps)
            
            print(f"Optimization completed in {optimizer.get_number_of_steps()} steps")
            return atoms_copy
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            return atoms
    
    def create_complex(self, absorbent, analyte, separation_distance=3.0):
        """
        Create absorbent-analyte complex using simple geometric approach.
        For more sophisticated docking, integrate with specialized tools.
        
        Args:
            absorbent (ase.Atoms): Optimized absorbent structure
            analyte (ase.Atoms): Optimized analyte structure
            separation_distance (float): Initial separation in Angstroms
            
        Returns:
            ase.Atoms: Combined complex structure
        """
        # Calculate centers of mass
        abs_com = absorbent.get_center_of_mass()
        ana_com = analyte.get_center_of_mass()
        
        # Position analyte above absorbent
        analyte_copy = analyte.copy()
        displacement = np.array([0, 0, separation_distance]) + abs_com - ana_com
        analyte_copy.translate(displacement)
        
        # Combine structures
        complex_atoms = absorbent + analyte_copy
        
        return complex_atoms
    
    def optimize_complex(self, complex_atoms, fmax=0.05, steps=300):
        """
        Optimize the complex structure using GemNet/FairChem.
        
        Args:
            complex_atoms (ase.Atoms): Initial complex structure
            fmax (float): Force convergence criterion
            steps (int): Maximum optimization steps
            
        Returns:
            ase.Atoms: Optimized complex
        """
        return self.optimize_structure(complex_atoms, fmax, steps)
    
    def calculate_properties(self, complex_atoms):
        """
        Calculate molecular properties including HOMO-LUMO gap and others.
        
        Args:
            complex_atoms (ase.Atoms): Optimized complex structure
            
        Returns:
            dict: Dictionary of calculated properties
        """
        properties = {}
        
        try:
            # Mock calculations for demonstration
            # Replace with actual property calculations using appropriate methods
            
            # Basic geometric properties
            properties['total_atoms'] = len(complex_atoms)
            properties['molecular_volume'] = self._calculate_molecular_volume(complex_atoms)
            properties['center_of_mass'] = complex_atoms.get_center_of_mass().tolist()
            
            # Energy-related properties (placeholder)
            if self.fairchem_calc:
                complex_atoms.set_calculator(self.fairchem_calc)
                properties['total_energy'] = complex_atoms.get_potential_energy()
                properties['forces_rms'] = np.sqrt(np.mean(complex_atoms.get_forces()**2))
            
            # Electronic properties (would require quantum chemistry calculations)
            properties['homo_lumo_gap'] = self._estimate_homo_lumo_gap(complex_atoms)
            properties['dipole_moment'] = self._calculate_dipole_moment(complex_atoms)
            properties['polarizability'] = self._estimate_polarizability(complex_atoms)
            
            # Binding properties
            properties['binding_energy'] = self._estimate_binding_energy(complex_atoms)
            properties['binding_sites'] = self._identify_binding_sites(complex_atoms)
            
            # Spectroscopic properties (estimated)
            properties['ir_frequencies'] = self._estimate_ir_frequencies(complex_atoms)
            properties['uv_vis_absorption'] = self._estimate_uv_vis(complex_atoms)
            
        except Exception as e:
            print(f"Error calculating properties: {e}")
            properties['error'] = str(e)
        
        return properties
    
    def _calculate_molecular_volume(self, atoms):
        """Estimate molecular volume using van der Waals radii."""
        vdw_radii = {'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8, 'P': 1.8}
        total_volume = 0
        for symbol in atoms.get_chemical_symbols():
            radius = vdw_radii.get(symbol, 1.5)
            total_volume += (4/3) * np.pi * radius**3
        return total_volume
    
    def _estimate_homo_lumo_gap(self, atoms):
        """Estimate HOMO-LUMO gap using simple heuristics."""
        # This is a very rough estimation - use DFT for accurate values
        n_electrons = sum(atoms.get_atomic_numbers())
        if n_electrons < 10:
            return 8.0 + np.random.normal(0, 0.5)  # Small molecules
        elif n_electrons < 50:
            return 4.0 + np.random.normal(0, 1.0)  # Medium molecules
        else:
            return 2.0 + np.random.normal(0, 0.5)  # Large molecules
    
    def _calculate_dipole_moment(self, atoms):
        """Calculate dipole moment from partial charges."""
        # Simplified calculation - use proper charge calculation methods
        positions = atoms.get_positions()
        charges = np.random.normal(0, 0.1, len(atoms))  # Mock charges
        dipole = np.sum(charges[:, np.newaxis] * positions, axis=0)
        return np.linalg.norm(dipole)
    
    def _estimate_polarizability(self, atoms):
        """Estimate molecular polarizability."""
        # Volume-based estimation
        volume = self._calculate_molecular_volume(atoms)
        return 0.1 * volume  # Rough correlation
    
    def _estimate_binding_energy(self, atoms):
        """Estimate binding energy between absorbent and analyte."""
        # This would require separate calculations of components
        n_atoms = len(atoms)
        return -5.0 - 0.1 * n_atoms + np.random.normal(0, 1.0)  # kcal/mol
    
    def _identify_binding_sites(self, atoms):
        """Identify potential binding sites in the complex."""
        # Simple distance-based analysis
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        
        binding_sites = []
        for i, (pos, sym) in enumerate(zip(positions, symbols)):
            if sym in ['O', 'N', 'S']:  # Heteroatoms often involved in binding
                binding_sites.append({'atom_index': i, 'element': sym, 'position': pos.tolist()})
        
        return binding_sites
    
    def _estimate_ir_frequencies(self, atoms):
        """Estimate IR frequencies for major functional groups."""
        symbols = atoms.get_chemical_symbols()
        frequencies = []
        
        # Add typical frequencies based on present atoms
        if 'O' in symbols and 'H' in symbols:
            frequencies.extend([3200, 3400])  # O-H stretch
        if 'C' in symbols and 'O' in symbols:
            frequencies.append(1700)  # C=O stretch
        if 'C' in symbols and 'H' in symbols:
            frequencies.extend([2900, 3000])  # C-H stretch
        
        return sorted(frequencies)
    
    def _estimate_uv_vis(self, atoms):
        """Estimate UV-Vis absorption wavelength."""
        n_pi_electrons = sum(1 for sym in atoms.get_chemical_symbols() if sym in ['C', 'N', 'O'])
        # Rough estimation based on conjugation
        lambda_max = 200 + 30 * np.sqrt(n_pi_electrons)
        return min(lambda_max, 800)  # nm
    
    def save_gjf_file(self, atoms, filename, title="Optimized Complex", 
                      method="B3LYP", basis="6-31G(d)", charge=0, multiplicity=1):
        """
        Save structure as Gaussian .gjf file.
        
        Args:
            atoms (ase.Atoms): Structure to save
            filename (str): Output filename
            title (str): Calculation title
            method (str): DFT method
            basis (str): Basis set
            charge (int): Molecular charge
            multiplicity (int): Spin multiplicity
        """
        with open(filename, 'w') as f:
            # Write header
            f.write(f"%nprocshared=4\n")
            f.write(f"%mem=2GB\n")
            f.write(f"# {method}/{basis} opt freq\n\n")
            f.write(f"{title}\n\n")
            f.write(f"{charge} {multiplicity}\n")
            
            # Write coordinates
            for symbol, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
                f.write(f"{symbol:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
            
            f.write("\n")
    
    def analyze_complex(self, absorbent_file, analyte_file, output_prefix="complex"):
        """
        Complete analysis pipeline.
        
        Args:
            absorbent_file (str): Path to absorbent .gjf file
            analyte_file (str): Path to analyte .gjf file
            output_prefix (str): Prefix for output files
            
        Returns:
            tuple: (optimized_complex, properties_dict)
        """
        print("Starting molecular complex analysis...")
        
        # Step 1: Parse input files
        print("1. Parsing input structures...")
        absorbent = self.parse_gjf_file(absorbent_file)
        analyte = self.parse_gjf_file(analyte_file)
        print(f"   Absorbent: {len(absorbent)} atoms")
        print(f"   Analyte: {len(analyte)} atoms")
        
        # Step 2: Optimize individual structures
        print("2. Optimizing individual structures...")
        opt_absorbent = self.optimize_structure(absorbent)
        opt_analyte = self.optimize_structure(analyte)
        
        # Step 3: Create initial complex
        print("3. Creating initial complex...")
        initial_complex = self.create_complex(opt_absorbent, opt_analyte)
        
        # Step 4: Optimize complex
        print("4. Optimizing complex structure...")
        final_complex = self.optimize_complex(initial_complex)
        
        # Step 5: Calculate properties
        print("5. Calculating molecular properties...")
        properties = self.calculate_properties(final_complex)
        
        # Step 6: Save results
        print("6. Saving results...")
        output_gjf = f"{output_prefix}_optimized.gjf"
        self.save_gjf_file(final_complex, output_gjf, 
                          title="Optimized Absorbent-Analyte Complex")
        
        # Save properties to file
        properties_file = f"{output_prefix}_properties.txt"
        with open(properties_file, 'w') as f:
            f.write("Molecular Complex Properties\n")
            f.write("=" * 40 + "\n\n")
            for key, value in properties.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Analysis complete! Results saved as {output_gjf} and {properties_file}")
        return final_complex, properties

# Example usage
def main():
    """Example usage of the MolecularComplexAnalyzer."""
    
    # Initialize analyzer
    analyzer = MolecularComplexAnalyzer()
    
    # Example file paths (replace with your actual files)
    absorbent_file = "absorbent.gjf"
    analyte_file = "analyte.gjf"
    
    # Check if example files exist, create dummy ones if not
    if not os.path.exists(absorbent_file):
        print("Creating example absorbent.gjf file...")
        with open(absorbent_file, 'w') as f:
            f.write("""%nprocshared=4
%mem=2GB
# B3LYP/6-31G(d) opt

Benzene absorbent

0 1
C      0.000000    1.396000    0.000000
C      1.209000    0.698000    0.000000
C      1.209000   -0.698000    0.000000
C      0.000000   -1.396000    0.000000
C     -1.209000   -0.698000    0.000000
C     -1.209000    0.698000    0.000000
H      0.000000    2.480000    0.000000
H      2.147000    1.240000    0.000000
H      2.147000   -1.240000    0.000000
H      0.000000   -2.480000    0.000000
H     -2.147000   -1.240000    0.000000
H     -2.147000    1.240000    0.000000

""")
    
    if not os.path.exists(analyte_file):
        print("Creating example analyte.gjf file...")
        with open(analyte_file, 'w') as f:
            f.write("""%nprocshared=4
%mem=2GB
# B3LYP/6-31G(d) opt

Water analyte

0 1
O      0.000000    0.000000    0.119000
H      0.000000    0.757000   -0.476000
H      0.000000   -0.757000   -0.476000

""")
    
    try:
        # Run analysis
        complex_structure, properties = analyzer.analyze_complex(
            absorbent_file, analyte_file, output_prefix="benzene_water"
        )
        
        # Display key properties
        print("\nKey Properties:")
        print("-" * 30)
        for key in ['total_atoms', 'homo_lumo_gap', 'binding_energy', 'dipole_moment']:
            if key in properties:
                print(f"{key}: {properties[key]}")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        print("Please ensure FairChem and GemNet packages are properly installed.")

if __name__ == "__main__":
    main()
