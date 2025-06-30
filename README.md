# Molecular Complex Analysis Pipeline

A comprehensive Python toolkit for analyzing molecular complexes using FairChem and GemNet models. This pipeline optimizes absorbent and analyte structures, forms complexes, and predicts various molecular properties including HOMO-LUMO gaps, binding energies, and spectroscopic properties.

## 🚀 Features

- **Structure Optimization**: Uses FairChem models for molecular geometry optimization
- **Complex Formation**: Automated creation of absorbent-analyte complexes
- **Property Prediction**: Comprehensive molecular property calculations
- **Multiple Output Formats**: Generates optimized .gjf files and detailed property reports
- **Extensible Design**: Easy to add new property calculators and optimization methods

## 🏗️ Key Capabilities

### Structural Analysis
- Gaussian .gjf file parsing and writing
- Geometry optimization using state-of-the-art ML models
- Complex formation with configurable binding geometries

### Property Calculations
- **Electronic Properties**: HOMO-LUMO gap, dipole moment, polarizability
- **Energetic Properties**: Binding energy, total energy, force analysis
- **Spectroscopic Properties**: IR frequencies, UV-Vis absorption
- **Geometric Properties**: Molecular volume, center of mass, binding sites

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Quick Install
```bash
git clone https://github.com/Safi-ullah-majid/molecular-complex-analyzer.git
cd molecular-complex-analyzer
pip install -e .
