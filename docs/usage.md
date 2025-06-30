# Usage Guide

This guide explains how to use the Molecular Complex Analysis Pipeline to analyze molecular complexes, calculate properties, and generate outputs.

---

## 📦 1. Python Usage (Script/Notebook)

You can import the `MolecularComplexAnalyzer` class in any Python script:

```python
from molecular_analyzer import MolecularComplexAnalyzer

# Initialize the analyzer
analyzer = MolecularComplexAnalyzer()

# Analyze a molecular complex
structure, properties = analyzer.analyze_complex(
    absorbent_path="examples/benzene.gjf",
    analyte_path="examples/water.gjf",
    output_prefix="benzene_water"
)

# Output the results
print("Computed Properties:", properties)
