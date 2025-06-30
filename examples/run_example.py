from molecular_analyzer import MolecularComplexAnalyzer

analyzer = MolecularComplexAnalyzer()
structure, props = analyzer.analyze_complex("examples/benzene.gjf", "examples/water.gjf", "test_output")
print(props)
