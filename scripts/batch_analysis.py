import os
from molecular_analyzer import MolecularComplexAnalyzer

analyzer = MolecularComplexAnalyzer()

folder = "input_folder"
for absorbent in os.listdir(folder):
    if absorbent.endswith(".gjf"):
        for analyte in os.listdir(folder):
            if analyte.endswith(".gjf") and analyte != absorbent:
                prefix = f"{absorbent[:-4]}_{analyte[:-4]}"
                structure, props = analyzer.analyze_complex(
                    os.path.join(folder, absorbent),
                    os.path.join(folder, analyte),
                    output_prefix=prefix
                )
                print(f"{prefix} → {props}")
