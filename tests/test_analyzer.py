import unittest
from molecular_analyzer import MolecularComplexAnalyzer

class TestAnalyzer(unittest.TestCase):
    def test_initialization(self):
        analyzer = MolecularComplexAnalyzer()
        self.assertIsNotNone(analyzer)

if __name__ == '__main__':
    unittest.main()
