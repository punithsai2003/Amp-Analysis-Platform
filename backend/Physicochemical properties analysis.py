"""Physicochemical properties analysis"""
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def analyze_physicochemical(sequence):
    """Calculate all physicochemical properties"""
    try:
        analyzer = ProteinAnalysis(sequence)
        
        return {
            'molecular_weight': round(analyzer.molecular_weight(), 2),
            'isoelectric_point': round(analyzer.isoelectric_point(), 2),
            'aromaticity': round(analyzer.aromaticity(), 4),
            'instability_index': round(analyzer.instability_index(), 2),
            'gravy': round(analyzer.gravy(), 4),
            'helix_fraction': round(analyzer.secondary_structure_fraction()[0] * 100, 2),
            'sheet_fraction': round(analyzer.secondary_structure_fraction()[1] * 100, 2),
            'coil_fraction': round(analyzer.secondary_structure_fraction()[2] * 100, 2)
        }
    except Exception as e:
        raise Exception(f"Physicochemical analysis failed: {e}")