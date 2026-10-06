"""Functional predictions"""
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def predict_antimicrobial(sequence):
    """Predict antimicrobial activity"""
    # Simple heuristic-based prediction
    # In production, replace with actual CAMPR3 API or ML model
    
    analyzer = ProteinAnalysis(sequence)
    
    # AMP characteristics: positive charge, amphipathic, moderate hydrophobicity
    pos_residues = sequence.count('K') + sequence.count('R')
    hydrophobic = sequence.count('A') + sequence.count('L') + sequence.count('I') + sequence.count('V')
    length = len(sequence)
    
    charge_score = min(pos_residues / length * 2, 1.0)
    hydro_score = min(hydrophobic / length * 2, 1.0)
    
    amp_score = (charge_score + hydro_score) / 2
    
    return {
        'ampScore': round(amp_score, 3),
        'amp_classification': 'AMP' if amp_score > 0.5 else 'Non-AMP'
    }

def predict_toxicity(sequence):
    """Predict toxicity"""
    # Simple rule-based
    toxic_motifs = ['WW', 'FF', 'YY', 'CCC', 'PPP']
    toxic_count = sum(sequence.count(motif) for motif in toxic_motifs)
    
    is_toxic = toxic_count > 2 or sequence.count('C') > len(sequence) * 0.15
    
    return {
        'toxic': 'Toxic' if is_toxic else 'Non-Toxic'
    }

def predict_allergenicity(sequence):
    """Predict allergenicity"""
    # Simple heuristic
    allergenic_residues = sequence.count('E') + sequence.count('D') + sequence.count('Q')
    ratio = allergenic_residues / len(sequence)
    
    return {
        'allergen': 'Allergen' if ratio > 0.2 else 'Non-Allergen'
    }

def predict_hemolytic(sequence):
    """Predict hemolytic activity"""
    analyzer = ProteinAnalysis(sequence)
    gravy = analyzer.gravy()
    
    return {
        'hemolytic': 'Hemolytic' if gravy > 0.5 else 'Non-Hemolytic'
    }