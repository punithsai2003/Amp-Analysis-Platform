"""Sequence validation"""

VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')

def validate_sequence(sequence):
    """Validate protein sequence"""
    sequence = sequence.upper().strip()
    
    if not sequence:
        return False, "Empty sequence"
    
    if len(sequence) < 5:
        return False, "Sequence too short (minimum 5 amino acids)"
    
    if len(sequence) > 1000:
        return False, "Sequence too long (maximum 1000 amino acids)"
    
    invalid_chars = set(sequence) - VALID_AA
    if invalid_chars:
        return False, f"Invalid characters: {invalid_chars}"
    
    return True, "Valid"