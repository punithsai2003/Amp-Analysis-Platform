"""File parsing utilities"""
from Bio import SeqIO

def parse_fasta(filepath):
    """Parse FASTA file"""
    sequences = []
    for record in SeqIO.parse(filepath, 'fasta'):
        sequences.append({
            'id': record.id,
            'sequence': str(record.seq).upper()
        })
    return sequences

def parse_csv(filepath):
    """Parse CSV file"""
    import csv
    sequences = []
    
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for i, row in enumerate(reader, 1):
            if len(row) >= 2:
                sequences.append({
                    'id': row[0] or f'Seq_{i}',
                    'sequence': row[1].strip().upper()
                })
    
    return sequences