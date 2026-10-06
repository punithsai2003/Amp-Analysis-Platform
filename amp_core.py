"""
AMP Analysis Platform - core science logic.

Pure functions ported 1:1 from the original Flask backend (backend/app.py).
No Streamlit imports here, so this module is unit-testable on its own.
"""

from __future__ import annotations

import random
import re
import time
from collections import Counter
from typing import Any

import requests

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

AA_WEIGHTS = {
    "A": 89, "R": 174, "N": 132, "D": 133, "C": 121, "E": 147,
    "Q": 146, "G": 75, "H": 155, "I": 131, "L": 131, "K": 146,
    "M": 149, "F": 165, "P": 115, "S": 105, "T": 119, "W": 204,
    "Y": 181, "V": 117,
}

ESMFOLD_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
HF_PROTGPT2_URL = "https://api-inference.huggingface.co/models/nferruz/ProtGPT2"


# ==========================================
# FILE PARSING
# ==========================================

def parse_fasta(content: str) -> list[str]:
    """Parse FASTA format into a list of sequences."""
    sequences: list[str] = []
    current: list[str] = []

    for line in content.split("\n"):
        line = line.strip()
        if line.startswith(">"):
            if current:
                sequences.append("".join(current))
            current = []
        elif line:
            current.append(line.upper())

    if current:
        sequences.append("".join(current))

    return sequences


def parse_csv(content: str) -> list[str]:
    """Parse CSV, pulling the first valid amino-acid string out of each row."""
    sequences: list[str] = []

    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        parts = [p.strip().upper() for p in line.split(",")]

        header_words = ("SEQUENCE", "SEQ", "PEPTIDE", "AMP", "ID", "NAME")
        if any(word in parts[0].upper() for word in header_words):
            continue

        for part in parts:
            if part and len(part) >= 5 and all(c in AA_ALPHABET for c in part):
                sequences.append(part)
                break

    return sequences


def parse_plain_text(content: str) -> list[str]:
    """One sequence per line, whitespace separated. Used for pasted input."""
    sequences: list[str] = []
    for token in re.split(r"[\s,;]+", content.upper()):
        token = token.strip()
        if token and len(token) >= 5 and all(c in AA_ALPHABET for c in token):
            sequences.append(token)
    return sequences


def parse_uploaded(filename: str, content: str) -> list[str]:
    """Dispatch on file extension."""
    lower = filename.lower()
    if lower.endswith((".fasta", ".fa", ".faa")):
        return parse_fasta(content)
    if lower.endswith((".csv", ".tsv")):
        return parse_csv(content)
    if lower.endswith(".txt"):
        # A .txt file may still hold FASTA
        if content.lstrip().startswith(">"):
            return parse_fasta(content)
        return parse_plain_text(content)
    raise ValueError(f"Unsupported file type: {filename}")


# ==========================================
# SEQUENCE GENERATION
# ==========================================

def analyze_sequence_patterns(sequences: list[str]) -> dict[str, Any]:
    """Extract length distribution, N-terminal motifs and bigram transitions."""
    stats: dict[str, Any] = {
        "lengths": [len(s) for s in sequences],
        "n_terminal_patterns": [],
        "c_terminal_patterns": [],
        "bigrams": {},
        "amino_acid_freq": {},
    }

    for seq in sequences:
        if len(seq) >= 3:
            stats["n_terminal_patterns"].append(seq[:3])
        if len(seq) >= 5:
            stats["c_terminal_patterns"].append(seq[-5:])

    for seq in sequences:
        for i in range(len(seq) - 1):
            stats["bigrams"].setdefault(seq[i], []).append(seq[i + 1])

    all_aas = "".join(sequences)
    counts = Counter(all_aas)
    stats["amino_acid_freq"] = {aa: counts.get(aa, 0) for aa in AA_ALPHABET}

    # Guard against training data with no usable N-terminal motifs
    if not stats["n_terminal_patterns"]:
        stats["n_terminal_patterns"] = ["KLW"]
    if not stats["lengths"]:
        stats["lengths"] = [25]

    return stats


def choose_next_amino_acid(current_aa: str, bigrams: dict[str, list[str]]) -> str:
    """Sample the next residue from observed transitions, falling back to AMP-common residues."""
    options = bigrams.get(current_aa)
    if options:
        return random.choice(options)
    return random.choice("KLRWFGAIV")


def is_valid_generated_sequence(sequence: str, training_sequences: set[str] | list[str]) -> bool:
    """Reject sequences that are too short/long, duplicated, or have 5x homopolymer runs."""
    if not 10 <= len(sequence) <= 100:
        return False
    if sequence in training_sequences:
        return False
    if "X" in sequence:
        return False
    for aa in AA_ALPHABET:
        if aa * 5 in sequence:
            return False
    return True


def generate_via_patterns(
    training_sequences: list[str],
    num_generate: int,
    progress_cb=None,
) -> list[str]:
    """Markov/bigram generation seeded on the uploaded training set."""
    stats = analyze_sequence_patterns(training_sequences)
    training_set = set(training_sequences)

    generated: list[str] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = max(num_generate * 20, 200)

    while len(generated) < num_generate and attempts < max_attempts:
        attempts += 1

        length = random.choice(stats["lengths"])
        length = max(10, min(100, length))
        sequence = random.choice(stats["n_terminal_patterns"])

        guard = 0
        while len(sequence) < length and guard < 200:
            guard += 1
            sequence += choose_next_amino_acid(sequence[-1], stats["bigrams"])

        sequence = sequence[:length]

        if sequence in seen:
            continue
        if is_valid_generated_sequence(sequence, training_set):
            generated.append(sequence)
            seen.add(sequence)
            if progress_cb:
                progress_cb(len(generated), num_generate)

    return generated


def generate_via_huggingface(
    training_sequences: list[str],
    num_generate: int,
    api_key: str,
    progress_cb=None,
) -> list[str] | None:
    """Generate with the ProtGPT2 Inference API. Returns None if unusable."""
    if not api_key:
        return None

    headers = {"Authorization": f"Bearer {api_key}"}
    sample_prompts = random.sample(training_sequences, min(10, len(training_sequences)))
    generated: list[str] = []

    for i in range(num_generate):
        prompt = random.choice(sample_prompts)[:10]
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 50,
                "temperature": 0.8,
                "top_p": 0.9,
                "num_return_sequences": 1,
            },
        }

        try:
            response = requests.post(HF_PROTGPT2_URL, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and result:
                    seq = result[0].get("generated_text", "").upper()
                    seq = "".join(c for c in seq if c in AA_ALPHABET)
                    if 10 <= len(seq) <= 100:
                        generated.append(seq)
                        if progress_cb:
                            progress_cb(len(generated), num_generate)
            elif response.status_code in (401, 403):
                # Bad token - no point retrying
                return None
        except requests.RequestException:
            continue

        time.sleep(1)

    return generated or None


# ==========================================
# QUICK SCREENING
# ==========================================

def quick_screen_sequence(sequence: str) -> dict[str, Any]:
    """Fast heuristic scoring used to rank candidates before deep analysis."""
    length = len(sequence)
    if length == 0:
        raise ValueError("Empty sequence")

    pos_charged = sum(sequence.count(aa) for aa in "KRH")
    neg_charged = sum(sequence.count(aa) for aa in "DE")
    hydrophobic = sum(sequence.count(aa) for aa in "AILMFWV")
    aromatic = sum(sequence.count(aa) for aa in "FWY")

    net_charge = (pos_charged - neg_charged) / length
    hydrophobic_ratio = hydrophobic / length
    aromatic_ratio = aromatic / length

    amp_score = 0.0
    if net_charge > 0.15:
        amp_score += 0.3
    elif net_charge > 0.05:
        amp_score += 0.15
    if 0.4 <= hydrophobic_ratio <= 0.6:
        amp_score += 0.3
    elif 0.3 <= hydrophobic_ratio <= 0.7:
        amp_score += 0.15
    if 10 <= length <= 50:
        amp_score += 0.2
    if aromatic_ratio > 0.1:
        amp_score += 0.2

    quality_score = amp_score * 0.6
    if 15 <= length <= 40:
        quality_score += 0.2
    if 0.4 <= hydrophobic_ratio <= 0.55:
        quality_score += 0.1
    if 0.1 <= net_charge <= 0.3:
        quality_score += 0.1

    novelty_score = 0.8

    return {
        "length": length,
        "net_charge": round(net_charge, 3),
        "hydrophobic_ratio": round(hydrophobic_ratio, 3),
        "aromatic_ratio": round(aromatic_ratio, 3),
        "amp_score_quick": round(amp_score, 3),
        "quality_score": round(quality_score, 3),
        "novelty_score": round(novelty_score, 3),
        "overall_score": round((quality_score + novelty_score + amp_score) / 3, 3),
    }


# ==========================================
# PHYSICOCHEMICAL + FUNCTIONAL PREDICTIONS
# ==========================================

def calculate_physicochemical(sequence: str) -> dict[str, Any]:
    """Approximate molecular weight, isoelectric point and secondary-structure fractions."""
    mw = sum(AA_WEIGHTS.get(aa, 110) for aa in sequence)

    pos = sum(sequence.count(aa) for aa in "KRH")
    neg = sum(sequence.count(aa) for aa in "DE")
    pi = 7.0 + ((pos - neg) / len(sequence)) * 5.0

    return {
        "molecular_weight": round(mw, 1),
        "isoelectric_point": round(max(3.0, min(11.0, pi)), 2),
        "instability_index": 35.0,
        "helix": 35.0,
        "sheet": 10.0,
        "coil": 55.0,
    }


def predict_amp_detailed(sequence: str) -> dict[str, Any]:
    """Charge- and hydrophobicity-weighted AMP likelihood."""
    pos = sum(sequence.count(aa) for aa in "KRH")
    hydro = sum(sequence.count(aa) for aa in "AILMFWV")
    length = len(sequence)

    charge_score = min(pos / length * 2, 1.0)
    hydro_score = min(hydro / length * 1.5, 1.0)

    return {"ampScore": round(charge_score * 0.6 + hydro_score * 0.4, 3)}


def predict_toxicity(sequence: str) -> dict[str, Any]:
    toxic_count = sum(sequence.count(m) for m in ("WW", "FF", "YY", "CCC"))
    return {"toxic": "Toxic" if toxic_count > 2 else "Non-Toxic"}


def predict_allergenicity(sequence: str) -> dict[str, Any]:
    acidic = sum(sequence.count(aa) for aa in "DE") / len(sequence)
    return {"allergen": "Allergen" if acidic > 0.2 else "Non-Allergen"}


def predict_hemolytic(sequence: str) -> dict[str, Any]:
    hydro = sum(sequence.count(aa) for aa in "AILMFWV") / len(sequence)
    return {"hemolytic": "Hemolytic" if hydro > 0.6 else "Non-Hemolytic"}


# ==========================================
# STRUCTURE PREDICTION
# ==========================================

def extract_alphafold_metrics(pdb_content: str) -> dict[str, float]:
    """
    Pull pLDDT / pTM / ipTM out of a PDB file.

    pLDDT lives in the B-factor column (chars 61-66) of ATOM records.
    pTM / ipTM are only present if the predictor wrote REMARK lines; ESMFold
    does not, so they are estimated from mean pLDDT.
    """
    plddt_scores: list[float] = []
    ptm_score: float | None = None
    iptm_score: float | None = None

    for line in pdb_content.split("\n"):
        if line.startswith("ATOM"):
            try:
                plddt_scores.append(float(line[60:66].strip()))
            except (ValueError, IndexError):
                pass
        elif line.startswith("REMARK"):
            upper = line.upper()
            if "PTM" in upper:
                numbers = re.findall(r"\d+\.\d+", line)
                if numbers:
                    ptm_score = float(numbers[0])
            if "IPTM" in upper or "INTERFACE" in upper:
                numbers = re.findall(r"\d+\.\d+", line)
                if numbers:
                    iptm_score = float(numbers[0])

    avg_plddt = sum(plddt_scores) / len(plddt_scores) if plddt_scores else 50.0

    if ptm_score is None:
        if avg_plddt > 70:
            ptm_score = 0.6 + (avg_plddt - 70) / 30 * 0.3
        elif avg_plddt > 50:
            ptm_score = 0.4 + (avg_plddt - 50) / 20 * 0.2
        else:
            ptm_score = 0.2 + avg_plddt / 50 * 0.2

    if iptm_score is None:
        iptm_score = ptm_score * 0.9

    return {
        "pLDDT": round(avg_plddt, 1),
        "pTM": round(min(max(ptm_score, 0.0), 1.0), 3),
        "ipTM": round(min(max(iptm_score, 0.0), 1.0), 3),
        "plddt_per_atom": plddt_scores,
    }


def predict_structure_esmfold(sequence: str, timeout: int = 120) -> dict[str, Any]:
    """
    Fold a sequence with the public ESMFold API.

    Returns a dict with pLDDT/pTM/ipTM, the raw PDB text, and an error string
    if the call failed (in which case conservative placeholder metrics are used).
    """
    fallback = {
        "pLDDT": 50.0,
        "pTM": 0.5,
        "ipTM": 0.5,
        "pdb_content": None,
        "structure_error": None,
    }

    try:
        response = requests.post(ESMFOLD_URL, data=sequence, timeout=timeout)
    except requests.Timeout:
        fallback["structure_error"] = f"ESMFold timed out after {timeout}s"
        return fallback
    except requests.RequestException as exc:
        fallback["structure_error"] = f"Network error: {exc}"
        return fallback

    if response.status_code != 200:
        fallback["structure_error"] = f"ESMFold returned HTTP {response.status_code}"
        return fallback

    pdb_content = response.text

    if not pdb_content or len(pdb_content) < 100:
        fallback["structure_error"] = "ESMFold response too short to be a PDB file"
        return fallback

    if not pdb_content.startswith("HEADER") and "ATOM" not in pdb_content:
        fallback["structure_error"] = "ESMFold response was not in PDB format"
        return fallback

    metrics = extract_alphafold_metrics(pdb_content)
    metrics["pdb_content"] = pdb_content
    metrics["structure_error"] = None
    return metrics


def deep_analysis_sequence(
    sequence: str,
    seq_id: str,
    include_structure: bool = True,
    retries: int = 2,
) -> dict[str, Any]:
    """Full characterisation of a single candidate."""
    result: dict[str, Any] = {
        "id": seq_id,
        "sequence": sequence,
        "length": len(sequence),
    }

    result.update(calculate_physicochemical(sequence))

    # Keep the result schema stable whether or not folding runs, so downstream
    # code (tables, CSV export) never has to special-case missing keys.
    result.update({
        "pLDDT": None,
        "pTM": None,
        "ipTM": None,
        "pdb_content": None,
        "structure_error": "Structure prediction skipped",
    })

    if include_structure:
        structure: dict[str, Any] = {}
        for attempt in range(retries):
            if attempt > 0:
                time.sleep(5)
            structure = predict_structure_esmfold(sequence)
            if structure.get("pdb_content"):
                break
        result.update(structure)

    result.update(predict_amp_detailed(sequence))
    result.update(predict_toxicity(sequence))
    result.update(predict_allergenicity(sequence))
    result.update(predict_hemolytic(sequence))

    return result


# ==========================================
# EXPORT
# ==========================================

CSV_HEADER = (
    "ID,Sequence,Length,pLDDT,pTM,ipTM,MW,pI,Helix%,"
    "AMP_Score,Toxicity,Allergen,Hemolytic"
)


def _cell(value: Any) -> str:
    """Render a CSV cell, mapping missing/None values to N/A rather than 'None'."""
    return "N/A" if value is None else str(value)


def results_to_csv(results: list[dict[str, Any]]) -> str:
    """Serialise deep-analysis results to CSV, matching the original endpoint."""
    lines = [CSV_HEADER]
    for r in results:
        cells = [
            r["id"], r["sequence"], r["length"],
            r.get("pLDDT"), r.get("pTM"), r.get("ipTM"),
            r.get("molecular_weight"), r.get("isoelectric_point"),
            r.get("helix"), r.get("ampScore"),
            r.get("toxic"), r.get("allergen"), r.get("hemolytic"),
        ]
        lines.append(",".join(_cell(c) for c in cells))
    return "\n".join(lines)


def screened_to_csv(screened: list[dict[str, Any]]) -> str:
    """Serialise the full screened candidate pool."""
    lines = [
        "Rank,ID,Sequence,Length,Net_Charge,Hydrophobic_Ratio,"
        "Aromatic_Ratio,AMP_Score_Quick,Quality_Score,Novelty_Score,Overall_Score"
    ]
    for rank, s in enumerate(screened, 1):
        lines.append(
            f"{rank},{s['id']},{s['sequence']},{s['length']},"
            f"{s['net_charge']},{s['hydrophobic_ratio']},{s['aromatic_ratio']},"
            f"{s['amp_score_quick']},{s['quality_score']},{s['novelty_score']},"
            f"{s['overall_score']}"
        )
    return "\n".join(lines)
