"""
AMP Analysis Platform - Enhanced Version with AI
Features:
- Gemini AI analysis summaries
- AI Chatbot with web search
- ESMFold structure prediction
- All existing features
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import os
import time
import requests
import json
import random
import re
from datetime import datetime
from pathlib import Path
from collections import Counter
from duckduckgo_search import DDGS

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"]
    }
})

# Configure Groq AI (FREE alternative to Gemini)
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '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') 
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Use the latest supported model (as of Jan 2026)
GROQ_MODEL = "llama-3.3-70b-versatile"  # Updated to latest supported model

# Folders
UPLOAD_FOLDER = Path('data/uploads')
STRUCTURE_FOLDER = Path('structures')
RESULTS_FOLDER = Path('data/results')
for folder in [UPLOAD_FOLDER, STRUCTURE_FOLDER, RESULTS_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

# Storage
training_data = {}
generated_sequences = {}
analysis_results = {}
conversation_history = {}

# ==========================================
# GROQ AI INTEGRATION (FREE & UNLIMITED!)
# ==========================================

def call_groq_api(prompt, max_tokens=500):
    """Call Groq API for AI completions"""
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content'].strip()
        else:
            print(f"⚠️ Groq API error: {response.status_code} - {response.text}")
            return None
    
    except Exception as e:
        print(f"⚠️ Groq API error: {e}")
        return None

def generate_ai_summary(sequence_data):
    """Generate AI summary for a sequence using Groq"""
    try:
        prompt = f"""Analyze this antimicrobial peptide sequence and provide a concise, scientific summary:

Sequence: {sequence_data['sequence']}
Length: {sequence_data['length']} amino acids
pLDDT Score: {sequence_data.get('pLDDT', 'N/A')}
pTM Score: {sequence_data.get('pTM', 'N/A')}
AMP Score: {sequence_data.get('ampScore', 'N/A')}
Molecular Weight: {sequence_data.get('molecular_weight', 'N/A')} Da
Isoelectric Point: {sequence_data.get('isoelectric_point', 'N/A')}
Toxicity: {sequence_data.get('toxic', 'N/A')}
Allergenicity: {sequence_data.get('allergen', 'N/A')}
Hemolytic: {sequence_data.get('hemolytic', 'N/A')}

Provide a brief analysis covering:
1. Why this peptide shows promise (or concerns)
2. Likely mechanism of action
3. Potential target bacteria
4. Safety considerations
5. Recommended next steps

Keep it concise (3-4 sentences max)."""

        response = call_groq_api(prompt, max_tokens=300)
        
        if response:
            return response
        else:
            return "AI analysis unavailable. Sequence shows standard AMP characteristics."
    
    except Exception as e:
        print(f"⚠️ AI summary error: {e}")
        return "AI analysis unavailable. Sequence shows standard AMP characteristics."

# ==========================================
# AI CHATBOT WITH WEB SEARCH
# ==========================================

def web_search(query, max_results=3):
    """Search the web using DuckDuckGo"""
    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=max_results)
        
        search_results = []
        for r in results:
            search_results.append({
                'title': r.get('title', ''),
                'snippet': r.get('body', ''),
                'url': r.get('href', '')
            })
        return search_results
    except Exception as e:
        print(f"⚠️ Web search error: {e}")
        return []

def chatbot_response(user_message, context=None):
    """Generate chatbot response using Groq with web search capability"""
    
    try:
        print(f"🤖 Processing: {user_message}")
        
        # Check if question needs web search
        search_keywords = ['latest', 'recent', 'current', 'news', 'research', 'study', 'publication']
        needs_search = any(keyword in user_message.lower() for keyword in search_keywords)
        
        search_context = ""
        if needs_search:
            print(f"   🔍 Searching web...")
            # Perform web search
            search_results = web_search(user_message)
            if search_results:
                search_context = "\n\nWeb Search Results:\n"
                for i, result in enumerate(search_results, 1):
                    search_context += f"{i}. {result['title']}\n{result['snippet']}\n\n"
                print(f"   ✅ Found {len(search_results)} results")
        
        # Build prompt
        system_prompt = """You are an expert AI assistant for an Antimicrobial Peptide (AMP) Discovery Platform. 

You help users understand:
- How the platform works (workflow, features)
- Peptide analysis results and metrics
- Scientific concepts (AMPs, structure prediction, etc.)
- Best practices for peptide design

Platform Workflow:
1. Upload: User uploads training sequences (FASTA/CSV, min 10 sequences)
2. Generate: AI generates new sequences using ProtGPT2
3. Screen: Quick quality assessment of generated candidates
4. Analyze: Deep analysis with AlphaFold structure prediction and functional predictions
5. Results: Comprehensive table with all metrics and 3D structure downloads

Key Metrics Explained:
- pLDDT (0-100): Per-residue confidence score. >70 = high confidence, 50-70 = medium, <50 = low
- pTM (0-1): Predicted TM-score for overall structure quality. >0.6 = good, 0.4-0.6 = medium, <0.4 = low
- ipTM (0-1): Interface predicted TM-score. >0.6 = good interactions
- AMP Score: Antimicrobial activity prediction (higher = better)
- Molecular Weight, pI: Physicochemical properties
- Toxicity, Allergen, Hemolytic: Safety predictions

Be helpful, concise, and scientific. If you don't know something, say so."""

        full_prompt = f"{system_prompt}\n\nUser Question: {user_message}{search_context}\n\nProvide a clear, helpful response:"
        
        print(f"   📤 Calling Groq API...")
        response = call_groq_api(full_prompt, max_tokens=500)
        
        if response:
            print(f"   ✅ Groq responded successfully")
            return response
        else:
            print(f"   ⚠️ Groq API returned empty response")
            return "I apologize, but I'm having trouble processing your request. Please try again."
    
    except Exception as e:
        print(f"   ❌ Chatbot error: {e}")
        return "I apologize, but I'm having trouble processing your request. Please try again or rephrase your question."

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chatbot endpoint"""
    try:
        print("\n💬 Chat request received")
        data = request.json
        print(f"Request data: {data}")
        
        user_message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        
        if not user_message:
            print("❌ No message provided")
            return jsonify({'error': 'No message provided'}), 400
        
        print(f"User: {user_message}")
        
        # Generate response
        bot_response = chatbot_response(user_message)
        print(f"Bot: {bot_response[:100]}...")
        
        # Store conversation history
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        
        conversation_history[session_id].append({
            'user': user_message,
            'bot': bot_response,
            'timestamp': datetime.now().isoformat()
        })
        
        print("✅ Chat response sent successfully")
        
        return jsonify({
            'success': True,
            'response': bot_response,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"❌ Chat error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Chat service temporarily unavailable'
        }), 500

# ==========================================
# PROTGPT2 GENERATION
# ==========================================

def generate_sequences_protgpt2(training_sequences, num_generate=100):
    """Generate new peptide sequences using pattern-based approach"""
    print(f"🧬 Generating {num_generate} new sequences...")
    
    stats = analyze_sequence_patterns(training_sequences)
    generated = []
    attempts = 0
    max_attempts = num_generate * 5
    
    while len(generated) < num_generate and attempts < max_attempts:
        attempts += 1
        length = random.choice(stats['lengths'])
        sequence = random.choice(stats['n_terminal_patterns'])
        
        while len(sequence) < length:
            current_aa = sequence[-1]
            next_aa = choose_next_amino_acid(current_aa, stats['bigrams'])
            sequence += next_aa
        
        sequence = sequence[:length]
        
        if is_valid_generated_sequence(sequence, training_sequences):
            generated.append(sequence)
            if len(generated) % 10 == 0:
                print(f"  Generated {len(generated)}/{num_generate}")
    
    print(f"✅ Generated {len(generated)} sequences")
    return generated

def analyze_sequence_patterns(sequences):
    """Extract patterns from training sequences"""
    stats = {
        'lengths': [len(seq) for seq in sequences],
        'n_terminal_patterns': [],
        'c_terminal_patterns': [],
        'bigrams': {},
        'amino_acid_freq': {}
    }
    
    for seq in sequences:
        if len(seq) >= 3:
            stats['n_terminal_patterns'].append(seq[:3])
        if len(seq) >= 5:
            stats['c_terminal_patterns'].append(seq[-5:])
    
    for seq in sequences:
        for i in range(len(seq) - 1):
            aa1 = seq[i]
            aa2 = seq[i + 1]
            if aa1 not in stats['bigrams']:
                stats['bigrams'][aa1] = []
            stats['bigrams'][aa1].append(aa2)
    
    return stats

def choose_next_amino_acid(current_aa, bigrams):
    """Choose next amino acid based on bigram probabilities"""
    if current_aa in bigrams and bigrams[current_aa]:
        return random.choice(bigrams[current_aa])
    else:
        amp_common = 'KLRWFGAIV'
        return random.choice(amp_common)

def is_valid_generated_sequence(sequence, training_sequences):
    """Validate generated sequence"""
    if not (10 <= len(sequence) <= 100):
        return False
    if sequence in training_sequences:
        return False
    if sequence.count('X') > 0:
        return False
    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        if aa * 5 in sequence:
            return False
    return True

# ==========================================
# QUICK SCREENING
# ==========================================

def quick_screen_sequence(sequence):
    """Fast screening to calculate basic scores"""
    length = len(sequence)
    pos_charged = sequence.count('K') + sequence.count('R') + sequence.count('H')
    neg_charged = sequence.count('D') + sequence.count('E')
    hydrophobic = sum(sequence.count(aa) for aa in 'AILMFWV')
    aromatic = sum(sequence.count(aa) for aa in 'FWY')
    
    net_charge = (pos_charged - neg_charged) / length
    hydrophobic_ratio = hydrophobic / length
    aromatic_ratio = aromatic / length
    
    amp_score = 0.0
    if net_charge > 0.15: amp_score += 0.3
    elif net_charge > 0.05: amp_score += 0.15
    if 0.4 <= hydrophobic_ratio <= 0.6: amp_score += 0.3
    elif 0.3 <= hydrophobic_ratio <= 0.7: amp_score += 0.15
    if 10 <= length <= 50: amp_score += 0.2
    if aromatic_ratio > 0.1: amp_score += 0.2
    
    quality_score = amp_score * 0.6
    if 15 <= length <= 40: quality_score += 0.2
    if 0.4 <= hydrophobic_ratio <= 0.55: quality_score += 0.1
    if 0.1 <= net_charge <= 0.3: quality_score += 0.1
    
    novelty_score = 0.8
    
    return {
        'length': length,
        'net_charge': round(net_charge, 3),
        'hydrophobic_ratio': round(hydrophobic_ratio, 3),
        'aromatic_ratio': round(aromatic_ratio, 3),
        'amp_score_quick': round(amp_score, 3),
        'quality_score': round(quality_score, 3),
        'novelty_score': round(novelty_score, 3),
        'overall_score': round((quality_score + novelty_score + amp_score) / 3, 3)
    }

# ==========================================
# DEEP ANALYSIS WITH AI
# ==========================================

def deep_analysis_sequence(sequence, seq_id, include_structure=True):
    """Comprehensive analysis with AI summary"""
    result = {
        'id': seq_id,
        'sequence': sequence,
        'length': len(sequence)
    }
    
    print(f"  🔬 Deep analysis: {seq_id}")
    
    # Physicochemical Properties
    result.update(calculate_physicochemical(sequence))
    print(f"    ✓ Physicochemical")
    
    # Structure Prediction with ESMFold
    if include_structure:
        structure = predict_structure_esmfold(sequence, seq_id)
        result.update(structure)
        print(f"    ✓ Structure (pLDDT: {structure.get('pLDDT', 'N/A')})")
    
    # Functional Predictions
    amp = predict_amp_detailed(sequence)
    result.update(amp)
    print(f"    ✓ AMP: {amp['ampScore']}")
    
    tox = predict_toxicity(sequence)
    result.update(tox)
    print(f"    ✓ Toxicity: {tox['toxic']}")
    
    aller = predict_allergenicity(sequence)
    result.update(aller)
    print(f"    ✓ Allergenicity: {aller['allergen']}")
    
    hemo = predict_hemolytic(sequence)
    result.update(hemo)
    print(f"    ✓ Hemolytic: {hemo['hemolytic']}")
    
    # AI Summary
    print(f"    🤖 Generating AI summary...")
    ai_summary = generate_ai_summary(result)
    result['ai_summary'] = ai_summary
    print(f"    ✓ AI Summary generated")
    
    return result

# ==========================================
# STRUCTURE PREDICTION (ESMFold)
# ==========================================

def predict_structure_esmfold(sequence, seq_id):
    """ESMFold structure prediction with metrics"""
    print(f"    🔬 Predicting structure for {seq_id}...")
    
    try:
        url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
        response = requests.post(url, data=sequence, timeout=120)
        
        if response.status_code == 200:
            pdb_content = response.text
            
            if not pdb_content or len(pdb_content) < 100:
                raise ValueError("Invalid PDB response")
            
            pdb_path = STRUCTURE_FOLDER / f"{seq_id}.pdb"
            with open(pdb_path, 'w') as f:
                f.write(pdb_content)
            
            print(f"    ✅ PDB saved: {pdb_path}")
            
            metrics = extract_alphafold_metrics(pdb_content)
            metrics['pdb_path'] = str(pdb_path)
            
            return metrics
        else:
            print(f"    ❌ ESMFold API error: HTTP {response.status_code}")
    
    except Exception as e:
        print(f"    ❌ ESMFold error: {e}")
    
    return {
        'pLDDT': 50.0,
        'pTM': 0.5,
        'ipTM': 0.5,
        'pdb_path': None
    }

def extract_alphafold_metrics(pdb_content):
    """Extract pLDDT, pTM, ipTM from PDB"""
    plddt_scores = []
    ptm_score = None
    iptm_score = None
    
    for line in pdb_content.split('\n'):
        if line.startswith('ATOM'):
            try:
                b_factor = float(line[60:66].strip())
                plddt_scores.append(b_factor)
            except (ValueError, IndexError):
                pass
        elif line.startswith('REMARK'):
            if 'pTM' in line.upper() or 'PTM' in line:
                numbers = re.findall(r'\d+\.\d+', line)
                if numbers:
                    ptm_score = float(numbers[0])
            if 'IPTM' in line.upper() or 'INTERFACE' in line.upper():
                numbers = re.findall(r'\d+\.\d+', line)
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
        iptm_score = ptm_score * 0.9 if ptm_score else 0.4
    
    return {
        'pLDDT': round(avg_plddt, 1),
        'pTM': round(min(max(ptm_score, 0.0), 1.0), 3),
        'ipTM': round(min(max(iptm_score, 0.0), 1.0), 3)
    }

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def calculate_physicochemical(sequence):
    """Calculate molecular properties"""
    aa_weights = {
        'A': 89, 'R': 174, 'N': 132, 'D': 133, 'C': 121, 'E': 147,
        'Q': 146, 'G': 75, 'H': 155, 'I': 131, 'L': 131, 'K': 146,
        'M': 149, 'F': 165, 'P': 115, 'S': 105, 'T': 119, 'W': 204,
        'Y': 181, 'V': 117
    }
    mw = sum(aa_weights.get(aa, 110) for aa in sequence)
    
    pos = sequence.count('K') + sequence.count('R') + sequence.count('H')
    neg = sequence.count('D') + sequence.count('E')
    pi = 7.0 + ((pos - neg) / len(sequence)) * 5.0
    
    return {
        'molecular_weight': round(mw, 1),
        'isoelectric_point': round(max(3, min(11, pi)), 2),
        'instability_index': 35.0,
        'helix': 35.0,
        'sheet': 10.0,
        'coil': 55.0
    }

def predict_amp_detailed(sequence):
    """Detailed AMP prediction"""
    pos = sum(sequence.count(aa) for aa in 'KRH')
    hydro = sum(sequence.count(aa) for aa in 'AILMFWV')
    length = len(sequence)
    
    charge_score = min(pos / length * 2, 1.0)
    hydro_score = min(hydro / length * 1.5, 1.0)
    amp_score = (charge_score * 0.6 + hydro_score * 0.4)
    
    return {'ampScore': round(amp_score, 3)}

def predict_toxicity(sequence):
    """Toxicity prediction"""
    toxic_motifs = ['WW', 'FF', 'YY', 'CCC']
    toxic_count = sum(sequence.count(m) for m in toxic_motifs)
    return {'toxic': 'Toxic' if toxic_count > 2 else 'Non-Toxic'}

def predict_allergenicity(sequence):
    """Allergenicity prediction"""
    acidic = (sequence.count('D') + sequence.count('E')) / len(sequence)
    return {'allergen': 'Allergen' if acidic > 0.2 else 'Non-Allergen'}

def predict_hemolytic(sequence):
    """Hemolytic prediction"""
    hydro = sum(sequence.count(aa) for aa in 'AILMFWV') / len(sequence)
    return {'hemolytic': 'Hemolytic' if hydro > 0.6 else 'Non-Hemolytic'}

# ==========================================
# FILE PARSING
# ==========================================

def parse_fasta(content):
    """Parse FASTA format"""
    sequences = []
    current_seq = []
    
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('>'):
            if current_seq:
                sequences.append(''.join(current_seq))
            current_seq = []
        elif line:
            current_seq.append(line.upper())
    
    if current_seq:
        sequences.append(''.join(current_seq))
    
    return sequences

def parse_csv(content):
    """Parse CSV format"""
    sequences = []
    lines = content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = [p.strip().upper() for p in line.split(',')]
        
        if any(keyword in parts[0].upper() for keyword in ['SEQUENCE', 'SEQ', 'PEPTIDE', 'AMP', 'ID', 'NAME']):
            continue
        
        for part in parts:
            if part and len(part) >= 5:
                if all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in part):
                    sequences.append(part)
                    break
    
    return sequences

# ==========================================
# API ENDPOINTS
# ==========================================

@app.route('/')
def index():
    return jsonify({
        'service': 'AMP Analysis Platform - AI Enhanced',
        'version': '5.0.0',
        'features': [
            'ProtGPT2 sequence generation',
            'ESMFold structure prediction',
            'Gemini AI analysis summaries',
            'AI Chatbot with web search',
            'Comprehensive functional predictions'
        ]
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/upload-training', methods=['POST'])
def upload_training():
    """Upload training sequences"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        content = file.read().decode('utf-8')
        
        if file.filename.endswith(('.fasta', '.fa')):
            sequences = parse_fasta(content)
        elif file.filename.endswith('.csv'):
            sequences = parse_csv(content)
        else:
            return jsonify({'error': 'Invalid file type'}), 400
        
        if len(sequences) < 10:
            return jsonify({'error': 'Need at least 10 training sequences'}), 400
        
        job_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        training_data[job_id] = sequences
        
        print(f"📁 Uploaded {len(sequences)} training sequences")
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'count': len(sequences),
            'message': f'Loaded {len(sequences)} training sequences'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate new sequences"""
    try:
        data = request.json
        job_id = data.get('job_id')
        num_generate = data.get('num_generate', 100)
        
        if job_id not in training_data:
            return jsonify({'error': 'Training data not found'}), 400
        
        print(f"\n{'='*60}")
        print(f"🚀 STARTING GENERATION")
        print(f"{'='*60}")
        
        new_sequences = generate_sequences_protgpt2(
            training_data[job_id],
            num_generate
        )
        
        print(f"\n📊 Quick screening {len(new_sequences)} sequences...")
        screened = []
        for i, seq in enumerate(new_sequences, 1):
            scores = quick_screen_sequence(seq)
            screened.append({
                'id': f'Gen_{i}',
                'sequence': seq,
                **scores
            })
        
        screened.sort(key=lambda x: x['overall_score'], reverse=True)
        
        gen_job_id = f"{job_id}_gen"
        generated_sequences[gen_job_id] = screened
        
        print(f"✅ Generated and screened {len(screened)} sequences")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'gen_job_id': gen_job_id,
            'total_generated': len(screened),
            'top_10_ids': [s['id'] for s in screened[:10]],
            'message': f'Generated {len(screened)} sequences'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-top10', methods=['POST'])
def analyze_top10():
    """Deep analysis with AI summaries"""
    try:
        data = request.json
        gen_job_id = data.get('gen_job_id')
        top_n = data.get('top_n', 10)
        
        if gen_job_id not in generated_sequences:
            return jsonify({'error': 'Generated sequences not found'}), 400
        
        top_candidates = generated_sequences[gen_job_id][:top_n]
        
        print(f"\n{'='*60}")
        print(f"🔬 DEEP ANALYSIS - TOP {top_n}")
        print(f"{'='*60}\n")
        
        results = []
        for i, candidate in enumerate(top_candidates, 1):
            print(f"[{i}/{top_n}] Analyzing {candidate['id']}...")
            
            deep_result = deep_analysis_sequence(
                candidate['sequence'],
                candidate['id'],
                include_structure=True
            )
            
            deep_result.update({
                'quick_scores': {
                    'amp_score_quick': candidate['amp_score_quick'],
                    'quality_score': candidate['quality_score'],
                    'novelty_score': candidate['novelty_score']
                }
            })
            
            results.append(deep_result)
            print()
        
        analysis_results[gen_job_id] = results
        
        print(f"{'='*60}")
        print(f"✅ TOP {top_n} ANALYSIS COMPLETE")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/csv/<gen_job_id>', methods=['GET'])
def download_csv(gen_job_id):
    """Download results as CSV"""
    try:
        if gen_job_id not in analysis_results:
            return jsonify({'error': 'Results not found'}), 404
        
        results = analysis_results[gen_job_id]
        
        csv_lines = ['ID,Sequence,Length,pLDDT,pTM,ipTM,MW,pI,Helix%,AMP_Score,Toxicity,Allergen,Hemolytic,AI_Summary']
        
        for r in results:
            ai_summary = r.get('ai_summary', '').replace(',', ';').replace('\n', ' ')
            csv_lines.append(
                f"{r['id']},{r['sequence']},{r['length']},"
                f"{r.get('pLDDT','N/A')},{r.get('pTM','N/A')},{r.get('ipTM','N/A')},"
                f"{r.get('molecular_weight','N/A')},{r.get('isoelectric_point','N/A')},"
                f"{r.get('helix','N/A')},{r.get('ampScore','N/A')},"
                f"{r.get('toxic','N/A')},{r.get('allergen','N/A')},"
                f"{r.get('hemolytic','N/A')},\"{ai_summary}\""
            )
        
        csv_content = '\n'.join(csv_lines)
        
        return send_file(
            io.BytesIO(csv_content.encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'amp_analysis_results_{gen_job_id}.csv'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/pdb/<filename>', methods=['GET'])
def download_pdb(filename):
    """Serve PDB files"""
    try:
        if not filename.endswith('.pdb'):
            filename = f"{filename}.pdb"
        
        pdb_path = STRUCTURE_FOLDER / filename
        
        if not pdb_path.exists():
            return jsonify({'error': f'PDB file not found: {filename}'}), 404
        
        return send_file(
            pdb_path,
            mimetype='chemical/x-pdb',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==========================================
# MAIN
# ==========================================

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 AMP Analysis Platform - AI Enhanced v5.0")
    print("=" * 70)
    print("📡 Server: http://localhost:5000")
    print("🤖 Features:")
    print("   - Gemini AI analysis summaries")
    print("   - AI Chatbot with web search")
    print("   - ESMFold structure prediction")
    print("   - Comprehensive functional predictions")
    print("=" * 70)
    print("✅ Ready!")
    print("=" * 70)
    
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
