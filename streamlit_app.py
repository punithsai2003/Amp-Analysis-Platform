"""
AMP Analysis Platform - Streamlit edition
==========================================

Single-process replacement for the previous Flask API + static HTML frontend.
Everything runs in one process, so there is no CORS configuration, no second
server and no API_URL to keep in sync.

Deploy: push to GitHub, point Streamlit Community Cloud at streamlit_app.py.
Run locally: streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime

import pandas as pd
import streamlit as st

import amp_core as core

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AMP Analysis Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main > div { padding-top: 1.5rem; }
    .stMetric { background: rgba(128,128,128,0.08); padding: 0.75rem 1rem;
                border-radius: 0.5rem; }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; }
    .seq-mono { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
                font-size: 0.85rem; word-break: break-all; line-height: 1.6; }
    .pill { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
            font-size: 0.75rem; font-weight: 600; margin-right: 0.35rem; }
    .pill-good { background: #10b98122; color: #059669; }
    .pill-bad  { background: #ef444422; color: #dc2626; }
    .pill-mid  { background: #f59e0b22; color: #d97706; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================
# SESSION STATE
# ==========================================

DEFAULTS = {
    "training": [],          # list[str] of uploaded sequences
    "training_source": "",   # filename or "pasted"
    "screened": [],          # list[dict] ranked candidates
    "results": [],           # list[dict] deep-analysis results
    "gen_meta": {},          # generation method / timestamp
}

for key, default in DEFAULTS.items():
    st.session_state.setdefault(key, default)


def reset_downstream(from_stage: str) -> None:
    """Invalidate later stages when an earlier one changes."""
    if from_stage == "training":
        st.session_state.screened = []
        st.session_state.results = []
        st.session_state.gen_meta = {}
    elif from_stage == "generate":
        st.session_state.results = []


# ==========================================
# HELPERS
# ==========================================

def get_secret(name: str, default: str = "") -> str:
    """
    Read a Streamlit secret without exploding when no secrets file exists.

    st.secrets raises StreamlitSecretNotFoundError (not KeyError) if there is
    no secrets.toml at all, which is the normal case on a fresh deploy.
    """
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def pill(label: str, kind: str) -> str:
    return f'<span class="pill pill-{kind}">{label}</span>'


def verdict_pill(value: str) -> str:
    """Colour a categorical prediction. 'Non-' prefixed answers are the good ones."""
    if value is None:
        return pill("Unknown", "mid")
    return pill(value, "good" if value.startswith("Non-") else "bad")


def plddt_band(plddt: float | None) -> tuple[str, str]:
    """AlphaFold's standard confidence bands."""
    if plddt is None:
        return "Not predicted", "mid"
    if plddt >= 90:
        return "Very high", "good"
    if plddt >= 70:
        return "Confident", "good"
    if plddt >= 50:
        return "Low", "mid"
    return "Very low", "bad"


def fmt(value: float | None, spec: str = ".3f", missing: str = "—") -> str:
    """Format a number that may legitimately be absent."""
    return missing if value is None else format(value, spec)


def render_3d_structure(pdb_content: str, height: int = 420) -> None:
    """Embed a py3Dmol cartoon view coloured by pLDDT confidence."""
    try:
        import py3Dmol
    except ImportError:
        st.info("Install `py3Dmol` to enable the interactive 3D viewer.")
        return

    view = py3Dmol.view(width="100%", height=height)
    view.addModel(pdb_content, "pdb")
    # AlphaFold/ESMFold convention: B-factor holds pLDDT, so colour by it.
    view.setStyle(
        {"cartoon": {
            "colorscheme": {
                "prop": "b",
                "gradient": "roygb",
                "min": 50,
                "max": 90,
            }
        }}
    )
    view.zoomTo()
    view.spin(False)
    st.components.v1.html(view._make_html(), height=height + 20)

    st.caption(
        "Coloured by pLDDT confidence: blue/green = high, "
        "yellow = moderate, orange/red = low."
    )


def structures_zip(results: list[dict]) -> bytes | None:
    """Bundle every successfully predicted PDB into a single download."""
    with_pdb = [r for r in results if r.get("pdb_content")]
    if not with_pdb:
        return None

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in with_pdb:
            zf.writestr(f"{r['id']}.pdb", r["pdb_content"])
    return buffer.getvalue()


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.title("🧬 AMP Platform")
    st.caption("Generate, screen and characterise antimicrobial peptides.")

    st.divider()
    st.subheader("Pipeline status")

    stages = [
        ("Training data", len(st.session_state.training), "sequences"),
        ("Generated pool", len(st.session_state.screened), "candidates"),
        ("Deep analysis", len(st.session_state.results), "characterised"),
    ]
    for label, count, unit in stages:
        icon = "✅" if count else "⬜"
        st.markdown(f"{icon} **{label}** — {count} {unit}" if count
                    else f"{icon} {label}")

    st.divider()
    st.subheader("Settings")

    hf_key = get_secret("HF_API_KEY")
    use_hf = st.toggle(
        "Use ProtGPT2 (Hugging Face)",
        value=bool(hf_key),
        disabled=not hf_key,
        help=(
            "Requires HF_API_KEY in Streamlit secrets. Without it the app uses "
            "bigram pattern generation trained on your uploaded sequences, "
            "which is faster and needs no external service."
        ),
    )
    if not hf_key:
        st.caption("No `HF_API_KEY` secret found — using pattern generation.")

    st.divider()
    if st.button("🔄 Reset everything", width="stretch"):
        for key, default in DEFAULTS.items():
            st.session_state[key] = default
        st.rerun()


# ==========================================
# HEADER
# ==========================================

st.title("AMP Analysis Platform")
st.caption(
    "Antimicrobial peptide design pipeline — sequence generation, rapid screening, "
    "ESMFold structure prediction and functional characterisation."
)

tab_upload, tab_generate, tab_analyse, tab_export = st.tabs([
    "1 · Training data",
    "2 · Generate & screen",
    "3 · Deep analysis",
    "4 · Export",
])


# ==========================================
# TAB 1 - TRAINING DATA
# ==========================================

with tab_upload:
    st.subheader("Upload known antimicrobial peptides")
    st.write(
        "The generator learns residue-transition patterns from these sequences. "
        "At least 10 are required; 50+ gives noticeably better output."
    )

    col_file, col_paste = st.columns(2)

    with col_file:
        uploaded = st.file_uploader(
            "FASTA, CSV or TXT",
            type=["fasta", "fa", "faa", "csv", "tsv", "txt"],
            help="FASTA headers are ignored. CSV: first valid peptide per row is taken.",
        )
        if uploaded is not None:
            try:
                content = uploaded.getvalue().decode("utf-8", errors="replace")
                sequences = core.parse_uploaded(uploaded.name, content)
            except Exception as exc:
                st.error(f"Could not parse **{uploaded.name}**: {exc}")
            else:
                if len(sequences) < 10:
                    st.error(
                        f"Only {len(sequences)} valid sequences found — "
                        "at least 10 are needed. Check the file uses the standard "
                        "20 amino-acid letters."
                    )
                elif sequences != st.session_state.training:
                    st.session_state.training = sequences
                    st.session_state.training_source = uploaded.name
                    reset_downstream("training")
                    st.rerun()

    with col_paste:
        pasted = st.text_area(
            "…or paste sequences",
            height=150,
            placeholder="GIGKFLHSAKKFGKAFVGEIMNS\nKWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK\n…",
            help="One per line, or FASTA.",
        )
        if st.button("Load pasted sequences", disabled=not pasted.strip()):
            sequences = (
                core.parse_fasta(pasted) if pasted.lstrip().startswith(">")
                else core.parse_plain_text(pasted)
            )
            if len(sequences) < 10:
                st.error(f"Only {len(sequences)} valid sequences found — need at least 10.")
            else:
                st.session_state.training = sequences
                st.session_state.training_source = "pasted input"
                reset_downstream("training")
                st.rerun()

    training = st.session_state.training

    if training:
        st.success(
            f"**{len(training)}** training sequences loaded "
            f"from {st.session_state.training_source}."
        )

        lengths = [len(s) for s in training]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sequences", len(training))
        m2.metric("Mean length", f"{sum(lengths) / len(lengths):.1f}")
        m3.metric("Shortest", min(lengths))
        m4.metric("Longest", max(lengths))

        with st.expander("Length distribution"):
            hist = pd.Series(lengths).value_counts().sort_index()
            hist.index.name = "Length (residues)"
            st.bar_chart(hist, y_label="Count")

        with st.expander(f"Preview all {len(training)} sequences"):
            st.dataframe(
                pd.DataFrame({
                    "#": range(1, len(training) + 1),
                    "Sequence": training,
                    "Length": lengths,
                }),
                hide_index=True,
                width="stretch",
            )
    else:
        st.info("Upload or paste training sequences to begin.")


# ==========================================
# TAB 2 - GENERATE & SCREEN
# ==========================================

with tab_generate:
    st.subheader("Generate novel candidates")

    if not st.session_state.training:
        st.warning("Load training data on the first tab before generating.")
    else:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            num_generate = st.slider(
                "Number of sequences to generate", 20, 500, 100, step=20,
                help="Pattern generation is near-instant. ProtGPT2 takes ~1s each.",
            )
        with col_b:
            st.write("")
            st.write("")
            go = st.button("🚀 Generate", type="primary", width="stretch")

        if go:
            progress = st.progress(0.0, text="Starting…")

            def on_progress(done: int, total: int) -> None:
                progress.progress(done / total, text=f"Generated {done}/{total}")

            generated = None
            method = "Bigram pattern generation"

            if use_hf and hf_key:
                with st.spinner("Calling ProtGPT2 via Hugging Face…"):
                    generated = core.generate_via_huggingface(
                        st.session_state.training, num_generate, hf_key, on_progress
                    )
                if generated:
                    method = "ProtGPT2 (Hugging Face Inference API)"
                else:
                    st.warning(
                        "Hugging Face API unavailable or token rejected — "
                        "falling back to pattern generation."
                    )

            if not generated:
                generated = core.generate_via_patterns(
                    st.session_state.training, num_generate, on_progress
                )

            progress.progress(1.0, text="Screening…")

            screened = []
            for i, seq in enumerate(generated, 1):
                screened.append({"id": f"Gen_{i}", "sequence": seq,
                                 **core.quick_screen_sequence(seq)})
            screened.sort(key=lambda x: x["overall_score"], reverse=True)

            st.session_state.screened = screened
            st.session_state.gen_meta = {
                "method": method,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "requested": num_generate,
            }
            reset_downstream("generate")
            progress.empty()
            st.rerun()

    screened = st.session_state.screened

    if screened:
        meta = st.session_state.gen_meta
        st.success(
            f"**{len(screened)}** candidates generated and screened "
            f"via {meta.get('method', 'unknown')} at {meta.get('timestamp', '')}."
        )
        if len(screened) < meta.get("requested", 0):
            st.caption(
                f"Requested {meta['requested']} but produced {len(screened)} — "
                "duplicates and sequences failing validity checks were discarded."
            )

        df = pd.DataFrame(screened)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Candidates", len(df))
        m2.metric("Best overall score", f"{df['overall_score'].max():.3f}")
        m3.metric("Mean AMP score", f"{df['amp_score_quick'].mean():.3f}")
        m4.metric("Mean length", f"{df['length'].mean():.1f}")

        st.markdown("##### Ranked candidates")
        st.dataframe(
            df[[
                "id", "sequence", "length", "net_charge", "hydrophobic_ratio",
                "aromatic_ratio", "amp_score_quick", "quality_score", "overall_score",
            ]].rename(columns={
                "id": "ID", "sequence": "Sequence", "length": "Len",
                "net_charge": "Net charge", "hydrophobic_ratio": "Hydrophobic",
                "aromatic_ratio": "Aromatic", "amp_score_quick": "AMP score",
                "quality_score": "Quality", "overall_score": "Overall",
            }),
            hide_index=True,
            width="stretch",
            height=420,
            column_config={
                "Overall": st.column_config.ProgressColumn(
                    "Overall", min_value=0.0, max_value=1.0, format="%.3f"
                ),
            },
        )


# ==========================================
# TAB 3 - DEEP ANALYSIS
# ==========================================

with tab_analyse:
    st.subheader("Structure prediction & functional characterisation")

    if not st.session_state.screened:
        st.warning("Generate candidates on the previous tab first.")
    else:
        st.info(
            "ESMFold is a free public API and can take 30–120 seconds per sequence. "
            "Start with 3 to confirm it's responding, then raise the count.",
            icon="⏱️",
        )

        col_a, col_b, col_c = st.columns([2, 1, 1])
        with col_a:
            top_n = st.slider("Analyse top N candidates", 1, 10, 3)
        with col_b:
            include_structure = st.toggle("Predict 3D structure", value=True)
        with col_c:
            st.write("")
            run = st.button("🔬 Run analysis", type="primary", width="stretch")

        if run:
            candidates = st.session_state.screened[:top_n]
            progress = st.progress(0.0, text="Starting…")
            status = st.empty()
            results = []

            for i, candidate in enumerate(candidates, 1):
                status.markdown(
                    f"**[{i}/{top_n}]** Analysing `{candidate['id']}` "
                    f"({candidate['length']} residues)…"
                )
                result = core.deep_analysis_sequence(
                    candidate["sequence"],
                    candidate["id"],
                    include_structure=include_structure,
                )
                result["quick_scores"] = {
                    "amp_score_quick": candidate["amp_score_quick"],
                    "quality_score": candidate["quality_score"],
                    "novelty_score": candidate["novelty_score"],
                }
                results.append(result)
                progress.progress(i / top_n, text=f"Completed {i}/{top_n}")
                st.session_state.results = results  # partial results survive a rerun

            progress.empty()
            status.empty()
            st.rerun()

    results = st.session_state.results

    if results:
        SKIPPED = "Structure prediction skipped"
        skipped = [r for r in results if r.get("structure_error") == SKIPPED]
        failed = [r for r in results
                  if r.get("structure_error") and r["structure_error"] != SKIPPED]

        if skipped:
            st.info(
                f"Structure prediction was turned off, so pLDDT, pTM and ipTM "
                f"are blank for {len(skipped)} candidate(s). "
                "Physicochemical and functional predictions are unaffected.",
                icon="ℹ️",
            )
        if failed:
            st.warning(
                f"{len(failed)} of {len(results)} structure predictions did not "
                "return a model, so those rows fall back to conservative "
                "placeholder metrics (pLDDT 50, pTM 0.5) — do not report them "
                "as measured values. Reason: "
                + "; ".join(sorted({r["structure_error"] for r in failed}))
            )

        st.markdown("##### Summary")
        summary = pd.DataFrame([{
            "ID": r["id"],
            "Length": r["length"],
            "pLDDT": r.get("pLDDT"),
            "pTM": r.get("pTM"),
            "ipTM": r.get("ipTM"),
            "MW (Da)": r.get("molecular_weight"),
            "pI": r.get("isoelectric_point"),
            "AMP score": r.get("ampScore"),
            "Toxicity": r.get("toxic"),
            "Allergen": r.get("allergen"),
            "Hemolytic": r.get("hemolytic"),
            "Structure": "✅" if r.get("pdb_content") else "❌",
        } for r in results])

        st.dataframe(
            summary, hide_index=True, width="stretch",
            column_config={
                "pLDDT": st.column_config.ProgressColumn(
                    "pLDDT", min_value=0, max_value=100, format="%.1f"
                ),
                "AMP score": st.column_config.ProgressColumn(
                    "AMP score", min_value=0.0, max_value=1.0, format="%.3f"
                ),
            },
        )

        st.markdown("##### Per-candidate detail")

        for r in results:
            plddt = r.get("pLDDT")
            band, band_kind = plddt_band(plddt)
            header = (
                f"{r['id']} — {r['length']} aa · "
                f"pLDDT {fmt(plddt, '.1f')} ({band})"
            )

            with st.expander(header):
                st.markdown(
                    f'<div class="seq-mono">{r["sequence"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    verdict_pill(r.get("toxic"))
                    + verdict_pill(r.get("allergen"))
                    + verdict_pill(r.get("hemolytic"))
                    + pill(f"pLDDT {band}", band_kind),
                    unsafe_allow_html=True,
                )

                st.write("")
                left, right = st.columns([1, 1])

                with left:
                    st.markdown("**Confidence metrics**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("pLDDT", fmt(plddt, ".1f"))
                    c2.metric("pTM", fmt(r.get("pTM")))
                    c3.metric("ipTM", fmt(r.get("ipTM")))

                    st.markdown("**Physicochemical**")
                    c4, c5, c6 = st.columns(3)
                    c4.metric("MW", f"{fmt(r.get('molecular_weight'), ',.0f')} Da")
                    c5.metric("pI", fmt(r.get("isoelectric_point"), ".2f"))
                    c6.metric("AMP score", fmt(r.get("ampScore")))

                    st.markdown("**Predicted secondary structure**")
                    st.bar_chart(
                        pd.DataFrame({
                            "Fraction (%)": [
                                r.get("helix", 0), r.get("sheet", 0), r.get("coil", 0),
                            ]
                        }, index=["Helix", "Sheet", "Coil"]),
                        horizontal=True,
                    )

                with right:
                    if r.get("pdb_content"):
                        st.markdown("**Predicted structure**")
                        render_3d_structure(r["pdb_content"])
                        st.download_button(
                            f"⬇️ {r['id']}.pdb",
                            data=r["pdb_content"],
                            file_name=f"{r['id']}.pdb",
                            mime="chemical/x-pdb",
                            key=f"pdb_{r['id']}",
                        )
                    else:
                        st.markdown("**Predicted structure**")
                        st.error(
                            r.get("structure_error", "No structure available."),
                            icon="⚠️",
                        )


# ==========================================
# TAB 4 - EXPORT
# ==========================================

with tab_export:
    st.subheader("Download results")

    if not st.session_state.screened and not st.session_state.results:
        st.warning("Nothing to export yet — run the pipeline first.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if st.session_state.screened:
        st.markdown("##### Screened candidate pool")
        st.caption(
            f"All {len(st.session_state.screened)} generated sequences with "
            "quick-screening scores, ranked by overall score."
        )
        st.download_button(
            "⬇️ Download candidates CSV",
            data=core.screened_to_csv(st.session_state.screened),
            file_name=f"amp_candidates_{stamp}.csv",
            mime="text/csv",
        )

        fasta = "\n".join(
            f">{s['id']} overall_score={s['overall_score']}\n{s['sequence']}"
            for s in st.session_state.screened
        )
        st.download_button(
            "⬇️ Download candidates FASTA",
            data=fasta,
            file_name=f"amp_candidates_{stamp}.fasta",
            mime="text/plain",
        )

    if st.session_state.results:
        st.divider()
        st.markdown("##### Deep-analysis results")
        st.caption(
            f"{len(st.session_state.results)} fully characterised candidates "
            "with AlphaFold-style confidence metrics."
        )
        st.download_button(
            "⬇️ Download analysis CSV",
            data=core.results_to_csv(st.session_state.results),
            file_name=f"amp_analysis_{stamp}.csv",
            mime="text/csv",
        )

        zip_bytes = structures_zip(st.session_state.results)
        if zip_bytes:
            count = sum(1 for r in st.session_state.results if r.get("pdb_content"))
            st.download_button(
                f"⬇️ Download {count} PDB structures (ZIP)",
                data=zip_bytes,
                file_name=f"amp_structures_{stamp}.zip",
                mime="application/zip",
            )
        else:
            st.caption("No PDB structures available to bundle.")

    st.divider()
    with st.expander("Method notes — read before publishing"):
        st.markdown(
            """
**Structure prediction.** Models come from the public
[ESMFold API](https://esmatlas.com/). pLDDT is read from the B-factor column
of the returned PDB.

**pTM and ipTM are estimated, not measured.** ESMFold's API does not return
these values, so they are derived from mean pLDDT using a monotonic mapping.
Treat them as an ordering heuristic, not as reportable AlphaFold metrics.
ipTM in particular is not meaningful for single-chain peptides — it describes
inter-chain interfaces.

**Functional predictions are heuristics.** Toxicity, allergenicity and
hemolytic activity use simple motif and composition rules, not trained
classifiers. For publication-grade calls, cross-check against ToxinPred,
AllerTOP and HemoPI.

**Physicochemical values are approximations.** Molecular weight sums residue
masses without subtracting peptide-bond water; isoelectric point uses a linear
charge approximation rather than Henderson–Hasselbalch. Instability index and
the helix/sheet/coil split are fixed placeholder constants inherited from the
original implementation — they do not vary per sequence.
"""
        )
