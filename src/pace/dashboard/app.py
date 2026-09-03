"""
PACE Dashboard - Streamlit Application

Interactive dashboard for visualizing PACE pipeline results.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from pathlib import Path


# Page configuration
st.set_page_config(
    page_title="PACE Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_results(results_dir="results"):
    """Load experiment results"""
    results = {}
    
    # Load experiment CSVs
    for exp_file in ["exp1.csv", "exp2.csv", "exp3.csv"]:
        path = os.path.join(results_dir, exp_file)
        if os.path.exists(path):
            results[exp_file.replace('.csv', '')] = pd.read_csv(path)
    
    # Load metric summaries
    summaries = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('_metricssummary.json'):
                with open(os.path.join(root, file), 'r') as f:
                    summary = json.load(f)
                    summary['dataset'] = file.replace('_metricssummary.json', '')
                    summaries.append(summary)
    
    if summaries:
        results['summaries'] = pd.DataFrame(summaries)
    
    return results


def page_overview(results):
    """Overview page"""
    st.title("⚖️ PACE Lakehouse Dashboard")
    st.markdown("### Ethical AI Data Governance Pipeline")
    
    col1, col2, col3 = st.columns(3)
    
    # Display metrics if available
    if 'summaries' in results and not results['summaries'].empty:
        avg_score = results['summaries']['score'].mean()
        avg_bronze = results['summaries'].apply(lambda x: x.get('components', {}).get('bronze', 0), axis=1).mean()
        avg_silver = results['summaries'].apply(lambda x: x.get('components', {}).get('silver', 0), axis=1).mean()
        
        col1.metric("Avg PACE Score", f"{avg_score:.3f}")
        col2.metric("Avg Bronze Score", f"{avg_bronze:.3f}")
        col3.metric("Avg Silver Score", f"{avg_silver:.3f}")
    else:
        col1.info("No results available. Run experiments first.")
    
    st.markdown("---")
    
    # Architecture diagram
    st.subheader("Pipeline Architecture")
    st.markdown("""
    ```
    Raw Data → Bronze (Ingest + PII) → Silver (Anonymize + Causal) → Gold (Fairness + Features) → ML/Analytics
                ↓ SB                    ↓ SS                          ↓ SG
                └────────────────────────┴──────────────────────────────→ PACE Score
    ```
    """)
    
    # Layer descriptions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🟤 Bronze Layer**")
        st.markdown("- Data ingestion")
        st.markdown("- PII detection")
        st.markdown("- Provenance tracking")
    
    with col2:
        st.markdown("**🔘 Silver Layer**")
        st.markdown("- Anonymization (k-anonymity, DP)")
        st.markdown("- Utility assessment")
        st.markdown("- Causal validation")
    
    with col3:
        st.markdown("**🟡 Gold Layer**")
        st.markdown("- Bias mitigation")
        st.markdown("- Fairness metrics")
        st.markdown("- Feature engineering")


def page_experiments(results):
    """Experiments page"""
    st.title("🧪 Experiments")
    
    tab1, tab2, tab3 = st.tabs(["Ablation Study", "Benchmarking", "Regulatory"])
    
    with tab1:
        st.subheader("Experiment 1: Ablation Study")
        if 'exp1' in results:
            df = results['exp1']
            
            # PACE scores by config
            fig = px.bar(df.groupby('config')['pacescore'].mean().reset_index(),
                        x='pacescore', y='config', orientation='h',
                        title='PACE Score by Configuration',
                        labels={'pacescore': 'PACE Score', 'config': 'Configuration'})
            fig.add_vline(x=0.85, line_dash="dash", line_color="green", annotation_text="EXCELLENT")
            fig.add_vline(x=0.70, line_dash="dash", line_color="orange", annotation_text="ACCEPTABLE")
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Run Experiment 1 first: `python experiments/scripts/runexperiment1.py`")
    
    with tab2:
        st.subheader("Experiment 2: Multi-Dataset Benchmarking")
        if 'exp2' in results:
            df = results['exp2']
            
            # Layer scores by dataset
            grouped = df.groupby('dataset')[['SB', 'SS', 'SG', 'pacescore']].mean().reset_index()
            fig = px.bar(grouped, x='dataset', y=['SB', 'SS', 'SG', 'pacescore'],
                        title='Layer Scores by Dataset',
                        labels={'value': 'Score', 'variable': 'Layer'},
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Run Experiment 2 first: `python experiments/scripts/runexperiment2.py`")
    
    with tab3:
        st.subheader("Experiment 3: Regulatory Compliance")
        if 'exp3' in results:
            df = results['exp3']
            
            # Compliance by regulation
            fig = px.bar(df.groupby('regulation')['pacescore'].mean().reset_index(),
                        x='regulation', y='pacescore',
                        title='PACE Score by Regulation',
                        labels={'pacescore': 'PACE Score', 'regulation': 'Regulation'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Compliance status
            compliance_summary = df.groupby('regulation')['compliant'].value_counts().unstack(fill_value=0)
            st.write("**Compliance Summary:**")
            st.dataframe(compliance_summary, use_container_width=True)
            
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Run Experiment 3 first: `python experiments/scripts/runexperiment3.py`")


def page_pace_score(results):
    """PACE Score page"""
    st.title("📊 PACE Score")
    
    if 'summaries' in results and not results['summaries'].empty:
        df = results['summaries']
        
        # Score distribution
        fig = px.histogram(df, x='score', nbins=20,
                          title='PACE Score Distribution',
                          labels={'score': 'PACE Score', 'count': 'Frequency'})
        fig.add_vline(x=0.85, line_dash="dash", line_color="green", annotation_text="EXCELLENT")
        fig.add_vline(x=0.70, line_dash="dash", line_color="orange", annotation_text="ACCEPTABLE")
        st.plotly_chart(fig, use_container_width=True)
        
        # Layer scores comparison
        st.subheader("Layer Scores by Dataset")
        layer_data = []
        for _, row in df.iterrows():
            components = row.get('components', {})
            layer_data.append({
                'dataset': row['dataset'],
                'Bronze': components.get('bronze', 0),
                'Silver': components.get('silver', 0),
                'Gold': components.get('gold', 0)
            })
        
        layer_df = pd.DataFrame(layer_data)
        fig = px.bar(layer_df, x='dataset', y=['Bronze', 'Silver', 'Gold'],
                    title='Layer Scores by Dataset',
                    labels={'value': 'Score', 'variable': 'Layer'},
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics
        st.subheader("Detailed Metrics")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No results available. Run the pipeline first.")


def main():
    """Main dashboard function"""
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "Experiments", "PACE Score"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    PACE Lakehouse combines FAIR principles with Causality, Anonymity, 
    Regulatory-compliance, and Ethics for ethical AI data governance.
    """)
    
    # Load results
    results_dir = st.sidebar.text_input("Results Directory", "results")
    results = load_results(results_dir)
    
    # Display selected page
    if page == "Overview":
        page_overview(results)
    elif page == "Experiments":
        page_experiments(results)
    elif page == "PACE Score":
        page_pace_score(results)


if __name__ == "__main__":
    main()
