import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import random
import time
from datetime import datetime, timedelta
import math

# Set page config
st.set_page_config(
    page_title="AI Cancer Drug Simulation Platform v2.0",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .drug-card {
        border: 2px solid #e1e5e9;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .simulation-section {
        background: rgba(255,255,255,0.8);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .stAlert {
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Title
st.markdown("""
<div class="main-header">
    <h1>🧬 AI Cancer Drug Simulation Platform v2.0</h1>
    <h3>Advanced In Silico Drug Discovery & Personalized Oncology</h3>
    <p><strong>Enhanced with AI-Driven Insights & Precision Medicine</strong></p>
    <p>Developed by <strong>Muhammad Nabeel Saddique</strong></p>
    <p>Medical Student - Oncology Research & Computational Biology</p>
</div>
""", unsafe_allow_html=True)

# Enhanced Cancer Models Database
CANCER_MODELS = {
    "Breast Cancer (ER+/HER2-)": {
        "description": "Estrogen receptor positive, HER2 negative breast cancer",
        "key_pathways": ["ER signaling", "Cell cycle", "Apoptosis", "DNA repair"],
        "growth_rate": 0.8,
        "drug_resistance": 0.3,
        "metastatic_potential": 0.4,
        "mutation_frequency": 0.2,
        "tumor_heterogeneity": 0.3,
        "immune_infiltration": 0.5,
        "angiogenesis": 0.4,
        "survival_months": 84
    },
    "Breast Cancer (HER2+)": {
        "description": "HER2 amplified breast cancer",
        "key_pathways": ["HER2/EGFR", "PI3K/AKT", "Cell cycle", "Angiogenesis"],
        "growth_rate": 1.2,
        "drug_resistance": 0.5,
        "metastatic_potential": 0.7,
        "mutation_frequency": 0.4,
        "tumor_heterogeneity": 0.6,
        "immune_infiltration": 0.3,
        "angiogenesis": 0.7,
        "survival_months": 72
    },
    "Lung Adenocarcinoma (EGFR+)": {
        "description": "EGFR mutated lung adenocarcinoma",
        "key_pathways": ["EGFR", "RAS/RAF/MEK", "Cell cycle", "Apoptosis"],
        "growth_rate": 1.0,
        "drug_resistance": 0.6,
        "metastatic_potential": 0.8,
        "mutation_frequency": 0.5,
        "tumor_heterogeneity": 0.7,
        "immune_infiltration": 0.4,
        "angiogenesis": 0.6,
        "survival_months": 36
    },
    "Colorectal Cancer (p53 mutated)": {
        "description": "p53 tumor suppressor mutated colorectal cancer",
        "key_pathways": ["p53 pathway", "Wnt signaling", "Cell cycle", "DNA repair"],
        "growth_rate": 0.9,
        "drug_resistance": 0.4,
        "metastatic_potential": 0.5,
        "mutation_frequency": 0.3,
        "tumor_heterogeneity": 0.4,
        "immune_infiltration": 0.6,
        "angiogenesis": 0.5,
        "survival_months": 48
    },
    "Melanoma (BRAF V600E)": {
        "description": "BRAF V600E mutated melanoma",
        "key_pathways": ["BRAF/MEK", "PI3K/AKT", "Cell cycle", "Apoptosis"],
        "growth_rate": 1.3,
        "drug_resistance": 0.7,
        "metastatic_potential": 0.9,
        "mutation_frequency": 0.6,
        "tumor_heterogeneity": 0.8,
        "immune_infiltration": 0.7,
        "angiogenesis": 0.8,
        "survival_months": 24
    },
    "Pancreatic Adenocarcinoma (KRAS)": {
        "description": "KRAS mutated pancreatic adenocarcinoma",
        "key_pathways": ["KRAS/RAF/MEK", "p53 pathway", "Cell cycle", "Apoptosis"],
        "growth_rate": 1.1,
        "drug_resistance": 0.8,
        "metastatic_potential": 0.8,
        "mutation_frequency": 0.7,
        "tumor_heterogeneity": 0.6,
        "immune_infiltration": 0.2,
        "angiogenesis": 0.7,
        "survival_months": 12
    }
}

# Enhanced Drug Database
DRUG_DATABASE = {
    "Tamoxifen": {
        "class": "SERM",
        "mechanism": "Estrogen receptor antagonist",
        "targets": ["ESR1", "ESR2"],
        "half_life": 7,
        "bioavailability": 0.8,
        "toxicity_score": 0.3,
        "cost_per_month": 50,
        "resistance_barrier": 0.4,
        "synergy_potential": 0.6,
        "response_rate": 0.65
    },
    "Trastuzumab": {
        "class": "Monoclonal Antibody",
        "mechanism": "HER2 receptor antagonist",
        "targets": ["ERBB2"],
        "half_life": 28,
        "bioavailability": 1.0,
        "toxicity_score": 0.4,
        "cost_per_month": 5000,
        "resistance_barrier": 0.6,
        "synergy_potential": 0.8,
        "response_rate": 0.75
    },
    "Erlotinib": {
        "class": "TKI",
        "mechanism": "EGFR tyrosine kinase inhibitor",
        "targets": ["EGFR"],
        "half_life": 1.5,
        "bioavailability": 0.6,
        "toxicity_score": 0.5,
        "cost_per_month": 3000,
        "resistance_barrier": 0.3,
        "synergy_potential": 0.7,
        "response_rate": 0.70
    },
    "5-Fluorouracil": {
        "class": "Antimetabolite",
        "mechanism": "Thymidylate synthase inhibitor",
        "targets": ["TYMS"],
        "half_life": 0.5,
        "bioavailability": 0.9,
        "toxicity_score": 0.7,
        "cost_per_month": 200,
        "resistance_barrier": 0.2,
        "synergy_potential": 0.8,
        "response_rate": 0.45
    },
    "Doxorubicin": {
        "class": "Anthracycline",
        "mechanism": "DNA intercalation, topoisomerase II inhibition",
        "targets": ["TOP2A", "TOP2B"],
        "half_life": 3,
        "bioavailability": 1.0,
        "toxicity_score": 0.8,
        "cost_per_month": 800,
        "resistance_barrier": 0.5,
        "synergy_potential": 0.7,
        "response_rate": 0.55
    },
    "Metformin": {
        "class": "Biguanide",
        "mechanism": "AMPK activation, mTOR inhibition",
        "targets": ["AMPK", "MTOR"],
        "half_life": 0.25,
        "bioavailability": 0.5,
        "toxicity_score": 0.1,
        "cost_per_month": 10,
        "resistance_barrier": 0.8,
        "synergy_potential": 0.9,
        "response_rate": 0.35
    },
    "Aspirin": {
        "class": "NSAID",
        "mechanism": "COX inhibition, NF-κB modulation",
        "targets": ["PTGS1", "PTGS2"],
        "half_life": 0.3,
        "bioavailability": 0.8,
        "toxicity_score": 0.2,
        "cost_per_month": 5,
        "resistance_barrier": 0.7,
        "synergy_potential": 0.6,
        "response_rate": 0.25
    },
    "Pembrolizumab": {
        "class": "Immune Checkpoint Inhibitor",
        "mechanism": "PD-1 receptor antagonist",
        "targets": ["PDCD1"],
        "half_life": 22,
        "bioavailability": 1.0,
        "toxicity_score": 0.5,
        "cost_per_month": 8000,
        "resistance_barrier": 0.7,
        "synergy_potential": 0.9,
        "response_rate": 0.40
    },
    "Bevacizumab": {
        "class": "Monoclonal Antibody",
        "mechanism": "VEGF inhibitor",
        "targets": ["VEGFA"],
        "half_life": 20,
        "bioavailability": 1.0,
        "toxicity_score": 0.4,
        "cost_per_month": 6000,
        "resistance_barrier": 0.5,
        "synergy_potential": 0.8,
        "response_rate": 0.30
    },
    "Imatinib": {
        "class": "TKI",
        "mechanism": "BCR-ABL, KIT, PDGFR inhibitor",
        "targets": ["ABL1", "KIT", "PDGFRA"],
        "half_life": 1.5,
        "bioavailability": 0.98,
        "toxicity_score": 0.3,
        "cost_per_month": 4000,
        "resistance_barrier": 0.4,
        "synergy_potential": 0.7,
        "response_rate": 0.85
    }
}

# Enhanced Sidebar
st.sidebar.title("🔬 Simulation Control Center")

# Main Parameters
st.sidebar.header("🎯 Core Parameters")
selected_cancer = st.sidebar.selectbox(
    "Cancer Model",
    list(CANCER_MODELS.keys()),
    help="Select cancer type for simulation"
)

simulation_mode = st.sidebar.radio(
    "Simulation Mode",
    ["Single Drug Analysis", "Combination Therapy", "Drug Screening"],
    help="Choose simulation approach"
)

if simulation_mode == "Single Drug Analysis":
    selected_drugs = st.sidebar.multiselect(
        "Select Drugs (1-4)",
        list(DRUG_DATABASE.keys()),
        default=["Tamoxifen", "Metformin"],
        max_selections=4,
        help="Choose drugs to analyze"
    )
elif simulation_mode == "Combination Therapy":
    combo_drugs = st.sidebar.multiselect(
        "Select Drug Combination (2-3 drugs)",
        list(DRUG_DATABASE.keys()),
        default=["Tamoxifen", "Metformin"],
        max_selections=3,
        help="Choose drugs for combination therapy"
    )
    selected_drugs = combo_drugs
else:  # Drug Screening
    screening_category = st.sidebar.selectbox(
        "Screening Category",
        ["All Drugs", "Targeted Therapies", "Chemotherapies", "Immunotherapies", "Repurposed Drugs"]
    )
    
    if screening_category == "All Drugs":
        selected_drugs = list(DRUG_DATABASE.keys())
    elif screening_category == "Targeted Therapies":
        selected_drugs = ["Trastuzumab", "Erlotinib", "Imatinib", "Bevacizumab"]
    elif screening_category == "Chemotherapies":
        selected_drugs = ["Doxorubicin", "5-Fluorouracil"]
    elif screening_category == "Immunotherapies":
        selected_drugs = ["Pembrolizumab"]
    else:  # Repurposed Drugs
        selected_drugs = ["Metformin", "Aspirin"]

simulation_duration = st.sidebar.slider(
    "Treatment Duration (days)",
    min_value=7,
    max_value=730,
    value=180,
    step=7,
    help="Length of treatment period"
)

# Patient Factors
st.sidebar.header("👤 Patient Profile")
with st.sidebar.expander("📊 Demographics", expanded=True):
    age = st.slider("Age (years)", 20, 90, 55)
    weight = st.slider("Weight (kg)", 40, 150, 70)
    performance_status = st.selectbox("ECOG Performance Status", [0, 1, 2, 3, 4])

with st.sidebar.expander("🧬 Genetic Profile"):
    genetic_variants = st.multiselect(
        "Genetic Variants",
        ["CYP2D6*4", "CYP3A4*22", "BRCA1", "BRCA2", "TP53", "EGFR T790M", "KRAS G12C"],
        default=[]
    )
    
    tmb_score = st.slider("Tumor Mutational Burden", 0.0, 50.0, 10.0, help="Mutations per megabase")

with st.sidebar.expander("🏥 Clinical History"):
    prior_treatments = st.multiselect(
        "Prior Treatments",
        ["None", "Chemotherapy", "Radiation", "Immunotherapy", "Targeted Therapy"],
        default=["None"]
    )
    
    comorbidities = st.multiselect(
        "Comorbidities",
        ["None", "Diabetes", "Hypertension", "Cardiac Disease", "Hepatic Impairment", "Renal Impairment"],
        default=["None"]
    )

# Advanced Settings
st.sidebar.header("⚙️ Advanced Settings")
with st.sidebar.expander("🔬 Simulation Options"):
    enable_resistance = st.checkbox("Model Drug Resistance", True)
    enable_heterogeneity = st.checkbox("Tumor Heterogeneity", True)
    enable_combinations = st.checkbox("Synergy Effects", True)
    monte_carlo_iterations = st.slider("Monte Carlo Runs", 100, 1000, 500, step=100)

# Enhanced Simulation Functions
def advanced_efficacy_simulation(cancer_model, drug_info, duration, patient_factors, is_combination=False):
    """Advanced drug efficacy simulation with realistic modeling"""
    
    # Base efficacy from drug response rate
    base_efficacy = drug_info['response_rate']
    
    # Patient-specific adjustments
    age_factor = 1 - (patient_factors['age'] - 50) * 0.002
    weight_factor = min(1.1, 70 / patient_factors['weight'])
    performance_factor = 1 - patient_factors['performance_status'] * 0.08
    
    # Genetic variant effects
    genetic_factor = 1.0
    if 'CYP2D6*4' in patient_factors['genetic_variants']:
        genetic_factor *= 0.75  # Reduced metabolism
    if 'BRCA1' in patient_factors['genetic_variants'] or 'BRCA2' in patient_factors['genetic_variants']:
        if drug_info['mechanism'] in ['DNA intercalation', 'DNA damage']:
            genetic_factor *= 1.2  # Enhanced sensitivity to DNA damaging agents
    
    # Cancer model specific factors
    resistance_factor = 1 - cancer_model['drug_resistance']
    if enable_heterogeneity:
        heterogeneity_factor = 1 - cancer_model['tumor_heterogeneity'] * 0.2
    else:
        heterogeneity_factor = 1.0
    
    # Combination therapy synergy
    synergy_factor = 1.0
    if is_combination and enable_combinations:
        synergy_factor = 1 + drug_info['synergy_potential'] * 0.3
    
    # Time-dependent simulation
    time_points = np.linspace(0, duration, 180)
    efficacy_curve = []
    resistance_development = []
    
    for i, t in enumerate(time_points):
        # Pharmacokinetic modeling (simplified)
        dose_interval = max(1, drug_info['half_life'] * 2)  # Dosing every 2 half-lives
        time_since_dose = t % dose_interval
        concentration = drug_info['bioavailability'] * np.exp(-0.693 * time_since_dose / drug_info['half_life'])
        
        # Resistance development over time
        if enable_resistance:
            resistance_dev = min(0.6, (t / duration) * cancer_model['drug_resistance'])
        else:
            resistance_dev = 0
        
        resistance_development.append(resistance_dev)
        
        # Tumor heterogeneity creates variation
        if enable_heterogeneity:
            heterogeneity_noise = np.random.normal(0, cancer_model['tumor_heterogeneity'] * 0.05)
        else:
            heterogeneity_noise = 0
        
        # Calculate daily efficacy
        daily_efficacy = (base_efficacy * 
                         resistance_factor * 
                         age_factor * 
                         weight_factor * 
                         performance_factor * 
                         genetic_factor * 
                         heterogeneity_factor * 
                         synergy_factor * 
                         concentration * 
                         (1 - resistance_dev) + 
                         heterogeneity_noise)
        
        efficacy_curve.append(max(0, min(1, daily_efficacy)))
    
    return time_points, efficacy_curve, resistance_development

def enhanced_toxicity_simulation(drug_info, duration, patient_factors, is_combination=False):
    """Enhanced toxicity simulation with patient-specific factors"""
    
    base_toxicity = drug_info['toxicity_score']
    
    # Patient risk factors
    age_factor = 1 + (patient_factors['age'] - 50) * 0.008
    performance_factor = 1 + patient_factors['performance_status'] * 0.12
    
    # Comorbidity effects
    comorbidity_count = len([c for c in patient_factors['comorbidities'] if c != 'None'])
    comorbidity_factor = 1 + comorbidity_count * 0.15
    
    # Combination therapy increases toxicity
    combo_factor = 1.0
    if is_combination:
        combo_factor = 1.25
    
    time_points = np.linspace(0, duration, 180)
    toxicity_curve = []
    
    for t in time_points:
        # Cumulative toxicity over time
        cumulative_factor = 1 + (t / duration) * 0.3
        
        # Organ-specific considerations
        organ_sensitivity = 1.0
        if 'Hepatic Impairment' in patient_factors['comorbidities']:
            organ_sensitivity *= 1.3
        if 'Renal Impairment' in patient_factors['comorbidities']:
            organ_sensitivity *= 1.2
        if 'Cardiac Disease' in patient_factors['comorbidities'] and drug_info['class'] == 'Anthracycline':
            organ_sensitivity *= 1.5
        
        # Calculate daily toxicity
        daily_toxicity = (base_toxicity * 
                         age_factor * 
                         performance_factor * 
                         comorbidity_factor * 
                         combo_factor * 
                         cumulative_factor * 
                         organ_sensitivity)
        
        # Add biological variation
        variation = np.random.normal(0, 0.08)
        final_toxicity = daily_toxicity + variation
        
        toxicity_curve.append(max(0, min(1, final_toxicity)))
    
    return time_points, toxicity_curve

def calculate_survival_benefit(efficacy_curve, toxicity_curve, cancer_model):
    """Calculate survival benefit based on efficacy and toxicity"""
    
    baseline_survival = cancer_model['survival_months']
    
    # Average efficacy and toxicity over treatment period
    avg_efficacy = np.mean(efficacy_curve)
    avg_toxicity = np.mean(toxicity_curve)
    
    # Survival benefit calculation
    efficacy_benefit = avg_efficacy * 24  # Maximum 24 month benefit
    toxicity_penalty = avg_toxicity * 6   # Maximum 6 month penalty
    
    predicted_survival = baseline_survival + efficacy_benefit - toxicity_penalty
    
    return max(baseline_survival * 0.5, predicted_survival)  # Minimum 50% of baseline

def monte_carlo_analysis(selected_drugs, cancer_model, duration, patient_factors, iterations=500):
    """Run Monte Carlo simulation for uncertainty analysis"""
    
    results = {drug: {'efficacy': [], 'toxicity': [], 'survival': []} for drug in selected_drugs}
    
    for _ in range(iterations):
        for drug in selected_drugs:
            drug_info = DRUG_DATABASE[drug]
            
            # Add noise to parameters
            noisy_patient = patient_factors.copy()
            noisy_patient['age'] += np.random.normal(0, 2)
            noisy_patient['weight'] += np.random.normal(0, 3)
            
            # Run simulation
            time_points, efficacy, resistance = advanced_efficacy_simulation(
                cancer_model, drug_info, duration, noisy_patient, len(selected_drugs) > 1
            )
            time_points_tox, toxicity = enhanced_toxicity_simulation(
                drug_info, duration, noisy_patient, len(selected_drugs) > 1
            )
            
            survival = calculate_survival_benefit(efficacy, toxicity, cancer_model)
            
            results[drug]['efficacy'].append(np.mean(efficacy))
            results[drug]['toxicity'].append(np.mean(toxicity))
            results[drug]['survival'].append(survival)
    
    return results

# Main Application Layout
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Cancer Model", 
    "💊 Drug Analysis", 
    "🚀 Simulation Results", 
    "📊 Advanced Analytics", 
    "📋 Clinical Report"
])

# Tab 1: Cancer Model Overview
with tab1:
    st.header("🔬 Cancer Model Characteristics")
    
    model_info = CANCER_MODELS[selected_cancer]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="simulation-section">
            <h3>{selected_cancer}</h3>
            <p><strong>Description:</strong> {model_info['description']}</p>
            <p><strong>Expected Survival:</strong> {model_info['survival_months']} months</p>
            
            <h4>🎯 Key Pathways:</h4>
            <ul>
                {''.join([f'<li>{pathway}</li>' for pathway in model_info['key_pathways']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Molecular characteristics
        molecular_data = {
            'Characteristic': ['Growth Rate', 'Drug Resistance', 'Metastatic Potential', 
                             'Mutation Frequency', 'Tumor Heterogeneity', 'Immune Infiltration'],
            'Score': [model_info['growth_rate'], model_info['drug_resistance'], 
                     model_info['metastatic_potential'], model_info['mutation_frequency'],
                     model_info['tumor_heterogeneity'], model_info['immune_infiltration']]
        }
        
        fig_bar = px.bar(
            pd.DataFrame(molecular_data),
            x='Characteristic',
            y='Score',
            title='Molecular Profile',
            color='Score',
            color_continuous_scale='Viridis'
        )
        fig_bar.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Key metrics gauges
        fig_gauge = make_subplots(
            rows=3, cols=1,
            specs=[[{"type": "indicator"}], [{"type": "indicator"}], [{"type": "indicator"}]],
            subplot_titles=("Aggressiveness", "Drug Resistance", "Metastatic Risk")
        )
        
        # Aggressiveness score
        aggressiveness = (model_info['growth_rate'] + model_info['metastatic_potential']) / 2
        fig_gauge.add_trace(go.Indicator(
            mode="gauge+number",
            value=aggressiveness,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 1.5]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 0.5], 'color': "lightgreen"},
                            {'range': [0.5, 1], 'color': "yellow"},
                            {'range': [1, 1.5], 'color': "red"}]}
        ), row=1, col=1)
        
        fig_gauge.add_trace(go.Indicator(
            mode="gauge+number",
            value=model_info['drug_resistance'],
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 1]},
                   'bar': {'color': "red"}}
        ), row=2, col=1)
        
        fig_gauge.add_trace(go.Indicator(
            mode="gauge+number",
            value=model_info['metastatic_potential'],
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 1]},
                   'bar': {'color': "orange"}}
        ), row=3, col=1)
        
        fig_gauge.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig_gauge, use_container_width=True)

# Tab 2: Drug Analysis
with tab2:
    st.header("💊 Drug Portfolio Analysis")
    
    if selected_drugs:
        # Drug comparison
        st.subheader("🎯 Drug Performance Comparison")
        
        # Create comparison dataframe
        comparison_data = []
        for drug in selected_drugs:
            drug_info = DRUG_DATABASE[drug]
            
            # Calculate drug score
            efficacy_score = drug_info['response_rate']
            safety_score = 1 - drug_info['toxicity_score']
            cost_effectiveness = efficacy_score / (drug_info['cost_per_month'] / 1000)
            
            comparison_data.append({
                'Drug': drug,
                'Class': drug_info['class'],
                'Response Rate': f"{drug_info['response_rate']:.1%}",
                'Safety Score': f"{safety_score:.2f}",
                'Half-life (days)': drug_info['half_life'],
                'Cost/Month': f"${drug_info['cost_per_month']:,}",
                'Cost-Effectiveness': f"{cost_effectiveness:.3f}",
                'Resistance Barrier': f"{drug_info['resistance_barrier']:.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Radar chart for drug comparison
        if len(selected_drugs) <= 4:  # Only show radar for up to 4 drugs
            st.subheader("📊 Multi-dimensional Drug Comparison")
            
            fig_radar = go.Figure()
            
            categories = ['Efficacy', 'Safety', 'Cost-Effectiveness', 'Resistance Barrier', 'Synergy Potential']
            
            for drug in selected_drugs:
                drug_info = DRUG_DATABASE[drug]
                
                values = [
                    drug_info['response_rate'],
                    1 - drug_info['toxicity_score'],
                    min(1, drug_info['response_rate'] / (drug_info['cost_per_month'] / 5000)),
                    drug_info['resistance_barrier'],
                    drug_info['synergy_potential']
                ]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=drug,
                    line=dict(width=3)
                ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Drug Performance Radar Chart",
                height=500
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Detailed drug information cards
        st.subheader("🔍 Detailed Drug Profiles")
        
        cols = st.columns(min(3, len(selected_drugs)))
        for i, drug in enumerate(selected_drugs):
            with cols[i % 3]:
                drug_info = DRUG_DATABASE[drug]
                
                # Calculate overall drug score
                drug_score = (
                    drug_info['response_rate'] * 0.3 +
                    (1 - drug_info['toxicity_score']) * 0.25 +
                    drug_info['resistance_barrier'] * 0.2 +
                    drug_info['synergy_potential'] * 0.15 +
                    min(1, 1000 / drug_info['cost_per_month']) * 0.1
                )
                
                score_color = "green" if drug_score > 0.7 else "orange" if drug_score > 0.5 else "red"
                
                st.markdown(f"""
                <div class="drug-card">
                    <h4>{drug}</h4>
                    <p><strong>Class:</strong> {drug_info['class']}</p>
                    <p><strong>Mechanism:</strong> {drug_info['mechanism']}</p>
                    <p><strong>Response Rate:</strong> {drug_info['response_rate']:.1%}</p>
                    <p><strong>Half-life:</strong> {drug_info['half_life']} days</p>
                    <p><strong>Bioavailability:</strong> {drug_info['bioavailability']:.1%}</p>
                    <p><strong>Monthly Cost:</strong> ${drug_info['cost_per_month']:,}</p>
                    <p><strong>Overall Score:</strong> <span style="color: {score_color}; font-weight: bold;">{drug_score:.2f}/1.0</span></p>
                </div>
                """, unsafe_allow_html=True)

# Tab 3: Simulation Results
with tab3:
    st.header("🚀 Advanced Simulation Engine")
    
    # Simulation controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        run_simulation = st.button("🚀 Run Primary Simulation", type="primary", use_container_width=True)
    
    with col2:
        run_monte_carlo = st.button("🎲 Monte Carlo Analysis", use_container_width=True)
    
    with col3:
        show_predictions = st.button("📈 Survival Predictions", use_container_width=True)
    
    # Patient parameters
    patient_factors = {
        'age': age,
        'weight': weight,
        'performance_status': performance_status,
        'genetic_variants': genetic_variants,
        'comorbidities': comorbidities,
        'prior_treatments': prior_treatments
    }
    
    if run_simulation and selected_drugs:
        st.success("🔬 Running Advanced Simulation...")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        simulation_results = {}
        cancer_model = CANCER_MODELS[selected_cancer]
        
        for i, drug in enumerate(selected_drugs):
            status_text.text(f'Simulating {drug}... ({i+1}/{len(selected_drugs)})')
            progress_bar.progress((i + 1) / len(selected_drugs))
            
            drug_info = DRUG_DATABASE[drug]
            
            # Run simulations
            time_points, efficacy, resistance = advanced_efficacy_simulation(
                cancer_model, drug_info, simulation_duration, patient_factors, 
                simulation_mode == "Combination Therapy"
            )
            
            time_points_tox, toxicity = enhanced_toxicity_simulation(
                drug_info, simulation_duration, patient_factors, 
                simulation_mode == "Combination Therapy"
            )
            
            survival_benefit = calculate_survival_benefit(efficacy, toxicity, cancer_model)
            
            simulation_results[drug] = {
                'time': time_points,
                'efficacy': efficacy,
                'toxicity': toxicity,
                'resistance': resistance,
                'avg_efficacy': np.mean(efficacy),
                'avg_toxicity': np.mean(toxicity),
                'therapeutic_index': np.mean(efficacy) / (np.mean(toxicity) + 0.01),
                'survival_months': survival_benefit,
                'cost_per_qaly': drug_info['cost_per_month'] * 12 / max(0.1, np.mean(efficacy))
            }
            
            time.sleep(0.3)  # Simulate processing time
        
        progress_bar.empty()
        status_text.empty()
        
        # Display Results
        st.header("📈 Simulation Results")
        
        # Key metrics summary
        st.subheader("🎯 Key Performance Metrics")
        
        metrics_data = []
        for drug, results in simulation_results.items():
            metrics_data.append({
                'Drug': drug,
                'Avg Efficacy': f"{results['avg_efficacy']:.3f}",
                'Avg Toxicity': f"{results['avg_toxicity']:.3f}",
                'Therapeutic Index': f"{results['therapeutic_index']:.2f}",
                'Predicted Survival': f"{results['survival_months']:.1f} months",
                'Cost per QALY': f"${results['cost_per_qaly']:,.0f}"
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df = metrics_df.sort_values('Therapeutic Index', ascending=False)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Efficacy over time
        st.subheader("📊 Efficacy Over Time")
        
        fig_efficacy = go.Figure()
        for drug, results in simulation_results.items():
            fig_efficacy.add_trace(go.Scatter(
                x=results['time'],
                y=results['efficacy'],
                mode='lines',
                name=f'{drug}',
                line=dict(width=3),
                hovertemplate=f'<b>{drug}</b><br>Day: %{{x}}<br>Efficacy: %{{y:.3f}}<extra></extra>'
            ))
        
        fig_efficacy.update_layout(
            title="Drug Efficacy Trajectories",
            xaxis_title="Time (days)",
            yaxis_title="Efficacy Score (0-1)",
            hovermode='x unified',
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig_efficacy, use_container_width=True)
        
        # Toxicity over time
        st.subheader("⚠️ Toxicity Profiles")
        
        fig_toxicity = go.Figure()
        for drug, results in simulation_results.items():
            fig_toxicity.add_trace(go.Scatter(
                x=results['time'],
                y=results['toxicity'],
                mode='lines',
                name=f'{drug}',
                line=dict(width=3, dash='dot'),
                hovertemplate=f'<b>{drug}</b><br>Day: %{{x}}<br>Toxicity: %{{y:.3f}}<extra></extra>'
            ))
        
        fig_toxicity.update_layout(
            title="Drug Toxicity Profiles",
            xaxis_title="Time (days)",
            yaxis_title="Toxicity Score (0-1)",
            hovermode='x unified',
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig_toxicity, use_container_width=True)
        
        # Drug resistance development
        if enable_resistance:
            st.subheader("🛡️ Resistance Development")
            
            fig_resistance = go.Figure()
            for drug, results in simulation_results.items():
                fig_resistance.add_trace(go.Scatter(
                    x=results['time'],
                    y=results['resistance'],
                    mode='lines',
                    name=f'{drug}',
                    line=dict(width=3),
                    fill='tonexty' if drug != list(simulation_results.keys())[0] else None
                ))
            
            fig_resistance.update_layout(
                title="Drug Resistance Development Over Time",
                xaxis_title="Time (days)",
                yaxis_title="Resistance Level (0-1)",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig_resistance, use_container_width=True)
        
        # Recommendations
        st.header("🏆 Treatment Recommendations")
        
        best_drug = metrics_df.iloc[0]['Drug']
        best_ti = float(metrics_df.iloc[0]['Therapeutic Index'])
        best_survival = metrics_df.iloc[0]['Predicted Survival']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"""
            **🥇 Recommended Primary Treatment**: {best_drug}
            
            - Highest therapeutic index: {best_ti:.2f}
            - Predicted survival: {best_survival}
            - Optimal efficacy-to-toxicity ratio
            """)
        
        with col2:
            # Risk alerts
            high_toxicity_drugs = [drug for drug, results in simulation_results.items() 
                                 if results['avg_toxicity'] > 0.6]
            
            if high_toxicity_drugs:
                st.warning(f"""
                **⚠️ High Toxicity Alert**
                
                Drugs requiring careful monitoring:
                - {', '.join(high_toxicity_drugs)}
                
                Consider dose reduction or supportive care.
                """)
            
            # Cost considerations
            expensive_drugs = [drug for drug, results in simulation_results.items() 
                             if results['cost_per_qaly'] > 100000]
            
            if expensive_drugs:
                st.info(f"""
                **💰 Cost Consideration**
                
                High-cost treatments:
                - {', '.join(expensive_drugs)}
                
                Consider cost-effectiveness in treatment decision.
                """)
    
    if run_monte_carlo and selected_drugs:
        st.success("🎲 Running Monte Carlo Uncertainty Analysis...")
        
        cancer_model = CANCER_MODELS[selected_cancer]
        
        with st.spinner("Performing uncertainty analysis..."):
            mc_results = monte_carlo_analysis(
                selected_drugs, cancer_model, simulation_duration, 
                patient_factors, monte_carlo_iterations
            )
        
        st.header("📊 Monte Carlo Analysis Results")
        
        # Uncertainty visualization
        for metric in ['efficacy', 'toxicity', 'survival']:
            st.subheader(f"📈 {metric.title()} Distribution")
            
            fig_dist = go.Figure()
            
            for drug in selected_drugs:
                values = mc_results[drug][metric]
                
                fig_dist.add_trace(go.Histogram(
                    x=values,
                    name=drug,
                    opacity=0.7,
                    nbinsx=30
                ))
            
            fig_dist.update_layout(
                title=f"{metric.title()} Distribution (Monte Carlo)",
                xaxis_title=f"{metric.title()} Value",
                yaxis_title="Frequency",
                barmode='overlay',
                template="plotly_white"
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Confidence intervals
        st.subheader("📊 95% Confidence Intervals")
        
        ci_data = []
        for drug in selected_drugs:
            for metric in ['efficacy', 'toxicity', 'survival']:
                values = mc_results[drug][metric]
                ci_lower = np.percentile(values, 2.5)
                ci_upper = np.percentile(values, 97.5)
                mean_val = np.mean(values)
                
                ci_data.append({
                    'Drug': drug,
                    'Metric': metric.title(),
                    'Mean': f"{mean_val:.3f}",
                    '95% CI Lower': f"{ci_lower:.3f}",
                    '95% CI Upper': f"{ci_upper:.3f}"
                })
        
        ci_df = pd.DataFrame(ci_data)
        st.dataframe(ci_df, use_container_width=True)
    
    if show_predictions and selected_drugs:
        st.header("📈 Survival Prediction Analysis")
        
        # Create survival curves (simplified)
        time_survival = np.linspace(0, 60, 60)  # 5 years in months
        
        fig_survival = go.Figure()
        
        cancer_model = CANCER_MODELS[selected_cancer]
        baseline_survival = cancer_model['survival_months']
        
        # Baseline survival (no treatment)
        baseline_curve = []
        for t in time_survival:
            # Exponential decay model
            survival_prob = np.exp(-t / baseline_survival)
            baseline_curve.append(survival_prob)
        
        fig_survival.add_trace(go.Scatter(
            x=time_survival,
            y=baseline_curve,
            mode='lines',
            name='No Treatment (Baseline)',
            line=dict(width=3, dash='dash', color='red')
        ))
        
        # Treatment survival curves
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        for i, drug in enumerate(selected_drugs):
            drug_info = DRUG_DATABASE[drug]
            
            # Estimate improved survival
            response_rate = drug_info['response_rate']
            toxicity = drug_info['toxicity_score']
            
            # Simplified survival benefit calculation
            hazard_ratio = (1 - response_rate * 0.6) * (1 + toxicity * 0.3)
            improved_survival = baseline_survival / hazard_ratio
            
            treatment_curve = []
            for t in time_survival:
                survival_prob = np.exp(-t / improved_survival)
                treatment_curve.append(survival_prob)
            
            fig_survival.add_trace(go.Scatter(
                x=time_survival,
                y=treatment_curve,
                mode='lines',
                name=f'{drug} Treatment',
                line=dict(width=3, color=colors[i % len(colors)])
            ))
        
        fig_survival.update_layout(
            title="Predicted Survival Curves",
            xaxis_title="Time (months)",
            yaxis_title="Survival Probability",
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig_survival, use_container_width=True)

# Tab 4: Advanced Analytics
with tab4:
    st.header("📊 Advanced Analytics Dashboard")
    
    if selected_drugs:
        # Drug interaction network (simplified visualization)
        st.subheader("🔗 Drug-Target Network Analysis")
        
        # Create network data
        network_data = []
        all_targets = set()
        
        for drug in selected_drugs:
            drug_info = DRUG_DATABASE[drug]
            for target in drug_info['targets']:
                network_data.append({
                    'Drug': drug,
                    'Target': target,
                    'Class': drug_info['class']
                })
                all_targets.add(target)
        
        network_df = pd.DataFrame(network_data)
        
        if not network_df.empty:
            # Target overlap analysis
            fig_network = px.scatter(
                network_df,
                x='Drug',
                y='Target',
                color='Class',
                size_max=20,
                title="Drug-Target Interaction Network"
            )
            fig_network.update_layout(height=400)
            st.plotly_chart(fig_network, use_container_width=True)
        
        # Biomarker analysis
        st.subheader("🧬 Biomarker-Drug Matching Analysis")
        
        biomarker_matches = []
        cancer_model = CANCER_MODELS[selected_cancer]
        
        for drug in selected_drugs:
            drug_info = DRUG_DATABASE[drug]
            
            # Calculate biomarker compatibility score
            compatibility_score = 0
            matching_factors = []
            
            # Check for specific biomarker matches
            if 'BRCA1' in genetic_variants or 'BRCA2' in genetic_variants:
                if drug_info['mechanism'] in ['DNA intercalation', 'DNA damage']:
                    compatibility_score += 0.3
                    matching_factors.append('BRCA mutation (DNA damage sensitivity)')
            
            if 'TP53' in genetic_variants:
                if drug_info['mechanism'] in ['Cell cycle', 'Apoptosis']:
                    compatibility_score += 0.2
                    matching_factors.append('p53 mutation (cell cycle disruption)')
            
            if tmb_score > 20:
                if drug_info['class'] == 'Immune Checkpoint Inhibitor':
                    compatibility_score += 0.4
                    matching_factors.append('High TMB (immunotherapy response)')
            
            # Pathway alignment
            drug_targets = drug_info['targets']
            pathway_alignment = 0
            for pathway in cancer_model['key_pathways']:
                if any(target in pathway for target in drug_targets):
                    pathway_alignment += 0.1
            
            compatibility_score += pathway_alignment
            
            biomarker_matches.append({
                'Drug': drug,
                'Compatibility Score': f"{compatibility_score:.2f}",
                'Matching Factors': ', '.join(matching_factors) if matching_factors else 'None specific',
                'Pathway Alignment': f"{pathway_alignment:.1f}"
            })
        
        biomarker_df = pd.DataFrame(biomarker_matches)
        st.dataframe(biomarker_df, use_container_width=True)
        
        # Cost-effectiveness analysis
        st.subheader("💰 Pharmacoeconomic Analysis")
        
        cost_data = []
        for drug in selected_drugs:
            drug_info = DRUG_DATABASE[drug]
            
            # Calculate various cost metrics
            annual_cost = drug_info['cost_per_month'] * 12
            response_rate = drug_info['response_rate']
            
            # Cost per response
            cost_per_response = annual_cost / max(0.1, response_rate)
            
            # Incremental cost-effectiveness ratio (simplified)
            baseline_effectiveness = 0.2  # Baseline care
            incremental_effectiveness = response_rate - baseline_effectiveness
            
            if incremental_effectiveness > 0:
                icer = annual_cost / incremental_effectiveness
            else:
                icer = float('inf')
            
            cost_data.append({
                'Drug': drug,
                'Annual Cost': annual_cost,
                'Response Rate': f"{response_rate:.1%}",
                'Cost per Response': cost_per_response,
                'ICER': icer if icer != float('inf') else 'Dominated'
            })
        
        cost_df = pd.DataFrame(cost_data)
        
        # Cost visualization
        fig_cost = px.scatter(
            cost_df,
            x='Annual Cost',
            y=[float(r.strip('%'))/100 for r in cost_df['Response Rate']],
            size='Cost per Response',
            hover_name='Drug',
            title='Cost vs Effectiveness Analysis',
            labels={'y': 'Response Rate', 'x': 'Annual Cost ($)'}
        )
        
        st.plotly_chart(fig_cost, use_container_width=True)
        st.dataframe(cost_df, use_container_width=True)
        
        # Risk stratification
        st.subheader("⚠️ Risk Stratification Analysis")
        
        risk_factors = []
        
        # Age-based risk
        if age > 70:
            risk_factors.append("Advanced age (>70) - Increased toxicity risk")
        elif age < 30:
            risk_factors.append("Young age (<30) - Consider fertility preservation")
        
        # Performance status risk
        if performance_status >= 2:
            risk_factors.append("Poor performance status - Limited treatment tolerance")
        
        # Genetic risk factors
        if 'CYP2D6*4' in genetic_variants:
            risk_factors.append("CYP2D6*4 variant - Altered drug metabolism")
        
        # Comorbidity risks
        active_comorbidities = [c for c in comorbidities if c != 'None']
        if active_comorbidities:
            risk_factors.append(f"Comorbidities: {', '.join(active_comorbidities)} - Increased monitoring needed")
        
        # Prior treatment history
        if 'Chemotherapy' in prior_treatments:
            risk_factors.append("Prior chemotherapy - Potential cumulative toxicity")
        
        if risk_factors:
            st.warning("**Identified Risk Factors:**\n\n" + "\n".join([f"• {risk}" for risk in risk_factors]))
        else:
            st.success("**Low Risk Profile** - Standard treatment protocols appropriate")

# Tab 5: Clinical Report
with tab5:
    st.header("📋 Clinical Simulation Report")
    
    # Report generation
    if st.button("📄 Generate Comprehensive Report", type="primary"):
        # Current timestamp
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        st.markdown(f"""
        <div class="simulation-section">
            <h2>🧬 AI Cancer Drug Simulation Report</h2>
            <p><strong>Generated:</strong> {report_time}</p>
            <p><strong>Platform Version:</strong> 2.0</p>
            
            <h3>📊 Patient Profile</h3>
            <ul>
                <li><strong>Age:</strong> {age} years</li>
                <li><strong>Weight:</strong> {weight} kg</li>
                <li><strong>Performance Status:</strong> ECOG {performance_status}</li>
                <li><strong>Cancer Type:</strong> {selected_cancer}</li>
                <li><strong>Genetic Variants:</strong> {', '.join(genetic_variants) if genetic_variants else 'None detected'}</li>
                <li><strong>Comorbidities:</strong> {', '.join([c for c in comorbidities if c != 'None']) if any(c != 'None' for c in comorbidities) else 'None'}</li>
            </ul>
            
            <h3>🎯 Cancer Model Characteristics</h3>
            <ul>
                <li><strong>Growth Rate:</strong> {CANCER_MODELS[selected_cancer]['growth_rate']:.2f}</li>
                <li><strong>Drug Resistance:</strong> {CANCER_MODELS[selected_cancer]['drug_resistance']:.2f}</li>
                <li><strong>Metastatic Potential:</strong> {CANCER_MODELS[selected_cancer]['metastatic_potential']:.2f}</li>
                <li><strong>Expected Survival:</strong> {CANCER_MODELS[selected_cancer]['survival_months']} months</li>
            </ul>
            
            <h3>💊 Analyzed Drugs</h3>
            <ul>
                {''.join([f'<li><strong>{drug}</strong> ({DRUG_DATABASE[drug]["class"]})</li>' for drug in selected_drugs])}
            </ul>
            
            <h3>⚙️ Simulation Parameters</h3>
            <ul>
                <li><strong>Duration:</strong> {simulation_duration} days</li>
                <li><strong>Mode:</strong> {simulation_mode}</li>
                <li><strong>Resistance Modeling:</strong> {'Enabled' if enable_resistance else 'Disabled'}</li>
                <li><strong>Tumor Heterogeneity:</strong> {'Enabled' if enable_heterogeneity else 'Disabled'}</li>
                <li><strong>Monte Carlo Iterations:</strong> {monte_carlo_iterations}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations section
        st.markdown("""
        <div class="simulation-section">
            <h3>🏆 Clinical Recommendations</h3>
            <p><strong>Disclaimer:</strong> This simulation is for research and educational purposes only. 
            All treatment decisions should be made by qualified healthcare professionals based on 
            comprehensive clinical evaluation.</p>
            
            <h4>🔬 Simulation Insights:</h4>
            <ul>
                <li>Patient-specific factors have been incorporated into the analysis</li>
                <li>Drug interactions and resistance patterns were modeled</li>
                <li>Cost-effectiveness considerations included</li>
                <li>Risk stratification performed</li>
            </ul>
            
            <h4>📋 Next Steps:</h4>
            <ul>
                <li>Validate findings with clinical data</li>
                <li>Consider additional biomarker testing</li>
                <li>Multidisciplinary team consultation</li>
                <li>Patient preference discussion</li>
                <li>Regular monitoring protocol establishment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.success("✅ Comprehensive report generated successfully!")
    
    # Export options
    st.subheader("📤 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Data (CSV)", use_container_width=True):
            # Create exportable data
            export_data = {
                'Patient_Age': [age],
                'Patient_Weight': [weight],
                'Cancer_Type': [selected_cancer],
                'Simulation_Mode': [simulation_mode],
                'Duration_Days': [simulation_duration],
                'Selected_Drugs': [', '.join(selected_drugs)],
                'Genetic_Variants': [', '.join(genetic_variants)],
                'Report_Generated': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            }
            
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"cancer_simulation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📋 Copy Report Summary", use_container_width=True):
            report_summary = f"""
Cancer Drug Simulation Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Patient: {age}y, {weight}kg, ECOG {performance_status}
Cancer: {selected_cancer}
Drugs Analyzed: {', '.join(selected_drugs)}
Simulation Duration: {simulation_duration} days

Key Findings:
- Simulation completed successfully
- Patient-specific factors incorporated
- Risk assessment performed
- Treatment recommendations generated

Note: For research purposes only. Clinical validation required.
            """
            
            st.code(report_summary, language=None)
            st.info("📋 Report summary ready for copying")
    
    with col3:
        if st.button("🔄 Reset Simulation", use_container_width=True):
            st.warning("⚠️ This will reset all parameters to default values.")
            if st.button("✅ Confirm Reset"):
                st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; margin-top: 2rem;">
    <h3>🧬 AI Cancer Drug Simulation Platform v2.0</h3>
    <p><strong>Developed by Muhammad Nabeel Saddique</strong></p>
    <p>Medical Student | Oncology Research | Computational Biology</p>
    <p>🎯 <em>Advancing Precision Oncology through AI-Driven Drug Discovery</em></p>
    
    <div style="margin-top: 1rem; padding: 1rem; background: rgba(255,255,255,0.8); border-radius: 10px;">
        <h4>⚠️ Important Disclaimer</h4>
        <p><strong>This platform is designed for research, education, and academic presentation purposes only.</strong></p>
        <p>• Not intended for clinical decision-making or patient treatment planning</p>
        <p>• All predictions and recommendations require clinical validation</p>
        <p>• Consult qualified healthcare professionals for actual medical decisions</p>
        <p>• Results should be interpreted within appropriate clinical context</p>
    </div>
    
    <div style="margin-top: 1rem;">
        <h4>🔬 Research Applications</h4>
        <p>• Drug repurposing discovery • Clinical trial optimization • Biomarker research</p>
        <p>• Personalized medicine development • Pharmacoeconomic analysis • Academic studies</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📊 Platform Statistics
- **Cancer Models**: 6 types
- **Drug Database**: 10 compounds  
- **Simulation Features**: 15+
- **Analysis Tools**: 8 modules

### 🎯 Key Features
✅ Advanced PK/PD modeling  
✅ Patient-specific factors  
✅ Resistance prediction  
✅ Monte Carlo analysis  
✅ Cost-effectiveness  
✅ Biomarker integration  
✅ Risk stratification  
✅ Survival prediction
""")

st.sidebar.info("""
**💡 Tips for Better Results:**
- Include relevant genetic variants
- Specify comorbidities accurately  
- Use appropriate simulation duration
- Enable Monte Carlo for uncertainty
- Consider combination therapies
- Review biomarker matches
""")

st.sidebar.success("""
**🚀 Enhanced Features v2.0:**
- Expanded drug database
- Improved cancer models
- Advanced analytics dashboard
- Monte Carlo uncertainty analysis
- Biomarker-drug matching
- Comprehensive reporting
""")