import streamlit as st
import pandas as pd
import time
import plotly.express as px
import os
import asyncio
from utils.data_loader import load_transactional_data, get_preprocessing_function, get_target_column
from agents.ab_test_agent import ABTestAgent
from agents.detection_agent import DetectionAgent, get_expert_rules_function
from agents.explanation_agent import ExplanationAgent
from agents.evaluator_agent import EvaluatorAgent
from utils.llm_service import LLMService

# --- Page Config ---
st.set_page_config(
    page_title="FinGuard-AI: Self-Optimizing Fraud Detection",
    page_icon="🤖",
    layout="wide",
)

# --- Constants ---
BATCH_SIZE = 5

# (The rest of the initial setup functions remain the same)
# ... (get_all_transactional_data, get_llm_service, initialize_agents, initialize_session_state) ...

# (Re-paste the setup functions here to ensure they are included)
@st.cache_data
def get_all_transactional_data():
    """Loads all transactional data from CSV files."""
    return load_transactional_data()

@st.cache_resource
def get_llm_service():
    """Initializes and caches the LLMService."""
    try:
        return LLMService()
    except ValueError as e:
        st.error(f"LLM Service Error: {e}. Please ensure GROQ_API_KEY is set as an environment variable.")
        st.stop()

def initialize_agents(selected_dataset_name, llm_service_instance):
    """Initializes all agents with dataset-specific configurations and LLM service."""
    expert_rules_func = get_expert_rules_function(selected_dataset_name)
    
    detection_variants = ["A", "B"]
    explanation_variants = ["C", "D"]
    
    agents = {
        "ab_test_detection": ABTestAgent(detection_variants),
        "ab_test_explanation": ABTestAgent(explanation_variants),
        "detection": DetectionAgent(expert_rules=expert_rules_func, llm_service=llm_service_instance),
        "explanation": ExplanationAgent(llm_service=llm_service_instance, expert_rules_func=expert_rules_func),
        "evaluator": EvaluatorAgent(),
    }
    return agents

def initialize_session_state():
    """Initializes or resets the session state for the application."""
    if "llm_service" not in st.session_state:
        st.session_state.llm_service = get_llm_service()

    if "all_raw_data" not in st.session_state:
        st.session_state.all_raw_data = get_all_transactional_data()
        st.session_state.available_datasets = list(st.session_state.all_raw_data.keys())
        if "creditcard" in st.session_state.available_datasets:
            st.session_state.selected_dataset = "creditcard"
        elif st.session_state.available_datasets:
            st.session_state.selected_dataset = st.session_state.available_datasets[0]
        else:
            st.error("No transactional data found in the 'data/' directory.")
            st.stop()
    
    if ("selected_dataset" not in st.session_state) or \
       ("agents" not in st.session_state) or \
       (st.session_state.agents["detection"].expert_rules.__name__ != get_expert_rules_function(st.session_state.selected_dataset).__name__):
        
        st.session_state.agents = initialize_agents(st.session_state.selected_dataset, st.session_state.llm_service)
        
        raw_df = st.session_state.all_raw_data[st.session_state.selected_dataset].copy()
        preprocessing_func = get_preprocessing_function(st.session_state.selected_dataset)
        preprocessed_df = preprocessing_func(raw_df)
        
        target_column = get_target_column(st.session_state.selected_dataset)
        
        if target_column not in preprocessed_df.columns:
            st.error(f"Target column '{target_column}' not found in {st.session_state.selected_dataset} after preprocessing.")
            st.stop()

        golden_preference_column = None
        if "golden_explanation_preference" in preprocessed_df.columns:
            golden_preference_column = preprocessed_df["golden_explanation_preference"]
            preprocessed_df = preprocessed_df.drop("golden_explanation_preference", axis=1)

        st.session_state.data = {
            "X": preprocessed_df.drop(target_column, axis=1),
            "y": preprocessed_df[target_column],
            "golden_preference": golden_preference_column
        }
        
        st.session_state.current_index = 0
        st.session_state.transaction_history = []
        st.session_state.metrics_history = []
        st.session_state.variant_history_detection = {v: [] for v in st.session_state.agents["ab_test_detection"].variants}
        st.session_state.variant_history_explanation = {v: [] for v in st.session_state.agents["ab_test_explanation"].variants}

    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "transaction_history" not in st.session_state:
        st.session_state.transaction_history = []
    if "metrics_history" not in st.session_state:
        st.session_state.metrics_history = []
    
    if "variant_history_detection" not in st.session_state:
        st.session_state.variant_history_detection = {v: [] for v in st.session_state.agents["ab_test_detection"].variants}
    if "variant_history_explanation" not in st.session_state:
        st.session_state.variant_history_explanation = {v: [] for v in st.session_state.agents["ab_test_explanation"].variants}

# --- Main App Execution ---
initialize_session_state()
agents = st.session_state.agents
X = st.session_state.data["X"]
y = st.session_state.data["y"]
golden_preference = st.session_state.data["golden_preference"]

# --- UI Sidebar ---
st.sidebar.title("Configuration")
selected_dataset_ui = st.sidebar.selectbox(
    "Select Dataset",
    options=st.session_state.available_datasets,
    index=st.session_state.available_datasets.index(st.session_state.selected_dataset)
)

if selected_dataset_ui != st.session_state.selected_dataset:
    st.session_state.selected_dataset = selected_dataset_ui
    st.rerun()

st.sidebar.title("Simulation Controls")
start_button = st.sidebar.button("Start Full Simulation", key="start_simulation_btn")
if st.sidebar.button("Reset Simulation", key="reset_btn"):
    for key in list(st.session_state.keys()):
        if key not in ["all_raw_data", "available_datasets", "selected_dataset", "llm_service"]:
            del st.session_state[key]
    initialize_session_state()
    st.rerun()

st.sidebar.title("Simulation Progress")
progress_bar = st.sidebar.progress(0)
progress_metric = st.sidebar.metric("Transactions Processed", f"0 / {len(X)}")

async def run_full_simulation(placeholder):
    """Runs the entire simulation asynchronously in batches."""
    total_transactions = len(X)
    
    for i in range(0, total_transactions, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total_transactions)
        X_batch = X.iloc[i:batch_end]
        y_batch = y.iloc[i:batch_end]
        
        # --- A/B Test Variant Selection ---
        detection_variant = agents["ab_test_detection"].choose_variant()
        explanation_variants = [agents["ab_test_explanation"].choose_variant() for _ in range(len(X_batch))]

        # --- Process Transaction Batch ---
        detection_results = await agents["detection"].detect_batch(X_batch, detection_variant)
        explanations = await agents["explanation"].explain_batch(X_batch, detection_results, explanation_variants, st.session_state.selected_dataset)

        # --- Evaluate and Update Agents ---
        for j, (detection_result, explanation_variant) in enumerate(zip(detection_results, explanation_variants)):
            current_index = i + j
            ground_truth = y_batch.iloc[j]
            
            detection_reward = agents["evaluator"].evaluate_detection(detection_result, ground_truth)
            agents["ab_test_detection"].update(detection_variant, detection_reward)

            explanation_reward = 0.5
            if detection_result[0] == 1:
                if golden_preference is not None:
                    preference = golden_preference.iloc[current_index]
                    explanation_reward = agents["evaluator"].evaluate_explanation(explanation_variant, preference)
                
                st.session_state.transaction_history.append({
                    "transaction_index": current_index,
                    "detection_variant": detection_variant,
                    "explanation_variant": explanation_variant,
                    "explanation": explanations[j],
                    "ground_truth": "Fraud" if ground_truth == 1 else "Not Fraud",
                })
            
            agents["ab_test_explanation"].update(explanation_variant, explanation_reward)

        # --- Periodic UI Update ---
        metrics = agents["evaluator"].get_metrics()
        st.session_state.metrics_history.append(metrics)

        # Append history for charts
        for variant in agents["ab_test_detection"].variants:
            successes = agents["ab_test_detection"].successes[variant]
            failures = agents["ab_test_detection"].failures[variant]
            st.session_state.variant_history_detection[variant].append(successes / (successes + failures + 1e-6))

        for variant in agents["ab_test_explanation"].variants:
            successes = agents["ab_test_explanation"].successes[variant]
            failures = agents["ab_test_explanation"].failures[variant]
            st.session_state.variant_history_explanation[variant].append(successes / (successes + failures + 1e-6))

        progress_bar.progress(batch_end / total_transactions)
        progress_metric.metric("Transactions Processed", f"{batch_end} / {total_transactions}")
        with placeholder.container():
            display_results()

    st.balloons()

def display_results():
    """Renders the results area with the latest data from session state."""
    st.header(f"Live Metrics for '{st.session_state.selected_dataset}'")
    metrics_cols = st.columns(4)
    metrics = agents["evaluator"].get_metrics()
    with metrics_cols[0]:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2f}")
    with metrics_cols[1]:
        st.metric("Flagged", f"{metrics.get('false_positives', 0)}")
    with metrics_cols[2]:
        st.metric("Trust Metric", f"{metrics.get('trust_metric', 0):.2f}")
    with metrics_cols[3]:
        st.metric("Total Transactions", f"{metrics.get('total_transactions', 0)}")
    
    # --- Visualizations ---
    st.header("Performance Over Time")
    viz_cols = st.columns(2)

    with viz_cols[0]:
        st.subheader("Detection Variant Performance")
        if st.session_state.variant_history_detection.get("A"): 
            fig = px.line(title="Detection Variant Success Rate")
            for variant, history in st.session_state.variant_history_detection.items():
                fig.add_scatter(y=history, name=variant)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run simulation to see detection variant performance.")

    with viz_cols[1]:
        st.subheader("Explanation Variant Performance")
        if st.session_state.variant_history_explanation.get("C"):
            fig = px.line(title="Explanation Variant Selection Rate")
            for variant, history in st.session_state.variant_history_explanation.items():
                fig.add_scatter(y=history, name=variant)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run simulation to see explanation variant performance.")

    # --- Flagged Transactions ---
    st.header("Flagged Transactions")
    if st.session_state.transaction_history:
        for item in reversed(st.session_state.transaction_history[-10:]):
            with st.expander(f"Transaction {item['transaction_index']} (Detected: {item['detection_variant']}, Explained: {item['explanation_variant']})"):
                st.write(item["explanation"])
    else:
        st.info("No transactions flagged yet. Start the simulation!")

# --- Main App Logic ---
placeholder = st.empty()
with placeholder.container():
    display_results()

if start_button:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    loop.run_until_complete(run_full_simulation(placeholder))
