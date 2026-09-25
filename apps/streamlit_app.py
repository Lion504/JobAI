"""
JobAI Streamlit Dashboard: AI Agent Forecast Chatbot & Labour Market Analytics
"""
import os
import sys
from pathlib import Path
import json
import pandas as pd
import streamlit as st

# Locate JobAI repository root
def find_repo_root():
    env = os.environ.get("JOBAI_REPO")
    if env and Path(env).is_dir():
        return Path(env).resolve()
    current = Path.cwd().resolve()
    for cand in (current, *current.parents):
        if (cand / "configs" / "data.yaml").is_file() or (cand / "jobai").is_dir():
            return cand
    return current

REPO = find_repo_root()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="JobAI — Finnish Labour Market Forecast & Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar: Settings & Theme Toggle
st.sidebar.title("JobAI Settings")
dark_mode = st.sidebar.toggle("Dark Mode", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "### Project Specs\n"
    "- **Model**: `Qwen3-4B` (4-bit QLoRA)\n"
    "- **Training**: 151,727 panel examples\n"
    "- **Horizons**: 1Q (3m), 2Q (6m), 4Q (12m)\n"
    "- **RAG**: BGE-M3 + TEM/KEHA Bulletins\n"
)
st.sidebar.markdown("---")
st.sidebar.info("Tip: Use the tabs above to toggle between the **Chatbot** and **Analytics**.")

# Custom Styling: RGB Animated Gradient Border & Theme Tokens
theme_vars = """
    --bg-color: #0E1117;
    --text-color: #F3F4F6;
    --container-bg: #161B22;
    --card-bg: #1F2937;
    --border-color: #374151;
    --table-header: #374151;
""" if dark_mode else """
    --bg-color: #F8FAFC;
    --text-color: #0F172A;
    --container-bg: #FFFFFF;
    --card-bg: #F1F5F9;
    --border-color: #E2E8F0;
    --table-header: #E2E8F0;
"""

custom_css = f"""
<style>
:root {{
    {theme_vars}
}}

/* RGB Animated Gradient Border around main container */
@keyframes gradientBorderAnimation {{
    0% {{
        border-color: #F08080;
        box-shadow: 0 0 16px rgba(240, 128, 128, 0.45);
    }}
    50% {{
        border-color: #ADD8E6;
        box-shadow: 0 0 18px rgba(173, 216, 230, 0.55);
    }}
    100% {{
        border-color: #F08080;
        box-shadow: 0 0 16px rgba(240, 128, 128, 0.45);
    }}
}}

.animated-main-container {{
    border: 3.5px solid #F08080;
    border-radius: 16px;
    padding: 24px;
    background-color: var(--container-bg);
    animation: gradientBorderAnimation 4s ease-in-out infinite;
    margin-bottom: 24px;
    transition: all 0.3s ease-in-out;
}}

.metric-card {{
    background-color: var(--card-bg);
    border-radius: 10px;
    padding: 16px;
    border: 1px solid var(--border-color);
    text-align: center;
}}

.stApp {{
    background-color: var(--bg-color);
    color: var(--text-color);
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Top Navigation Bar
tab_chat, tab_analytics = st.tabs(["AI Agent Chatbot", "Figures & Analytics"])

# Tab 1: AI Agent Chatbot
with tab_chat:
    st.markdown(
        """
        <div class="animated-main-container">
            <h2 style="margin-top:0;">Labour Market Forecasting Assistant</h2>
            <p style="color:gray;">
                Ask questions about Finnish quarterly job vacancy forecasts (1Q, 2Q, 4Q) by region, occupation, or industry.
                The AI agent computes numerical predictions using the fine-tuned <b>Qwen3-4B</b> adapter and cites official 
                <b>Ministry of Economic Affairs & Employment (TEM)</b> bulletins to explain the forecast.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Cached service loader to ensure GPU weights load once
    @st.cache_resource(show_spinner="Connecting to AI Agent & Vector Index...")
    def load_forecast_service(repo_dir):
        try:
            from jobai.chat import ForecastService
            service = ForecastService(repo_dir, allow_embedding_download=True)
            return service, None
        except Exception as e:
            return None, str(e)

    service, service_err = load_forecast_service(REPO)

    # Initialize chat session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "**Terve! I am JobAI.**\n\n"
                    "I forecast registered job vacancies in Finland and explain trends using official bulletins.\n"
                    "Try asking:\n"
                    "- *'What is the vacancy outlook for nurses in Uusimaa?'*\n"
                    "- *'How are software development jobs looking in Pirkanmaa for the next 2 quarters?'*\n"
                    "- *'What is the trend for construction workers in North Ostrobothnia?'*"
                ),
                "forecasts": None,
                "rag": None
            }
        ]
    if "chat_context" not in st.session_state:
        st.session_state.chat_context = None

    # Display historical chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("forecasts") is not None and not msg["forecasts"].empty:
                st.dataframe(msg["forecasts"], use_container_width=True)
            if msg.get("rag") and msg["rag"].get("passages"):
                passages = msg["rag"]["passages"]
                with st.expander(f"View {len(passages)} Quoted Official Sources"):
                    for idx, p in enumerate(passages, 1):
                        st.markdown(f"**Source {idx}:** [{p.get('title', 'Bulletin')}]({p.get('url', '#')}) *({p.get('published', 'N/A')})*")
                        st.caption(f"> \"{p.get('text', '')[:320]}...\"")

    # Chat Input Box
    if prompt := st.chat_input("Ask a forecast question (e.g., 'What is the outlook for nurses in Uusimaa?')..."):
        # Record user message
        st.session_state.messages.append({"role": "user", "content": prompt, "forecasts": None, "rag": None})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant response block
        with st.chat_message("assistant"):
            if service is None:
                err_msg = (
                    f"**Forecast Service could not be initialized:** `{service_err}`\n\n"
                    "Please ensure you are connected to a **GPU runtime** in Colab and that the dependencies "
                    "from `notebooks/12_forecast_chat_with_rag.ipynb` are installed."
                )
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg, "forecasts": None, "rag": None})
            else:
                with st.spinner("Routing query, forecasting with Qwen3-4B, and retrieving bulletin context..."):
                    try:
                        result = service.answer_question(prompt, context=st.session_state.chat_context)
                        st.session_state.chat_context = result.get("context")
                        answer_text = result.get("answer", "No response generated.")
                        st.markdown(answer_text)

                        forecasts_df = None
                        if result.get("status") == "answered" and result.get("forecasts"):
                            raw_df = pd.DataFrame(result["forecasts"])
                            show_cols = [c for c in ["origin_quarter", "target_quarter", "horizon_q", "last_value", "y_pred"] if c in raw_df.columns]
                            forecasts_df = raw_df[show_cols].round(2)
                            st.dataframe(forecasts_df, use_container_width=True)

                        rag_info = result.get("rag", {})
                        if rag_info.get("passages"):
                            passages = rag_info["passages"]
                            with st.expander(f"View {len(passages)} Quoted Official Sources"):
                                for idx, p in enumerate(passages, 1):
                                    st.markdown(f"**Source {idx}:** [{p.get('title', 'Bulletin')}]({p.get('url', '#')}) *({p.get('published', 'N/A')})*")
                                    st.caption(f"> \"{p.get('text', '')[:320]}...\"")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer_text,
                            "forecasts": forecasts_df,
                            "rag": rag_info
                        })
                    except Exception as ex:
                        fail_msg = f"**An error occurred during inference:** `{ex}`"
                        st.error(fail_msg)
                        st.session_state.messages.append({"role": "assistant", "content": fail_msg, "forecasts": None, "rag": None})

# Tab 2: Figures & Analytics Dashboard
with tab_analytics:
    st.markdown(
        """
        <div class="animated-main-container">
            <h2 style="margin-top:0;">Labour Market Figures & Model Analytics</h2>
            <p style="color:gray;">
                Explore Statistics Finland data trends, baseline comparisons, and the fine-tuned model's evaluation metrics.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Helper function to locate images across figures/ and reports/figures/
    def find_figure_path(candidates):
        search_dirs = [
            REPO / "figures",
            REPO / "reports" / "figures",
            REPO / "reports"
        ]
        for sdir in search_dirs:
            for name in candidates:
                p = sdir / name
                if p.is_file():
                    return p
        return None

    # Primary Historical Vacancy Trends
    st.subheader("Primary Historical Vacancy Trends")
    fig_col1, fig_col2 = st.columns(2)

    # National Vacancies
    national_fig = find_figure_path([
        "vacancies_national.png",
        "02_national_vacancy_measures_over_time.png",
        "national_vacancies.png"
    ])
    with fig_col1:
        st.markdown("#### 🇫🇮 National Job Vacancies")
        if national_fig:
            st.image(str(national_fig), caption="National Vacancies over Time (Statistics Finland 11l1)", use_container_width=True)
        else:
            st.warning("National vacancies figure not found in `figures/` or `reports/figures/`.")

    # Regional Vacancies
    regional_fig = find_figure_path([
        "vacancies_regional.png",
        "02_regional_total_vacancies_over_time.png",
        "regional_vacancies.png"
    ])
    with fig_col2:
        st.markdown("#### 🇫🇮 Regional Job Vacancies")
        if regional_fig:
            st.image(str(regional_fig), caption="Regional Total Vacancies across Finland (11n1)", use_container_width=True)
        else:
            st.warning("Regional vacancies figure not found in `figures/` or `reports/figures/`.")

    st.markdown("---")

    # Dropdown Selector for Additional Figures
    st.subheader("Inspect Additional Model & Data Analytics")

    FIGURE_CATALOG = {
        "Forecasting Comparison: Model vs Baselines": [
            "forecasting_comparison.png",
            "model_vs_baselines_test_mae_old.png",
            "shared_models_smape_mae.png"
        ],
        "Regional Breakdown: Dataset A Target Series": [
            "regional_breakdown.png",
            "03_dataset_a_targets.png",
            "03_dataset_a_targets_old.png"
        ],
        "Forecast Accuracy: MAE by Horizon": [
            "mae_by_horizon.png",
            "04_test_mae_by_horizon.png"
        ],
        "Vacancy Count Distribution across Series": [
            "vacancy_distribution.png",
            "02_total_vacancies_distribution.png"
        ],
        "Series Selection Quality Diagnostics": [
            "03_panel_selection_diagnostics.png"
        ],
        "Training Size vs Baselines Comparison": [
            "qwen3_4b_training_size_vs_baselines_mae_old.png"
        ],
        "Horizon-Specific Adapters vs Baselines": [
            "qwen3_4b_horizon_adapters_vs_baselines_mae_old.png"
        ],
        "Matched Test Performance Across Adapters": [
            "qwen3_4b_previous_vs_horizon_adapters_matched_mae_old.png"
        ]
    }

    selected_option = st.selectbox(
        "Choose an analytical figure to view:",
        options=["-- Select a figure to inspect --"] + list(FIGURE_CATALOG.keys())
    )

    if selected_option and selected_option != "-- Select a figure to inspect --":
        target_files = FIGURE_CATALOG[selected_option]
        found_file = find_figure_path(target_files)
        if found_file:
            st.image(str(found_file), caption=selected_option, use_container_width=True)
        else:
            st.warning(f"Image file for '{selected_option}' not found. Searched for: {target_files} in `figures/` and `reports/figures/`.")

    # Baselines & Model Performance Scorecard
    st.markdown("---")
    st.subheader("Baselines & Model Performance Scorecard")

    baseline_csv_candidates = [
        REPO / "reports" / "model_vs_baselines.csv",
        REPO / "reports" / "baselines.csv",
        REPO / "reports" / "model_evaluations" / "comparisons" / "comparison_c64a6d3e5c17" / "adapter_test_predictions.csv"
    ]
    loaded_table = None
    for b_path in baseline_csv_candidates:
        if b_path.is_file():
            try:
                loaded_table = pd.read_csv(b_path)
                st.caption(f"Loaded scorecard from `{b_path.relative_to(REPO)}`")
                st.dataframe(loaded_table, use_container_width=True)
                break
            except Exception:
                pass
    if loaded_table is None:
        st.info("Baseline scorecard CSV not found in `reports/`. Run Notebook 04 or 07 to generate detailed metrics tables.")
