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
st.sidebar.markdown("### System Diagnostics")

# Check GPU acceleration
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        st.sidebar.success(f"🟢 **GPU:** {gpu_name}")
    else:
        st.sidebar.warning("🟡 **CPU Mode:** Select T4 GPU in Colab for full inference.")
except Exception:
    st.sidebar.warning("⚪ PyTorch not initialized.")

# Check Chroma Vector Store
chroma_dir = REPO / "data/processed/rag/chroma"
if chroma_dir.is_dir() and any(chroma_dir.iterdir()):
    st.sidebar.success("🟢 **RAG:** 6,982 docs ready")
else:
    st.sidebar.error("🔴 **RAG:** Chroma index missing")

# Check Pinned LoRA Adapter
adapter_dir = REPO / "models/adapters/qwen-qwen3-4b__20260919T125120Z/final_adapter"
if adapter_dir.is_dir():
    st.sidebar.success("🟢 **Adapter:** Pinned & verified")
else:
    st.sidebar.error("🔴 **Adapter:** Checkpoint missing")

st.sidebar.markdown("---")
st.sidebar.info("Tip: Switch tabs above for **Analytics**.")

# Custom Styling: RGB Animated Gradient Border & Theme Tokens
theme_vars = """
    --bg-color: #0E1117;
    --text-color: #FFFFFF;
    --subtext-color: #E2E8F0;
    --container-bg: #161B22;
    --card-bg: #1F2937;
    --border-color: #374151;
    --table-header: #374151;
    --chat-bg: #161B22;
    --chat-text: #FFFFFF;
    --tab-text: #CBD5E1;
    --tab-text-active: #FFFFFF;
    --tab-border-active: #F08080;
    --header-bg: rgba(14, 17, 23, 0.82);
    --chat-input-bg: rgba(255, 255, 255, 0.90);
""" if dark_mode else """
    --bg-color: #F8FAFC;
    --text-color: #000000;
    --subtext-color: #1E293B;
    --container-bg: #FFFFFF;
    --card-bg: #F1F5F9;
    --border-color: #CBD5E1;
    --table-header: #E2E8F0;
    --chat-bg: #FFFFFF;
    --chat-text: #000000;
    --tab-text: #334155;
    --tab-text-active: #000000;
    --tab-border-active: #2563EB;
    --header-bg: rgba(248, 250, 252, 0.85);
    --chat-input-bg: rgba(255, 255, 255, 0.92);
"""

custom_css = f"""
<style>
:root {{
    {theme_vars}
}}

/* Sidebar radiant animated border glow */
@keyframes radiantSidebarGlow {{
    0% {{
        border-right-color: #F08080;
        box-shadow: 4px 0 16px rgba(240, 128, 128, 0.45);
    }}
    50% {{
        border-right-color: #ADD8E6;
        box-shadow: 4px 0 18px rgba(173, 216, 230, 0.55);
    }}
    100% {{
        border-right-color: #F08080;
        box-shadow: 4px 0 16px rgba(240, 128, 128, 0.45);
    }}
}}

/* Sidebar Width: exactly 1.5 out of 9 with radiant color-changing border */
section[data-testid="stSidebar"], [data-testid="stSidebar"] {{
    width: calc(100vw * 1.5 / 9) !important;
    min-width: 220px !important;
    max-width: calc(100vw * 1.5 / 9) !important;
    background-color: #F8FAFC !important; /* Crisp light background */
    border-right-width: 2.5px !important;
    border-right-style: solid !important;
    animation: radiantSidebarGlow 4s ease-in-out infinite !important;
}}

/* Sidebar compact layout: tight vertical spacing so everything fits on one screen */
section[data-testid="stSidebar"] .block-container {{
    padding-top: 1rem !important;
}}

section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
    gap: 0.35rem !important;
}}

section[data-testid="stSidebar"] hr {{
    margin: 0.35rem 0 !important;
    border-color: #CBD5E1 !important;
}}

/* All white/light background elements: crisp black text in dark mode & light mode */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] strong,
section[data-testid="stSidebar"] b,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
section[data-testid="stSidebar"] [data-testid="stAlert"] * {{
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}}

section[data-testid="stSidebar"] code {{
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    background-color: #E2E8F0 !important;
    font-size: 0.76rem !important;
    padding: 1px 4px !important;
}}

/* Radiant animated glow for fixed top header bar */
@keyframes radiantHeaderGlow {{
    0% {{
        border-bottom-color: #F08080;
        box-shadow: 0 4px 14px rgba(240, 128, 128, 0.35);
    }}
    50% {{
        border-bottom-color: #ADD8E6;
        box-shadow: 0 4px 16px rgba(173, 216, 230, 0.45);
    }}
    100% {{
        border-bottom-color: #F08080;
        box-shadow: 0 4px 14px rgba(240, 128, 128, 0.35);
    }}
}}

/* Chat input typing bar: radiant glowing border, translucent frosted glass, pinned to bottom */
div[data-testid="stChatInput"] {{
    position: fixed !important;
    bottom: 20px !important;
    left: calc(max(220px, 100vw * 1.5 / 9) + 2rem) !important;
    right: 2rem !important;
    width: calc(100vw - max(220px, 100vw * 1.5 / 9) - 4rem) !important;
    max-width: 100% !important;
    z-index: 999 !important;
    background-color: var(--chat-input-bg) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border-width: 2.5px !important;
    border-style: solid !important;
    border-radius: 14px !important;
    animation: gradientBorderAnimation 4s ease-in-out infinite !important;
}}

section[data-testid="stSidebar"][aria-expanded="false"] ~ .main div[data-testid="stChatInput"] {{
    left: 2rem !important;
    width: calc(100vw - 4rem) !important;
}}

[data-testid="stChatInput"] > div {{
    background-color: transparent !important;
    border: none !important;
}}

[data-testid="stChatInput"] textarea {{
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    caret-color: #000000 !important;
    background-color: transparent !important;
}}

[data-testid="stChatInput"] textarea::placeholder {{
    color: #4B5563 !important;
    -webkit-text-fill-color: #4B5563 !important;
    opacity: 1 !important;
}}

[data-testid="stChatInput"] button svg {{
    fill: #000000 !important;
    color: #000000 !important;
}}

div[data-testid="stBottom"] {{
    background-color: transparent !important;
}}

/* Main container: expands cleanly to fill the remaining 7.5 out of 9 with padding for fixed top bar and bottom chat bar */
.main .block-container {{
    padding-top: 3.6rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-bottom: 6.5rem !important;
    max-width: 100% !important;
}}

/* Top header bar: translucent frosted glass with radiant bottom glow */
header[data-testid="stHeader"] {{
    background-color: var(--header-bg) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border-bottom-width: 2.5px !important;
    border-bottom-style: solid !important;
    animation: radiantHeaderGlow 4s ease-in-out infinite !important;
    z-index: 999990 !important;
}}

header[data-testid="stHeader"] > div {{
    background-color: transparent !important;
}}

/* Prevent parent containers from trapping fixed tabs inside main area */
[data-testid="stAppViewContainer"],
section.main,
.main,
.block-container,
div[data-testid="stTabs"] {{
    contain: none !important;
}}

/* Tabs Navigation: Fixed inside the top header bar next to three dots option */
div[data-testid="stTabs"] {{
    margin-top: 0 !important;
    position: static !important;
}}

div[data-testid="stTabs"] > div:first-child,
div[data-baseweb="tab-list"] {{
    position: fixed !important;
    top: 0 !important;
    left: calc(max(220px, 100vw * 1.5 / 9) + 1.5rem) !important;
    right: 80px !important;
    height: 2.875rem !important;
    z-index: 999999 !important;
    background-color: transparent !important;
    display: flex !important;
    align-items: center !important;
    border-bottom: none !important;
    gap: 16px !important;
    padding: 0 !important;
    margin: 0 !important;
}}

section[data-testid="stSidebar"][aria-expanded="false"] ~ .main div[data-baseweb="tab-list"],
section[data-testid="stSidebar"][aria-expanded="false"] ~ .main div[data-testid="stTabs"] > div:first-child {{
    left: 4rem !important;
}}

/* Remove default rogue white/gray tab border */
div[data-baseweb="tab-border"] {{
    display: none !important;
}}

/* Active tab highlight line sits properly UNDER the tab text */
@keyframes radiantTabLine {{
    0% {{ background-color: #F08080; }}
    50% {{ background-color: #ADD8E6; }}
    100% {{ background-color: #F08080; }}
}}

div[data-baseweb="tab-highlight"] {{
    animation: radiantTabLine 4s ease-in-out infinite !important;
    top: auto !important;
    bottom: 0px !important;
    height: 3px !important;
    border-radius: 2px !important;
}}

/* Tabs Styling: Crisp text, 100% opacity, zero blur or washed-out look */
button[data-baseweb="tab"] {{
    color: var(--tab-text) !important;
    opacity: 1 !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    background: transparent !important;
    border: none !important;
    padding: 6px 14px !important;
    height: 2.875rem !important;
    line-height: 2.875rem !important;
    transition: all 0.2s ease-in-out !important;
}}

button[data-baseweb="tab"] div,
button[data-baseweb="tab"] p,
button[data-baseweb="tab"] span {{
    color: var(--tab-text) !important;
    opacity: 1 !important;
}}

button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--tab-text-active) !important;
    font-weight: 700 !important;
}}

button[data-baseweb="tab"][aria-selected="true"] div,
button[data-baseweb="tab"][aria-selected="true"] p,
button[data-baseweb="tab"][aria-selected="true"] span {{
    color: var(--tab-text-active) !important;
    opacity: 1 !important;
}}

button[data-baseweb="tab"]:hover div,
button[data-baseweb="tab"]:hover p,
button[data-baseweb="tab"]:hover span {{
    color: var(--tab-text-active) !important;
}}

/* Chat Messages: Pure black/white contrast matching active theme */
[data-testid="stChatMessage"] {{
    background-color: var(--chat-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    color: var(--chat-text) !important;
}}

[data-testid="stChatMessageContent"] {{
    color: var(--chat-text) !important;
}}

[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] span,
[data-testid="stChatMessageContent"] div {{
    color: var(--chat-text) !important;
    opacity: 1 !important;
    font-size: 0.96rem !important;
    line-height: 1.6 !important;
}}

[data-testid="stChatMessageContent"] strong,
[data-testid="stChatMessageContent"] b {{
    color: var(--text-color) !important;
}}

/* Subtitles in Header Containers */
.container-subtitle {{
    color: var(--subtext-color) !important;
    font-size: 0.95rem !important;
    line-height: 1.55 !important;
    opacity: 1 !important;
    margin-bottom: 0 !important;
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

.animated-main-container h2 {{
    color: var(--text-color) !important;
}}

.metric-card {{
    background-color: var(--card-bg);
    border-radius: 10px;
    padding: 16px;
    border: 1px solid var(--border-color);
    text-align: center;
}}

.stApp, [data-testid="stAppViewContainer"], .main {{
    background-color: var(--bg-color) !important;
    color: var(--text-color);
}}

.stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
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
            <p class="container-subtitle">
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
            service = ForecastService(repo_dir, allow_embedding_download=True, allow_base_download=True)
            return service, None
        except Exception as e:
            return None, str(e)

    service, service_err = load_forecast_service(REPO)

    if service is None:
        st.error(
            f"⚠️ **Forecast Service could not be initialized:** `{service_err}`\n\n"
            "- Ensure you are connected to a **GPU runtime** in Google Colab (`Runtime > Change runtime type > T4 GPU`).\n"
            "- Ensure the required packages are installed (`pip install -r requirements.txt`)."
        )

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
                "choices": None,
                "rag": None
            }
        ]
    if "chat_context" not in st.session_state:
        st.session_state.chat_context = None

    # Container for all chat messages: ensures all messages appear BEFORE the typing text bar
    messages_container = st.container()

    # Display historical chat messages inside the container
    with messages_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if isinstance(msg.get("forecasts"), pd.DataFrame) and not msg["forecasts"].empty:
                    st.dataframe(msg["forecasts"], use_container_width=True)
                if msg.get("choices"):
                    st.caption("Available matching series to specify:")
                    st.dataframe(pd.DataFrame(msg["choices"]), use_container_width=True)
                if (msg.get("rag") or {}).get("passages"):
                    passages = msg["rag"]["passages"]
                    with st.expander(f"View {len(passages)} Quoted Official Sources"):
                        for idx, p in enumerate(passages, 1):
                            st.markdown(f"**Source {idx}:** [{p.get('title', 'Bulletin')}]({p.get('url', '#')}) *({p.get('published', 'N/A')})*")
                            st.caption(f"> \"{p.get('text', '')[:320]}...\"")

    # Chat Input Box (Fixed to stay docked at the bottom of the viewport)
    if prompt := st.chat_input("Ask a forecast question (e.g., 'What is the outlook for nurses in Uusimaa?')..."):
        # Record user message
        st.session_state.messages.append({"role": "user", "content": prompt, "forecasts": None, "choices": None, "rag": None})

        # Display user message and generate assistant response inside messages_container (BEFORE chat input)
        with messages_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                if service is None:
                    err_msg = (
                        f"**Forecast Service could not be initialized:** `{service_err}`\n\n"
                        "Please ensure you are connected to a **GPU runtime** in Colab and that the dependencies "
                        "from `notebooks/12_forecast_chat_with_rag.ipynb` are installed."
                    )
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg, "forecasts": None, "choices": None, "rag": None})
                else:
                    with st.spinner("Routing query, forecasting with Qwen3-4B, and retrieving bulletin context..."):
                        try:
                            result = service.answer_question(prompt, context=st.session_state.chat_context)
                            if result is None:
                                raise ValueError("Service returned an empty result.")
                            st.session_state.chat_context = result.get("context")
                            answer_text = result.get("answer", "No response generated.")
                            st.markdown(answer_text)

                            forecasts_df = None
                            if result.get("status") == "answered" and result.get("forecasts"):
                                raw_df = pd.DataFrame(result["forecasts"])
                                show_cols = [c for c in ["origin_quarter", "target_quarter", "horizon_q", "last_value", "y_pred"] if c in raw_df.columns]
                                forecasts_df = raw_df[show_cols].round(2)
                                st.dataframe(forecasts_df, use_container_width=True)

                            choices = (result.get("request") or {}).get("choices")
                            if choices:
                                st.caption("Available matching series to specify:")
                                st.dataframe(pd.DataFrame(choices), use_container_width=True)

                            rag_info = result.get("rag") or {}
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
                                "choices": choices,
                                "rag": rag_info
                            })
                        except Exception as ex:
                            fail_msg = f"**An error occurred during inference:** `{ex}`"
                            st.error(fail_msg)
                            st.session_state.messages.append({"role": "assistant", "content": fail_msg, "forecasts": None, "choices": None, "rag": None})

        # Trigger clean rerun so message history persists neatly in messages_container above input bar
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()

# Tab 2: Figures & Analytics Dashboard
with tab_analytics:
    st.markdown(
        """
        <div class="animated-main-container">
            <h2 style="margin-top:0;">Labour Market Figures & Model Analytics</h2>
            <p class="container-subtitle">
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
