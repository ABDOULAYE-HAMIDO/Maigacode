import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import torch

from model.chat import SuperCodeurChat

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MaigaCode",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: #0e1117; }
    .main .block-container { max-width: 900px; padding: 2rem 1rem; }
    .stChatMessage { border: 1px solid #2a2a2a; border-radius: 12px; margin: 0.5rem 0; }
    [data-testid="stChatMessageContent"] p { font-size: 0.95rem; line-height: 1.6; }
    code { background: #1e1e1e !important; border-radius: 6px; padding: 2px 6px; }
    pre {
        background: #1a1a2e !important;
        border: 1px solid #2a2a3e;
        border-radius: 10px;
        padding: 1rem !important;
        overflow-x: auto;
    }
    .stSidebar { background: #0a0a0f; border-right: 1px solid #1a1a2a; }
    .stButton>button {
        background: #1a1a2e;
        border: 1px solid #3a3a5e;
        border-radius: 8px;
        color: #c0c0ff;
        font-size: 0.85rem;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background: #2a2a4e;
        border-color: #6a6aff;
        color: #ffffff;
    }
    .stSlider>div>div>div { background: #4a4aff !important; }
    .status-badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 12px; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600;
    }
    .badge-ready { background: #1a3a2a; color: #4aff8a; border: 1px solid #2a5a3a; }
    .badge-waiting { background: #3a2a1a; color: #ffaa4a; border: 1px solid #5a3a2a; }
    .msg-meta {
        font-size: 0.7rem; color: #666; margin-top: 0.3rem;
        display: flex; gap: 8px; align-items: center;
    }
    .sidebar-title {
        font-size: 1.2rem; font-weight: 700;
        background: linear-gradient(135deg, #4a4aff, #aa4aff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sidebar-sub {
        font-size: 0.75rem; color: #666; margin-bottom: 1.5rem;
    }
    hr { border-color: #1a1a2a; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────
if "chat" not in st.session_state:
    st.session_state.chat = SuperCodeurChat()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_ready" not in st.session_state:
    st.session_state.model_ready = False
if "config_name" not in st.session_state:
    st.session_state.config_name = "100M"

chat = st.session_state.chat

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚡ MaigaCode</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Assistant code IA · local</div>',
                unsafe_allow_html=True)

    st.markdown("### 🎛️ Configuration")

    config_name = st.selectbox(
        "Modèle",
        ["nano", "10M", "50M", "100M", "300M", "350M"],
        index=3,
        help="Taille du modèle. 100M+ recommandé.",
    )
    st.session_state.config_name = config_name

    with st.expander("⚙️ Paramètres génération", expanded=False):
        temperature = st.slider("Température", 0.0, 1.5, 0.6, 0.05,
                                help="0 = déterministe, 1.5 = créatif")
        top_k = st.slider("Top-k", 1, 100, 40, 1)
        top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.05)
        repetition_penalty = st.slider("Répétition", 1.0, 2.0, 1.15, 0.05)
        max_new_tokens = st.slider("Max tokens", 32, 512, 200, 16)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Charger", use_container_width=True):
            with st.spinner(f"Chargement {config_name}..."):
                try:
                    chat.load(config_name=config_name)
                    st.session_state.model_ready = True
                    st.session_state.messages = []
                    chat.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur: {e}")
    with col2:
        if st.button("🧹 Effacer", use_container_width=True):
            st.session_state.messages = []
            chat.clear()
            st.rerun()

    status_class = "badge-ready" if st.session_state.model_ready else "badge-waiting"
    status_text = "✅ Prêt" if st.session_state.model_ready else "⏳ En attente"
    st.markdown(
        f'<div class="status-badge {status_class}">{status_text}</div>',
        unsafe_allow_html=True,
    )

    if chat.is_loaded:
        st.markdown(f"<small>"
                    f"⚙️ {chat.config.total_params_estimated/1e6:.0f}M · "
                    f"{'⚡ GPU' if chat.device=='cuda' else '💻 CPU'}"
                    f"</small>", unsafe_allow_html=True)

# ── Main chat area ───────────────────────────────────────────────────────
st.title("💬 Assistant Code IA")

# Welcome / status
if not st.session_state.model_ready:
    st.info(
        "👋 **Bienvenue sur MaigaCode**\n\n"
        "1. Configure le modèle dans la barre latérale\n"
        "2. Clique sur **📥 Charger**\n"
        "3. Envoie ton code ou ta question\n\n"
        "*Le modèle s'entraîne actuellement sur Kaggle.* "
        "Tu peux charger un checkpoint intermédiaire si disponible."
    )
else:
    st.caption(
        f"Modèle **{st.session_state.config_name}** actif · "
        f"Conversation: {len(chat.conversation.turns)} échanges"
    )

# Afficher l'historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "meta" in msg:
            st.markdown(f'<div class="msg-meta">{msg["meta"]}</div>',
                        unsafe_allow_html=True)

# Input
if prompt := st.chat_input("Écris ton code ou ta question ici...", disabled=not st.session_state.model_ready):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🤔 *Réflexion...*")
        try:
            response = chat.respond(
                query=prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
            )
            tokens = len(chat.tokenizer.encode(prompt + response))
            meta = f"⚡ {tokens} tokens · 🌡️ {temperature} · 📏 {top_k}"
            placeholder.markdown(response)
            st.markdown(f'<div class="msg-meta">{meta}</div>',
                        unsafe_allow_html=True)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "meta": meta,
            })
        except Exception as e:
            placeholder.error(f"Erreur: {e}")

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Stats")
    if chat.is_loaded:
        st.markdown(f"<small>Paramètres: {chat.config.total_params_estimated:,}</small>",
                    unsafe_allow_html=True)
        st.markdown(f"<small>Couches: {chat.config.num_layers}</small>",
                    unsafe_allow_html=True)
        st.markdown(f"<small>Cache: {len(chat.conversation.turns)} échanges</small>",
                    unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<small>MaigaCode v1.0 · 2026</small>", unsafe_allow_html=True)
