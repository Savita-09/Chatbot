"""
Nexus AI — Streamlit Frontend (fully integrated with Flask + MySQL backend)
"""

import os, re, json, random, requests, io
from datetime import datetime, date
from typing import Annotated, Any, Optional

import streamlit as st
from PIL import Image as PILImage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()


GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
BACKEND        = os.getenv("BACKEND", "http://localhost:5000")

if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

def _headers():
    token = st.session_state.get("token", "")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

def _parse(r):
    """Safely parse JSON from a requests.Response."""
    try:
        return r.json()
    except Exception:
        return {"error": f"HTTP {r.status_code}: {r.text[:300]}"}

def api_post(path, body=None, auth=True):
    try:
        h = _headers() if auth else {"Content-Type": "application/json"}
        r = requests.post(f"{BACKEND}{path}", json=body or {}, headers=h, timeout=12)
        return _parse(r), r.status_code
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot reach backend. Is Flask running on port 5000?"}, 503
    except requests.exceptions.Timeout:
        return {"error": "Backend timed out."}, 504
    except Exception as e:
        return {"error": str(e)}, 500

def api_get(path):
    try:
        r = requests.get(f"{BACKEND}{path}", headers=_headers(), timeout=12)
        return _parse(r), r.status_code
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot reach backend."}, 503
    except Exception as e:
        return {"error": str(e)}, 500

def api_put(path, body=None):
    try:
        r = requests.put(f"{BACKEND}{path}", json=body or {}, headers=_headers(), timeout=12)
        return _parse(r), r.status_code
    except Exception as e:
        return {"error": str(e)}, 500

def api_delete(path):
    try:
        r = requests.delete(f"{BACKEND}{path}", headers=_headers(), timeout=12)
        return _parse(r), r.status_code
    except Exception as e:
        return {"error": str(e)}, 500

PLANS = {
    "free": {
        "name": "Free", "price": 0, "price_label": "Free forever",
        "color": "#a0aec0", "gradient": "linear-gradient(135deg,#2d3748,#4a5568)",
        "icon": "✦", "chat_limit": 20, "search_limit": 5, "image_limit": 3,
        "models": ["llama-3.3-70b-versatile","meta-llama/llama-prompt-guard-2-22m", "openai/gpt-oss-120b", "meta-llama/llama-4-scout-17b-16e-instruct","qwen/qwen3-32b"],
        "image_models": ["flux"],
        "features": [
            "20 AI chats per day", "5 web searches per day",
            "3 image generations per day", "Basic models (8B–9B)",
            "Standard response speed", "Community support",
        ],
        "missing": [
            "Advanced 70B models", "Priority inference",
            "Unlimited searches", "API access", "Custom system prompts",
        ],
    },
    "pro": {
        "name": "Pro", "price": 9, "price_label": "$9 / month",
        "color": "#4f8cff", "gradient": "linear-gradient(135deg,#1e3a8a,#4f8cff)",
        "icon": "⚡", "chat_limit": 9999, "search_limit": 9999, "image_limit": 30,
        "models": ["llama-3.1-8b-instant", "mixtral-8x7b-32768",
                   "gemma2-9b-it", "llama3-70b-8192", "llama3-8b-8192"],
        "image_models": ["flux", "turbo", "stable-diffusion", "dall-e-3-free"],
        "features": [
            "Unlimited AI chats", "Unlimited web searches",
            "30 image generations per day", "All models incl. Llama 3.3 70B",
            "Priority inference speed", "Custom system prompts",
            "Conversation export", "Email support",
        ],
        "missing": ["API access", "Dedicated infrastructure", "Team management"],
    },
    "enterprise": {
        "name": "Enterprise", "price": 49, "price_label": "$49 / month",
        "color": "#a78bfa", "gradient": "linear-gradient(135deg,#4c1d95,#a78bfa)",
        "icon": "🏢", "chat_limit": 99999, "search_limit": 99999, "image_limit": 99999,
        "models": ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile",
                   "llama-3.1-8b-instant", "mixtral-8x7b-32768",
                   "gemma2-9b-it", "llama3-70b-8192", "llama3-8b-8192"],
        "image_models": ["flux", "turbo", "stable-diffusion", "dall-e-3-free"],
        "features": [
            "Unlimited everything", "REST API access", "All Pro features included",
            "Team management (up to 20 seats)", "Dedicated infrastructure",
            "Custom fine-tuned models", "SLA & uptime guarantee", "Priority phone support",
        ],
        "missing": [],
    },
}

CHAT_MODELS_ALL = {
    "openai/gpt-oss-120b" : "🚀 Groq: GPT-OSS 120B",
    "meta-llama/llama-prompt-guard-2-22m" : "🚀 Groq: Llama Prompt Guard 2 22M",
    "meta-llama/llama-4-scout-17b-16e-instruct": "🚀 Groq: Llama 4 Scout 17B 16E Instruct",
    "llama-3.3-70b-versatile": "⚡ Groq: Llama 3.3 70B",
    "qwen/qwen3-32b": "Qwen : Qwen2-32b",
    "llama3.1-8b-instant": "🚀 Groq: Llama 3.1 8B",
    "llama3": "🦙 Ollama: Llama 3 (Local)",
    "mistral": "🌀 Ollama: Mistral (Local)",
}

SEARCH_MODELS_ALL = {
    "llama-3.3-70b-versatile": "⚡ Llama 3.3 70B + Tavily",
}
IMAGE_MODELS_ALL = {
    "flux"            : "🎨 FLUX",
    "turbo"           : "⚡ FLUX Turbo",
    "stable-diffusion": "🖼️ Stable Diffusion",
    "dall-e-3-free"   : "✨ DALL·E Style",
}


class State(BaseModel):
    messages: Annotated[list, add_messages]
    image   : Any  = None
    sources : list = []
    mode    : str  = "chat"

@st.cache_resource(show_spinner=False)
def get_llm(model_id: str):
    # logic to decide provider
    ollama_models = ["llama3", "mistral", "gemma2", "phi3"] 
    
    if model_id in ollama_models:
        return ChatOllama(
            model=model_id,
            base_url=OLLAMA_BASE_URL,
            temperature=0.7
        )
    else:
        return ChatGroq(
            model=model_id, 
            api_key=GROQ_API_KEY,
            temperature=0.7, 
            max_tokens=4096, 
            timeout=60, 
            max_retries=2
        )

@st.cache_resource(show_spinner=False)
def get_search_tool():
    return TavilySearchResults(max_results=6) if TAVILY_API_KEY else None

def generate_image_free(prompt: str, model: str = "flux") -> Optional[PILImage.Image]:
    pm  = {"flux":"flux","turbo":"turbo","stable-diffusion":"stable-diffusion","dall-e-3-free":"flux"}.get(model,"flux")
    url = (f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}"
           f"?model={pm}&seed={random.randint(1000,99999)}&width=1024&height=1024&nologo=true")
    try:
        r = requests.get(url, timeout=60); r.raise_for_status()
        return PILImage.open(io.BytesIO(r.content))
    except Exception:
        return None

IMAGE_KW  = ["image","draw","generate image","picture","illustration","photo","paint",
             "artwork","portrait","futuristic","design a","create a visual","make an image","sketch","render"]
SEARCH_KW = ["search","find","latest","news","today","current","who is","what is","when did",
             "price of","weather","score","update","recent","2024","2025","2026","tell me about"]

def detect_mode(text: str) -> str:
    tl = text.lower()
    if any(k in tl for k in IMAGE_KW):  return "image"
    if any(k in tl for k in SEARCH_KW): return "search"
    return "chat"

def chat_node(state: State) -> State:
    # Get model from session state (UI selection)
    selected_model = st.session_state.get("chat_model", "llama-3.1-8b-instant")
    llm = get_llm(selected_model)
    return State(messages=[llm.invoke(state.messages)], mode="chat")

def search_node(state: State) -> State:
    from langchain_core.messages import SystemMessage
    tool = get_search_tool()
    query = state.messages[-1].content if hasattr(state.messages[-1], "content") else str(state.messages[-1])
    
    sources, context = [], ""
    if tool:
        try:
            results = tool.invoke(query)
            for r in (results or [])[:5]:
                sources.append({"title": r.get("title",""), "url": r.get("url","#")})
                context += f"\n- {r.get('title','')}: {r.get('content','')[:300]}"
        except Exception: pass
        
    aug = ([SystemMessage(content=f"Use these search results:\n{context}\nAnswer comprehensively.")] + list(state.messages)) if context else list(state.messages)
    
    # Use the search model selected in UI
    search_model = st.session_state.get("search_model", "llama-3.3-70b-versatile")
    return State(messages=[get_llm(search_model).invoke(aug)], sources=sources, mode="search")

def image_node(state: State) -> State:
    from langchain_core.messages import AIMessage
    prompt = state.messages[-1].content if hasattr(state.messages[-1], "content") else str(state.messages[-1])
    img = generate_image_free(prompt, st.session_state.get("image_model", "flux"))
    return State(messages=[AIMessage(content=f"🎨 Generated image for: **{prompt[:80]}**")], image=img, mode="image")

def router_fn(state: State):
    return detect_mode(state.messages[-1].content if hasattr(state.messages[-1], "content") else str(state.messages[-1]))

@st.cache_resource(show_spinner=False)
def build_graph():
    gb = StateGraph(State)
    gb.add_node("chat_node",   chat_node)
    gb.add_node("search_node", search_node)
    gb.add_node("image_node",  image_node)
    gb.add_conditional_edges(START, router_fn, {"chat":"chat_node","search":"search_node","image":"image_node"})
    gb.add_edge("chat_node", END)
    gb.add_edge("search_node", END)
    gb.add_edge("image_node", END)
    return gb.compile()


st.set_page_config(page_icon="✦", page_title="Nexus AI", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');
:root{--bg0:#080c14;--bg1:#0d1220;--bg2:#131929;--bg3:#1a2235;--border:rgba(99,120,180,0.13);--border-hi:rgba(99,120,180,0.30);--accent:#4f8cff;--accent2:#a78bfa;--green:#34d399;--amber:#fbbf24;--red:#f87171;--text-1:#eef2ff;--text-2:#a0aec0;--text-3:#4a5568;--radius:14px;--radius-sm:8px;--shadow:0 8px 32px rgba(0,0,0,.5);--font-head:'Syne',sans-serif;--font-body:'DM Sans',sans-serif}
html,body,[class*="css"],.stApp{font-family:var(--font-body)!important;background:var(--bg0)!important;color:var(--text-1)!important}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:var(--bg1)}::-webkit-scrollbar-thumb{background:var(--bg3);border-radius:99px}
section[data-testid="stSidebar"]{background:var(--bg1)!important;border-right:1px solid var(--border)!important;padding-top:0!important}
section[data-testid="stSidebar"]>div:first-child{padding-top:0}
.logo-wrap{padding:20px 18px 14px;display:flex;align-items:center;gap:10px;border-bottom:1px solid var(--border);margin-bottom:10px}
.logo-mark{font-family:var(--font-head);font-size:21px;font-weight:800;background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.logo-dot{width:8px;height:8px;border-radius:50%;background:var(--accent);box-shadow:0 0 8px var(--accent);animation:pulse 2s ease infinite;flex-shrink:0}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(1.5)}}
.plan-badge{display:inline-flex;align-items:center;gap:4px;padding:4px 10px;border-radius:99px;font-size:11px;font-weight:700;font-family:var(--font-head);text-transform:uppercase;letter-spacing:.06em}
.badge-free{background:rgba(74,85,104,.2);color:#a0aec0;border:1px solid rgba(74,85,104,.4)}
.badge-pro{background:rgba(79,140,255,.15);color:var(--accent);border:1px solid rgba(79,140,255,.3)}
.badge-enterprise{background:rgba(167,139,250,.15);color:var(--accent2);border:1px solid rgba(167,139,250,.3)}
.usage-row{margin:5px 0}
.usage-label{font-size:11px;color:var(--text-3);display:flex;justify-content:space-between;margin-bottom:3px}
.usage-track{background:var(--bg3);border-radius:99px;height:4px;overflow:hidden}
.usage-fill{height:4px;border-radius:99px;transition:width .4s ease}
.fill-ok{background:linear-gradient(90deg,var(--accent),var(--green))}
.fill-warn{background:linear-gradient(90deg,var(--amber),#f59e0b)}
.fill-full{background:linear-gradient(90deg,var(--red),#dc2626)}
.section-label{font-family:var(--font-head);font-size:10px;font-weight:700;color:var(--text-3);text-transform:uppercase;letter-spacing:.1em;margin:14px 0 5px;padding:0 2px}
.stButton>button{font-family:var(--font-body)!important;font-weight:500!important;border-radius:var(--radius-sm)!important;transition:all .18s!important}
.btn-primary>button{background:linear-gradient(135deg,var(--accent),#2563eb)!important;color:white!important;border:none!important;font-weight:600!important}
.btn-primary>button:hover{transform:translateY(-1px);box-shadow:0 4px 18px rgba(79,140,255,.4)!important}
.btn-upgrade>button{background:linear-gradient(135deg,#7c3aed,var(--accent2))!important;color:white!important;border:none!important;font-weight:600!important}
.btn-upgrade>button:hover{transform:translateY(-1px);box-shadow:0 4px 18px rgba(167,139,250,.4)!important}
.btn-danger>button{background:transparent!important;color:var(--red)!important;border:1px solid rgba(248,113,113,.25)!important}
.btn-danger>button:hover{background:rgba(248,113,113,.08)!important}
.btn-ghost>button{background:var(--bg3)!important;color:var(--text-2)!important;border:1px solid var(--border-hi)!important}
.btn-ghost>button:hover{border-color:var(--accent)!important;color:var(--accent)!important}
div[data-testid="stChatInput"] textarea{background:var(--bg3)!important;border-color:var(--border-hi)!important;border-radius:var(--radius)!important;color:var(--text-1)!important;font-family:var(--font-body)!important;font-size:15px!important}
div[data-testid="stChatInput"] textarea:focus{border-color:var(--accent)!important;box-shadow:0 0 0 2px rgba(79,140,255,.18)!important}
.stTextInput input{background:var(--bg3)!important;border-color:var(--border-hi)!important;border-radius:var(--radius-sm)!important;color:var(--text-1)!important}
.stTextInput input:focus{border-color:var(--accent)!important;box-shadow:0 0 0 2px rgba(79,140,255,.18)!important}
label{color:var(--text-2)!important;font-size:13px!important}
div[data-baseweb="select"]>div{background:var(--bg3)!important;border-color:var(--border-hi)!important;border-radius:var(--radius-sm)!important;color:var(--text-1)!important;font-size:13px!important}
.msg-wrap{display:flex;gap:10px;margin:14px 0;animation:fadeUp .28s ease}
.msg-wrap.user{flex-direction:row-reverse}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.avatar{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:14px;flex-shrink:0}
.avatar.bot{background:linear-gradient(135deg,var(--accent),var(--accent2));color:white}
.avatar.user{background:var(--bg3);color:var(--text-1);border:1px solid var(--border-hi)}
.bubble{max-width:78%;padding:13px 16px;border-radius:16px;font-size:15px;line-height:1.65}
.bubble.bot{background:var(--bg2);border:1px solid var(--border);border-radius:4px 16px 16px 16px}
.bubble.user{background:linear-gradient(135deg,#1e3a8a,#1d4ed8);color:white;border-radius:16px 4px 16px 16px}
.bubble.bot pre{background:var(--bg0);border-radius:8px;padding:12px;overflow-x:auto;font-size:13px;border:1px solid var(--border);margin:8px 0}
.bubble.bot code{background:rgba(79,140,255,.12);padding:2px 6px;border-radius:4px;font-size:13px;color:#93c5fd}
.bubble.bot h1,.bubble.bot h2,.bubble.bot h3{font-family:var(--font-head);margin:12px 0 6px}
.bubble.bot ul,.bubble.bot ol{padding-left:20px;margin:6px 0}
.bubble.bot li{margin-bottom:4px}
.bubble.bot strong{color:var(--accent);font-weight:600}
.bubble.bot a{color:var(--accent);text-decoration:none}
.bubble.bot blockquote{border-left:3px solid var(--accent2);padding-left:12px;color:var(--text-2);margin:8px 0;font-style:italic}
.mode-badge{display:inline-flex;align-items:center;gap:4px;font-size:10px;font-weight:700;font-family:var(--font-head);padding:3px 8px;border-radius:99px;margin-bottom:6px;text-transform:uppercase;letter-spacing:.06em}
.badge-chat{background:rgba(79,140,255,.15);color:var(--accent)}
.badge-search{background:rgba(52,211,153,.15);color:var(--green)}
.badge-image{background:rgba(167,139,250,.15);color:var(--accent2)}
.sources-wrap{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.src-pill{background:var(--bg3);border:1px solid var(--border-hi);border-radius:99px;padding:3px 10px;font-size:11px;color:var(--text-2);text-decoration:none;white-space:nowrap;max-width:190px;overflow:hidden;text-overflow:ellipsis;display:inline-block}
.src-pill:hover{border-color:var(--accent);color:var(--accent)}
.welcome{text-align:center;padding:50px 20px 28px;max-width:700px;margin:auto}
.welcome-title{font-family:var(--font-head);font-size:30px;font-weight:800;background:linear-gradient(135deg,var(--text-1),var(--accent));-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:10px}
.welcome-sub{font-size:15px;color:var(--text-2);line-height:1.7;max-width:500px;margin:0 auto 34px}
.cap-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;max-width:640px;margin:auto}
.cap-card{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);padding:18px 14px;text-align:center;transition:all .2s}
.cap-card:hover{border-color:var(--border-hi);transform:translateY(-2px);box-shadow:var(--shadow)}
.cap-icon{font-size:26px;margin-bottom:7px}
.cap-title{font-family:var(--font-head);font-size:13px;font-weight:700;margin-bottom:4px}
.limit-warn{background:rgba(251,191,36,.07);border:1px solid rgba(251,191,36,.28);border-radius:var(--radius-sm);padding:11px 15px;font-size:13px;color:var(--amber);margin:8px 0}
.limit-block{background:rgba(248,113,113,.07);border:1px solid rgba(248,113,113,.28);border-radius:var(--radius-sm);padding:13px 16px;font-size:14px;color:var(--red);margin:10px 0;text-align:center}
.price-card{border-radius:20px;padding:28px 22px;border:1px solid var(--border);position:relative;transition:all .22s;background:var(--bg2);min-height:490px}
.price-card:hover{transform:translateY(-4px);box-shadow:var(--shadow)}
.price-card.popular{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent),0 16px 40px rgba(79,140,255,.18)}
.popular-badge{position:absolute;top:-13px;left:50%;transform:translateX(-50%);background:linear-gradient(135deg,var(--accent),#2563eb);color:white;font-size:11px;font-weight:700;font-family:var(--font-head);padding:4px 14px;border-radius:99px;white-space:nowrap;text-transform:uppercase;letter-spacing:.05em}
.plan-icon{font-size:30px;margin-bottom:10px}
.plan-name{font-family:var(--font-head);font-size:19px;font-weight:800;margin-bottom:4px}
.plan-price{font-family:var(--font-head);font-size:30px;font-weight:800;margin:10px 0 3px}
.plan-period{font-size:13px;color:var(--text-2);margin-bottom:18px}
.feature-list{list-style:none;padding:0;margin:0 0 22px}
.feature-list li{font-size:13px;color:var(--text-2);padding:5px 0;display:flex;align-items:flex-start;gap:8px}
.check{color:var(--green);font-size:14px;flex-shrink:0;margin-top:1px}
.cross-icon{color:var(--text-3);font-size:12px;flex-shrink:0;margin-top:2px}
.missing-item{opacity:.4}
.stTabs [data-baseweb="tab-list"]{background:var(--bg2);border-radius:var(--radius-sm);padding:4px;border:1px solid var(--border)}
.stTabs [data-baseweb="tab"]{background:transparent;color:var(--text-2);border-radius:8px;font-family:var(--font-body);font-size:14px}
.stTabs [aria-selected="true"]{background:var(--bg3)!important;color:var(--text-1)!important}
.auth-logo{font-family:var(--font-head);font-size:26px;font-weight:800;text-align:center;background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:6px}
.auth-sub{text-align:center;font-size:14px;color:var(--text-2);margin-bottom:22px}
.conv-btn button{text-align:left!important;background:transparent!important;border:none!important;color:var(--text-2)!important;font-size:13px!important;padding:8px 12px!important;border-radius:var(--radius-sm)!important}
.conv-btn button:hover{background:var(--bg3)!important;color:var(--text-1)!important}
.conv-btn.active button{background:rgba(79,140,255,.15)!important;color:var(--accent)!important}
</style>
""", unsafe_allow_html=True)


def init():
    today = str(date.today())
    defs  = {
        "page": "login", "username": None, "token": None, "user_id": None,
        "avatar_color": "#4f8cff", "plan": "free",
        "active_session": None, "local_sessions": {},
        "error": "", "success": "",
        "chat_model": "llama-3.1-8b-instant",
        "search_model": "llama-3.3-70b-versatile",
        "image_model": "flux",
        "show_upgrade": False, "upgrade_reason": "",
        "usage_date": today, "usage_chat": 0, "usage_search": 0, "usage_image": 0,
        "billing_email": "", "billing_card": "", "sub_start": None, "pending_plan": "pro",
    }
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v
    # reset local counters on new day
    if st.session_state.usage_date != today:
        st.session_state.usage_date  = today
        st.session_state.usage_chat  = 0
        st.session_state.usage_search = 0
        st.session_state.usage_image  = 0

init()

# ─────────────────────────────────────────────────────────────
# PLAN / USAGE HELPERS
# ─────────────────────────────────────────────────────────────
def my_plan():
    return PLANS[st.session_state.plan]

def sync_usage():
    """Pull today's usage & plan from backend."""
    if not st.session_state.get("token"):
        return
    data, code = api_get("/user/usage")
    if code == 200:
        u = data.get("usage", {})
        st.session_state.usage_chat   = u.get("chat_count", 0)
        st.session_state.usage_search = u.get("search_count", 0)
        st.session_state.usage_image  = u.get("image_count", 0)
        st.session_state.plan         = data.get("plan", "free")

def check_limit(mode):
    plan  = my_plan()
    used  = st.session_state.get(f"usage_{mode}", 0)
    limit = plan[f"{mode}_limit"]
    if used >= limit:
        return False, f"You've used all **{limit} {mode}s** on the **{plan['name']} plan** today."
    return True, ""

def bump_usage_backend(mode):
    """Increment counter on backend; fall back to local if guest."""
    if st.session_state.get("token"):
        data, code = api_put("/user/usage", {"mode": mode})
        if code == 429:
            return False
        if code == 200:
            st.session_state[f"usage_{mode}"] = data.get("used", st.session_state.get(f"usage_{mode}", 0) + 1)
            return True
    # guest — local only
    st.session_state[f"usage_{mode}"] = st.session_state.get(f"usage_{mode}", 0) + 1
    return True

def plan_allows_model(model_id, kind="chat"):
    key = "image_models" if kind == "image" else "models"
    return model_id in my_plan()[key]


def _esc(t):
    return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def _inline(t):
    t = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
    t = re.sub(r'\*(.+?)\*',     r'<em>\1</em>',         t)
    t = re.sub(r'`([^`]+)`',     lambda m: f'<code>{_esc(m.group(1))}</code>', t)
    t = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', t)
    return t

def md_to_html(text):
    text = re.sub(r'```(\w+)?\n?([\s\S]*?)```',
                  lambda m: f'<pre><code>{_esc(m.group(2).strip())}</code></pre>', text)
    lines, out = text.split('\n'), []
    in_ul = in_ol = False
    for line in lines:
        if re.match(r'^# ',   line): out.append(f'<h1>{_inline(line[2:])}</h1>');  continue
        if re.match(r'^## ',  line): out.append(f'<h2>{_inline(line[3:])}</h2>');  continue
        if re.match(r'^### ', line): out.append(f'<h3>{_inline(line[4:])}</h3>');  continue
        if re.match(r'^[-*] ',line):
            if not in_ul: out.append('<ul>'); in_ul = True
            out.append(f'<li>{_inline(line[2:])}</li>'); continue
        if in_ul: out.append('</ul>'); in_ul = False
        if re.match(r'^\d+\. ', line):
            if not in_ol: out.append('<ol>'); in_ol = True
            out.append(f'<li>{_inline(re.sub(r"^\d+\. ","",line))}</li>'); continue
        if in_ol: out.append('</ol>'); in_ol = False
        if re.match(r'^> ', line): out.append(f'<blockquote>{_inline(line[2:])}</blockquote>'); continue
        if line.strip(): out.append(f'<p>{_inline(line)}</p>')
    if in_ul: out.append('</ul>')
    if in_ol: out.append('</ol>')
    return ''.join(out)


def load_sessions():
    """Load all sessions from backend into local state."""
    if not st.session_state.get("token"):
        return
    data, code = api_get("/sessions")
    if code == 200 and isinstance(data, list):
        for s in data:
            sid = str(s["id"])
            if sid not in st.session_state.local_sessions:
                st.session_state.local_sessions[sid] = {
                    "title": s["title"], "messages": None  # lazy load
                }

def new_session(title="New Chat"):
    if st.session_state.get("token"):
        data, code = api_post("/sessions", {"title": title})
        if code == 201:
            sid = str(data["id"])
            st.session_state.local_sessions[sid] = {"title": title, "messages": []}
            st.session_state.active_session = sid
            return sid
    # guest fallback
    sid = str(datetime.now().timestamp())
    st.session_state.local_sessions[sid] = {"title": title, "messages": []}
    st.session_state.active_session = sid
    return sid

def current_msgs():
    sid = st.session_state.active_session
    if not sid:
        return []
    sess = st.session_state.local_sessions.get(sid, {})
    if sess.get("messages") is None:
        # lazy fetch from backend
        data, code = api_get(f"/sessions/{sid}")
        if code == 200:
            msgs = data.get("messages", [])
            # normalise backend messages to local format
            for m in msgs:
                m.setdefault("image", None)
                m.setdefault("ts", "")
                try:
                    m["sources"] = json.loads(m["sources"]) if isinstance(m["sources"], str) else (m["sources"] or [])
                except Exception:
                    m["sources"] = []
            st.session_state.local_sessions[sid]["messages"] = msgs
        else:
            st.session_state.local_sessions[sid]["messages"] = []
    return st.session_state.local_sessions[sid].get("messages", [])

def add_msg(role, content, mode="chat", image=None, sources=None):
    sid = st.session_state.active_session
    if not sid:
        return
    msg = {
        "role": role, "content": content, "mode": mode,
        "image": image, "sources": sources or [],
        "ts": datetime.now().strftime("%H:%M"),
    }
    st.session_state.local_sessions[sid].setdefault("messages", []).append(msg)
    # persist to backend (skip image blobs)
    if st.session_state.get("token") and image is None:
        api_post(f"/sessions/{sid}/messages",
                 {"role": role, "content": content, "mode": mode, "sources": sources or []})

def rename_first(msg):
    sid = st.session_state.active_session
    if not sid:
        return
    sess = st.session_state.local_sessions.get(sid, {})
    if sess.get("title") == "New Chat":
        new_title = msg[:32] + ("…" if len(msg) > 32 else "")
        st.session_state.local_sessions[sid]["title"] = new_title
        if st.session_state.get("token"):
            api_put(f"/sessions/{sid}/rename", {"title": new_title})

def delete_session(sid):
    if st.session_state.get("token"):
        api_delete(f"/sessions/{sid}")
    st.session_state.local_sessions.pop(sid, None)
    if st.session_state.active_session == sid:
        st.session_state.active_session = None


def process_message(user_text):
    from langchain_core.messages import HumanMessage
    mode = detect_mode(user_text)

    allowed, reason = check_limit(mode)
    if not allowed:
        st.session_state.show_upgrade   = True
        st.session_state.upgrade_reason = reason
        add_msg("assistant",
                f"⚠️ **Daily limit reached.**\n\n{reason}\n\nUpgrade to **Pro** for unlimited access.",
                mode=mode)
        return

    model_key = "image_model" if mode == "image" else "chat_model"
    if not plan_allows_model(st.session_state.get(model_key, "llama-3.1-8b-instant"), mode):
        st.session_state[model_key] = "flux" if mode == "image" else "llama-3.1-8b-instant"
        add_msg("assistant",
                "⚠️ That model requires **Pro plan**. Switched to a free model.",
                mode=mode)
        return

    if not GROQ_API_KEY:
        add_msg("assistant",
                "⚠️ **GROQ_API_KEY not set.** Add it to your `.env` file.",
                mode="chat")
        return

    ok = bump_usage_backend(mode)
    if not ok:
        add_msg("assistant",
                "⚠️ Daily limit reached on the server. Upgrade to Pro for unlimited access.",
                mode=mode)
        return

    try:
        graph   = build_graph()
        history = [HumanMessage(content=m["content"])
                   for m in current_msgs()[-8:] if m["role"] == "user"]
        history.append(HumanMessage(content=user_text))
        result  = graph.invoke(State(messages=history))

        msgs     = result.get("messages",[]) if isinstance(result,dict) else getattr(result,"messages",[])
        img_out  = result.get("image")       if isinstance(result,dict) else getattr(result,"image",None)
        sources  = result.get("sources",[])  if isinstance(result,dict) else getattr(result,"sources",[])
        res_mode = result.get("mode","chat") if isinstance(result,dict) else getattr(result,"mode","chat")
        reply    = msgs[-1].content if msgs and hasattr(msgs[-1],"content") else ""

        add_msg("assistant", reply, mode=res_mode, image=img_out, sources=sources)
    except Exception as e:
        add_msg("assistant", f"⚠️ **Error:** `{e}`", mode="chat")


@st.dialog("✦ Choose Your Plan", width="large")
def upgrade_modal():
    reason = st.session_state.get("upgrade_reason", "")
    if reason:
        st.markdown(f'<div class="limit-warn">⚠️ {reason}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;padding:8px 0 20px">
      <div style="font-family:var(--font-head);font-size:22px;font-weight:800;
           background:linear-gradient(135deg,var(--text-1),var(--accent));
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:6px">
        Upgrade Nexus AI
      </div>
      <div style="font-size:14px;color:var(--text-2)">Unlock unlimited chats, searches &amp; premium models</div>
    </div>""", unsafe_allow_html=True)

    cols = st.columns(3)
    for i, pk in enumerate(["free","pro","enterprise"]):
        p    = PLANS[pk]
        pop  = "popular" if pk == "pro" else ""
        pop_h = '<div class="popular-badge">⚡ Most Popular</div>' if pk == "pro" else ""
        feats = "".join(f'<li><span class="check">✓</span>{f}</li>' for f in p["features"])
        miss  = "".join(f'<li class="missing-item"><span class="cross-icon">✕</span>{f}</li>' for f in p.get("missing",[]))
        price_num = p["price_label"].split("/")[0].strip()
        with cols[i]:
            st.markdown(f"""
            <div class="price-card {pop}">
              {pop_h}
              <div class="plan-icon">{p['icon']}</div>
              <div class="plan-name" style="color:{p['color']}">{p['name']}</div>
              <div class="plan-price">{price_num}</div>
              <div class="plan-period">{'per month · cancel anytime' if p['price']>0 else 'No credit card needed'}</div>
              <ul class="feature-list">{feats}{miss}</ul>
            </div>""", unsafe_allow_html=True)

            cur_plan = st.session_state.plan == pk
            if cur_plan:
                st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
                st.button("✓ Current Plan", key=f"cur_{pk}", use_container_width=True, disabled=True)
                st.markdown('</div>', unsafe_allow_html=True)
            elif pk == "free":
                st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
                if st.button("Downgrade to Free", key=f"sel_{pk}", use_container_width=True):
                    if st.session_state.get("token"):
                        api_post("/billing/cancel")
                    st.session_state.plan = "free"
                    st.session_state.chat_model  = "llama-3.1-8b-instant"
                    st.session_state.image_model = "flux"
                    st.session_state.show_upgrade = False
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                cls = "btn-upgrade" if pk == "pro" else "btn-primary"
                st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
                if st.button(f"Get {p['name']} →", key=f"sel_{pk}", use_container_width=True):
                    st.session_state.pending_plan  = pk
                    st.session_state.show_upgrade  = False
                    st.session_state.page = "billing"
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)
    st.caption("🔒 Secure payment via Stripe · Cancel anytime · No hidden fees")
    st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
    if st.button("✕ Close", key="close_modal", use_container_width=True):
        st.session_state.show_upgrade = False
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def render_billing():
    pending = st.session_state.get("pending_plan", "pro")
    plan    = PLANS[pending]
    _, c2, _ = st.columns([1, 2, 1])
    with c2:
        st.markdown(f"""
        <div style="text-align:center;padding:36px 0 22px">
          <div style="font-family:var(--font-head);font-size:26px;font-weight:800;
               background:linear-gradient(135deg,var(--accent),var(--accent2));
               -webkit-background-clip:text;-webkit-text-fill-color:transparent">✦ Nexus AI</div>
          <div style="font-size:15px;color:var(--text-2);margin-top:4px">
            Subscribe to <strong style="color:{plan['color']}">{plan['name']} Plan</strong>
          </div>
        </div>""", unsafe_allow_html=True)

        feats_html = "".join(
            f"<li style='font-size:13px;color:var(--text-2);padding:3px 0;list-style:none;display:flex;gap:8px'>"
            f"<span style='color:var(--green)'>✓</span>{f}</li>"
            for f in plan["features"][:4]
        )
        st.markdown(f"""
        <div style="background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);padding:22px 26px;margin-bottom:20px">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px">
            <div>
              <div style="font-family:var(--font-head);font-weight:800;font-size:16px">{plan['icon']} {plan['name']} Plan</div>
              <div style="font-size:12px;color:var(--text-3);margin-top:2px">Billed monthly · Cancel anytime</div>
            </div>
            <div style="font-family:var(--font-head);font-size:22px;font-weight:800;color:var(--accent)">{plan['price_label']}</div>
          </div>
          <ul style="padding:0;margin:0 0 14px">{feats_html}</ul>
          <div style="border-top:1px solid var(--border);padding-top:12px;display:flex;justify-content:space-between;font-weight:700;font-size:15px">
            <span>Total today</span><span style="color:var(--accent)">{plan['price_label']}</span>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div style="font-size:11px;font-weight:700;color:var(--text-3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px">💳 Payment Details</div>', unsafe_allow_html=True)

        with st.form("billing_form"):
            email  = st.text_input("Email address", placeholder="you@example.com",
                                   value=st.session_state.billing_email)
            card   = st.text_input("Card number", placeholder="1234 5678 9012 3456")
            c1, c2_ = st.columns(2)
            with c1:  expiry = st.text_input("Expiry", placeholder="MM / YY")
            with c2_: cvc    = st.text_input("CVC",    placeholder="123")
            name = st.text_input("Name on card", placeholder="Jane Doe")
            st.markdown('<div style="font-size:11px;color:var(--text-3);margin:6px 0 10px">🔒 Secured by Stripe · 256-bit SSL encryption</div>', unsafe_allow_html=True)
            submitted = st.form_submit_button(
                f"Subscribe to {plan['name']} — {plan['price_label']} →",
                use_container_width=True, type="primary"
            )

        if submitted:
            if email and card and expiry and cvc and name:
                if st.session_state.get("token"):
                    data, code = api_post("/billing/subscribe", {
                        "plan": pending, "email": email,
                        "card": card, "expiry": expiry,
                        "cvc": cvc, "name": name,
                    })
                    if code == 200:
                        st.session_state.plan         = pending
                        st.session_state.billing_email = email
                        st.session_state.billing_card  = data.get("card_last4", card[-4:])
                        st.session_state.sub_start     = data.get("sub_start", str(date.today()))
                        st.session_state.chat_model    = "llama-3.3-70b-versatile"
                        st.session_state.image_model   = "flux"
                        st.session_state.page          = "chat"
                        st.session_state.pending_plan  = None
                        st.session_state.success       = f"🎉 Welcome to {plan['name']}! All features unlocked."
                        st.rerun()
                    else:
                        st.error(data.get("error", "Subscription failed. Try again."))
                else:
                    # guest — local upgrade
                    st.session_state.plan         = pending
                    st.session_state.billing_email = email
                    st.session_state.billing_card  = card.replace(" ","")[-4:]
                    st.session_state.sub_start     = str(date.today())
                    st.session_state.chat_model    = "llama-3.3-70b-versatile"
                    st.session_state.page          = "chat"
                    st.session_state.success       = f"🎉 Welcome to {plan['name']}!"
                    st.rerun()
            else:
                st.error("Please fill in all fields to continue.")

        st.markdown('<div class="btn-ghost" style="margin-top:8px">', unsafe_allow_html=True)
        if st.button("← Back to Chat", use_container_width=True):
            st.session_state.page = "chat"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;font-size:12px;color:var(--text-3);margin-top:14px">By subscribing you agree to our Terms · No refunds on partial months</div>', unsafe_allow_html=True)


def render_account():
    render_sidebar()
    plan = my_plan()
    _, c2, _ = st.columns([1, 2.5, 1])
    with c2:
        if st.session_state.success:
            st.success(st.session_state.success); st.session_state.success = ""
        st.markdown('<div style="padding:28px 0 18px"><div style="font-family:var(--font-head);font-size:24px;font-weight:800">Account Settings</div><div style="color:var(--text-2);font-size:14px;margin-top:3px">Manage your plan, profile and usage</div></div>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["👤  Profile", "💳  Subscription", "📊  Usage"])

        with tab1:
            st.markdown("#### Profile Information")
            uname = st.text_input("Username", value=st.session_state.username or "")
            email = st.text_input("Email",    value=st.session_state.billing_email or "")
            st.markdown("**Avatar color**")
            colors = ["#4f8cff","#34d399","#a78bfa","#f87171","#fbbf24","#f97316"]
            cc = st.columns(6)
            for i, c in enumerate(colors):
                with cc[i]:
                    if st.button(" ", key=f"col_{i}"):
                        st.session_state.avatar_color = c; st.rerun()
                    sel = "3px solid white" if st.session_state.avatar_color == c else "2px solid transparent"
                    st.markdown(f'<div style="width:28px;height:28px;border-radius:50%;background:{c};margin:auto;border:{sel};margin-top:4px"></div>', unsafe_allow_html=True)
            st.markdown('<div class="btn-primary" style="margin-top:12px">', unsafe_allow_html=True)
            if st.button("Save Profile", use_container_width=True):
                if st.session_state.get("token"):
                    data, code = api_put("/user/profile", {"username": uname, "email": email,
                                                            "avatar_color": st.session_state.avatar_color})
                    if code == 200:
                        st.session_state.username = uname
                        st.session_state.billing_email = email
                        st.success("✓ Profile saved!")
                    else:
                        st.error(data.get("error", "Save failed"))
                else:
                    if uname: st.session_state.username = uname
                    if email: st.session_state.billing_email = email
                    st.success("✓ Profile saved!")
            st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            st.markdown(f"""
            <div style="background:{plan['gradient']};border-radius:var(--radius);padding:22px 24px;margin-bottom:18px;position:relative;overflow:hidden">
              <div style="font-family:var(--font-head);font-size:18px;font-weight:800;color:white;margin-bottom:3px">{plan['icon']} {plan['name']} Plan</div>
              <div style="font-size:13px;color:rgba(255,255,255,.7)">{plan['price_label']}</div>
              {'<div style="font-size:12px;color:rgba(255,255,255,.55);margin-top:5px">Active since: '+str(st.session_state.sub_start)+'</div>' if st.session_state.plan!="free" and st.session_state.sub_start else ""}
              <div style="position:absolute;right:-16px;top:-16px;font-size:90px;opacity:.07">{plan['icon']}</div>
            </div>""", unsafe_allow_html=True)

            if st.session_state.plan == "free":
                st.markdown("**Upgrade to Pro** for unlimited access to all features:")
                for f in PLANS["pro"]["features"][:5]: st.markdown(f"✓ {f}")
                st.markdown("")
                st.markdown('<div class="btn-upgrade">', unsafe_allow_html=True)
                if st.button("⚡ Upgrade to Pro — $9/mo", use_container_width=True):
                    st.session_state.pending_plan = "pro"
                    st.session_state.page = "billing"; st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<div class="btn-ghost" style="margin-top:6px">', unsafe_allow_html=True)
                if st.button("🏢 View Enterprise Plan", use_container_width=True):
                    st.session_state.pending_plan = "enterprise"
                    st.session_state.page = "billing"; st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                crd = st.session_state.billing_card
                st.markdown(f"**Payment method:** •••• {crd}" if crd else "**Payment method:** Not on file")
                st.markdown(f"**Renewal:** {plan['price_label']} on next billing date")
                st.markdown("")
                st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
                if st.button("Cancel Subscription", use_container_width=True):
                    if st.session_state.get("token"):
                        api_post("/billing/cancel")
                    st.session_state.plan        = "free"
                    st.session_state.chat_model  = "llama-3.1-8b-instant"
                    st.session_state.image_model = "flux"
                    st.success("Subscription cancelled. You're now on Free."); st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        with tab3:
            sync_usage()
            st.markdown("#### Usage Today")
            for label, mode, icon in [("AI Chats","chat","💬"),("Web Searches","search","🔍"),("Image Gens","image","🎨")]:
                used  = st.session_state.get(f"usage_{mode}", 0)
                limit = plan[f"{mode}_limit"]
                lstr  = "∞" if limit > 9000 else str(limit)
                pct   = min(used / max(limit if limit < 9000 else 100, 1) * 100, 100)
                cls   = "fill-ok" if pct < 70 else ("fill-warn" if pct < 100 else "fill-full")
                st.markdown(f"""
                <div style="margin:12px 0">
                  <div style="display:flex;justify-content:space-between;font-size:14px;margin-bottom:5px">
                    <span>{icon} {label}</span>
                    <span style="color:var(--text-2)">{used} / {lstr}</span>
                  </div>
                  <div class="usage-track"><div class="usage-fill {cls}" style="width:{pct}%"></div></div>
                </div>""", unsafe_allow_html=True)
            st.caption("Counters reset daily at midnight UTC.")
            if st.session_state.plan == "free":
                st.markdown('<div style="height:10px"/>', unsafe_allow_html=True)
                st.markdown('<div class="btn-upgrade">', unsafe_allow_html=True)
                if st.button("⚡ Get unlimited with Pro", use_container_width=True):
                    st.session_state.pending_plan = "pro"
                    st.session_state.page = "billing"; st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:16px"/>', unsafe_allow_html=True)
        st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
        if st.button("← Back to Chat", use_container_width=True):
            st.session_state.page = "chat"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


def render_sidebar():
    plan  = my_plan()
    pname = plan["name"].lower()
    with st.sidebar:
        st.markdown(f"""
        <div class="logo-wrap">
          <div class="logo-dot"></div>
          <div class="logo-mark">Nexus AI</div>
          <div style="margin-left:auto">
            <span class="plan-badge badge-{pname}">{plan['icon']} {plan['name']}</span>
          </div>
        </div>""", unsafe_allow_html=True)

        if st.session_state.success:
            st.success(st.session_state.success); st.session_state.success = ""

        if st.session_state.plan == "free":
            st.markdown('<div class="section-label">📊 Daily Usage</div>', unsafe_allow_html=True)
            for label, mode, icon in [("Chats","chat","💬"),("Searches","search","🔍"),("Images","image","🎨")]:
                used  = st.session_state.get(f"usage_{mode}", 0)
                limit = plan[f"{mode}_limit"]
                pct   = min(used / max(limit, 1) * 100, 100)
                cls   = "fill-ok" if pct < 70 else ("fill-warn" if pct < 100 else "fill-full")
                st.markdown(f"""
                <div class="usage-row">
                  <div class="usage-label"><span>{icon} {label}</span><span>{used}/{limit}</span></div>
                  <div class="usage-track"><div class="usage-fill {cls}" style="width:{pct}%"></div></div>
                </div>""", unsafe_allow_html=True)
            st.markdown('<div style="height:6px"/>', unsafe_allow_html=True)
            st.markdown('<div class="btn-upgrade">', unsafe_allow_html=True)
            if st.button("⚡ Upgrade to Pro", use_container_width=True, key="sb_upgrade"):
                st.session_state.show_upgrade   = True
                st.session_state.upgrade_reason = ""
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:rgba(79,140,255,.08);border:1px solid rgba(79,140,255,.2);
                 border-radius:var(--radius-sm);padding:10px 14px;margin:6px 0 10px;font-size:13px">
              {plan['icon']} <strong>{plan['name']}</strong> active
              <div style="font-size:11px;color:var(--text-3);margin-top:3px">
                {'∞ chats · ∞ searches · 30 images/day' if plan['name']=='Pro' else '∞ everything · API access'}
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
        if st.button("✦  New Chat", use_container_width=True, key="new_chat"):
            new_session(); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        allowed_chat = my_plan()["models"]
        allowed_imgs = my_plan()["image_models"]

        st.markdown('<div class="section-label">💬 Chat Model</div>', unsafe_allow_html=True)
        ckeys = list(CHAT_MODELS_ALL.keys())
        clabs = [f"{'🔒 ' if k not in allowed_chat else ''}{v}" for k, v in CHAT_MODELS_ALL.items()]
        ci    = ckeys.index(st.session_state.chat_model) if st.session_state.chat_model in ckeys else 0
        chosen = st.selectbox("Chat", clabs, index=ci, label_visibility="collapsed", key="sel_chat")
        ck = ckeys[clabs.index(chosen)]
        if ck not in allowed_chat:
            st.markdown('<div style="font-size:11px;color:var(--amber);margin-top:-8px;margin-bottom:4px">🔒 Pro only</div>', unsafe_allow_html=True)
        st.session_state.chat_model = ck

        st.markdown('<div class="section-label">🔍 Search Model</div>', unsafe_allow_html=True)
        skeys = list(SEARCH_MODELS_ALL.keys()); slabs = list(SEARCH_MODELS_ALL.values())
        si    = skeys.index(st.session_state.search_model) if st.session_state.search_model in skeys else 0
        st.session_state.search_model = skeys[
            slabs.index(st.selectbox("Search", slabs, index=si, label_visibility="collapsed", key="sel_search"))
        ]

        st.markdown('<div class="section-label">🎨 Image Model</div>', unsafe_allow_html=True)
        ikeys = list(IMAGE_MODELS_ALL.keys())
        ilabs = [f"{'🔒 ' if k not in allowed_imgs else ''}{v}" for k, v in IMAGE_MODELS_ALL.items()]
        ii    = ikeys.index(st.session_state.image_model) if st.session_state.image_model in ikeys else 0
        chosen_i = st.selectbox("Image", ilabs, index=ii, label_visibility="collapsed", key="sel_image")
        ik = ikeys[ilabs.index(chosen_i)]
        if ik not in allowed_imgs:
            st.markdown('<div style="font-size:11px;color:var(--amber);margin-top:-8px;margin-bottom:4px">🔒 Pro only</div>', unsafe_allow_html=True)
        st.session_state.image_model = ik

        st.markdown('<div class="section-label">💬 Conversations</div>', unsafe_allow_html=True)
        sessions = st.session_state.local_sessions
        if not sessions:
            st.markdown('<div style="font-size:12px;color:var(--text-3);padding:5px 0">No chats yet.</div>', unsafe_allow_html=True)
        else:
            for sid, data in reversed(list(sessions.items())):
                is_active = sid == st.session_state.active_session
                c1, c2_ = st.columns([5, 1])
                with c1:
                    cls = "conv-btn active" if is_active else "conv-btn"
                    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
                    if st.button(f"💬 {data.get('title','Chat')[:24]}", key=f"sess_{sid}", use_container_width=True):
                        st.session_state.active_session = sid; st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                with c2_:
                    st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
                    if st.button("🗑", key=f"del_{sid}"):
                        delete_session(sid); st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        uname = st.session_state.username or "Guest"
        color = st.session_state.avatar_color
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:4px 0">
          <div class="avatar user" style="background:{color};flex-shrink:0">{uname[0].upper()}</div>
          <div>
            <div style="font-size:14px;font-weight:600">{uname}</div>
            <div style="font-size:11px;color:var(--text-3)">{plan['icon']} {plan['name']} · Groq + Pollinations</div>
          </div>
        </div>""", unsafe_allow_html=True)

        ca, cb = st.columns(2)
        with ca:
            st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
            if st.button("⚙ Account", use_container_width=True, key="acct_btn"):
                st.session_state.page = "account"; st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with cb:
            st.markdown('<div class="btn-danger">', unsafe_allow_html=True)
            if st.button("🚪 Logout", use_container_width=True, key="logout_btn"):
                if st.session_state.get("token"):
                    api_post("/auth/logout")
                for k in ["username","token","user_id"]:
                    st.session_state[k] = None
                st.session_state.plan            = "free"
                st.session_state.local_sessions  = {}
                st.session_state.active_session  = None
                st.session_state.page            = "login"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


def render_auth():
    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.markdown('<div class="auth-logo">✦ Nexus AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Your free intelligent multi-modal assistant</div>', unsafe_allow_html=True)

        t1, t2 = st.columns(2)
        with t1:
            if st.button("🔑 Login", use_container_width=True,
                         type="primary" if st.session_state.page == "login" else "secondary",
                         key="go_login"):
                st.session_state.page = "login"; st.rerun()
        with t2:
            if st.button("✨ Sign Up", use_container_width=True,
                         type="primary" if st.session_state.page == "signup" else "secondary",
                         key="go_signup"):
                st.session_state.page = "signup"; st.rerun()

        if st.session_state.error:
            st.error(st.session_state.error); st.session_state.error = ""
        if st.session_state.success:
            st.success(st.session_state.success); st.session_state.success = ""

        st.markdown("<div style='height:10px'/>", unsafe_allow_html=True)

        if st.session_state.page == "login":
            with st.form("lf"):
                em = st.text_input("Email", placeholder="you@example.com")
                pw = st.text_input("Password", type="password", placeholder="••••••••")
                c1, c2 = st.columns(2)
                with c1: sub   = st.form_submit_button("Login →", use_container_width=True, type="primary")
                with c2: guest = st.form_submit_button("Guest Mode", use_container_width=True)

            if sub:
                if em and pw:
                    data, code = api_post("/auth/login", {"email": em, "password": pw}, auth=False)
                    if code == 200:
                        st.session_state.token        = data["token"]
                        st.session_state.user_id      = data["user_id"]
                        st.session_state.username     = data["username"]
                        st.session_state.plan         = data["plan"]
                        st.session_state.avatar_color = data.get("avatar_color", "#4f8cff")
                        sync_usage()
                        load_sessions()
                        if not st.session_state.local_sessions:
                            new_session()
                        else:
                            st.session_state.active_session = next(iter(
                                reversed(list(st.session_state.local_sessions.keys()))
                            ))
                        st.session_state.page = "chat"; st.rerun()
                    else:
                        st.session_state.error = data.get("error", "Login failed"); st.rerun()
                else:
                    st.session_state.error = "Please enter email and password"; st.rerun()

            elif guest:
                st.session_state.username = "Guest"
                st.session_state.plan     = "free"
                st.session_state.page     = "chat"
                new_session(); st.rerun()

        else:  # signup
            with st.form("sf"):
                un = st.text_input("Username", placeholder="nexus_user")
                em = st.text_input("Email",    placeholder="you@example.com")
                pw = st.text_input("Password", type="password", placeholder="Min. 6 characters")
                sub = st.form_submit_button("Create Account →", use_container_width=True, type="primary")

            if sub:
                if un and em and pw:
                    data, code = api_post("/auth/register",
                                          {"username": un, "email": em, "password": pw},
                                          auth=False)
                    if code == 201:
                        st.session_state.token        = data["token"]
                        st.session_state.user_id      = data["user_id"]
                        st.session_state.username     = data["username"]
                        st.session_state.plan         = "free"
                        st.session_state.avatar_color = data.get("avatar_color", "#4f8cff")
                        sync_usage()
                        new_session()
                        st.session_state.page = "chat"; st.rerun()
                    else:
                        st.session_state.error = data.get("error", "Registration failed"); st.rerun()
                else:
                    st.session_state.error = "Please fill in all fields"; st.rerun()

        st.markdown("""
        <div style="text-align:center;font-size:13px;color:var(--text-3);margin-top:18px">
          Free · 20 chats/day · No credit card required<br>
          <span style="color:var(--accent)">⚡ Pro from $9/mo</span> for unlimited everything
        </div>""", unsafe_allow_html=True)


def render_bubble(msg):
    role    = msg["role"]; content = msg["content"]
    mode    = msg.get("mode", "chat")
    sources = msg.get("sources", []); img = msg.get("image"); ts = msg.get("ts", "")
    is_user = role == "user"

    badge_html = ""
    if not is_user:
        bm = {"chat":("badge-chat","✦ Nexus"),"search":("badge-search","🔍 Search"),"image":("badge-image","🎨 Image")}
        bc, bl = bm.get(mode, ("badge-chat","✦ Nexus"))
        badge_html = f'<div class="mode-badge {bc}">{bl}</div>'

    body     = md_to_html(content) if not is_user else _esc(content)
    src_html = ""
    if sources:
        pills    = "".join(f'<a class="src-pill" href="{s.get("url","#")}" target="_blank">{s.get("url","")[:45]}</a>' for s in sources[:5])
        src_html = f'<div class="sources-wrap">{pills}</div>'

    if img is not None:
        st.markdown(
            f'<div class="msg-wrap"><div class="avatar bot">✦</div>'
            f'<div style="max-width:78%">{badge_html}'
            f'<div class="bubble bot">{body}</div></div></div>',
            unsafe_allow_html=True
        )
        st.image(img, width=420)
        return

    st.markdown(f"""
    <div class="msg-wrap {'user' if is_user else ''}">
      <div class="avatar {'user' if is_user else 'bot'}">{'you' if is_user else '✦'}</div>
      <div style="max-width:78%">
        {badge_html}
        {body}
        {src_html}
      </div>
    </div>""", unsafe_allow_html=True)


EXAMPLES = [
    ("💬","Explain ML transformers","Explain how transformer architecture works in machine learning"),
    ("🔍","Latest AI news (search)","search for the latest AI breakthroughs and news in 2026"),
    ("🎨","Futuristic city (image)","generate a futuristic neon cyberpunk city at night"),
]

def render_chat():
    render_sidebar()
    if st.session_state.get("show_upgrade"):
        upgrade_modal()

    msgs = current_msgs()
    if not msgs:
        uname = (st.session_state.username or "there").split()[0]
        plan  = my_plan()
        lim   = f"{plan['chat_limit']} chats/day" if plan['chat_limit'] < 9000 else "unlimited chats"
        st.markdown(f"""
        <div class="welcome">
          <div style="font-size:52px;margin-bottom:14px">✦</div>
          <div class="welcome-title">Hello, {uname}!</div>
          <div class="welcome-sub">
            <strong style="color:var(--accent)">{plan['icon']} {plan['name']}</strong> plan · {lim}.
          </div>
        </div>""", unsafe_allow_html=True)

        cols = st.columns(3)
        for i, (icon, label, prompt) in enumerate(EXAMPLES):
            with cols[i % 3]:
                st.markdown(f'<div class="cap-card"><div class="cap-icon">{icon}</div><div class="cap-title">{label}</div></div>', unsafe_allow_html=True)
                if st.button("Try →", key=f"ex_{i}", use_container_width=True):
                    if not st.session_state.active_session: new_session()
                    add_msg("user", prompt); rename_first(prompt)
                    with st.spinner("✦ Nexus is thinking…"):
                        process_message(prompt)
                    st.rerun()
    else:
        for msg in msgs:
            render_bubble(msg)

    if st.session_state.plan == "free":
        plan = my_plan()
        for mode, icon in [("chat","💬"),("search","🔍"),("image","🎨")]:
            used  = st.session_state.get(f"usage_{mode}", 0)
            limit = plan[f"{mode}_limit"]
            if 0 < limit - used <= 2:
                st.markdown(
                    f'<div class="limit-warn">{icon} Only <strong>{limit-used} '
                    f'{mode}{"s" if limit-used!=1 else ""}</strong> left today. '
                    f'Upgrade to Pro for unlimited.</div>',
                    unsafe_allow_html=True
                )

    user_input = st.chat_input("Message Nexus… chat, search the web, or generate an image")
    if user_input and user_input.strip():
        if not st.session_state.active_session:
            new_session()
        add_msg("user", user_input.strip())
        rename_first(user_input.strip())
        with st.spinner("✦ Nexus is thinking…"):
            process_message(user_input.strip())
        st.rerun()


def main():
    page = st.session_state.page
    if not st.session_state.username or page in ("login","signup"):
        render_auth()
    elif page == "billing":
        render_billing()
    elif page == "account":
        render_account()
    else:
        render_chat()

if __name__ == "__main__":
    main()
