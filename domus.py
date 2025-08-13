import os
import re
import json
import base64
import time
import smtplib
from datetime import datetime

import numpy as np
import pandas as pd
import faiss
import streamlit as st
import streamlit.components.v1 as components
from bs4 import BeautifulSoup
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from sentence_transformers import SentenceTransformer
import requests

# ==================== UTIL: LOAD BASE64 IMAGES ====================

def load_base64_image(path: str) -> str:
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        return ""

logo_lbn_base64 = load_base64_image("assets/lbn_logo.png")
logo_uel_base64 = load_base64_image("assets/uel_logo.png")
logo_domus_base64 = load_base64_image("assets/logo1.png")

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Domus for Newham",
    page_icon=(f"data:image/png;base64,{logo_domus_base64}" if logo_domus_base64 else None),
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== HEADER ====================

st.markdown(f"""
<style>
.main-header {{
    display: flex; justify-content: space-between; align-items: center; padding: 10px;
}}
.header-content {{ display: flex; align-items: center; }}
.logo-section {{ display: flex; align-items: center; }}
.header-text {{ margin-left: 10px; }}
.right-logos {{ display: flex; gap: 10px; }}
.header-logo {{ height: 60px; }}
</style>

<div class="main-header">
    <div class="header-content">
        <div class="logo-section">
            {'<img src="data:image/png;base64,' + logo_domus_base64 + '" class="header-logo" />' if logo_domus_base64 else ''}
            <div class="header-text">
                <h1 class="main-title">Domus Portal</h1>
                <p class="subtitle">AI-Powered Housing Solutions for London Borough of Newham</p>
            </div>
        </div>
    </div>
    <div class="right-logos">
        {'<img src="data:image/png;base64,' + logo_uel_base64 + '" class="header-logo" />' if logo_uel_base64 else ''}
        {'<img src="data:image/png;base64,' + logo_lbn_base64 + '" class="header-logo" />' if logo_lbn_base64 else ''}
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== STYLING ====================

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); font-family: 'Inter', sans-serif; min-height: 100vh; }
    .main-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2.5rem 2rem; border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3); color: white; position: relative; overflow: hidden; }
    .main-header::before { content: ''; position: absolute; inset: 0; background: url('data:image/svg+xml,<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 100 100\"><defs><pattern id=\"grain\" width=\"100\" height=\"100\" patternUnits=\"userSpaceOnUse\"><circle cx=\"50\" cy=\"50\" r=\"1\" fill=\"%23ffffff\" opacity=\"0.1\"/></pattern></defs><rect width=\"100\" height=\"100\" fill=\"url(%23grain)\"/></svg>') repeat; opacity: 0.1; }
    .header-content { display: flex; justify-content: space-between; align-items: center; position: relative; z-index: 1; }
    .logo-section { display: flex; align-items: center; gap: 1.5rem; }
    .header-logo { width: 80px; height: 80px; border-radius: 50%; background: rgba(255,255,255,0.2); padding: 10px; backdrop-filter: blur(10px); box-shadow: 0 8px 32px rgba(0,0,0,0.1); }
    .main-title { font-size: 3rem !important; font-weight: 800 !important; margin: 0 !important; color: white !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); letter-spacing: -1px; }
    .subtitle { font-size: 1.2rem; opacity: 0.9; margin: 0.5rem 0 0 0; font-weight: 400; }
    .stExpander { background: white !important; border: 1px solid #e2e8f0 !important; border-radius: 15px !important; margin-bottom: 1rem; box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important; transition: all 0.3s ease; }
    .stExpander:hover { box-shadow: 0 8px 30px rgba(0,0,0,0.12) !important; transform: translateY(-2px); }
    .stButton > button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; color: white !important; border-radius: 12px !important; border: none !important; padding: 0.75rem 2rem !important; font-weight: 600 !important; box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3) !important; transition: all 0.3s ease !important; font-family: 'Inter', sans-serif !important; font-size: 0.95rem !important; }
    .stButton > button:hover { background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%) !important; box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4) !important; transform: translateY(-2px) !important; }
    .stDataFrame { background: white !important; border: 1px solid #e2e8f0 !important; border-radius: 15px !important; box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important; overflow: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ==================== LOAD INDEXES ====================

@st.cache_resource
def load_indexes():
    household_index = faiss.read_index("household_index.faiss")
    property_index = faiss.read_index("property_index.faiss")
    household_df = pd.read_csv("household_vectors.csv")
    property_df = pd.read_csv("property_vectors.csv")
    return household_index, property_index, household_df, property_df

household_index, property_index, household_df, property_df = load_indexes()
model = SentenceTransformer("all-MiniLM-L6-v2")

# ==================== EMAIL CONFIG ====================
# Move credentials to .streamlit/secrets.toml. Fallback to env vars.
FROM_EMAIL = st.secrets.get("email", {}).get("from_email") or os.getenv("DOMUS_FROM_EMAIL", "")
APP_PASSWORD = st.secrets.get("email", {}).get("app_password") or os.getenv("DOMUS_APP_PASSWORD", "")

# ==================== LIVE PRICE SCRAPER ====================

EXCLUDE = (
    "deposit",
    "holding",
    "tenancy",
    "council",
    "tax",
    "fee",
    "fees",
    "bill",
    "bills",
    "per person",
)
WEEKLY_RE = re.compile(r"£\s*([\d,]+(?:\.\d{2})?)\s*(pw|per\s*week)\b", re.I)
MONTHLY_RE = re.compile(r"£\s*([\d,]+(?:\.\d{2})?)\s*(pcm|per\s*month)\b", re.I)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-GB,en;q=0.9",
            "Referer": "https://www.rightmove.co.uk/",
            "Connection": "keep-alive",
        }
    )
    return s


def _get(url: str, timeout: int = 12):
    try:
        return _session().get(url, timeout=timeout, allow_redirects=True)
    except requests.RequestException:
        return None


def _extract_from_json(soup: BeautifulSoup) -> str | None:
    # JSON-LD
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for it in items:
            if not isinstance(it, dict):
                continue
            offer = it.get("offers")
            if isinstance(offer, dict):
                price = offer.get("price") or offer.get("lowPrice") or offer.get("highPrice")
                spec = offer.get("priceSpecification") or {}
                unit = (spec.get("unitText") or "").lower()
                if price:
                    try:
                        val = float(str(price).replace(",", ""))
                    except Exception:
                        continue
                    if "week" in unit or "pw" in unit:
                        return f"£{int(round(val))} per week"
                    if "month" in unit or "pcm" in unit:
                        return f"£{int(round(val))} per month"
                    return f"£{int(round(val))}"
    # Embedded app JSON
    for tag in soup.find_all("script"):
        txt = (tag.string or tag.text or "")
        if not txt:
            continue
        m = re.search(r'"displayPrice"\s*:\s*"([^"]+)"', txt)
        if m:
            return m.group(1).strip()
        m2 = re.search(r'"price"\s*:\s*([\d\.]+)\s*,\s*"priceFrequency"\s*:\s*"([^"]+)"', txt)
        if m2:
            val = float(m2.group(1))
            unit = m2.group(2).lower()
            if "week" in unit:
                return f"£{int(round(val))} per week"
            if "month" in unit:
                return f"£{int(round(val))} per month"
    return None


def _ok_line(t: str) -> bool:
    tl = t.lower()
    return ("£" in tl) and (("pw" in tl or "per week" in tl or "pcm" in tl or "per month" in tl)) and not any(
        x in tl for x in EXCLUDE
    )


def _extract_from_html(soup: BeautifulSoup) -> str | None:
    selectors = [
        '[data-testid="price"]',
        '[data-testid="listing-price"]',
        '.property-header-price',
        '.price',
        '.rent',
        'h1',
        'h2',
        'h3',
        'span',
        'p',
    ]
    candidates: list[str] = []

    for sel in selectors:
        for el in soup.select(sel):
            txt = el.get_text(" ", strip=True)
            if _ok_line(txt):
                candidates.append(txt)

    if not candidates:
        for ln in (ln.strip() for ln in soup.get_text("\n", strip=True).splitlines() if ln.strip()):
            if _ok_line(ln):
                candidates.append(ln)

    if not candidates:
        return None

    def score(t: str) -> float:
        tl = t.lower()
        s = 0.0
        if "per week" in tl or " pw" in tl:
            s += 3.0
        if "per month" in tl or " pcm" in tl:
            s += 2.0
        s += min(len(t) / 40.0, 1.0)
        if any(x in tl for x in EXCLUDE):
            s -= 3.0
        return s

    candidates.sort(key=score, reverse=True)
    return candidates[0]


@st.cache_data(ttl=1800)
def get_exact_display_price(url: str, _refresh_key: int = 0) -> tuple[str | None, str]:
    if not url or not url.startswith("http"):
        return None, "fallback"
    r = _get(url)
    if not r or r.status_code != 200 or not r.text:
        return None, "fallback"
    soup = BeautifulSoup(r.text, "html.parser")
    dp = _extract_from_json(soup)
    if dp:
        return dp, "json"
    dp = _extract_from_html(soup)
    if dp:
        return dp, "html"
    return None, "fallback"


def to_weekly(dp: str | None) -> str | None:
    if not dp:
        return None
    t = dp.strip()
    m = WEEKLY_RE.search(t)
    if m:
        val = float(m.group(1).replace(",", ""))
        return f"£{int(round(val))} pw"
    m = MONTHLY_RE.search(t)
    if m:
        monthly = float(m.group(1).replace(",", ""))
        weekly = monthly * 12.0 / 52.0
        return f"£{int(round(weekly))} pw"
    return None


def weekly_price_from_url(url: str, stored_weekly) -> tuple[str, str]:
    dp, src = get_exact_display_price(url, _refresh_key=0)
    weekly = to_weekly(dp)
    if weekly:
        return weekly, src
    if stored_weekly is not None and str(stored_weekly).lower() != "nan":
        try:
            return f"£{int(round(float(stored_weekly)))} pw", "fallback"
        except Exception:
            pass
    return "—", "fallback"


# Back-compat shim

def display_price_weekly(url, stored_weekly, refresh_key: int = 0):
    weekly_text, _ = weekly_price_from_url(url, stored_weekly)
    return weekly_text


# ==================== PREVENTION DATA ====================


def load_prevention_entries():
    if os.path.exists("prevention_list.csv"):
        return pd.read_csv("prevention_list.csv")
    return pd.DataFrame(
        columns=[
            "Ref ID",
            "Name",
            "Age",
            "Gender",
            "Employment",
            "Income",
            "Household Size",
            "Adult Couples",
            "Single Adults",
            "Children <10",
            "Boys 10-15",
            "Girls 10-15",
            "Overnight Carer",
            "Bedroom Need",
            "Affordability",
            "Zone",
            "Feature Code",
            "Match Prompt",
            "Date",
        ]
    )


def save_prevention_entries(df):
    df.to_csv("prevention_list.csv", index=False)


prevention_df = load_prevention_entries()

# ==================== MATCHING ====================

ZONES = {"A1", "A2", "B0", "C0"}


def parse_constraints(prompt: str):
    p = prompt or ""
    p_low = p.lower()

    # FIXED: closed regex string literal
    m_bed = re.search(r"(\d+)\s*[-\s]?(bed|beds|bedroom|bedrooms|bd)\b", p_low)
    beds = int(m_bed.group(1)) if m_bed else None

    m_pw = re.search(r"£\s*([\d,]+)\s*(pw|per\s*week|/week)?", p_low)
    budget = int(m_pw.group(1).replace(",", "")) if m_pw else None

    m_pcm = re.search(r"£\s*([\d,]+)\s*(pcm|per\s*month)", p_low)
    if budget is None and m_pcm:
        monthly = float(m_pcm.group(1).replace(",", ""))
        budget = int(round(monthly * 12 / 52))

    zone = None
    for z in ZONES:
        if re.search(rf"\b{z}\b", p, flags=re.I):
            zone = z
            break

    tokens = [t for t in re.findall(r"[a-z0-9]+", p_low) if len(t) > 2]
    return {"beds": beds, "budget": budget, "zone": zone, "tokens": tokens}


@st.cache_resource
def precompute_property_embeddings(property_df, _model):
    texts = property_df["match_prompt"].fillna("").tolist()
    embs = _model.encode(texts, batch_size=64, show_progress_bar=False)
    return np.array(embs, dtype="float32")


prop_embs = precompute_property_embeddings(property_df, model)


def keyword_boost(row, tokens):
    txt = " ".join([str(row.get("title", "")), str(row.get("address", "")), str(row.get("match_prompt", "")),]).lower()
    hits = sum(1 for t in set(tokens) if t in txt)
    return hits


def search_properties(prompt: str, top_k: int = 5):
    cons = parse_constraints(prompt)
    df = property_df.copy()

    # Hard filters
    if cons["beds"] is not None:
        df = df[df["bedrooms"] == cons["beds"]]
    if cons["budget"] is not None:
        df = df[df["rent"].astype(float) <= float(cons["budget"])]
    if cons["zone"] is not None:
        df = df[df["zone"].astype(str).str.upper() == cons["zone"]]

    # Relax if empty: drop zone -> +10% budget
    if df.empty and cons["zone"] is not None:
        df = property_df.copy()
        if cons["beds"] is not None:
            df = df[df["bedrooms"] == cons["beds"]]
        if cons["budget"] is not None:
            df = df[df["rent"].astype(float) <= float(cons["budget"] * 1.10)]

    if df.empty:
        df = property_df.copy()

    # Semantic similarity (MiniLM)
    q = model.encode([prompt]).astype("float32")  # (1, d)

    # Align indices to avoid mismatches after filtering
    idxer = property_df.index.get_indexer(df.index)
    sem = np.dot(prop_embs[idxer], q.T).flatten()

    df = df.assign(sem_score=sem)

    # Keyword boost (normalized)
    if cons["tokens"]:
        kb = df.apply(lambda r: keyword_boost(r, cons["tokens"]), axis=1)
        kb = kb / kb.max() if kb.max() > 0 else kb
        df = df.assign(kboost=kb)
    else:
        df = df.assign(kboost=0.0)

    # Final score
    df = df.assign(score=0.85 * df["sem_score"] + 0.15 * df["kboost"])\
           .sort_values("score", ascending=False)

    return df.head(int(top_k))


# ==================== UI ====================

# Keep top_k in session so it applies before search
if "top_k" not in st.session_state:
    st.session_state.top_k = 5

# Tabs
tab1, tab2 = st.tabs(["MatchBot", "Search and Save"])

# ------- MATCHBOT TAB -------
with tab1:
    st.title("Domus - AI-Driven Property MatchBot")

    # Number of results *before* entering prompt so it affects search
    st.session_state.top_k = st.slider("Number of Matches", min_value=1, max_value=10, value=st.session_state.top_k, key="top_k_slider")
    top_k = int(st.session_state.top_k)

    prompt = st.text_input("Enter Household Need Description")

    if prompt:
        with open("search_log.txt", "a") as log_file:
            log_file.write(f"{datetime.now()}: {prompt}\n")

        with st.spinner("🔎 Searching for matches..."):
            matches = search_properties(prompt, top_k=top_k)
    else:
        matches = pd.DataFrame()

    if not matches.empty:
        st.subheader("🔑 Matching properties available based on your description")
        for _, row in matches.iterrows():
            title_price = display_price_weekly(row.get("url"), row.get("rent"))
            with st.expander(f"{row['bedrooms']}-bed in {row['zone']} — {title_price}"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**📍 Address:** {row['address']}")
                    st.markdown(f"**📝 Match Reason:** {row['match_prompt']}")
                with col2:
                    st.markdown(f"[👉 View Property]({row['url']})", unsafe_allow_html=True)

        st.subheader("📨 Send Matches to Resident")
        recipient = st.text_input("Enter Email Address", key="email_tab1")

        if st.button("👁️ Preview Email"):
            st.markdown("#### 📧 Email Preview", unsafe_allow_html=True)
            for _, row in matches.iterrows():
                price_text = display_price_weekly(row.get("url"), row.get("rent"))
                st.markdown(
                    f"<li>{row['bedrooms']}-bed in zone {row['zone']} — {price_text} — <a href='{row['url']}'>View</a></li>",
                    unsafe_allow_html=True,
                )

        def send_email_with_links(to_email: str, properties_df: pd.DataFrame) -> str:
            if not FROM_EMAIL or not APP_PASSWORD:
                return "❌ Email not configured. Please set FROM_EMAIL and APP_PASSWORD in secrets or env vars."
            if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", to_email):
                return "❌ Invalid email address."

            body = [
                "<html><body style='font-family: Arial;'>",
                "<p>Dear Resident,</p>",
                "<p>We have identified the following properties that match your housing needs:</p><ul>",
            ]
            for _, r in properties_df.iterrows():
                price_text = display_price_weekly(r.get("url"), r.get("rent"))
                body.append(
                    f"<li>{r['bedrooms']}-bed in zone {r['zone']} — {price_text} — <a href='{r['url']}'>View Property</a></li>"
                )
            body += [
                "</ul>",
                "<p>Please contact the letting agent directly and inform the Council once you've selected a suitable property.</p>",
                "<p>Best regards,<br><strong>Temporary Accommodation Team</strong><br>London Borough of Newham</p>",
                "</body></html>",
            ]
            body = "".join(body)

            msg = MIMEMultipart("alternative")
            msg["Subject"] = "Your Property Matches from Newham Council"
            msg["From"] = FROM_EMAIL
            msg["To"] = to_email
            msg.attach(MIMEText(body, "html"))

            try:
                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                    server.login(FROM_EMAIL, APP_PASSWORD)
                    server.sendmail(FROM_EMAIL, to_email, msg.as_string())
                return "✅ Email sent successfully."
            except Exception as e:
                return f"❌ Failed to send email: {e}"

        if st.button("📤 Send Email", disabled=(not recipient)):
            status = send_email_with_links(recipient, matches)
            st.success(status) if status.startswith("✅") else st.error(status)

# ------- PREVENTION TOOL TAB -------
with tab2:
    st.title("New Request – Search and Save ")

    # Share same top_k slider for consistency
    st.session_state.top_k = st.slider(
        "Number of Matches (Prevention Tab)",
        min_value=1,
        max_value=10,
        value=st.session_state.top_k,
        key="top_k_slider_tab2",
    )
    top_k2 = int(st.session_state.top_k)

    with st.form("prevention_form"):
        name = st.text_input("Full Name")
        ref_id = st.text_input("Reference ID")
        age = st.number_input("Age", 0, 120)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        employment = st.text_input("Employment Status")
        income = st.number_input("Household Income (£/week)", 0)
        household_size = st.number_input("Total Household Size", 1)
        couples = st.number_input("Adult Couples", 0)
        single_adults = st.number_input("Single Adults", 0)
        children_u10 = st.number_input("Children Under 10", 0)
        boys_10_15 = st.number_input("Boys Aged 10–15", 0)
        girls_10_15 = st.number_input("Girls Aged 10–15", 0)
        carer = st.selectbox("Overnight Carer?", ["No", "Yes"])
        zone = st.selectbox("Preferred Zone", ["A1", "A2", "B0", "C0"])
        date = datetime.today().strftime("%Y-%m-%d")
        submitted = st.form_submit_button("Submit & Search Properties")

    if submitted:
        bedroom_need = max(
            1,
            couples
            + single_adults
            + (children_u10 // 2)
            + ((boys_10_15 + girls_10_15) // 2)
            + (1 if (boys_10_15 % 2 + girls_10_15 % 2) else 0)
            + (1 if carer == "Yes" else 0),
        )
        rent = income
        special_flag = 1 if carer == "Yes" else 0
        feature_code = f"N{bedroom_need}0{str(rent).zfill(4)}{zone}{special_flag}"
        match_prompt = (
            f"{bedroom_need}-bed needed for {name}, {gender}, age {age}, "
            f"household size {household_size}, affordability {rent}/week, zone {zone}"
        )
        new_row = pd.DataFrame(
            [
                {
                    "Ref ID": ref_id,
                    "Name": name,
                    "Age": age,
                    "Gender": gender,
                    "Employment": employment,
                    "Income": income,
                    "Household Size": household_size,
                    "Adult Couples": couples,
                    "Single Adults": single_adults,
                    "Children <10": children_u10,
                    "Boys 10-15": boys_10_15,
                    "Girls 10-15": girls_10_15,
                    "Overnight Carer": carer,
                    "Bedroom Need": bedroom_need,
                    "Affordability": rent,
                    "Zone": zone,
                    "Feature Code": feature_code,
                    "Match Prompt": match_prompt,
                    "Date": date,
                }
            ]
        )
        prevention_df = pd.concat([prevention_df, new_row], ignore_index=True)
        save_prevention_entries(prevention_df)
        st.success("✅ Household added and matching properties below...")

        with st.spinner("🔎 Searching for matches..."):
            top_matches = search_properties(match_prompt, top_k=top_k2)

        for _, row in top_matches.iterrows():
            weekly_price = display_price_weekly(row.get("url"), row.get("rent"))
            with st.expander(f"{row['bedrooms']}-bed in {row['zone']} — {weekly_price}"):
                st.markdown(f"**Address:** {row['address']}")
                st.markdown(f"**Match Reason:** {row['match_prompt']}")
                st.markdown(f"[View Property]({row['url']})")

        recipient2 = st.text_input("Enter Email Address to Send Matches", key="email_tab2")

        def send_email_with_links2(to_email: str, properties_df: pd.DataFrame) -> str:
            if not FROM_EMAIL or not APP_PASSWORD:
                return "❌ Email not configured. Please set FROM_EMAIL and APP_PASSWORD in secrets or env vars."
            if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", to_email):
                return "❌ Invalid email address."

            body = [
                "<html><body style='font-family: Arial;'>",
                "<p>Dear Resident,</p>",
                "<p>We have identified the following properties that match your housing needs:</p><ul>",
            ]
            for _, r in properties_df.iterrows():
                weekly_price = display_price_weekly(r.get("url"), r.get("rent"))
                body.append(
                    f"<li>{r['bedrooms']}-bed in zone {r['zone']} — {weekly_price} — <a href='{r['url']}'>View Property</a></li>"
                )
            body += [
                "</ul>",
                "<p>Please contact the letting agent directly and inform the Council once you've selected a suitable property.</p>",
                "<p>Best regards,<br><strong>Temporary Accommodation Team</strong><br>London Borough of Newham</p>",
                "</body></html>",
            ]
            body = "".join(body)

            msg = MIMEMultipart("alternative")
            msg["Subject"] = "Your Property Matches from Newham Council"
            msg["From"] = FROM_EMAIL
            msg["To"] = to_email
            msg.attach(MIMEText(body, "html"))

            try:
                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                    server.login(FROM_EMAIL, APP_PASSWORD)
                    server.sendmail(FROM_EMAIL, to_email, msg.as_string())
                return "✅ Email sent successfully."
            except Exception as e:
                return f"❌ Failed to send email: {e}"

        if st.button("📤 Send Email to Resident", disabled=(not recipient2)):
            status = send_email_with_links2(recipient2, top_matches)
            st.success(status) if status.startswith("✅") else st.error(status)

# ==================== FOOTER ====================

components.html(
    f"""
<style>
.domus-footer {{ background-color: #F8F9FA; padding: 30px 40px; border-top: 1px solid #E0E0E0; font-family: 'Arial', sans-serif; color: #444; margin-top: 50px; }}
.footer-content {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 2rem; }}
.footer-left {{ display: flex; align-items: center; gap: 15px; flex-wrap: wrap; }}
.footer-logo {{ height: 60px; width: auto; }}
.footer-text h2 {{ margin: 0; font-size: 20px; color: #1976D2; }}
.footer-text p {{ margin: 2px 0 0; font-size: 13px; color: #666; }}
.footer-badges {{ display: flex; gap: 12px; flex-wrap: wrap; justify-content: flex-end; }}
.footer-badge {{ background-color: #2196F3; color: white; border-radius: 4px; padding: 6px 12px; font-size: 12px; white-space: nowrap; }}
.footer-bottom {{ text-align: center; margin-top: 20px; font-size: 12px; color: #999; }}
</style>

<div class="domus-footer">
    <div class="footer-content">
        <div class="footer-left">
            {'<img src="data:image/png;base64,' + logo_lbn_base64 + '" class="footer-logo" />' if logo_lbn_base64 else ''}
            {'<img src="data:image/png;base64,' + logo_uel_base64 + '" class="footer-logo" />' if logo_uel_base64 else ''}
            <div class="footer-text">
                <h2>Domus MatchBot</h2>
                <p>AI-Powered Housing Solutions for London Borough of Newham</p>
            </div>
        </div>
        <div class="footer-badges">
            <div class="footer-badge">24/7 Available</div>
            <div class="footer-badge">AI Powered</div>
            <div class="footer-badge">Secure Platform</div>
        </div>
    </div>
    <div class="footer-bottom">
        &copy; {datetime.today().year} London Borough of Newham – Temporary Accommodation Matching System<br>
        A collaboration between London Borough of Newham and University of East London
    </div>
</div>
""",
    height=250,
)
