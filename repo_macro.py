# app.py — STIR (Repo & MMF) Dashboard — Python 3.8 compatible
import streamlit as st
import pandas as pd
import requests
from datetime import date, timedelta
import plotly.express as px
from typing import Optional, List, Tuple, Dict

st.set_page_config(page_title="STIR Dashboard (Repo & MMF)", layout="wide")
st.title("Sample Dashboard — Repo & Money Market Funds")

OFR_BASE = "https://data.financialresearch.gov/v1"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

# ---------- Generic HTTP ----------
def _get_json(url: str, params: Optional[Dict] = None) -> Optional[Dict]:
    try:
        r = requests.get(url, params=params or {}, timeout=30)
    except Exception as e:
        st.error("HTTP error: {}".format(e))
        return None
    if r.status_code != 200:
        st.error("{} -> HTTP {}: {}".format(url, r.status_code, r.text[:300]))
        return None
    try:
        return r.json()
    except Exception as e:
        st.error("JSON decode error: {}".format(e))
        return None

# ---------- OFR discovery: find correct mnemonics at runtime ----------
@st.cache_data(ttl=3600)
def ofr_search(query: str) -> List[Dict]:
    url = OFR_BASE + "/metadata/search"
    js = _get_json(url, {"query": query})
    return js if isinstance(js, list) else []

def _pick_first(series_list: List[Dict], contains_all: List[str]) -> Optional[str]:
    for row in series_list:
        name = (row.get("value") or "").lower()
        ds = row.get("dataset")
        mn = row.get("mnemonic")
        if ds != "repo" or not mn:
            continue
        if all(s.lower() in name for s in contains_all):
            return mn
    return None

@st.cache_data(ttl=3600)
def discover_repo_volume_mnemonics() -> Dict[str, str]:
    """
    Use metadata/search to find Triparty/DVP/GCF 'Transaction Volume' + 'Overnight' + 'Final' mnemonics.
    Falls back to Preliminary if Final not found.
    """
    # pull broad hits once (wildcards supported)
    hits = []
    for q in [
        "Transaction Volume*",
        "Tri-Party Transaction Volume*",
        "DVP Transaction Volume*",
        "GCF Transaction Volume*",
        "Overnight Transaction Volume*",
    ]:
        hits.extend(ofr_search(q))

    def find_for(venue: str) -> Optional[str]:
        # try Final first
        mn = _pick_first(hits, [venue, "Transaction Volume", "Overnight", "(Final)"])
        if mn:
            return mn
        # then Preliminary
        return _pick_first(hits, [venue, "Transaction Volume", "Overnight", "(Preliminary)"])

    mapping = {
        "TRI": find_for("Tri-Party"),
        "DVP": find_for("DVP"),
        "GCF": find_for("GCF"),
    }
    return mapping

@st.cache_data(ttl=3600)
def ofr_timeseries(mnemonic: str, periodicity: str = "D") -> pd.DataFrame:
    """Fetch a single series and normalize to ['date','value']."""
    if not mnemonic:
        return pd.DataFrame(columns=["date", "value"])
    url = OFR_BASE + "/series/timeseries"
    js = _get_json(url, {"mnemonic": mnemonic, "periodicity": periodicity, "remove_nulls": "true"})
    if not js:
        return pd.DataFrame(columns=["date", "value"])
    # The STFM single series endpoint can return either a list of [date, value] pairs
    # or an object with {"timeseries":{"aggregation":[...]}} depending on docs version.
    if isinstance(js, list):
        df = pd.DataFrame(js, columns=["date", "value"])
    else:
        data = js.get("timeseries", {}).get("aggregation", [])
        df = pd.DataFrame(data, columns=["date", "value"]) if data and isinstance(data[0], list) else pd.DataFrame(js.get("timeseries", {}).get("aggregation", []))
        if "date" not in df.columns or "value" not in df.columns and len(df.columns) == 2:
            df.columns = ["date", "value"]
    if df.empty:
        return pd.DataFrame(columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["date"])

# ---------- FRED ----------
@st.cache_data(ttl=3600)
def fred_series(series_id: str, rename_to: str) -> pd.DataFrame:
    url = FRED_CSV.format(sid=series_id)
    try:
        d = pd.read_csv(url)
    except Exception as e:
        st.error("FRED fetch failed for {}: {}".format(series_id, e))
        return pd.DataFrame(columns=["date", rename_to])
    date_col = "DATE" if "DATE" in d.columns else ("observation_date" if "observation_date" in d.columns else None)
    if date_col is None or series_id not in d.columns:
        st.error("Unexpected FRED format for {}. Columns: {}".format(series_id, list(d.columns)))
        return pd.DataFrame(columns=["date", rename_to])
    out = pd.DataFrame({
        "date": pd.to_datetime(d[date_col], errors="coerce"),
        rename_to: pd.to_numeric(d[series_id], errors="coerce"),
    }).dropna(subset=["date"])
    return out

# ---------- Load data ----------
with st.spinner("Discovering OFR mnemonics…"):
    mnems = discover_repo_volume_mnemonics()

with st.expander("OFR discovery — chosen mnemonics"):
    st.write(mnems)

with st.spinner("Loading OFR repo volumes and FRED series…"):
    tri = ofr_timeseries(mnems.get("TRI"))
    dvp = ofr_timeseries(mnems.get("DVP"))
    gcf = ofr_timeseries(mnems.get("GCF"))
    rrp = fred_series("RRPONTSYD", "ON_RRP_Bn")
    mmf = fred_series("MMMFFAQ027S", "MMF_Total_$Bn")

def merge_on_date(dfs, labels):
    frames = []
    for df, lab in zip(dfs, labels):
        if df is None or df.empty:
            continue
        d = df.rename(columns={"value": lab}).copy()
        if {"date", lab}.issubset(d.columns):
            frames.append(d[["date", lab]])
    if not frames:
        return pd.DataFrame(columns=["date"] + labels)
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="date", how="outer")
    return out.sort_values("date").reset_index(drop=True)

repo_vol = merge_on_date([tri, dvp, gcf], ["Triparty_Vol", "DVP_Vol", "GCF_Vol"])

# ---------- Controls ----------
st.sidebar.header("Controls")
start_default = date(2014, 1, 1)
start = st.sidebar.date_input("Start date", start_default)
end = st.sidebar.date_input("End date", date.today())

def mask(df, col="date"):
    if df is None or df.empty:
        return df
    return df[(df[col] >= pd.to_datetime(start)) & (df[col] <= pd.to_datetime(end))]

repo_vol_f = mask(repo_vol)
rrp_f = mask(rrp)
mmf_f = mmf  # quarterly; show all

with st.expander("Data sanity — head() and lengths"):
    st.write("repo_vol rows:", 0 if repo_vol_f is None else len(repo_vol_f))
    st.write("rrp rows:", 0 if rrp_f is None else len(rrp_f))
    st.write("mmf rows:", 0 if mmf_f is None else len(mmf_f))
    st.write(repo_vol_f.head())

# ---------- Charts ----------
tab1, tab2 = st.tabs(["Repo Market", "MMF & RRP"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Overnight Transaction Volume — by Venue")
        ycols = [c for c in ["Triparty_Vol", "DVP_Vol", "GCF_Vol"] if c in repo_vol_f.columns]
        if ycols:
            st.plotly_chart(px.line(repo_vol_f, x="date", y=ycols), use_container_width=True, key="repo_volume")
        else:
            st.warning("No repo volume series found. See the discovery box above.")

    with c2:
        st.subheader("Six-Month Volume Change (%)")
        if not repo_vol.empty:
            tmp = repo_vol.set_index("date").pct_change(180).mul(100).reset_index()
            tmp = mask(tmp, "date")
            y2 = [c for c in ["Triparty_Vol", "DVP_Vol", "GCF_Vol"] if c in tmp.columns]
            if y2:
                st.plotly_chart(px.line(tmp, x="date", y=y2), use_container_width=True, key="six_month_change")
            else:
                st.info("Not enough data in the selected window.")
        else:
            st.info("Repo base frame is empty.")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Tri / DVP Ratio")
        if {"Triparty_Vol", "DVP_Vol"}.issubset(repo_vol_f.columns) and not repo_vol_f["DVP_Vol"].isna().all():
            ratio = repo_vol_f[["date"]].copy()
            ratio["TRI_DVP_Ratio"] = repo_vol_f["Triparty_Vol"] / repo_vol_f["DVP_Vol"]
            st.plotly_chart(px.line(ratio, x="date", y="TRI_DVP_Ratio"), use_container_width=True, key="tri_dvp_ratio")
        else:
            st.info("Need both Tri-party and DVP to compute ratio.")

    with c4:
        st.subheader("GCF Share of Total (%)")
        if {"Triparty_Vol", "DVP_Vol", "GCF_Vol"}.issubset(repo_vol_f.columns):
            tot = repo_vol_f[["Triparty_Vol", "DVP_Vol", "GCF_Vol"]].sum(axis=1)
            share = repo_vol_f[["date"]].copy()
            share["GCF_Share_%"] = (repo_vol_f["GCF_Vol"] / tot) * 100
            st.plotly_chart(px.line(share, x="date", y="GCF_Share_%"), use_container_width=True, key="gcf_share")
        else:
            st.info("Need all three to compute GCF share.")

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("ON RRP (Daily, $Bn)")
        if rrp_f is not None and not rrp_f.empty:
            st.plotly_chart(px.line(rrp_f, x="date", y="ON_RRP_Bn"), use_container_width=True, key="on_rrp")
        else:
            st.info("No RRP data in the selected window.")
    with c2:
        st.subheader("MMF Total Financial Assets (Quarterly, $Bn)")
        if mmf_f is not None and not mmf_f.empty:
            st.plotly_chart(px.line(mmf_f, x="date", y="MMF_Total_$Bn"), use_container_width=True, key="mmf_assets")
        else:
            st.info("MMF series is empty.")

st.markdown("---")
st.caption(
    "Sources: OFR Short-Term Funding Monitor (discover mnemonics via /metadata/search, then /series/timeseries), "
    "FRED (RRPONTSYD; MMMFFAQ027S)."
)
