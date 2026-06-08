import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ═══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="WellPerf Pro — IPR/TPR Analyser",
    page_icon="⛽",
    layout="wide"
)

# ═══════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #070b13; color: #e2e8f0; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #1e293b;
    }

    /* Number inputs */
    input[type="number"] {
        background-color: #0a0f1a !important;
        color: #e2e8f0 !important;
        border: 1px solid #1e293b !important;
        border-radius: 5px !important;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #0d1117;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 12px 16px;
    }

    /* Headings */
    h1, h2, h3 { color: #f8fafc !important; }

    /* Divider */
    hr { border-color: #1e293b; }

    /* Radio buttons */
    .stRadio label { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  ENGINEERING FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def z_factor(P_psia, T_R, sg_gas):
    """Papay (1985) z-factor correlation."""
    Ppc = 677 + 15 * sg_gas - 37.5 * sg_gas**2
    Tpc = 168 + 325 * sg_gas - 12.5 * sg_gas**2
    Ppr = P_psia / Ppc
    Tpr = T_R   / Tpc
    z   = (1
           - (3.52 * Ppr) / (10 ** (0.9813 * Tpr))
           + (0.274 * Ppr**2) / (10 ** (0.8157 * Tpr)))
    return float(np.clip(z, 0.15, 1.5))


def vogel_qmax(Pr, q_test, Pwf_test):
    """AOF from Vogel (1968) using a single test point."""
    r = Pwf_test / Pr
    F = 1 - 0.2 * r - 0.8 * r**2
    if F <= 0:
        raise ValueError("Invalid test point — Pwf_test must be less than Pr.")
    return q_test / F


def vogel_pwf(q_arr, Pr, qmax):
    """
    Vogel inverse — Pwf given q array.
    Solves: 0.8r² + 0.2r − (1 − q/qmax) = 0
    """
    ratio = np.clip(q_arr / qmax, 0, 1)
    disc  = np.maximum(0, 0.04 + 3.2 * (1 - ratio))
    r     = (-0.2 + np.sqrt(disc)) / 1.6
    return np.maximum(0, r * Pr)


def linear_pwf(q_arr, Pr, J):
    """Linear (Darcy) IPR — Pwf given q array."""
    return np.maximum(0, Pr - q_arr / J)


def composite_pwf(q_arr, Pr, J, Pb):
    """
    Composite (Generalised Vogel) IPR — Pwf given q array.
    Linear above the bubble point, Vogel below it.
    """
    Pb = min(Pb, Pr)
    qb = J * (Pr - Pb)               # rate at the bubble point
    qv_max = J * Pb / 1.8            # Vogel contribution below Pb
    out = np.empty_like(q_arr, dtype=float)
    for i, q in enumerate(q_arr):
        if q <= qb:
            out[i] = max(0.0, Pr - q / J)
        else:
            qv = q - qb
            ratio = np.clip(qv / qv_max, 0, 1)
            r = (-0.2 + np.sqrt(max(0.0, 0.04 + 3.2 * (1 - ratio)))) / 1.6
            out[i] = max(0.0, r * Pb)
    return out


def composite_aof(Pr, J, Pb):
    """AOF for the composite IPR (q at Pwf = 0)."""
    Pb = min(Pb, Pr)
    return J * (Pr - Pb) + J * Pb / 1.8


def tpr_pwf_single(q, Pwh, H, d_in, WC, GOR,
                   API, SG_gas, SG_water, Bo,
                   T_res, T_surf, Pr):
    """
    TPR — Pwf for a single flow rate q (STB/d).
    Simplified multiphase: hydrostatic + Darcy-Weisbach friction (Blasius).
    """
    sg_oil  = 141.5 / (API + 131.5)
    T_avg_R = (T_res + T_surf) / 2 + 460    # °Rankine
    P_avg   = (Pwh + Pr) / 2                # average tubing pressure (psia)

    # Gas FVF
    z  = z_factor(P_avg, T_avg_R, SG_gas)
    Bg = 0.00504 * z * T_avg_R / P_avg      # res bbl/scf

    # In-situ densities (lb/ft³)
    rho_oil   = 62.4 * sg_oil   / Bo
    rho_water = 62.4 * SG_water
    rho_gas   = (SG_gas * 28.97 * P_avg) / (10.73 * T_avg_R)
    rho_L     = rho_oil * (1 - WC) + rho_water * WC

    # Zero rate — pure liquid static gradient
    if q <= 0:
        return Pwh + rho_L * H / 144

    # Volumetric reservoir rates (res bbl/d)
    q_oil   = q * (1 - WC)
    qL_res  = q_oil * Bo + q * WC
    qG_res  = q_oil * GOR * Bg
    q_tot   = qL_res + qG_res

    # Flow velocity (ft/s)
    d_ft = d_in / 12
    A    = np.pi / 4 * d_ft**2
    v    = (q_tot * 5.615) / (86400 * A)

    # No-slip liquid fraction, then a velocity-dependent slip holdup:
    # at low velocity gas slips and the column stays liquid-heavy (high HL),
    # at high velocity HL approaches the no-slip value. This produces the
    # characteristic J-shaped TPR (gravity-dominated low, friction high).
    lam     = qL_res / (q_tot + 1e-10)
    HL      = lam + (1 - lam) * np.exp(-0.55 * v)
    rho_ns  = rho_L * lam + rho_gas * (1 - lam)      # no-slip (friction term)
    rho_mix = rho_L * HL  + rho_gas * (1 - HL)       # slip (gravity term)

    # Swamee-Jain friction factor
    mu_cp = 1.5
    Re    = max(2300, (rho_ns * v * d_ft) / (mu_cp * 6.72e-4))
    eD    = 0.0006 / d_ft
    f     = 0.25 / (np.log10(eD / 3.7 + 5.74 / Re**0.9)) ** 2

    # Pressure drops (psi)
    dP_hyd  = rho_mix * H / 144
    dP_fric = (f * rho_ns * v**2 * H) / (2 * 32.174 * d_ft * 144)

    return Pwh + dP_hyd + dP_fric


def find_operating_point(q_arr, ipr_arr, tpr_arr):
    """
    Locate operating point via sign-change interpolation
    of (IPR − TPR).
    """
    diff = ipr_arr - tpr_arr
    for i in range(len(diff) - 1):
        if diff[i] > 0 and diff[i + 1] <= 0:
            # Linear interpolation
            frac = diff[i] / (diff[i] - diff[i + 1])
            op_q   = q_arr[i]   + frac * (q_arr[i + 1]   - q_arr[i])
            op_pwf = ipr_arr[i] + frac * (ipr_arr[i + 1] - ipr_arr[i])
            return op_q, op_pwf
    return None, None


def gilbert_choke_pwh(q_arr, GLR_Mscf, S_64):
    """
    Gilbert (1954) surface-choke correlation — wellhead pressure
    (psi) required for CRITICAL (sonic) flow through a bean:

        P_wh = 435 · R^0.546 · q / S^1.89

    R = GLR (Mscf/STB), q = liquid rate (STB/d), S = bean size (1/64 in).
    """
    R = max(GLR_Mscf, 1e-6)
    return (435.0 * R ** 0.546 * np.asarray(q_arr, dtype=float)) / (S_64 ** 1.89)


def find_wellhead_point(q_arr, pwh_avail, pwh_choke):
    """
    Wellhead-node operating point: where available wellhead pressure
    (reservoir through tubing) meets the choke demand curve.
    """
    diff = pwh_avail - pwh_choke
    for i in range(len(diff) - 1):
        if diff[i] > 0 and diff[i + 1] <= 0:
            frac = diff[i] / (diff[i] - diff[i + 1])
            op_q   = q_arr[i]    + frac * (q_arr[i + 1]    - q_arr[i])
            op_pwh = pwh_choke[i] + frac * (pwh_choke[i + 1] - pwh_choke[i])
            return op_q, op_pwh
    return None, None


# ═══════════════════════════════════════════════════════════════
#  VERTICAL LIFT PERFORMANCE — Gilbert pressure-traverse method
#  Multiphase gradient integrated down/up the tubing (Beggs-Brill
#  holdup + Jain friction). This is the computational equivalent of
#  reading the Gilbert pressure-distribution curve by "equivalent
#  depth": Method 1 (THP -> Pwf) and Method 2 (Pwf -> THP).
# ═══════════════════════════════════════════════════════════════

def jain_friction(Re, eD):
    """Jain (1976) explicit friction factor:
        1/sqrt(f) = 1.14 - 2 log(e/D + 21.25/Re^0.9)"""
    Re = max(Re, 1.0e3)
    inv = 1.14 - 2.0 * np.log10(eD + 21.25 / Re ** 0.9)
    return 1.0 / inv ** 2


def bb_gradient(P, T_R, q, vp):
    """Beggs & Brill (1973) two-phase pressure gradient (psi/ft) for
    upward vertical flow, acceleration term neglected (sinθ = 1)."""
    d  = vp["d_in"] / 12.0
    A  = np.pi / 4.0 * d * d
    sg_oil = 141.5 / (vp["API"] + 131.5)
    rho_L  = 62.4 * (sg_oil * (1 - vp["WC"]) / vp["Bo"] + vp["SG_water"] * vp["WC"])

    z   = z_factor(P, T_R, vp["SG_gas"])
    Bg  = 0.0283 * z * T_R / P                      # ft³/scf
    rho_g = vp["SG_gas"] * 28.97 * P / (10.732 * T_R * z)

    q_oil = q * (1 - vp["WC"])
    qL = (q_oil * vp["Bo"] + q * vp["WC"]) * 5.615 / 86400.0   # ft³/s
    qg = q_oil * vp["GOR"] * Bg / 86400.0                      # ft³/s
    qt = qL + qg
    if qt <= 0:
        return rho_L / 144.0

    vsL = qL / A; vsg = qg / A; vm = vsL + vsg
    lamL = min(max(vsL / vm, 1e-6), 1.0)
    Nfr  = vm * vm / (32.174 * d)
    NLv  = 1.938 * vsL * (rho_L / max(vp["sigma"], 1e-3)) ** 0.25

    L1 = 316 * lamL ** 0.302
    L2 = 0.0009252 * lamL ** (-2.4684)
    L3 = 0.10 * lamL ** (-1.4516)
    L4 = 0.5 * lamL ** (-6.738)

    if (lamL < 0.01 and Nfr < L1) or (lamL >= 0.01 and Nfr < L2):
        regime = "seg"
    elif lamL >= 0.01 and L2 <= Nfr <= L3:
        regime = "trans"
    elif (0.01 <= lamL < 0.4 and L3 < Nfr <= L1) or (lamL >= 0.4 and L3 < Nfr <= L4):
        regime = "int"
    else:
        regime = "dist"

    def HL0(rg):
        a, b, c = {"seg": (0.98, 0.4846, 0.0868),
                   "int": (0.845, 0.5351, 0.0173),
                   "dist": (1.065, 0.5824, 0.0609)}[rg]
        return max(a * lamL ** b / Nfr ** c, lamL)

    def HL_incl(rg, h0):
        if rg == "dist":
            return min(max(h0, lamL), 1.0)
        e, f, g_, h = {"seg": (0.011, -3.768, 3.539, -1.614),
                       "int": (2.96, 0.305, -0.4473, 0.0978)}[rg]
        C = (1 - lamL) * np.log(max(e * lamL ** f * NLv ** g_ * Nfr ** h, 1e-9))
        C = max(C, 0.0)
        th = np.radians(90.0)
        psi = 1 + C * (np.sin(1.8 * th) - 0.333 * np.sin(1.8 * th) ** 3)
        return min(max(h0 * psi, lamL), 1.0)

    if regime == "trans":
        Ai = (L3 - Nfr) / (L3 - L2)
        HL = Ai * HL_incl("seg", HL0("seg")) + (1 - Ai) * HL_incl("int", HL0("int"))
    else:
        HL = HL_incl(regime, HL0(regime))
    HL = min(max(HL, lamL), 1.0)

    rho_s = rho_L * HL + rho_g * (1 - HL)
    rho_n = rho_L * lamL + rho_g * (1 - lamL)
    mu_n  = vp["muL"] * lamL + vp["mug"] * (1 - lamL)
    Re    = 1488 * rho_n * vm * d / max(mu_n, 1e-4)
    fn    = jain_friction(Re, vp["eD"])

    y = lamL / HL ** 2
    if 1.0 < y < 1.2:
        S = np.log(2.2 * y - 1.2)
    else:
        ly = np.log(max(y, 1e-6))
        S = ly / (-0.0523 + 3.182 * ly - 0.8725 * ly ** 2 + 0.01853 * ly ** 4)
    ftp = fn * np.exp(np.clip(S, -5, 5))

    grav = rho_s / 144.0
    fric = ftp * rho_n * vm * vm / (2 * 32.174 * d) / 144.0
    return grav + fric


def integrate_traverse(P0, q, vp, direction, n_steps=80, record=False):
    """March the pressure gradient along the tubing.
    direction +1 = downward (THP→Pwf, Method 1),
    direction -1 = upward   (Pwf→THP, Method 2)."""
    dL = vp["depth"] / n_steps
    P  = float(P0)
    T_top = vp["Tsurf"] + 460.0
    T_bot = vp["Tres"] + 460.0
    depths, pres = [0.0 if direction > 0 else vp["depth"]], [P]
    for i in range(n_steps):
        L = (i + 0.5) * dL if direction > 0 else vp["depth"] - (i + 0.5) * dL
        T_R = T_top + (T_bot - T_top) * (L / vp["depth"])
        g = bb_gradient(max(P, 20.0), T_R, q, vp)
        P = max(P + direction * g * dL, 15.0)
        if record:
            md = (i + 1) * dL if direction > 0 else vp["depth"] - (i + 1) * dL
            depths.append(md); pres.append(P)
    if record:
        return np.array(depths), np.array(pres)
    return P


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR — INPUT PARAMETERS
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⛽ WellPerf Pro")
    st.markdown("*IPR / TPR Well Performance Analyser*")
    st.divider()

    # ── IPR Model ──────────────────────────────────────────────
    st.markdown("### 📈 IPR Model")
    ipr_model = st.radio(
        "Select IPR Method",
        ["Vogel", "Composite", "Linear"],
        horizontal=True,
        label_visibility="collapsed",
        help="Vogel: saturated reservoir. Composite: linear above Pb, Vogel below. Linear: Darcy PI.",
    )

    Pr = st.number_input("Reservoir Pressure, Pr (psia)", value=3000, min_value=100, step=50)

    if ipr_model == "Vogel":
        q_test   = st.number_input("Test Flow Rate, q_test (STB/d)", value=500, min_value=1, step=10)
        Pwf_test = st.number_input("Test FBHP, Pwf_test (psia)",     value=1800, min_value=0, step=50)
    elif ipr_model == "Composite":
        J  = st.number_input("Productivity Index, J (STB/d/psi)", value=1.50, min_value=0.001, step=0.05, format="%.3f")
        Pb = st.number_input("Bubble-Point Pressure, Pb (psia)",  value=2500, min_value=50,    step=50)
    else:
        J = st.number_input("Productivity Index, J (STB/d/psi)", value=0.40, min_value=0.001, step=0.01, format="%.3f")

    st.divider()

    # ── Fluid Properties ───────────────────────────────────────
    st.markdown("### 🛢 Fluid Properties")
    API      = st.number_input("Oil Gravity (°API)",         value=35.0, min_value=10.0, max_value=60.0, step=1.0)
    Bo       = st.number_input("Oil FVF, Bo (res bbl/STB)",  value=1.20, min_value=1.00, step=0.01, format="%.3f")
    GOR      = st.number_input("Solution GOR (scf/STB)",     value=800,  min_value=0,    step=10)
    WC       = st.number_input("Water Cut, WC (0 – 1)",      value=0.10, min_value=0.0,  max_value=0.99, step=0.01, format="%.2f")
    SG_gas   = st.number_input("Gas Specific Gravity (air=1)",  value=0.65, min_value=0.50, step=0.01, format="%.3f")
    SG_water = st.number_input("Water Specific Gravity",        value=1.05, min_value=1.00, step=0.01, format="%.3f")

    st.divider()

    # ── Well & Tubing ──────────────────────────────────────────
    st.markdown("### 🔩 Well & Tubing")
    H     = st.number_input("Well Depth (ft)",                value=6500, min_value=100, step=100)
    d_in  = st.number_input("Tubing Inner Diameter (inches)", value=2.441, min_value=0.5, step=0.001, format="%.3f")
    Pwh   = st.number_input("Wellhead Pressure (psia)",       value=150,  min_value=15,  step=10)
    T_res = st.number_input("Reservoir Temperature (°F)",     value=180,  min_value=60,  step=5)
    T_surf= st.number_input("Surface Temperature (°F)",       value=80,   min_value=32,  step=5)

    st.divider()

    # ── Surface Choke (Gilbert) ────────────────────────────────
    st.markdown("### 🎚 Surface Choke (Gilbert)")
    enable_choke = st.checkbox("Enable wellhead choke analysis", value=True)
    if enable_choke:
        choke_S = st.number_input("Choke / Bean Size (1/64 in)", value=32, min_value=4, max_value=128, step=2)

    st.divider()
    with st.expander("🌀 VLP / Multiphase properties"):
        muL   = st.number_input("Liquid viscosity, μL (cP)", value=2.0,  min_value=0.1, step=0.1, format="%.2f")
        mug   = st.number_input("Gas viscosity, μg (cP)",     value=0.018, min_value=0.001, step=0.001, format="%.3f")
        sigma = st.number_input("Liquid surface tension (dyne/cm)", value=30.0, min_value=1.0, step=1.0)
        rough = st.number_input("Tubing roughness, e (ft)",   value=0.00006, min_value=0.0, step=0.00001, format="%.5f")

    st.divider()
    st.markdown("🟢 **Live** — results update as you edit.")


# ═══════════════════════════════════════════════════════════════
#  MAIN PAGE
# ═══════════════════════════════════════════════════════════════

st.markdown("# ⛽ WellPerf Pro")
st.markdown("**IPR / TPR Well Performance Analyser** — SETP 3513 · Universiti Teknologi Malaysia")
st.divider()

# ── Calculations ───────────────────────────────────────────────

try:
    # IPR
    if ipr_model == "Vogel":
        qmax = vogel_qmax(Pr, q_test, Pwf_test)
    elif ipr_model == "Composite":
        qmax = composite_aof(Pr, J, Pb)
    else:
        qmax = J * Pr

    N       = 200
    q_arr   = np.linspace(0, qmax, N)

    if ipr_model == "Vogel":
        ipr_arr = vogel_pwf(q_arr, Pr, qmax)
    elif ipr_model == "Composite":
        ipr_arr = composite_pwf(q_arr, Pr, J, Pb)
    else:
        ipr_arr = linear_pwf(q_arr, Pr, J)

    # TPR (vectorised via list comprehension)
    tpr_arr = np.array([
        tpr_pwf_single(q, Pwh, H, d_in, WC, GOR,
                       API, SG_gas, SG_water, Bo,
                       T_res, T_surf, Pr)
        for q in q_arr
    ])

    # Operating point
    op_q, op_pwf = find_operating_point(q_arr, ipr_arr, tpr_arr)

except Exception as e:
    st.error(f"Calculation error: {e}")
    st.stop()

# build VLP parameter dict (shared multiphase + well props)
vp = dict(d_in=d_in, API=API, WC=WC, Bo=Bo, SG_water=SG_water, SG_gas=SG_gas,
          GOR=GOR, Tres=T_res, Tsurf=T_surf, depth=H, muL=muL, mug=mug,
          sigma=sigma, eD=rough / (d_in / 12.0))


# ── Generic IPR helpers (for tubing-size & well-life sweeps) ───────
JE = (1.8 * qmax / Pr) if ipr_model == "Vogel" else J

def ipr_q(q, Pr_f, Jv=None):
    Jv = JE if Jv is None else Jv
    if ipr_model == "Linear":
        return max(0.0, Pr_f - q / Jv)
    if ipr_model == "Vogel":
        qm = Jv * Pr_f / 1.8
        if q >= qm: return 0.0
        x = (-0.2 + np.sqrt(max(0.04 + 3.2 * (1 - q / qm), 0))) / 1.6
        return max(0.0, x * Pr_f)
    pb = min(Pb, Pr_f); qb = Jv * (Pr_f - pb); qm = Jv * pb / 1.8
    if q <= qb: return max(0.0, Pr_f - q / Jv)
    qv = q - qb
    if qv >= qm: return 0.0
    x = (-0.2 + np.sqrt(max(0.04 + 3.2 * (1 - qv / qm), 0))) / 1.6
    return max(0.0, x * pb)

def ipr_aof(Pr_f, Jv=None):
    Jv = JE if Jv is None else Jv
    if ipr_model == "Linear": return Jv * Pr_f
    if ipr_model == "Vogel":  return Jv * Pr_f / 1.8
    pb = min(Pb, Pr_f); return Jv * (Pr_f - pb) + Jv * pb / 1.8

def vp_with_d(dval):
    v = dict(vp); v["d_in"] = dval; v["eD"] = rough / (dval / 12.0); return v

def crossing(xg, ya, yb, want_down=True):
    """first q where (ya-yb) changes sign; ya,yb arrays over xg."""
    d = np.asarray(ya) - np.asarray(yb)
    for i in range(len(d) - 1):
        if (want_down and d[i] > 0 and d[i+1] <= 0) or (not want_down and d[i]*d[i+1] < 0):
            fr = d[i] / (d[i] - d[i+1])
            return xg[i] + fr * (xg[i+1] - xg[i]), i, fr
    return None, None, None

GLR_Mscf = (1 - WC) * GOR / 1000.0
DARK = dict(paper_bgcolor="#0d1117", plot_bgcolor="#070b13",
            font=dict(color="#cbd5e1"),
            legend=dict(font=dict(color="#94a3b8"), bgcolor="#0d1117",
                        bordercolor="#1e293b", borderwidth=1))
def axes(fig, xt, yt, y2t=None):
    fig.update_layout(**DARK, height=480, hovermode="x unified",
        xaxis=dict(title=xt, color="#94a3b8", gridcolor="#1e293b", zerolinecolor="#334155"),
        yaxis=dict(title=yt, color="#94a3b8", gridcolor="#1e293b", zerolinecolor="#334155"))
    if y2t:
        fig.update_layout(yaxis2=dict(title=y2t, overlaying="y", side="right",
                                      color="#f5a623", showgrid=False))
    return fig

tab1, tab2, tab3, tab4 = st.tabs([
    "🎚  Nodal + Choke (Method 2)",
    "📐  Optimum Tubing Size",
    "✅  Well Producibility",
    "⏳  Well Life",
])

# ════════════════════════════════════════════════════════════════
#  TAB 1 — Wellhead-node nodal analysis: IPR · VLP (Method 2) · Choke
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Nodal analysis at the wellhead node")
    st.caption("VLP **Method 2** (slide 56): for each q the IPR gives P_wf, then the multiphase "
               "gradient is marched **up** the tubing to the available wellhead pressure P_wh. "
               "The Gilbert choke demand (P_wh vs q) is overlaid — their intersection is the "
               "operating rate for each bean size.")
    try:
        qg = np.linspace(max(qmax * 0.02, 10.0), qmax, 30)
        ipr_g  = np.array([ipr_q(q, Pr) for q in qg])
        availP = np.array([integrate_traverse(max(ipr_g[i], 20.0), qg[i], vp, -1, 60)
                           for i in range(len(qg))])

        beans = sorted(set([16, 20, 24, 32, 48, 64, int(choke_S)])) if enable_choke else []
        rows = []
        for b in beans:
            ck = gilbert_choke_pwh(qg, GLR_Mscf, b)
            qb, _, _ = crossing(qg, availP, ck, want_down=True)
            rows.append((b, qb, np.interp(qb, qg, ck) if qb else None))

        fig = go.Figure()
        # IPR on right axis (Pwf) so the IPR method is visible
        fig.add_trace(go.Scatter(x=q_arr, y=ipr_arr, name="IPR (P_wf, right axis)",
                                 yaxis="y2", line=dict(color="#f5a623", width=2, dash="dot"),
                                 hovertemplate="q=%{x:.0f}<br>Pwf=%{y:.0f} psia<extra>IPR</extra>"))
        # VLP Method 2 — available wellhead pressure
        fig.add_trace(go.Scatter(x=qg, y=availP, name="VLP Method 2 — available P_wh",
                                 line=dict(color="#3ec1a8", width=3),
                                 hovertemplate="q=%{x:.0f}<br>Pwh=%{y:.0f} psia<extra>VLP M2</extra>"))
        # choke family
        for b in beans:
            sel = (b == int(choke_S))
            fig.add_trace(go.Scatter(x=qg, y=gilbert_choke_pwh(qg, GLR_Mscf, b),
                name=f"Choke {b}/64" + (" (selected)" if sel else ""),
                line=dict(color="#f59e0b" if sel else "#475569",
                          width=3 if sel else 1.3, dash="solid" if sel else "dot"),
                hovertemplate=f"Bean {b}/64<br>"+"q=%{x:.0f}<br>Pwh=%{y:.0f} psia<extra></extra>"))
        # operating markers per bean
        for b, qb, pb_ in rows:
            if qb:
                fig.add_trace(go.Scatter(x=[qb], y=[pb_], mode="markers",
                    marker=dict(color="#4ade80", size=11,
                                symbol="diamond" if b == int(choke_S) else "circle",
                                line=dict(color="#16a34a", width=1.5)),
                    name=f"q_opt {b}/64", showlegend=False,
                    hovertemplate=f"Bean {b}/64<br>q_opt=%{{x:.0f}} STB/d<br>Pwh=%{{y:.0f}} psia<extra></extra>"))
        axes(fig, "Liquid flow rate  q  (STB/d)", "Wellhead pressure  P_wh  (psia)", "P_wf (psia)")
        fig.update_layout(title=dict(text="<b>IPR · VLP Method 2 · Choke performance</b>",
                                     font=dict(color="#f8fafc", size=16)))
        st.plotly_chart(fig, width="stretch")

        # optimum flow rate for each choke
        st.markdown("#### Optimum flow rate for each choke")
        import pandas as pd
        df = pd.DataFrame([{"Bean (1/64 in)": b,
                            "Optimum q (STB/d)": f"{qb:,.0f}" if qb else "no flow",
                            "Wellhead P_wh (psia)": f"{pb_:,.0f}" if qb else "—"}
                           for b, qb, pb_ in rows])
        st.dataframe(df, width="stretch", hide_index=True)
        sel_row = next((r for r in rows if r[0] == int(choke_S)), None)
        if sel_row and sel_row[1]:
            st.success(f"Selected bean **{int(choke_S)}/64 in** → operating rate "
                       f"**{sel_row[1]:,.0f} STB/d** at P_wh ≈ {sel_row[2]:,.0f} psia.")
    except Exception as e:
        st.error(f"Tab 1 error: {e}")

# ════════════════════════════════════════════════════════════════
#  TAB 2 — Optimum tubing size selection (slide 53/59)
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Optimum tubing size selection")
    st.caption("IPR plotted against the VLP (P_wf vs q) for several tubing sizes. "
               "Each IPR∩VLP intersection is the optimum q for that size; the size giving "
               "the **highest** optimum q is preferred (slide 53).")
    try:
        sizes = sorted(set([1.9, 2.375, 2.875, 3.5, round(float(d_in), 3)]))
        qg2 = np.linspace(max(qmax * 0.02, 10.0), qmax, 24)
        ipr_g2 = np.array([ipr_q(q, Pr) for q in qg2])
        palette = ["#3ec1a8", "#60a5fa", "#f59e0b", "#e8743b", "#a78bfa", "#f472b6"]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=q_arr, y=ipr_arr, name="IPR",
                                  line=dict(color="#ef4444", width=3),
                                  hovertemplate="q=%{x:.0f}<br>Pwf=%{y:.0f}<extra>IPR</extra>"))
        results = []
        for k, dv in enumerate(sizes):
            pv = np.array([integrate_traverse(Pwh, q, vp_with_d(dv), +1, 60) for q in qg2])
            qo, _, _ = crossing(qg2, ipr_g2, pv, want_down=True)
            results.append((dv, qo))
            c = palette[k % len(palette)]
            fig2.add_trace(go.Scatter(x=qg2, y=pv, name=f'VLP {dv}"',
                line=dict(color=c, width=2.4),
                hovertemplate=f'{dv}" tubing<br>'+"q=%{x:.0f}<br>Pwf=%{y:.0f}<extra></extra>"))
            if qo:
                fig2.add_trace(go.Scatter(x=[qo], y=[float(np.interp(qo, qg2, ipr_g2))],
                    mode="markers", showlegend=False,
                    marker=dict(color=c, size=11, line=dict(color="#fff", width=1.3)),
                    hovertemplate=f'{dv}" → q_opt=%{{x:.0f}} STB/d<extra></extra>'))
        axes(fig2, "Liquid flow rate  q  (STB/d)", "Flowing BHP  P_wf  (psia)")
        fig2.update_layout(title=dict(text="<b>IPR vs VLP for candidate tubing sizes</b>",
                                      font=dict(color="#f8fafc", size=16)))
        st.plotly_chart(fig2, width="stretch")

        import pandas as pd
        valid = [(d, q) for d, q in results if q]
        dfa = pd.DataFrame([{"Tubing ID (in)": d,
                             "Optimum q (STB/d)": f"{q:,.0f}" if q else "no flow"}
                            for d, q in results])
        st.dataframe(dfa, width="stretch", hide_index=True)
        if valid:
            best = max(valid, key=lambda t: t[1])
            st.success(f"Optimum tubing size: **{best[0]} in** → highest rate "
                       f"**{best[1]:,.0f} STB/d**.")
    except Exception as e:
        st.error(f"Tab 2 error: {e}")

# ════════════════════════════════════════════════════════════════
#  TAB 3 — Well producibility (slide 54)
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Well producibility")
    st.caption("Does the well flow, and is the operating point stable? The operating point is "
               "IPR ∩ VLP; a point on the friction-dominated (right) side of the VLP minimum is "
               "stable, while one left of the minimum is unstable / loading up (slide 54).")
    try:
        qg3 = np.linspace(max(qmax * 0.02, 10.0), qmax, 30)
        ipr_g3 = np.array([ipr_q(q, Pr) for q in qg3])
        vlp3 = np.array([integrate_traverse(Pwh, q, vp, +1, 60) for q in qg3])
        qo3, _, _ = crossing(qg3, ipr_g3, vlp3, want_down=True)
        qmin_idx = int(np.argmin(vlp3)); q_min = qg3[qmin_idx]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Operating rate", f"{qo3:,.0f} STB/d" if qo3 else "Will not flow")
        with c2:
            st.metric("VLP minimum at", f"{q_min:,.0f} STB/d")
        with c3:
            if qo3:
                st.metric("Stability", "Stable" if qo3 >= q_min else "Unstable (loading)")
            else:
                st.metric("Stability", "—")

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=q_arr, y=ipr_arr, name="IPR",
                                  line=dict(color="#ef4444", width=3)))
        fig3.add_trace(go.Scatter(x=qg3, y=vlp3, name="VLP (current tubing)",
                                  line=dict(color="#3ec1a8", width=3)))
        if qo3:
            po3 = float(np.interp(qo3, qg3, ipr_g3))
            fig3.add_trace(go.Scatter(x=[qo3], y=[po3], mode="markers+text",
                marker=dict(color="#4ade80", size=14, line=dict(color="#16a34a", width=2)),
                text=[f"  {qo3:,.0f} STB/d"], textposition="middle right",
                textfont=dict(color="#4ade80"), name="Operating point"))
        axes(fig3, "Liquid flow rate  q  (STB/d)", "Flowing BHP  P_wf  (psia)")
        fig3.update_layout(title=dict(text="<b>Producibility — IPR ∩ VLP</b>",
                                      font=dict(color="#f8fafc", size=16)))
        st.plotly_chart(fig3, width="stretch")
        if qo3 and qo3 >= q_min:
            st.success(f"The well **produces** at a stable {qo3:,.0f} STB/d.")
        elif qo3:
            st.warning(f"Intersection at {qo3:,.0f} STB/d is **left of the VLP minimum** — "
                       "unstable, the well is prone to liquid loading.")
        else:
            st.error("No IPR∩VLP intersection — the well will **not flow naturally** "
                     "(needs artificial lift).")
    except Exception as e:
        st.error(f"Tab 3 error: {e}")

# ════════════════════════════════════════════════════════════════
#  TAB 4 — Well life determination (slide 55/61)
# ════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Well life determination")
    st.caption("As the reservoir depletes, P_r falls and the IPR shrinks toward the VLP. "
               "When a future IPR just touches the VLP at one point, the well dies. "
               "The chart shows present + future IPRs against the fixed VLP (slide 55).")
    try:
        qg4 = np.linspace(max(qmax * 0.02, 10.0), max(qmax, 50.0), 30)
        vlp4 = np.array([integrate_traverse(Pwh, q, vp, +1, 60) for q in qg4])

        Pr_fracs = [1.0, 0.85, 0.7, 0.55, 0.4]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=qg4, y=vlp4, name="VLP (fixed)",
                                  line=dict(color="#3ec1a8", width=3)))
        cols = ["#ef4444", "#f59e0b", "#a78bfa", "#60a5fa", "#94a3b8"]
        for k, fr in enumerate(Pr_fracs):
            Prf = Pr * fr
            iprf = np.array([ipr_q(q, Prf) for q in qg4])
            fig4.add_trace(go.Scatter(x=qg4, y=iprf, name=f"IPR @ P_r={Prf:,.0f}",
                line=dict(color=cols[k], width=2, dash="solid" if fr == 1 else "dash")))

        # find P_r at which the well dies (VLP tangent / last intersection)
        Pr_dead = None; q_dead = None
        for frac in np.linspace(1.0, 0.15, 60):
            Prf = Pr * frac
            iprf = np.array([ipr_q(q, Prf) for q in qg4])
            qo, _, _ = crossing(qg4, iprf, vlp4, want_down=True)
            if qo is None:
                break
            Pr_dead, q_dead = Prf, qo
        axes(fig4, "Liquid flow rate  q  (STB/d)", "Flowing BHP  P_wf  (psia)")
        fig4.update_layout(title=dict(text="<b>Well life — present & future IPR vs VLP</b>",
                                      font=dict(color="#f8fafc", size=16)))
        st.plotly_chart(fig4, width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Abandonment P_r (well dies near)",
                      f"{Pr_dead:,.0f} psia" if Pr_dead else "—",
                      help="Lowest reservoir pressure that still sustains flow.")
        with c2:
            st.metric("Rate at that P_r", f"{q_dead:,.0f} STB/d" if q_dead else "—")
        if Pr_dead:
            st.info(f"The well sustains flow down to about **P_r ≈ {Pr_dead:,.0f} psia** "
                    f"(~{Pr_dead/Pr*100:.0f}% of current). Below that the IPR no longer "
                    "reaches the VLP and the well dies.")
    except Exception as e:
        st.error(f"Tab 4 error: {e}")
