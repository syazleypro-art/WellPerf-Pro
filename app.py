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

tab1, tab2 = st.tabs(["📊  Nodal Analysis (IPR vs TPR)",
                      "🌀  Vertical Lift Performance — Gilbert Method"])

with tab1:
    # ── Result Cards ───────────────────────────────────────────────

    drawdown     = round(Pr - op_pwf, 0) if op_pwf else None
    drawdown_pct = round((drawdown / Pr) * 100, 1) if drawdown else None

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            label="AOF  (q_max)",
            value=f"{int(round(qmax)):,} STB/d",
            help="Absolute Open Flow — maximum possible rate if Pwf = 0"
        )
    with c2:
        if op_q:
            st.metric(
                label="Operating Rate",
                value=f"{int(round(op_q)):,} STB/d",
                delta=f"{round(op_q/qmax*100, 1)}% of AOF"
            )
        else:
            st.metric(label="Operating Rate", value="No Intersection")
    with c3:
        if op_pwf:
            st.metric(
                label="Operating FBHP",
                value=f"{int(round(op_pwf)):,} psia",
                delta=f"Drawdown: {int(drawdown):,} psi ({drawdown_pct}%)",
                delta_color="inverse"
            )
        else:
            st.metric(label="Operating FBHP", value="—")
    with c4:
        st.metric(label="IPR Method", value=ipr_model)

    st.divider()

    # ── Chart ──────────────────────────────────────────────────────

    fig = go.Figure()

    # IPR curve
    fig.add_trace(go.Scatter(
        x=q_arr, y=ipr_arr,
        mode="lines",
        name="IPR Curve",
        line=dict(color="#60a5fa", width=3),
        hovertemplate="q = %{x:.0f} STB/d<br>Pwf = %{y:.0f} psia<extra>IPR</extra>"
    ))

    # TPR curve
    fig.add_trace(go.Scatter(
        x=q_arr, y=tpr_arr,
        mode="lines",
        name="TPR Curve",
        line=dict(color="#f59e0b", width=3),
        hovertemplate="q = %{x:.0f} STB/d<br>Pwf = %{y:.0f} psia<extra>TPR</extra>"
    ))

    # ── VLP (Beggs-Brill) + Choke performance on a coarse grid ──
    q_co   = np.linspace(max(qmax * 0.02, 10.0), qmax, 32)
    ipr_co = np.interp(q_co, q_arr, ipr_arr)
    pwf_vlp = np.array([integrate_traverse(Pwh, q, vp, +1, 60) for q in q_co])
    fig.add_trace(go.Scatter(
        x=q_co, y=pwf_vlp, mode="lines", name="VLP (Beggs-Brill)",
        line=dict(color="#3ec1a8", width=3),
        hovertemplate="q = %{x:.0f} STB/d<br>Pwf = %{y:.0f} psia<extra>VLP</extra>"
    ))

    op_ck_q = op_ck_pwf = None
    if enable_choke:
        GLR_Mscf = (1 - WC) * GOR / 1000.0
        pwh_ck = gilbert_choke_pwh(q_co, GLR_Mscf, choke_S)
        pwf_ck = np.array([integrate_traverse(max(pwh_ck[i], 20.0), q_co[i], vp, +1, 60)
                           for i in range(len(q_co))])
        fig.add_trace(go.Scatter(
            x=q_co, y=pwf_ck, mode="lines",
            name=f"Choke {int(choke_S)}/64 in (at BH)",
            line=dict(color="#e8743b", width=2.5, dash="dot"),
            hovertemplate="q = %{x:.0f} STB/d<br>Pwf = %{y:.0f} psia<extra>Choke</extra>"
        ))
        dc = ipr_co - pwf_ck
        for i in range(len(dc) - 1):
            if dc[i] > 0 and dc[i + 1] <= 0:
                fr = dc[i] / (dc[i] - dc[i + 1])
                op_ck_q   = q_co[i]   + fr * (q_co[i + 1]   - q_co[i])
                op_ck_pwf = ipr_co[i] + fr * (ipr_co[i + 1] - ipr_co[i])
                break
        if op_ck_q:
            fig.add_trace(go.Scatter(
                x=[op_ck_q], y=[op_ck_pwf], mode="markers",
                name="Choked operating point",
                marker=dict(color="#e8743b", size=13, symbol="diamond",
                            line=dict(color="#ffffff", width=1.5)),
                hovertemplate=(f"Choked OP<br>q = {op_ck_q:,.0f} STB/d<br>"
                               f"Pwf = {op_ck_pwf:,.0f} psia<extra></extra>")
            ))

    # Operating point
    if op_q and op_pwf:
        fig.add_trace(go.Scatter(
            x=[op_q], y=[op_pwf],
            mode="markers+text",
            name="Operating Point",
            marker=dict(color="#4ade80", size=14, symbol="circle",
                        line=dict(color="#16a34a", width=2)),
            text=[f"  ({int(round(op_q)):,} STB/d, {int(round(op_pwf)):,} psia)"],
            textposition="middle right",
            textfont=dict(color="#4ade80", size=11),
            hovertemplate=(
                f"Operating Point<br>"
                f"q = {int(round(op_q)):,} STB/d<br>"
                f"Pwf = {int(round(op_pwf)):,} psia"
                "<extra></extra>"
            )
        ))

    # Layout
    fig.update_layout(
        title=dict(
            text="<b>Well Performance — IPR · TPR · VLP · Choke</b>",
            font=dict(color="#f8fafc", size=16)
        ),
        xaxis=dict(
            title="Liquid Flow Rate  q  (STB/d)",
            color="#94a3b8",
            gridcolor="#1e293b",
            zerolinecolor="#334155",
            title_font=dict(color="#94a3b8")
        ),
        yaxis=dict(
            title="Flowing BHP  Pwf  (psia)",
            color="#94a3b8",
            gridcolor="#1e293b",
            zerolinecolor="#334155",
            title_font=dict(color="#94a3b8")
        ),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#070b13",
        legend=dict(
            font=dict(color="#94a3b8"),
            bgcolor="#0d1117",
            bordercolor="#1e293b",
            borderwidth=1
        ),
        hovermode="x unified",
        height=480
    )

    st.plotly_chart(fig, width="stretch")

    # ── Surface Choke / Wellhead-Node Analysis (Gilbert) ───────────

    if enable_choke:
        st.divider()
        st.markdown("#### 🎚 Choke performance (Gilbert)")
        GLR_Mscf = (1 - WC) * GOR / 1000.0
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Choked operating rate",
                      f"{op_ck_q:,.0f} STB/d" if op_ck_q else "No intersection",
                      help="IPR ∩ choke-performance curve (referred to bottomhole).")
        with k2:
            st.metric("Choked P_wf", f"{op_ck_pwf:,.0f} psia" if op_ck_pwf else "—")
        with k3:
            st.metric("Bean size", f"{int(choke_S)}/64 in",
                      help=f"GLR used: {GLR_Mscf:.3f} Mscf/STB")
        st.caption("The choke curve on the chart is the Gilbert wellhead demand "
                   "(P_wh = 435·R^0.546·q / S^1.89) carried down the tubing to bottomhole. "
                   "Its intersection with the IPR is the throttled operating point — a smaller "
                   "bean lifts the curve and lowers the rate.")

    # ── Equations Reference ────────────────────────────────────────

    st.divider()
    st.markdown("### ∑ Governing Equations")

    e1, e2, e3 = st.columns(3)

    with e1:
        st.markdown("**Vogel IPR (1968)**")
        st.latex(r"\frac{q}{q_{max}} = 1 - 0.2\left(\frac{P_{wf}}{P_r}\right) - 0.8\left(\frac{P_{wf}}{P_r}\right)^2")
        st.markdown("**Composite IPR** *(linear above $P_b$, Vogel below)*")
        st.latex(r"q = J(P_r - P_b) + \frac{J\,P_b}{1.8}\left[1 - 0.2\tfrac{P_{wf}}{P_b} - 0.8\left(\tfrac{P_{wf}}{P_b}\right)^2\right]")
        st.markdown("**Linear IPR (Darcy)**")
        st.latex(r"q = J \times (P_r - P_{wf})")

    with e2:
        st.markdown("**TPR — Hydrostatic**")
        st.latex(r"\Delta P_{hyd} = \frac{\rho_{slip} \cdot H}{144} \quad \text{[psi]}")
        st.markdown("**TPR — Friction (Darcy-Weisbach)**")
        st.latex(r"\Delta P_f = \frac{f \cdot \rho_{ns} \cdot v^2 \cdot H}{2 \cdot g_c \cdot d \cdot 144} \quad \text{[psi]}")
        st.markdown("**Friction factor (Swamee-Jain)**")
        st.latex(r"f = \frac{0.25}{\left[\log_{10}\!\left(\frac{\varepsilon/d}{3.7} + \frac{5.74}{Re^{0.9}}\right)\right]^2}")

    with e3:
        st.markdown("**Gilbert Choke (1954)**")
        st.latex(r"P_{wh} = \frac{435 \cdot R^{0.546} \cdot q_L}{S^{1.89}} \quad \text{[psi]}")
        st.markdown("**Gas FVF**")
        st.latex(r"B_g = \frac{0.00504 \cdot z \cdot T}{P} \quad \text{[res bbl/scf]}")
        st.markdown("**z-factor (Papay, 1985)**")
        st.latex(r"z = 1 - \frac{3.52 \, P_{pr}}{10^{0.9813 T_{pr}}} + \frac{0.274 \, P_{pr}^2}{10^{0.8157 T_{pr}}}")

    st.divider()
    st.caption("TPR uses a simplified multiphase column — velocity-dependent liquid holdup (slip) for the "
               "hydrostatic head and Swamee-Jain friction. For rigorous design, apply Hagedorn-Brown or "
               "Beggs-Brill correlations. | SETP 3513 · UTM")



with tab2:
    import plotly.graph_objects as go

    st.markdown("### Vertical Lift Performance — Gilbert pressure-traverse method")
    st.caption("VLP curves built by integrating the multiphase pressure gradient "
               "(Beggs-Brill holdup · Jain friction) along the tubing — the "
               "computational form of reading the Gilbert pressure-distribution "
               "curve by equivalent depth.")

    vlp_method = st.radio(
        "Method",
        ["Method 1 — P_wf vs q (THP → P_wf)",
         "Method 2 — THP vs q (P_wf → THP)",
         "Both"],
        index=2, horizontal=True,
        help="Method 1: fix THP, march down for P_wf. "
             "Method 2: take P_wf from IPR, march up for THP.",
    )

    try:
        qg_vlp = np.linspace(max(qmax * 0.02, 10.0), qmax, 35)
        ipr_g  = np.interp(qg_vlp, q_arr, ipr_arr)

        # Method 1 — required P_wf (outflow) for fixed THP
        pwf_vlp = np.array([integrate_traverse(Pwh, q, vp, +1, 60) for q in qg_vlp])
        d1 = ipr_g - pwf_vlp
        op1_q = op1_pwf = None
        for i in range(len(d1) - 1):
            if d1[i] > 0 and d1[i + 1] <= 0:
                fr = d1[i] / (d1[i] - d1[i + 1])
                op1_q   = qg_vlp[i] + fr * (qg_vlp[i + 1] - qg_vlp[i])
                op1_pwf = ipr_g[i]  + fr * (ipr_g[i + 1]  - ipr_g[i])
                break

        # Method 2 — required THP from IPR P_wf, operating where THP = THP_set
        thp_vlp = np.array([integrate_traverse(max(ipr_g[i], 20.0), qg_vlp[i], vp, -1, 60)
                            for i in range(len(qg_vlp))])
        d2 = thp_vlp - Pwh
        op2_q = op2_pwf = None
        for i in range(len(d2) - 1):
            if d2[i] * d2[i + 1] < 0:
                fr = d2[i] / (d2[i] - d2[i + 1])
                op2_q   = qg_vlp[i] + fr * (qg_vlp[i + 1] - qg_vlp[i])
                op2_pwf = ipr_g[i]  + fr * (ipr_g[i + 1]  - ipr_g[i])
                break

        # ---- metrics ----
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Operating rate (M1)",
                      f"{op1_q:,.0f} STB/d" if op1_q else "No flow")
        with m2:
            st.metric("Operating P_wf (M1)",
                      f"{op1_pwf:,.0f} psia" if op1_pwf else "—")
        with m3:
            st.metric("Operating rate (M2)",
                      f"{op2_q:,.0f} STB/d" if op2_q else "No flow")

        # ---- Method 1 plot ----
        if vlp_method.startswith("Method 1") or vlp_method == "Both":
            f1 = go.Figure()
            f1.add_trace(go.Scatter(x=q_arr, y=ipr_arr, name="IPR (inflow)",
                                    line=dict(color="#f5a623", width=3)))
            f1.add_trace(go.Scatter(x=qg_vlp, y=pwf_vlp, name="VLP — Method 1 (P_wf vs q)",
                                    line=dict(color="#3ec1a8", width=3)))
            if op1_q:
                f1.add_trace(go.Scatter(x=[op1_q], y=[op1_pwf], mode="markers+text",
                                        marker=dict(color="#e8743b", size=13,
                                                    line=dict(color="#fff", width=1.5)),
                                        text=["  operating point"], textposition="middle right",
                                        textfont=dict(color="#e8743b"), name="Operating point"))
            f1.update_layout(template="plotly_dark", height=430,
                             title="Method 1 — P_wf vs q  (IPR ∩ VLP)",
                             xaxis_title="Flow rate, q (STB/d)",
                             yaxis_title="P_wf (psia)",
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             legend=dict(orientation="h", y=1.12))
            st.plotly_chart(f1, width="stretch")

        # ---- Method 2 plot ----
        if vlp_method.startswith("Method 2") or vlp_method == "Both":
            f2 = go.Figure()
            f2.add_trace(go.Scatter(x=qg_vlp, y=thp_vlp, name="VLP — Method 2 (THP vs q)",
                                    line=dict(color="#3ec1a8", width=3)))
            f2.add_hline(y=Pwh, line=dict(color="#f5a623", width=2, dash="dash"),
                         annotation_text=f"Available THP = {Pwh:.0f} psia",
                         annotation_position="top left")
            if op2_q:
                f2.add_trace(go.Scatter(x=[op2_q], y=[Pwh], mode="markers+text",
                                        marker=dict(color="#e8743b", size=13,
                                                    line=dict(color="#fff", width=1.5)),
                                        text=["  q at available THP"], textposition="middle right",
                                        textfont=dict(color="#e8743b"), name="Operating q"))
            f2.update_layout(template="plotly_dark", height=430,
                             title="Method 2 — THP vs q  (operating where THP = available THP)",
                             xaxis_title="Flow rate, q (STB/d)",
                             yaxis_title="THP / P_wh (psia)",
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             legend=dict(orientation="h", y=1.12))
            st.plotly_chart(f2, width="stretch")

        # ---- Gilbert pressure-traverse (gradient) curve ----
        st.markdown("#### Gilbert pressure-traverse curve")
        q_show = op1_q if op1_q else qg_vlp[len(qg_vlp) // 2]
        depths, pres = integrate_traverse(Pwh, q_show, vp, +1, 80, record=True)
        ft = go.Figure()
        ft.add_trace(go.Scatter(x=pres, y=depths, mode="lines",
                                line=dict(color="#ffcd6b", width=3), name="Pressure traverse"))
        ft.update_layout(template="plotly_dark", height=430,
                         title=f"Pressure vs depth at q = {q_show:,.0f} STB/d "
                               f"(GLR-consistent, d = {d_in}\")",
                         xaxis_title="Pressure (psia)",
                         yaxis_title="Depth (ft)",
                         yaxis=dict(autorange="reversed"),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         showlegend=False)
        st.plotly_chart(ft, width="stretch")
        st.caption(f"THP = {Pwh:.0f} psia at surface → P_wf = {pres[-1]:,.0f} psia at "
                   f"{vp['depth']:,.0f} ft. Curve steepens with depth as the gas "
                   "compresses and the column grows heavier — the same shape as Gilbert's "
                   "field gradient curves.")

    except Exception as e:
        st.error(f"VLP calculation error: {e}")
