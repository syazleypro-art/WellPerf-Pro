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
        text="<b>Well Performance Curves — IPR vs TPR</b>",
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

st.plotly_chart(fig, use_container_width=True)

# ── Surface Choke / Wellhead-Node Analysis (Gilbert) ───────────

if enable_choke:
    st.divider()
    st.markdown("### 🎚 Wellhead-Node Analysis — Gilbert Choke")

    # GLR (Mscf/STB) from solution GOR and water cut
    GLR_Mscf = (1 - WC) * GOR / 1000.0

    # Available wellhead pressure from reservoir through tubing:
    #   Pwh_avail(q) = Pwf_IPR(q) − ΔP_tubing(q),   ΔP_tubing = TPR(q) − Pwh
    dP_tubing = tpr_arr - Pwh
    pwh_avail = np.maximum(0, ipr_arr - dP_tubing)

    # Choke demand for the selected bean
    pwh_sel = gilbert_choke_pwh(q_arr, GLR_Mscf, choke_S)
    ck_q, ck_pwh = find_wellhead_point(q_arr, pwh_avail, pwh_sel)

    # Result cards
    k1, k2, k3 = st.columns(3)
    with k1:
        if ck_q:
            st.metric("Choked Operating Rate", f"{int(round(ck_q)):,} STB/d",
                      help="Where available wellhead pressure meets the Gilbert choke demand.")
        else:
            st.metric("Choked Operating Rate", "No Intersection")
    with k2:
        st.metric("Wellhead Pressure", f"{int(round(ck_pwh)):,} psia" if ck_pwh else "—")
    with k3:
        st.metric("Bean Size", f"{int(choke_S)}/64 in",
                  help=f"GLR used: {GLR_Mscf:.3f} Mscf/STB")

    # Chart: available-Pwh curve + choke family + selected bean + op point
    figc = go.Figure()
    figc.add_trace(go.Scatter(
        x=q_arr, y=pwh_avail, mode="lines", name="Available Pwh (tubing)",
        line=dict(color="#a78bfa", width=3),
        hovertemplate="q = %{x:.0f} STB/d<br>Pwh = %{y:.0f} psia<extra>Available</extra>"
    ))
    bean_family = [16, 24, 32, 48, 64]
    for S in bean_family:
        if S == int(choke_S):
            continue
        figc.add_trace(go.Scatter(
            x=q_arr, y=gilbert_choke_pwh(q_arr, GLR_Mscf, S),
            mode="lines", name=f"{S}/64 in",
            line=dict(color="#475569", width=1.3, dash="dot"),
            hovertemplate=f"Bean {S}/64<br>"+"q = %{x:.0f}<br>Pwh = %{y:.0f} psia<extra></extra>"
        ))
    figc.add_trace(go.Scatter(
        x=q_arr, y=pwh_sel, mode="lines", name=f"Choke {int(choke_S)}/64 in (selected)",
        line=dict(color="#f59e0b", width=3),
        hovertemplate=f"Bean {int(choke_S)}/64<br>"+"q = %{x:.0f}<br>Pwh = %{y:.0f} psia<extra>Choke</extra>"
    ))
    if ck_q and ck_pwh:
        figc.add_trace(go.Scatter(
            x=[ck_q], y=[ck_pwh], mode="markers+text", name="Choked Operating Point",
            marker=dict(color="#4ade80", size=14, line=dict(color="#16a34a", width=2)),
            text=[f"  ({int(round(ck_q)):,} STB/d, {int(round(ck_pwh)):,} psia)"],
            textposition="middle right", textfont=dict(color="#4ade80", size=11),
            hoverinfo="skip"
        ))
    figc.update_layout(
        title=dict(text="<b>Available Wellhead Pressure vs Gilbert Choke Demand</b>",
                   font=dict(color="#f8fafc", size=15)),
        xaxis=dict(title="Liquid Flow Rate  q  (STB/d)", color="#94a3b8",
                   gridcolor="#1e293b", zerolinecolor="#334155", title_font=dict(color="#94a3b8")),
        yaxis=dict(title="Wellhead Pressure  Pwh  (psia)", color="#94a3b8",
                   gridcolor="#1e293b", zerolinecolor="#334155", title_font=dict(color="#94a3b8")),
        paper_bgcolor="#0d1117", plot_bgcolor="#070b13",
        legend=dict(font=dict(color="#94a3b8"), bgcolor="#0d1117",
                    bordercolor="#1e293b", borderwidth=1),
        hovermode="closest", height=460
    )
    st.plotly_chart(figc, use_container_width=True)
    st.caption("Gilbert's correlation assumes **critical (sonic) flow** across the bean — valid when "
               "downstream pressure is below ≈ 0.5–0.7 × wellhead pressure. Larger beans pass more rate "
               "at lower wellhead pressure.")

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
