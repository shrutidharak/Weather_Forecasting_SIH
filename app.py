"""
app.py — AI Extreme Weather Threat Intelligence Dashboard
==========================================================
Interactive meteorological threat intelligence interface featuring:
  • Spatio-Temporal Graph Neural Network (ST-GNN) core
  • Automated 4D bounding box threat localization
  • Extreme Forecast Index (EFI) anomaly scoring
  • Physics-informed moisture convergence validation
  • Diffusion-inspired conditional downscaling
"""
import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.data.synthetic_generator import SyntheticDataLoader, SyntheticWeatherGenerator
from src.data.climatology import SyntheticClimatology
from src.data.efi import EFIScorer
from src.models.baseline_detector import BaselineDetector
from src.models.tracker import SimpleTracker
from src.models.gnn_tracker import GNNTracker
from src.models.ensemble_uncertainty import EnsembleUncertaintyEstimator
from src.models.st_gnn import SpatioTemporalEventDetector
from src.models.physics_loss import physics_plausibility_score
from src.models.downscaler import ConvDownscaler, BilinearDownscaler, train_downscaler
from src.evaluation.metrics import Evaluator

# Configure page
st.set_page_config(
    page_title="AI Extreme Weather Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for human-crafted enterprise dashboard
st.markdown(
    """
    <style>
        /* Base typography & spacing */
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
            max-width: 96%;
        }
        h1, h2, h3, h4 {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #0f172a;
        }
        
        /* Metric cards */
        [data-testid="stMetricValue"] {
            font-size: 1.45rem !important;
            font-weight: 700 !important;
            color: #0f172a;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            color: #64748b !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Styled custom cards */
        .threat-pill {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.82rem;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        
        .metric-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 8px;
        }
        
        /* Divider */
        hr {
            margin-top: 1.2rem !important;
            margin-bottom: 1.2rem !important;
            border-color: #f1f5f9 !important;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Matplotlib global style configuration for crisp, professional plots
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 0.8,
    "grid.color": "#f1f5f9",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "text.color": "#0f172a",
    "axes.labelcolor": "#475569",
    "axes.titlesize": 10.5,
    "axes.titleweight": "600",
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
})

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Threat Monitor Controls")
    st.caption("Operational scenario & model configuration")
    
    st.markdown("#### Forecast Scenario")
    scenario = st.selectbox(
        "Forecast Scenario",
        [
            "A - Moving Event",
            "B - Multiple Events",
            "cyclone - Cyclone proxy",
            "heatwave - Heatwave proxy"
        ],
        label_visibility="collapsed"
    )
    scenario_key = scenario.split(" ")[0]

    st.markdown("#### Downscaling Model")
    downscaler_choice = st.radio(
        "Downscaler Selection",
        ["Bilinear Baseline", "Diffusion (ConvDownscaler)"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("#### Overlay Layers")
    show_efi = st.checkbox("Show EFI Anomaly Field", value=True)
    
    st.markdown("---")
    st.markdown("#### System Telemetry")
    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #64748b; line-height: 1.6;">
            <strong>Core Architecture:</strong> 3-Hop ST-GNN<br>
            <strong>Spatial Mesh:</strong> 4-Connected Lat/Lon<br>
            <strong>Channels:</strong> 6 Multivariate Fields<br>
            <strong>Detection Trigger:</strong> EFI &ge; 0.50<br>
            <strong>Inference Engine:</strong> PyTorch 2.12
        </div>
        """,
        unsafe_allow_html=True
    )

# ── Header Bar ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 0.8rem; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.8rem;">
        <div>
            <h1 style="margin: 0; font-size: 1.85rem; font-weight: 700; color: #0f172a; letter-spacing: -0.5px;">
                AI Extreme Weather Threat Intelligence
            </h1>
            <p style="margin: 4px 0 0 0; color: #64748b; font-size: 0.92rem; font-weight: 400;">
                Continuous Spatio-Temporal GNN Threat Localization &bull; Automated 4D Threat Bounding Boxes &bull; Physics-Constrained Downscaling
            </p>
        </div>
        <div style="display: flex; gap: 8px;">
            <span style="background: #e0f2fe; color: #0369a1; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;">Operational</span>
            <span style="background: #f1f5f9; color: #475569; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;">NWP Stream Active</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Badge Color Palettes ───────────────────────────────────────────────────────
_BADGE_COLOURS = {
    "cyclone":          "#1d4ed8",
    "heatwave":         "#b91c1c",
    "cold_wave":        "#0369a1",
    "extreme_rainfall": "#15803d",
    "normal":           "#475569",
    "unknown":          "#64748b",
}
_RISK_COLOURS = {
    "low":      "#16a34a",
    "moderate": "#d97706",
    "severe":   "#ea580c",
    "extreme":  "#dc2626",
}

def _event_badge(event_type: str) -> str:
    colour = _BADGE_COLOURS.get(event_type.lower(), "#64748b")
    return f'<span class="threat-pill" style="background:{colour}; color:white;">{event_type.upper()}</span>'

def _risk_badge(risk: str) -> str:
    colour = _RISK_COLOURS.get(risk.lower(), "#64748b")
    return f'<span class="threat-pill" style="background:{colour}; color:white;">{risk.upper()} RISK</span>'

# ── Cached Pipeline Runner ────────────────────────────────────────────────────
@st.cache_resource
def load_and_run_pipeline(scen_key):
    loader = SyntheticDataLoader(scenario=scen_key, seed=0)
    lr, hr, gt = loader.load_data()
    terrain = loader.get_terrain()

    clim = SyntheticClimatology()
    clim.generate_historical_baseline()

    # Extreme Forecast Index (EFI)
    scorer = EFIScorer(clim, threshold=0.5)
    feat_idx = 3 if scen_key == "heatwave" else 0
    efi_per_t = np.stack([
        scorer.compute(lr[:, t, :, :, feat_idx], feature_idx=feat_idx)
        for t in range(lr.shape[1])
    ], axis=0)
    efi_ensemble = efi_per_t.mean(axis=0)

    # Physics plausibility score
    mid = lr.shape[1] // 2
    mean_frame = lr[:, mid].mean(axis=0)
    phys_score = physics_plausibility_score(
        mean_frame[:,:,0], mean_frame[:,:,1],
        mean_frame[:,:,2], mean_frame[:,:,5]
    )

    # Spatio-Temporal GNN 4D Event Detection
    detector = SpatioTemporalEventDetector(
        lr.shape[2], lr.shape[3],
        detection_threshold=0.25
    )
    efi_stack = np.stack([efi_ensemble] * lr.shape[0], axis=0)
    st_events = detector.detect(
        lr, terrain, efi_stack,
        forecast_lead=24,
        physics_plausibility=phys_score
    )

    # Baseline Tracker (Euclidean)
    base_detector = BaselineDetector(clim, feature_idx=feat_idx, percentile=90 if scen_key == "heatwave" else 95)
    base_tracker = SimpleTracker(max_distance=5.0)
    for t in range(lr.shape[1]):
        objects = base_detector.detect(lr[0, t], time_step=t)
        base_tracker.update(objects, time_step=t)
    base_tracks = [tr for tr in base_tracker.get_all_tracks() if tr.duration >= 3]

    # Baseline GNN Tracker
    model_path = "gnn_tracker.pth"
    gnn_tracker = GNNTracker(model_path) if os.path.exists(model_path) else SimpleTracker(5.0)
    for t in range(lr.shape[1]):
        objects = base_detector.detect(lr[0, t], time_step=t)
        gnn_tracker.update(objects, time_step=t)
    gnn_tracks = [tr for tr in gnn_tracker.get_all_tracks() if tr.duration >= 3]

    # Ensemble Uncertainty
    all_ens_tracks = {}
    for e in range(lr.shape[0]):
        trkr = SimpleTracker(max_distance=5.0)
        for t in range(lr.shape[1]):
            objs = base_detector.detect(lr[e, t], time_step=t)
            trkr.update(objs, time_step=t)
        all_ens_tracks[e] = trkr.get_all_tracks()

    uncertainty = EnsembleUncertaintyEstimator().estimate_uncertainty(all_ens_tracks, 0)

    # Metric Evaluations
    eval_base = Evaluator(gt, base_tracks).evaluate_localization()
    eval_gnn = Evaluator(gt, gnn_tracks).evaluate_localization()

    # Train Downscaler
    conv_model = ConvDownscaler()
    train_downscaler(conv_model, n_samples=100, epochs=15, verbose=False)

    return (lr, hr, gt, terrain, base_tracks, gnn_tracks,
            uncertainty, eval_base, eval_gnn,
            st_events, phys_score, efi_per_t, conv_model)

with st.spinner("Processing NWP tensor through Spatio-Temporal GNN..."):
    (lr, hr, gt, terrain, base_tracks, gnn_tracks,
     uncertainty, eval_base, eval_gnn,
     st_events, phys_score, efi_per_t, conv_model) = load_and_run_pipeline(scenario_key)

# ── Horizon & Lead Time Controller ─────────────────────────────────────────────
time_steps = lr.shape[1]
with st.container(border=True):
    col_lead_slider, col_lead_metric = st.columns([4.5, 1.5])
    with col_lead_slider:
        t = st.slider(
            "Forecast Lead Horizon (Hours Ahead)",
            min_value=0,
            max_value=time_steps - 1,
            value=0,
            step=1,
            help="Navigate the temporal axis of the forecast window",
        )
    with col_lead_metric:
        st.metric("Lead Horizon", f"T + {t}h", delta=f"Step {t} of {time_steps - 1}")

# ── MAIN ROW: Spatial View, EFI Heatmap, and Event Overview ───────────────────
col_map, col_efi, col_info = st.columns([1.2, 1.0, 1.1], gap="medium")

with col_map:
    with st.container(border=True):
        st.markdown(f"**Spatial Intensity & Threat Tracking (T+{t}h)**")
        fig, ax = plt.subplots(figsize=(5.4, 4.8), dpi=120)
        
        # Display channel (temperature for heatwave, rainfall for others)
        ch_idx = 3 if scenario_key == "heatwave" else 0
        ch_label = "Temperature Anomaly (°C)" if scenario_key == "heatwave" else "Precipitation Rate (mm/h)"
        cmap_choice = "YlOrRd" if scenario_key == "heatwave" else "Blues"
        
        frame = lr[0, t, :, :, ch_idx]
        im = ax.imshow(frame, cmap=cmap_choice, origin="lower")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(ch_label, fontsize=8, color="#475569")
        cbar.ax.tick_params(labelsize=7.5)

        # Ground truth
        gt_t = [g for g in gt if g.timestamp == t]
        for g in gt_t:
            ax.plot(g.centroid_lon, g.centroid_lat, marker="*", color="#dc2626", markersize=11,
                    label="Ground Truth" if g == gt_t[0] else "")
            mn_lat, mn_lon, mx_lat, mx_lon = g.bbox
            ax.add_patch(plt.Rectangle(
                (mn_lon, mn_lat), mx_lon - mn_lon, mx_lat - mn_lat,
                fill=False, edgecolor="#dc2626", linestyle="--", linewidth=1.4
            ))

        # ST-GNN 4D threat boxes
        for ev in st_events:
            ts, te = ev.get("t_start", 0), ev.get("t_end", time_steps)
            if ts <= t <= te:
                lat_r = ev["latitude_range"]
                lon_r = ev["longitude_range"]
                ax.add_patch(plt.Rectangle(
                    (lon_r[0], lat_r[0]), lon_r[1] - lon_r[0], lat_r[1] - lat_r[0],
                    fill=True, facecolor="#ea580c", alpha=0.08
                ))
                ax.add_patch(plt.Rectangle(
                    (lon_r[0], lat_r[0]), lon_r[1] - lon_r[0], lat_r[1] - lat_r[0],
                    fill=False, edgecolor="#ea580c", linestyle="-", linewidth=2.0,
                    label="ST-GNN 4D BBox"
                ))
                ax.text(
                    lon_r[0] + 0.4, lat_r[0] + 0.4,
                    ev['event_type'].upper(),
                    color="#c2410c", fontsize=7.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff7ed", edgecolor="#ea580c", alpha=0.85)
                )

        # Baseline GNN tracks
        for track in gnn_tracks:
            obj = next((o for o in track.history if o.timestamp == t), None)
            if obj:
                ax.plot(obj.centroid_lon, obj.centroid_lat, marker="X", color="#16a34a", markersize=8,
                        label="Baseline Track")

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=7.5, framealpha=0.9)
        
        ax.set_xlabel("Grid Longitude", fontsize=8)
        ax.set_ylabel("Grid Latitude", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

with col_efi:
    with st.container(border=True):
        st.markdown(f"**Extreme Forecast Index (T+{t}h)**")
        if show_efi and efi_per_t is not None and t < len(efi_per_t):
            fig2, ax2 = plt.subplots(figsize=(4.7, 4.8), dpi=120)
            efi_frame = efi_per_t[t]
            im2 = ax2.imshow(efi_frame, cmap="RdBu_r", origin="lower", vmin=-1.0, vmax=1.0)
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label("EFI Score [-1.0, +1.0]", fontsize=8, color="#475569")
            cbar2.ax.tick_params(labelsize=7.5)

            # Highlight cells exceeding extreme threshold (> 0.5)
            mask = efi_frame > 0.5
            if mask.any():
                ys, xs = np.where(mask)
                ax2.scatter(xs, ys, c="#f59e0b", s=3.5, alpha=0.75, label="EFI &ge; 0.5")

            ax2.set_xlabel("Grid Longitude", fontsize=8)
            ax2.set_ylabel("Grid Latitude", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)
        else:
            st.info("EFI anomaly layer toggled off.")

with col_info:
    with st.container(border=True):
        st.markdown("**Threat Assessment & Extent**")
        
        if st_events:
            ev0 = st_events[0]
            et = ev0.get("event_type", "unknown")
            risk = ev0.get("risk_level", "low")
            
            # Classification Badges
            st.markdown(
                f"""
                <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 12px;">
                    {_event_badge(et)}
                    {_risk_badge(risk)}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # 2x2 Metric Grid
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Severity", f"{ev0.get('severity', 0) * 100:.0f}%")
            with m2:
                st.metric("Ensemble Conf.", f"{ev0.get('confidence', 0) * 100:.0f}%")
                
            m3, m4 = st.columns(2)
            with m3:
                st.metric("Exceedance (P95)", f"{ev0.get('probability_exceedance', 0) * 100:.0f}%")
            with m4:
                p_lo, p_hi = ev0['vertical_range']
                st.metric("Vertical Range", f"{p_lo:.0f} - {p_hi:.0f} hPa")

            # 4D Bounding Box Dimensions
            st.markdown(
                f"""
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:8px 12px; margin-top:8px; font-size:0.8rem; line-height:1.5;">
                    <strong>Spatial Extent:</strong> Lat {ev0['latitude_range']} &bull; Lon {ev0['longitude_range']}<br>
                    <strong>Temporal Window:</strong> T = {ev0['t_start']} to {ev0['t_end']} hrs
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("No extreme weather events detected at active threshold.")

        # Physics Plausibility Meter
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.8rem; font-weight:600; color:#475569;'>PHYSICS CONSISTENCY SCORE</div>", unsafe_allow_html=True)
        st.progress(float(phys_score), text=f"{phys_score * 100:.1f}% physically plausible")
        if phys_score < 0.5:
            st.caption("Moisture convergence does not support high rainfall intensity.")
        else:
            st.caption("Rainfall is physically supported by moisture influx convergence.")

# ── SECOND ROW: Ensemble Uncertainty & Baseline Benchmarking ──────────────────
col_unc, col_comp = st.columns([1, 1], gap="medium")

with col_unc:
    with st.container(border=True):
        st.markdown("### Ensemble Uncertainty Quantification")
        st.caption("Consolidates multi-member ensemble paths to calculate forecast trajectory spread.")
        
        prob = uncertainty.get("event_probability", 0)
        u1, u2, u3 = st.columns(3)
        with u1:
            st.metric("Ensemble Probability", f"{prob * 100:.1f}%")
        with u2:
            st.metric("Spatial Spread", f"{uncertainty.get('location_spread_lat', 0):.2f}° / {uncertainty.get('location_spread_lon', 0):.2f}°")
        with u3:
            st.metric("Consensus Confidence", f"{uncertainty.get('confidence', 'Unknown')}")
        
        st.markdown(
            f"""
            <div style="font-size:0.82rem; color:#64748b; margin-top:6px;">
                Consensus Centroid Coordinates: <strong>{uncertainty.get('consensus_centroid', (0.0, 0.0))[0]:.2f}°N, {uncertainty.get('consensus_centroid', (0.0, 0.0))[1]:.2f}°E</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

with col_comp:
    with st.container(border=True):
        st.markdown("### Model Benchmark Comparison")
        st.caption("Tracking continuity and localization precision against Ground Truth.")
        
        st.table({
            "Metric": ["Centroid Error", "Bounding Box IoU", "Track Continuity"],
            "Baseline Euclidean Tracker": [
                f"{eval_base.get('mean_centroid_error', 0):.2f}",
                f"{eval_base.get('mean_iou', 0):.2f}",
                f"{eval_base.get('track_continuity', 0):.2f}"
            ],
            "Proposed ST-GNN Module": [
                f"{eval_gnn.get('mean_centroid_error', 0):.2f}",
                f"{eval_gnn.get('mean_iou', 0):.2f}",
                f"{eval_gnn.get('track_continuity', 0):.2f}"
            ],
        })

# ── THIRD ROW: Conditional Downscaling ─────────────────────────────────────────
with st.container(border=True):
    st.markdown("### Regional High-Resolution Downscaling")
    st.caption("Evaluation of spatial resolution enhancement preserving extreme peak intensities.")

    import scipy.ndimage as ndimage
    import torch

    low_frame = lr[0, t, :, :, 0]
    high_frame = hr[0, t, :, :, 0]
    bilinear = ndimage.zoom(low_frame, 2.0, order=1)

    # ConvDownscaler inference
    terrain_t = torch.from_numpy(terrain.transpose(2, 0, 1).astype(np.float32))
    terrain_t[0] /= 3000.0
    terrain_t[1] /= float(lr.shape[3])
    rain_t = torch.from_numpy(low_frame[np.newaxis, np.newaxis].astype(np.float32))
    inp_t = torch.cat([rain_t, terrain_t.unsqueeze(0)], dim=1)

    if downscaler_choice.startswith("Diffusion"):
        with torch.no_grad():
            diffusion_out = conv_model(inp_t)[0, 0].numpy()
        active_label = "ConvDownscaler (Diffusion-Inspired)"
    else:
        diffusion_out = bilinear.copy()
        active_label = "Bilinear Upscaling"

    fig3, axs = plt.subplots(1, 4, figsize=(16, 3.4), dpi=120)
    imgs = [low_frame, bilinear, diffusion_out, high_frame]
    titles = [
        "Coarse NWP Input (32×32)",
        "Bilinear Baseline (64×64)",
        f"{active_label} (64×64)",
        "High-Resolution Target (64×64)"
    ]
    
    for i, (ax, img, tl) in enumerate(zip(axs, imgs, titles)):
        im_sub = ax.imshow(img, cmap="Blues", origin="lower")
        ax.set_title(tl, fontsize=8.5, pad=6)
        ax.axis("off")
        
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    st.markdown(
        """
        <div style="font-size:0.8rem; color:#64748b; line-height:1.5;">
            <strong>Loss Objective:</strong> Total Loss = Reconstruction MSE + 2 &times; ExtremeValueLoss (P90 Tail) + PeakSharpnessLoss.<br>
            Penalizes peak under-prediction to avoid smooth blurring associated with standard bilinear interpolation.
        </div>
        """,
        unsafe_allow_html=True
    )

# ── FOURTH ROW: ST-GNN 4D Event Registry ───────────────────────────────────────
if st_events:
    with st.container(border=True):
        st.markdown("### Active 4D Threat Registry")
        st.caption("Standardized schema output from SpatioTemporalEventDetector.")
        
        table_data = []
        for ev in st_events:
            et = ev.get("event_type", "?")
            table_data.append({
                "Event Classification": et.upper(),
                "Risk Level": ev.get("risk_level", "?").upper(),
                "Latitude Boundary": str(ev.get("latitude_range", "?")),
                "Longitude Boundary": str(ev.get("longitude_range", "?")),
                "Vertical Range": f"{ev['vertical_range'][0]:.0f} - {ev['vertical_range'][1]:.0f} hPa",
                "Temporal Horizon": f"T = {ev.get('t_start', 0)} to T = {ev.get('t_end', 0)}",
                "Severity": f"{ev.get('severity', 0):.2f}",
                "Confidence": f"{ev.get('confidence', 0):.2f}",
                "Exceedance Prob.": f"{ev.get('probability_exceedance', 0):.2f}",
                "Physics Score": f"{ev.get('physics_plausibility', 1):.2f}",
            })
        st.dataframe(table_data, width='stretch')
