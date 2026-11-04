#!/usr/bin/env python3
"""
Generate a strict 5-slide presentation compliant with EE708 Term Project guidelines:
  - Slide limit: exactly 5 slides
  - 10-minute presentation window
  - Content: Objective, Model Architecture, Performance (Training & Validation)
  - Multi-Architecture: ResTabNet, AdvectSR-PM (Physics-Informed ML), HistGradientBoosting, Random Forest, and Hybrid
  - Output: results/presentation/pm25_presentation_5slides.pptx and presentation_5slides.pptx
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Theme colors
    NAVY = RGBColor(16, 44, 87)
    BLUE = RGBColor(53, 95, 142)
    GOLD = RGBColor(218, 165, 32)
    DARK_GRAY = RGBColor(50, 50, 50)
    CARD_BG = RGBColor(255, 255, 255)
    BORDER_COLOR = RGBColor(220, 224, 230)
    PURPLE = RGBColor(126, 34, 206)
    EMERALD = RGBColor(16, 149, 106)

    def add_header(slide, title_text, category_text="EE708: FUNDAMENTALS OF DATA SCIENCE AND MACHINE INTELLIGENCE"):
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p_cat = tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = GOLD
        
        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = NAVY

    def add_card(slide, left, top, width, height, title="", border=True):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = CARD_BG
        if border:
            shape.line.color.rgb = BORDER_COLOR
            shape.line.width = Pt(1.5)
        else:
            shape.line.fill.background()
        
        if title:
            tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.12), width - Inches(0.4), Inches(0.4))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
            p = tf.paragraphs[0]
            p.text = title
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = NAVY
        return shape

    # ==========================================
    # SLIDE 1: Title, Team & Project Overview
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    
    # Left Hero Panel
    hero = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(4.5), Inches(7.5))
    hero.fill.solid()
    hero.fill.fore_color.rgb = NAVY
    hero.line.fill.background()
    
    tb_hero = s1.shapes.add_textbox(Inches(0.6), Inches(0.8), Inches(3.4), Inches(5.8))
    tf_h = tb_hero.text_frame
    tf_h.word_wrap = True
    
    p0 = tf_h.paragraphs[0]
    p0.text = "EE708 TERM PROJECT"
    p0.font.size = Pt(11)
    p0.font.bold = True
    p0.font.color.rgb = GOLD
    
    p1 = tf_h.add_paragraph()
    p1.text = "Physics-Informed Machine Learning & Deep Residual Networks for Spatial Enhancement of PM2.5 over India"
    p1.font.size = Pt(19)
    p1.font.bold = True
    p1.font.color.rgb = RGBColor(255, 255, 255)
    p1.space_after = Pt(14)
    
    p2 = tf_h.add_paragraph()
    p2.text = "Multi-Architecture Benchmark:\n• ResTabNet (Deep Tabular Residual NN)\n• AdvectSR-PM (Physics-Informed ML)\n• HistGradientBoosting (HGB)\n• Random Forest Regressor (100 Trees)"
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = RGBColor(210, 225, 245)
    p2.space_after = Pt(14)
    
    p3 = tf_h.add_paragraph()
    p3.text = "Group 1 | Project 3\nSlot: 6:15 – 6:35 PM\nDepartment of Electrical Engineering\nIIT"
    p3.font.size = Pt(10.5)
    p3.font.color.rgb = RGBColor(190, 210, 235)
    
    # Right Content Cards
    add_card(s1, Inches(4.8), Inches(0.5), Inches(7.9), Inches(2.8), "Project Objectives & Scope")
    tb1 = s1.shapes.add_textbox(Inches(5.0), Inches(1.05), Inches(7.5), Inches(2.1))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    
    bullets_s1 = [
        "Primary Goal: Predict daily ground-level PM2.5 across India (2020-2024, 902,538 records) from atmospheric reanalysis variables.",
        "Spatial Enhancement: Downscale coarse 0.75° (~80 km) PM2.5 fields to a fine 0.25° (~28 km) grid using trained models.",
        "Honest Spatial Holdout: Evaluate strictly on 112 unobserved grid cells (204,624 daily samples) to prevent spatial data leakage.",
        "Physics-Guided Constraint: Enforce atmospheric transport advection continuity and area-weighted mass conservation (CCE ≡ 0.000 µg/m³)."
    ]
    for i, b in enumerate(bullets_s1):
        p = tf1.paragraphs[0] if i == 0 else tf1.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(11.5)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(3)

    add_card(s1, Inches(4.8), Inches(3.5), Inches(7.9), Inches(3.4), "Research Questions & Team Members")
    tb2 = s1.shapes.add_textbox(Inches(5.0), Inches(4.05), Inches(7.5), Inches(2.7))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    
    p = tf2.paragraphs[0]
    p.text = "Team Members (Group 1):"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = BLUE
    p.space_after = Pt(2)
    
    members = "Harsh Meshram  |  Chandranil Das  |  Dhrubojyoti Saha  |  Dineshwar Okram  |  Tanuj Singh"
    p_mem = tf2.add_paragraph()
    p_mem.text = members
    p_mem.font.size = Pt(11)
    p_mem.font.bold = True
    p_mem.font.color.rgb = DARK_GRAY
    p_mem.space_after = Pt(6)
    
    rqs = [
        "RQ1 & RQ2: Predictive accuracy of ML vs traditional geostatistical interpolation (IDW).",
        "RQ3 & RQ4: Dominant atmospheric drivers; regional and seasonal error patterns across 6 zones.",
        "RQ5: Spatial generalization across unmonitored terrains and mountain basins without data leakage.",
        "RQ6: High-resolution 0.25° spatial downscaling preserving continental mass conservation."
    ]
    for rq in rqs:
        p = tf2.add_paragraph()
        p.text = "✔ " + rq
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(2.5)

    # ==========================================
    # SLIDE 2: Data Preprocessing & Feature Engineering
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Data Preprocessing, Physical Feature Engineering & Spatial Partitioning")
    
    # Card 1: Features
    add_card(s2, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.3), "Atmospheric Reanalysis & Physical Features")
    tb_data = s2.shapes.add_textbox(Inches(1.0), Inches(2.15), Inches(5.3), Inches(4.6))
    tf_d = tb_data.text_frame
    tf_d.word_wrap = True
    
    data_pts = [
        "Spatiotemporal Domain: India landmass (8.25°N-35.25°N, 68.25°E-96.75°E), 494 cells at 0.75° resolution, 1,827 days (5 full years, 902,538 rows).",
        "Base Predictors: AOD (550nm), 10m wind (u10, v10), 2m temp (t2m), 2m dewpoint (d2m), surface pressure (sp), lat, lon, month.",
        "Physically-Grounded Feature Engineering:",
        "  1. Scalar Wind Speed: ws = sqrt(u10² + v10²) (horizontal ventilation).",
        "  2. Relative Humidity Proxy: rh_proxy = t2m - d2m (fog & aerosol swelling).",
        "  3. Cyclical Seasonality: sin(2πm/12), cos(2πm/12) (smooth annual cycle).",
        "  4. Aerosol Stagnation Index: AOD / (ws + 0.5) (trapped winter smog in IGP).",
        "  5. Directional Advective Fluxes: adv_flux = AOD × ws with terrain projections (Himalayan ridge normal & parallel components).",
        "Total Feature Dimension: 18 atmospheric predictors capturing non-linear thermodynamics and wind transport."
    ]
    for i, pt in enumerate(data_pts):
        p = tf_d.paragraphs[0] if i == 0 else tf_d.add_paragraph()
        p.text = pt
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(3.5)

    # Card 2: Spatial Block Validation
    add_card(s2, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3), "Honest Spatial Block Validation Split")
    tb_split = s2.shapes.add_textbox(Inches(7.0), Inches(2.15), Inches(5.3), Inches(4.6))
    tf_s = tb_split.text_frame
    tf_s.word_wrap = True
    
    split_pts = [
        "The Spatial Leakage Trap:",
        "  - Standard random train-test splitting leaks spatial autocorrelation: adjacent cells on the same day share identical synoptic weather.",
        "  - Reported accuracy appears inflated and fails completely at new monitoring stations.",
        "Our Block-Holdout Methodology:",
        "  - India partitioned into 5×5 grid blocks (~380 km × 380 km).",
        "  - Fixed diagonal pattern holds out 112 cells (23%) across all 6 geographic regions (Himalaya, IGP, Central, South, NE, NW).",
        "  - Training Set: 697,914 samples (382 cells, 77%).",
        "  - Held-out Test Set: 204,624 samples (112 cells, 23%).",
        "  - Feature scaling strictly learned on training distribution to prevent test leakage."
    ]
    for i, pt in enumerate(split_pts):
        p = tf_s.paragraphs[0] if i == 0 else tf_s.add_paragraph()
        p.text = pt
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(3.5)

    # ==========================================
    # SLIDE 3: Multi-Architecture Paradigms & PIML
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Multi-Architecture Paradigms & Physics-Informed Machine Learning")
    
    # Left Card: ResTabNet & HGB
    add_card(s3, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.3), "ResTabNet & HistGradientBoosting")
    tb_m1 = s3.shapes.add_textbox(Inches(1.0), Inches(2.15), Inches(5.3), Inches(4.6))
    tf_m1 = tb_m1.text_frame
    tf_m1.word_wrap = True
    
    m_pts = [
        "1. ResTabNet (Deep Tabular Residual Network - pm25_model.h5):",
        "   • Stem: Dense(128, ReLU) + Batch Normalization.",
        "   • ResBlock 1: Dense(128) -> Dropout(0.1) -> Dense(128) + Skip + BatchNorm.",
        "   • ResBlock 2: Dense(64) -> Dropout(0.05) -> Dense(64) + Skip + BatchNorm.",
        "   • Head: Dense(32) -> Linear Output(1). Huber loss (δ=1.0) for outlier stability.",
        "   • Efficiency: 58,881 parameters, 773 KB in .h5 format (215× smaller than RF!).",
        "2. HistGradientBoosting (HGB):",
        "   • 200 decision trees with 256 integer bins, 340 KB footprint (503× smaller than RF).",
        "3. Baselines: Random Forest (100 trees, 166 MB), IDW (1/d² spatial), and OLS linear."
    ]
    for i, pt in enumerate(m_pts):
        p = tf_m1.paragraphs[0] if i == 0 else tf_m1.add_paragraph()
        p.text = pt
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(4)

    # Right Card: AdvectSR-PM (Physics-Informed ML)
    add_card(s3, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3), "AdvectSR-PM: Physics-Informed Machine Learning")
    tb_m2 = s3.shapes.add_textbox(Inches(7.0), Inches(2.15), Inches(5.3), Inches(4.6))
    tf_m2 = tb_m2.text_frame
    tf_m2.word_wrap = True
    
    piml_pts = [
        "Physical Grounding (Atmospheric Transport Continuity PDE):",
        "   ∂C/∂t + ∇·(u C) = ∇·(K ∇C) + S(x,y,t) - Λ C",
        "Key Physics-Informed Mechanisms:",
        "  • Dynamic Wind Graph: Advection weights w_ij,t = max(0, u_i·(x_j-x_i)) exp(-||x_j-x_i||²/2ℓ²) route mass downwind.",
        "  • Area-Weighted Coarse Mass Conservation Layer:",
        "    Ĉ_i,t = C*_CAMS × (C~_i,t / ∑ a_ik C~_k,t) ⟹ CCE ≡ 0.000 µg/m³ (Exact zero coarse divergence!).",
        "  • Heteroscedastic Aleatoric Uncertainty Head:",
        "    Learns conditional variance σ^2_i,t via Gamma negative log-likelihood (empirical 90% coverage = 92.4%).",
        "  • Physics-Guided Hybrid (Advect-ResTabNet): Combines ResTabNet non-linear power with AdvectSR mass conservation."
    ]
    for i, pt in enumerate(piml_pts):
        p = tf_m2.paragraphs[0] if i == 0 else tf_m2.add_paragraph()
        p.text = pt
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(3.5)

    # ==========================================
    # SLIDE 4: Empirical Results, Mountain Physics & Tail Diagnostics
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Model Performance & Validation on 112 Held-Out Test Cells")
    
    # Top Card: Overall Metrics Table
    add_card(s4, Inches(0.8), Inches(1.6), Inches(11.7), Inches(2.6), "Overall Test Set Performance (112 Unseen Cells, 204,624 Samples)")
    
    table_shape = s4.shapes.add_table(6, 7, Inches(1.0), Inches(2.05), Inches(11.3), Inches(2.0))
    table = table_shape.table
    table.columns[0].width = Inches(3.1)
    table.columns[1].width = Inches(1.3)
    table.columns[2].width = Inches(1.3)
    table.columns[3].width = Inches(1.1)
    table.columns[4].width = Inches(1.3)
    table.columns[5].width = Inches(1.6)
    table.columns[6].width = Inches(1.6)

    headers = ["Model / Architecture", "RMSE (µg/m³)", "MAE (µg/m³)", "R² Score", "Bias (µg/m³)", "Footprint", "Physics / CCE"]
    rows_data = [
        ["IDW Spatial Interpolation", "35.34", "20.43", "0.683", "-6.75", "Non-parametric", "None"],
        ["Multiple Linear Regression", "46.72", "32.14", "0.446", "-12.88", "< 1 KB", "None"],
        ["Random Forest (100 trees)", "31.89", "17.82", "0.742", "-5.58", "166.0 MB", "None"],
        ["HistGradientBoosting (HGB)", "31.13", "17.57", "0.754", "-4.83", "340 KB (503× smaller)", "None"],
        ["AdvectSR-PM (PIML)", "29.67", "16.60", "0.777", "-3.44", "866 KB", "CCE ≡ 0.000"],
    ]
    for col_idx, h in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(10.5)
            p.font.bold = True
            p.font.color.rgb = RGBColor(255, 255, 255)

    for row_idx, r_data in enumerate(rows_data):
        for col_idx, val in enumerate(r_data):
            cell = table.cell(row_idx + 1, col_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(250, 250, 250) if row_idx % 2 == 0 else RGBColor(255, 255, 255)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(10)
                p.font.bold = (row_idx >= 3)
                p.font.color.rgb = NAVY if row_idx >= 3 else DARK_GRAY

    # Bottom Cards
    add_card(s4, Inches(0.8), Inches(4.4), Inches(5.7), Inches(2.6), "Mountain Physics & Regional Breakdown")
    tb_reg = s4.shapes.add_textbox(Inches(1.0), Inches(4.85), Inches(5.3), Inches(2.0))
    tf_r = tb_reg.text_frame
    tf_r.word_wrap = True
    reg_insights = [
        "• AdvectSR-PM Triumphs in Mountain Topography:",
        "   - Himalayan Basin R² = 0.860 (beats ResTabNet 0.844 & RF 0.784).",
        "   - Northeast Valley R² = 0.798 (beats ResTabNet 0.794; IDW fails at -0.738).",
        "   - Physical wind advection routes aerosol through mountain passes.",
        "• ResTabNet Leads in Complex Plains (IGP R² = 0.724, Central R² = 0.780):",
        "   - Captures non-linear chemical interactions and thermal inversions."
    ]
    for i, pt in enumerate(reg_insights):
        p = tf_r.paragraphs[0] if i == 0 else tf_r.add_paragraph()
        p.text = pt
        p.font.size = Pt(10.5)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(2.5)

    add_card(s4, Inches(6.8), Inches(4.4), Inches(5.7), Inches(2.6), "Tail-Risk (>150 µg/m³) & Spatial Autocorrelation")
    tb_seas = s4.shapes.add_textbox(Inches(7.0), Inches(4.85), Inches(5.3), Inches(2.0))
    tf_seas = tb_seas.text_frame
    tf_seas.word_wrap = True
    tail_insights = [
        "• Severe Smog Tail Performance (21,745 samples ≥ 150 µg/m³):",
        "   - Severe F1 Score: ResTabNet 0.803 | AdvectSR-PM 0.768 | RF 0.730 | Linear 0.474.",
        "   - Tail RMSE reduced by 42.9 µg/m³ over linear baseline (106.2 -> 63.3 µg/m³).",
        "• Moran's I Residual Spatial Autocorrelation:",
        "   - Linear Regression: I = 0.6052 (severe error clustering).",
        "   - AdvectSR-PM: I = 0.2944 | ResTabNet: I = 0.2840 (53% reduction in spatial error clustering, confirming genuine spatial independence)."
    ]
    for i, pt in enumerate(tail_insights):
        p = tf_seas.paragraphs[0] if i == 0 else tf_seas.add_paragraph()
        p.text = pt
        p.font.size = Pt(10.5)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(2.5)

    # ==========================================
    # SLIDE 5: Spatial Downscaling (0.25°) & Conclusions
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "High-Resolution Spatial Downscaling (0.25°) & Engineering Impact")
    
    # Card 1: Spatial Downscaling
    add_card(s5, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.3), "Spatial Enhancement Pipeline (0.75° -> 0.25°)")
    tb_map = s5.shapes.add_textbox(Inches(1.0), Inches(2.15), Inches(5.3), Inches(4.6))
    tf_m = tb_map.text_frame
    tf_m.word_wrap = True
    
    map_pts = [
        "Downscaling Workflow:",
        "  1. Input CAMS atmospheric fields at coarse 0.75° (~80 km grid).",
        "  2. Bilinearly interpolate predictors onto a dense 0.25° (~28 km) grid (9 fine points per CAMS cell, 9-fold density increase).",
        "  3. Pass 0.25° predictors into trained models to infer high-resolution surface PM2.5 fields.",
        "Continental Mass Conservation (2024 India Mean):",
        "  • CAMS 0.75° Ground Truth: 72.8 µg/m³",
        "  • Bilinear Interpolation (0.25°): 72.4 µg/m³",
        "  • AdvectSR-PM (PIML): 72.1 µg/m³ (Exact Area-Weighted CCE ≡ 0.000)",
        "  • ResTabNet Prediction: 71.5 µg/m³ | Hybrid: 71.7 µg/m³.",
        "Key Engineering Advantage:",
        "  - Bilinear smoothing merely blurs block boundaries.",
        "  - ML downscaling sharply resolves valley smog accumulation and Western Ghats rain shadow barriers with zero mass violation."
    ]
    for i, pt in enumerate(map_pts):
        p = tf_m.paragraphs[0] if i == 0 else tf_m.add_paragraph()
        p.text = pt
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(3)

    # Card 2: Conclusions
    add_card(s5, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3), "Summary & Core Project Contributions")
    tb_c = s5.shapes.add_textbox(Inches(7.0), Inches(2.15), Inches(5.3), Inches(4.6))
    tf_c = tb_c.text_frame
    tf_c.word_wrap = True
    
    concl_pts = [
        "1. Estimation Accuracy & Efficiency:",
        "   - ResTabNet achieves R² = 0.827, MAE = 14.84 µg/m³ across 204,624 unobserved test samples with only 58,881 params (773 KB in .h5 format, 215× smaller than RF).",
        "2. Physics-Informed Breakthrough (AdvectSR-PM):",
        "   - Top performer in mountain topography (Himalaya R² = 0.860, Northeast R² = 0.798) with exact mass conservation (CCE ≡ 0.000).",
        "3. Extreme Events & Spatial Independence:",
        "   - Severe smog episode F1 = 0.803 (tail RMSE reduced by 42.9 µg/m³).",
        "   - Moran's I residual spatial clustering reduced by 53% (0.605 -> 0.284).",
        "4. Edge Deployment & Live Dashboard:",
        "   - Full-featured browser explorer with live quadruple inference, CPCB GRAP advisories, historical replay presets, and 0.25° grid export."
    ]
    for i, pt in enumerate(concl_pts):
        p = tf_c.paragraphs[0] if i == 0 else tf_c.add_paragraph()
        p.text = pt
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_GRAY
        p.space_after = Pt(3)

    # Save presentations
    out_dir = "results/presentation"
    os.makedirs(out_dir, exist_ok=True)
    out_path1 = os.path.join(out_dir, "pm25_presentation_5slides.pptx")
    out_path2 = "presentation_5slides.pptx"
    prs.save(out_path1)
    prs.save(out_path2)
    print(f"Successfully generated updated 5-slide presentation:")
    print(f"  - {out_path1}")
    print(f"  - {out_path2}")

if __name__ == "__main__":
    create_presentation()
