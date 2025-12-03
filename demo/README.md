# Demo / Prototype Plan

## Goal
Deliver a minimal yet clear prototype demonstrating product-level dead stock risk scoring and recommended actions. Stretch goal: Streamlit app for bonus.

## Option A – Required Mockup (baseline)
- Tool: Canva/Figma dashboard mockup
- Screens:
  1. Inventory risk dashboard (table + risk gauge)
  2. Product detail panel with drivers & actions
- Include screenshots in presentation + report.

## Option B – Functional Streamlit Demo (bonus)
- Components:
  - CSV sample data (product_id, category, stock_age, weekly_sales, margin, promo_flag)
  - Feature engineering script (pandas)
  - Simple model (e.g., GradientBoostingClassifier) trained on synthetic labels
  - Streamlit UI: upload CSV, show risk scores, highlight top-k items, recommend actions.
- Deployment: Streamlit Cloud link.

## Implementation Steps
1. Design data schema & create synthetic dataset (50-100 SKUs).
2. Build notebook/script to generate labels + train baseline model.
3. Save pipeline artifacts (joblib) if going for functional demo.
4. Implement Streamlit UI with charts (plotly/seaborn).
5. Export screenshots for presentation.

## Next Actions
- Decide whether to pursue bonus option.
- Assign owner for mockup vs. coding tasks.
- Draft timeline so core deliverables finish before demo polish.
