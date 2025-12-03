# SE390 Midterm Report Outline

## Team Info
- **Team Name:** _TBD_
- **Members:**
  - _Name 1 (ID)_ – Problem research & business analysis
  - _Name 2 (ID)_ – AI solution design & technical feasibility
  - _Name 3 (ID)_ – Presentation/demo lead
- **Project Title:** _TBD_
- **Date:** 03.12.2025

## 1. Executive Summary (~0.5 page)
- Problem statement recap (dead stock risk in Turkish e-commerce)
- Proposed AI solution (1-2 sentences)
- Expected quantifiable impact (operations, finance, CX)

## 2. Problem Analysis (~1 page)
- Define dead stock and root causes (demand misforecast, promotions, seasonality)
- Business & operational pain points (storage cost ↑, turnover ↓, cash flow strain)
- Customer experience consequences (stockouts on best sellers, delays)
- Market context: Turkey e-commerce scale, references/metrics to research

## 3. Proposed AI Solution (1-2 pages)
- **Approach Overview:** risk scoring pipeline, prediction horizon, confidence bands
- **Data Requirements:** historical sales, pricing, promotions, inventory, supplier lead time, marketing spend, search trends, product metadata
- **ML Techniques:** candidate models (e.g., gradient boosted trees, Prophet + anomaly detection, survival analysis)
- **Feature Engineering:** velocity, seasonality, margin, demand volatility, stock age
- **System Architecture Diagram:** data ingestion → feature store → model training → inference API/dashboard
- **Technical Feasibility:**
  - Stack: Python, pandas, scikit-learn/XGBoost, Airflow, PostgreSQL, Streamlit/Gradio, Docker
  - Implementation approach: batch training, nightly inference, API endpoints
  - Data pipeline: ETL from ERP/WMS, cleaning, labeling dead stock
  - Scalability + integration: modular microservices, message queues, caching
  - Expected metrics: precision@top-k risk, recall, inventory turnover uplift

## 4. Business Impact & Implementation (~1 page)
- Operational improvements: freed warehouse space, reduced aged stock by X%
- Financial analysis: cost savings, working capital unlocked, ROI timeline
- CX benefits: better availability, faster delivery, fewer cancellations
- Implementation roadmap: pilot timeline, stakeholder roles, change management
- Risks/challenges + mitigation: data quality, adoption, integration complexity

## 5. Conclusion & Future Work (~0.5 page)
- Key takeaways from pilot hypothesis
- Next steps: PoC, data partnerships, predictive procurement
- Future enhancements: reinforcement learning reorder, supplier collaboration portal

## Appendices (optional)
- Glossary, references, supplemental diagrams, mockup screenshots

## Action Items
1. Gather Turkish e-commerce dead stock statistics (source citations).
2. Draft architecture diagram (Figma/Draw.io).
3. Decide on target ML technique for narrative.
4. Prepare mockup screenshots for report & presentation.
5. Review/rescope content to stay within 3-5 pages.
