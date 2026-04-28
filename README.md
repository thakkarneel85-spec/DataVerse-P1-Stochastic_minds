<div align="center">
  <img src="https://img.shields.io/badge/Status-Hackathon_Ready-success?style=for-the-badge" alt="Status" />
  <img src="https://img.shields.io/badge/Domain-Spatial_Data_Science-blue?style=for-the-badge" alt="Domain" />
  <img src="https://img.shields.io/badge/Tech-Python_%7C_Streamlit-orange?style=for-the-badge" alt="Tech" />
  <br/><br/>
  <h1>🌡️ Urban Heat Island (UHI) Intelligence Platform</h1>
  <p><strong>A Data-Driven Approach to Identifying, Mitigating, and Predicting Heat Vulnerability in Cities</strong></p>
</div>

---

## 🚨 The Problem: Cities are Baking
Urban Heat Islands (UHIs) are localized zones in cities that experience significantly higher temperatures than their rural surroundings. This isn't just an inconvenience—it's a public health crisis. 
* **Increased Mortality:** Extreme heat is a leading cause of weather-related deaths.
* **Energy Drain:** Higher temperatures demand more air conditioning, stressing power grids.
* **Inequality:** Vulnerable and low-income populations are often disproportionately affected.

## 💡 Our Solution
We built an **interactive, end-to-end analytical platform** that doesn't just show *where* it's hot, but explains *why* it's hot, and *what* it means for public health.

By combining spatial statistics, time-series forecasting, and machine learning, our dashboard provides city planners and public health officials with actionable intelligence:
1. **Identify hotspots** using spatial autocorrelation (LISA maps).
2. **Understand drivers** (e.g., asphalt vs. tree cover) using Quantile Regression.
3. **Forecast risks** using advanced SARIMAX modeling on an integrated Risk Index.
4. **Quantify impact** on health metrics like heatstroke and fatigue.

---

## ✨ Key Features & Analytical Modules

Our Streamlit dashboard brings complex academic models into an intuitive UI:

* 📊 **Urban Surface EDA:** Explore the relationships between 500+ neighbourhoods' structural factors (building density, tree cover, income).
* 📍 **Spatial Clusters (LISA):** Moran's I and LISA cluster maps to visually pinpoint "Hot-Hot" zones requiring immediate intervention.
* 🔬 **PCA Decomposition:** Dimensionality reduction to simplify complex urban indicators without losing information.
* 📈 **Quantile Regression:** Understand how variables like tree cover affect the *extreme* hottest days differently than average days.
* 📡 **SARIMAX Forecasting:** 30-day predictive modeling for both pure Surface Temperature and our composite **Heat Risk Index** (Temp + AQI + Health).
* 🏥 **Temperature → Health Impact:** Lag-adjusted OLS models proving the direct correlation between heat spikes and hospital admissions.

---

## 🚀 Quick Start (How to Run)

The project includes a stunning, interactive Streamlit dashboard that runs out-of-the-box (with built-in realistic synthetic data for immediate demonstration).

### Prerequisites
* Python 3.8+
* `pip`

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd DataVerse-P1-Stochastic_minds
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Dashboard:**
   ```bash
   streamlit run dashboard.py
   ```

*(Note: The dashboard automatically generates realistic synthetic urban data if real CSVs are not present, ensuring a flawless demo experience!)*

---

## 🛠️ Tech Stack
* **Frontend/Dashboard:** Streamlit, Plotly, HTML/CSS (Custom Premium Styling)
* **Data Manipulation:** Pandas, NumPy
* **Geospatial & Statistics:** GeoPandas, Scipy, Statsmodels (SARIMAX, OLS, Quantile Reg), Scikit-Learn (PCA)
* **Spatial Autocorrelation:** PySAL, ESDA

---

## 📚 Deep Dive (Jupyter Notebooks)
For judges interested in the rigorous statistical proofs behind our dashboard, the raw Jupyter Notebooks are provided:
* `UrbanSurface.ipynb` & `Auto_Correlation.ipynb`
* `Spatial Auto Correlation.ipynb` & `Principle Component Analysis.ipynb`
* `Quantile Regression.ipynb`
* `Times_series_modal.ipynb` & `Tempreture_Health.ipynb`

---
<div align="center">
  <i>Built with ❤️ for the DataVerse Hackathon</i>
</div>
