# Custom EDA Report Generator

A lightweight Python tool that automatically generates an Exploratory Data Analysis (EDA) report including:

- Dataset overview
- Data type summary
- Missing/duplicate analysis
- Correlation analysis
- Interactive Plotly visualisations
- HTML report export

---

## Installation

```bash
pip install -r requirements.txt


from custom_eda.reporting import generate_eda_report
from custom_eda.html_report import generate_html_report

report = generate_eda_report(df)
generate_html_report(report, "My EDA Report")