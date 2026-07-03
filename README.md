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

```
## Using on Colab

```
pip install git+https://github.com/DanialNayyar/auto-eda-report.git


from custom_eda.reporting import generate_eda_report
from custom_eda.html_report import generate_html_report

import pandas as pd

# Load dataset
df = pd.read_csv("your_dataset.csv")

# Generate report object
report = generate_eda_report(
    df,
    strong_corrs_for_scatter=True,
    correlation_method="pearson",
    correlation_threshold=0.7
)

# Generate HTML report
output_file = generate_html_report(
    report,
    report_title="My EDA Report"
)

print("Report saved as:", output_file)

```
