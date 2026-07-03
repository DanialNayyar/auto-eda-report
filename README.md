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
import pandas as pd

!git clone https://github.com/DanialNayyar/auto-eda-report.git
%cd auto-eda-report

from custom_eda.reporting import generate_eda_report

print("Import successful!")

df = pd.read_csv("your_dataset.csv")

report = generate_eda_report(
    df,
    strong_corrs_for_scatter=True,
    correlation_threshold=0.7
)

```
