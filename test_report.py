import pandas as pd

from custom_eda.reporting import generate_eda_report
from custom_eda.html_report import generate_html_report

# fake dataset
df = pd.DataFrame({
    "age": [20, 21, 22, 23, 24],
    "income": [20000, 21000, 25000, 24000, 30000],
    "passed_exam": [0, 1, 1, 1, 0],
    "gender": ["M", "F", "F", "M", "F"]
})

# 1. generate report dictionary
report = generate_eda_report(
    df,
    correlation_method="pearson",
    correlation_threshold=0.3,
    strong_corrs_for_scatter=True
)

# 2. generate HTML file
output_file = generate_html_report(
    report,
    report_title="Test EDA Report"
)

print("Report generated:", output_file)

print(report["plots"].keys())