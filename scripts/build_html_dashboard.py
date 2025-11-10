import os
import json
import csv
import boto3
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load project configuration file (region, bucket, Athena output path)
cfg = json.load(open("config/config.json"))
REGION = cfg["region"]
OUTPUT_PREFIX = cfg["athena_output"].replace("s3://", "")
BUCKET = OUTPUT_PREFIX.split("/")[0]
KEY_PREFIX = "/".join(OUTPUT_PREFIX.split("/")[1:])

# Creating local output directory for the dashboard
LOCAL_OUTPUT_DIR = os.path.join(os.getcwd(), "output")
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

# Defining the final HTML output path
LOCAL_DASHBOARD_PATH = os.path.join(LOCAL_OUTPUT_DIR, "olist_eda_dashboard.html")

# Initializing S3 client
s3 = boto3.client("s3", region_name=REGION)

# Function to read CSV results from S3 given a QueryExecutionId, returns header and rows
def read_csv_from_s3(qid):
    key = f"{KEY_PREFIX}{qid}.csv"
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    body = obj["Body"].read().decode("utf-8").splitlines()
    reader = csv.reader(body)
    data = list(reader)
    header, rows = data[0], data[1:]
    return header, rows

# Function to convert the value to float, returns 0.0 if conversion fails
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0
    
# Function to convert the value to int, returns 0.0 if conversion fails
def safe_int(x):
    try:
        return int(float(x))
    except Exception:
        return 0

# Main function to build the HTML dashboard
def main():
    ids = json.load(open("athena_query_ids.json"))
    if not ids:
        print("No EDA query IDs found. Skipping dashboard build.")
        return

    figs = {}
    kpis = {}

    # Orders Status Distribution 
    if "order_status_distribution" in ids:
        h, d = read_csv_from_s3(ids["order_status_distribution"])
        total_orders = sum(safe_int(r[1]) for r in d)
        kpis["Total Orders"] = f"{total_orders:,}"
        fig = go.Figure([go.Bar(x=[r[0] for r in d], y=[safe_int(r[1]) for r in d])])
        fig.update_layout(title="Order Status Distribution",
                          xaxis_title="Status", yaxis_title="Orders")
        figs["Order Status Distribution"] = fig

    # --- Monthly Orders & Revenue ---
    if "monthly_orders_revenue" in ids:
        h, d = read_csv_from_s3(ids["monthly_orders_revenue"])
        months = [r[0] for r in d]
        orders = [safe_int(r[1]) for r in d]
        revenue = [safe_float(r[2]) for r in d]
        total_rev = sum(revenue)
        kpis["Total Revenue"] = f"${total_rev:,.0f}"
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=months, y=orders, name="Orders"), secondary_y=False)
        fig.add_trace(go.Scatter(x=months, y=revenue, name="Revenue"), secondary_y=True)
        fig.update_layout(title="Monthly Orders & Revenue")
        fig.update_xaxes(title_text="Month")
        fig.update_yaxes(title_text="Orders", secondary_y=False)
        fig.update_yaxes(title_text="Revenue", secondary_y=True)
        figs["Monthly Orders & Revenue"] = fig

    # --- Payment Type Distribution ---
    if "payment_type_distribution" in ids:
        h, d = read_csv_from_s3(ids["payment_type_distribution"])
        fig = go.Figure([go.Pie(labels=[r[0] for r in d], values=[safe_float(r[2]) for r in d])])
        fig.update_layout(title="Payment Methods by Total Value")
        figs["Payment Methods"] = fig

    # --- Installments Distribution ---
    if "installments_distribution" in ids:
        h, d = read_csv_from_s3(ids["installments_distribution"])
        fig = go.Figure([go.Bar(x=[safe_int(r[0]) for r in d], y=[safe_int(r[1]) for r in d])])
        fig.update_layout(title="Installments Distribution",
                          xaxis_title="Installments", yaxis_title="Orders")
        figs["Installments Distribution"] = fig

    # --- Customers by State ---
    if "customers_by_state" in ids:
        h, d = read_csv_from_s3(ids["customers_by_state"])
        fig = go.Figure([go.Bar(x=[r[0] for r in d], y=[safe_int(r[1]) for r in d])])
        fig.update_layout(title="Customers by State",
                          xaxis_title="State", yaxis_title="Unique Customers")
        figs["Customers by State"] = fig

    # --- Top Product Categories by Sales ---
    if "top_categories_by_sales" in ids:
        h, d = read_csv_from_s3(ids["top_categories_by_sales"])
        fig = go.Figure([go.Bar(x=[r[0] for r in d], y=[safe_float(r[1]) for r in d])])
        fig.update_layout(title="Top Product Categories by Sales",
                          xaxis_title="Category", yaxis_title="Sales ($)")
        figs["Top Categories by Sales"] = fig

    # --- Review Score Distribution ---
    if "review_score_distribution" in ids:
        h, d = read_csv_from_s3(ids["review_score_distribution"])
        avg_score = sum(safe_int(r[0])*safe_int(r[1]) for r in d)/max(1,sum(safe_int(r[1]) for r in d))
        kpis["Avg Review Score"] = f"{avg_score:.2f}"
        fig = go.Figure([go.Bar(x=[safe_int(r[0]) for r in d], y=[safe_int(r[1]) for r in d])])
        fig.update_layout(title="Review Score Distribution",
                          xaxis_title="Score", yaxis_title="Reviews")
        figs["Review Score Distribution"] = fig

    # --- Delivery Delay Distribution ---
    if "delivery_delay_distribution" in ids:
        h, d = read_csv_from_s3(ids["delivery_delay_distribution"])
        delays = [safe_int(r[0]) for r in d for _ in range(safe_int(r[1]))]
        avg_delay = (sum(delays)/len(delays)) if delays else 0
        kpis["Avg Delivery Delay (days)"] = f"{avg_delay:.1f}"
        fig = go.Figure([go.Bar(x=[safe_int(r[0]) for r in d], y=[safe_int(r[1]) for r in d])])
        fig.update_layout(title="Delivery Delay Distribution (Days)",
                          xaxis_title="Delay Days", yaxis_title="Orders")
        figs["Delivery Delay Distribution"] = fig

    # --- Shipping Time Distribution ---
    if "shipping_time_distribution" in ids:
        h, d = read_csv_from_s3(ids["shipping_time_distribution"])
        fig = go.Figure([go.Bar(x=[safe_int(r[0]) for r in d], y=[safe_int(r[1]) for r in d])])
        fig.update_layout(title="Shipping Time Distribution (Days)",
                          xaxis_title="Days from Purchase to Delivery",
                          yaxis_title="Orders")
        figs["Shipping Time Distribution"] = fig

    # ----------- BUILD HTML ----------
    html_parts = [
"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Olist E-Commerce Dashboard</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body class="bg-light">
<div class="container my-4">
  <h1 class="text-center mb-3">Brazilian E-Commerce EDA Dashboard</h1>
  <p class="text-center text-muted">Orders & Sales • Payments • Customers • Products • Logistics</p>
  <hr>
  <div class="row text-center mb-4">
"""
    ]

    # KPI cards
    for name, val in kpis.items():
        html_parts.append(f"""
        <div class="col-md-3 mb-3">
          <div class="card shadow-sm border-0">
            <div class="card-body">
              <h6 class="text-muted">{name}</h6>
              <h4 class="fw-bold">{val}</h4>
            </div>
          </div>
        </div>
        """)

    html_parts.append("</div>")  # end KPI row

    # Section rendering
    for title, fig in figs.items():
        fig_html = fig.to_html(full_html=False, include_plotlyjs=False)
        html_parts.append(f"""
        <div class="accordion mb-3" id="accordion-{title.replace(' ','-')}">
          <div class="accordion-item">
            <h2 class="accordion-header" id="heading-{title}">
              <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse"
                data-bs-target="#collapse-{title.replace(' ','-')}" aria-expanded="false"
                aria-controls="collapse-{title.replace(' ','-')}">
                {title}
              </button>
            </h2>
            <div id="collapse-{title.replace(' ','-')}" class="accordion-collapse collapse"
                 aria-labelledby="heading-{title}" data-bs-parent="#accordion-{title.replace(' ','-')}">
              <div class="accordion-body">{fig_html}</div>
            </div>
          </div>
        </div>
        """)

    html_parts.append("""
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>
document.addEventListener("shown.bs.collapse", function(e){
  const plots = e.target.querySelectorAll('.js-plotly-plot');
  plots.forEach(p => { Plotly.Plots.resize(p); });
});
</script>
</body></html>
""")

    html = "\n".join(html_parts)
    with open(LOCAL_DASHBOARD_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard Saved Locally → {LOCAL_DASHBOARD_PATH}")

if __name__ == "__main__":
    main()
