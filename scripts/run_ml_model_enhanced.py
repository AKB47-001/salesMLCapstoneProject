"""
Enhanced ML pipeline for Olist dataset.

- Uses the same wrangled dataset and feature engineering logic
  as the existing ML script (Late Delivery Prediction).
- Trains multiple models:
    * Logistic Regression
    * Random Forest
    * Gradient Boosting
    * Support Vector Machine (SVC)
- Computes extended metrics:
    * Accuracy, AUC, Precision, Recall, F1-score, Training Time
- Builds an ML-focused HTML dashboard:
    * KPI cards
    * Model comparison table
    * ROC curves
    * Feature importance plots
    * Class imbalance chart
    * Correlation heatmap
- Saves trained models + scaler to output/models as .pkl files.
- Outputs HTML to: output/olist_ml_dashboard.html

This file is standalone and DOES NOT change or depend on the
existing run_ml_model.py; you can call it separately.
"""

import os
import json
import time
import boto3
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib

# -------------------------------------------------------------------
# Config & Paths
# -------------------------------------------------------------------

config = json.load(open("config/config.json"))
AWS_REGION = config["region"]
ATHENA_OUTPUT_PREFIX = config["athena_output"].replace("s3://", "")
S3_BUCKET = ATHENA_OUTPUT_PREFIX.split("/")[0]

# Local directories
LOCAL_OUTPUT_DIR = os.path.join(os.getcwd(), "output")
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

LOCAL_DASHBOARD_PATH = os.path.join(LOCAL_OUTPUT_DIR, "olist_ml_dashboard.html")

DATA_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR, exist_ok=True)
LOCAL_WRANGLED_CSV = os.path.join(DATA_DIR, "wrangled_sample.csv")

MODELS_DIR = os.path.join(LOCAL_OUTPUT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# AWS clients (kept for compatibility; Athena/S3 used only if we need to fetch data)
s3_client = boto3.client("s3", region_name=AWS_REGION)


# -------------------------------------------------------------------
# Data Fetching
# -------------------------------------------------------------------

def fetch_dataset_from_athena():
    """
    Fetches wrangled data from Athena using awswrangler and stores it locally.
    """
    import awswrangler as wr

    print(f"Exporting wrangled dataset from Athena ({config['catalog_database']}.wrangled)...")
    database_name = config["catalog_database"]

    sql_query = f"""
    SELECT
      customer_state, seller_state,
      price, freight_value, product_weight_g,
      product_category_name_english,
      order_delivered_customer_date, order_estimated_delivery_date
    FROM {database_name}.wrangled
    WHERE order_delivered_customer_date IS NOT NULL
    """

    df = wr.athena.read_sql_query(
        sql=sql_query,
        database=database_name,
        ctas_approach=False,
        workgroup="primary",
        s3_output=f"s3://{S3_BUCKET}/athena-results/",
        boto3_session=boto3.Session(region_name=AWS_REGION),
    )

    os.makedirs(os.path.dirname(LOCAL_WRANGLED_CSV), exist_ok=True)
    df.to_csv(LOCAL_WRANGLED_CSV, index=False)
    print(f"Data exported locally → {LOCAL_WRANGLED_CSV}")
    return df


def load_or_fetch_data():
    """
    Loads the wrangled dataset from local CSV if present,
    otherwise fetches it from Athena.
    """
    if os.path.exists(LOCAL_WRANGLED_CSV):
        print(f"Loaded existing dataset from {LOCAL_WRANGLED_CSV}")
        return pd.read_csv(LOCAL_WRANGLED_CSV)
    else:
        try:
            return fetch_dataset_from_athena()
        except Exception as error:
            print("Athena fetch failed with error", error)
            print("No data for ML model.")
            raise


# -------------------------------------------------------------------
# Feature Engineering
# -------------------------------------------------------------------

def prepare_features(df: pd.DataFrame):
    """
    Builds features and target for Late Delivery prediction.
    Target:
        is_late = 1 if actual delivery date > estimated date, else 0.
    """

    print("Preparing features for ML...")

    # Calculate delay in days and create binary target variable (is_late)
    df["delay_days"] = (
        pd.to_datetime(df["order_delivered_customer_date"])
        - pd.to_datetime(df["order_estimated_delivery_date"])
    ).dt.days

    # Preserve delay_days for correlation analysis (fill missing with 0)
    df["delay_days"] = df["delay_days"].fillna(0)

    # Create binary target
    df["is_late"] = (df["delay_days"] > 0).astype(int)

    # Select required columns for ML
    columns_needed = [
        "price",
        "freight_value",
        "product_weight_g",
        "customer_state",
        "seller_state",
        "product_category_name_english",
        "is_late",
        "delay_days"  # ✅ keep delay_days in dataframe
    ]

    # Keep only relevant columns and drop NA
    df = df[[col for col in columns_needed if col in df.columns]].dropna()

    # Convert numeric fields to float
    for numeric_col in ["price", "freight_value", "product_weight_g"]:
        if numeric_col in df.columns:
            df[numeric_col] = df[numeric_col].astype(float)

    # Encode categorical features numerically
    for cat_col in ["customer_state", "seller_state", "product_category_name_english"]:
        if cat_col in df.columns:
            df[cat_col] = LabelEncoder().fit_transform(df[cat_col].astype(str))

    # Prepare feature matrix and target
    X = df.drop(columns=["is_late"])
    y = df["is_late"]

    # Standardize numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Feature matrix shape: {X.shape}, Target positives (late deliveries): {y.sum()}")

    # ✅ Return df so correlation heatmap can access delay_days
    return X, X_scaled, y, scaler, df



# -------------------------------------------------------------------
# Model Training & Evaluation
# -------------------------------------------------------------------

def train_models(X_scaled, y):
    """
    Trains multiple models on the scaled feature set and returns a
    dictionary of metrics and fitted model objects.
    """

    print("Splitting train/test datasets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training models (LogReg, RandomForest, GradientBoosting, SVC)...")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, max_depth=10, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        # "SVC (RBF)": SVC(probability=True, kernel="rbf", random_state=42),
    }

    model_results = {}

    for model_name, model_instance in models.items():
        print(f" → Training {model_name}...")
        t0 = time.time()
        model_instance.fit(X_train, y_train)
        train_time = time.time() - t0      

        y_pred = model_instance.predict(X_test)
        y_pred_proba = model_instance.predict_proba(X_test)[:, 1]

        auc_score = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        model_results[model_name] = {
            "model": model_instance,
            "y_pred": y_pred,
            "y_proba": y_pred_proba,
            "auc": auc_score,
            "acc": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "train_time": train_time,
            "conf_matrix": confusion_matrix(y_test, y_pred),
        }

        print(
            f"{model_name}: "
            f"AUC={auc_score:.3f}, Acc={accuracy:.3f}, "
            f"Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}, "
            f"TrainTime={train_time:.2f}s"
        )

    return model_results, (X_train, X_test, y_train, y_test)


# -------------------------------------------------------------------
# Plotly Visualizations
# -------------------------------------------------------------------

def build_roc_figure(model_results):
    fig = go.Figure()
    for model_name, metrics in model_results.items():
        fpr, tpr, _ = roc_curve(metrics["y_pred"], metrics["y_proba"])
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{model_name} (AUC={metrics['auc']:.3f})",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    return fig


def build_confusion_matrix_figure(model_results):
    """
    Builds a 2x2 grid of heatmaps (or fewer, depending on models) for confusion matrices.
    """
    model_names = list(model_results.keys())
    n_models = len(model_names)
    cols = min(2, n_models)
    rows = (n_models + 1) // 2

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=model_names,
    )

    idx = 0
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            if idx >= n_models:
                break
            name = model_names[idx]
            cm = model_results[name]["conf_matrix"]
            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    x=["Pred OnTime", "Pred Late"],
                    y=["Actual OnTime", "Actual Late"],
                    showscale=False,
                ),
                row=r,
                col=c,
            )
            idx += 1

    fig.update_layout(title="Confusion Matrices")
    return fig


def build_feature_importance_figures(model_results, feature_names):
    figs = {}

    # Random Forest
    if "Random Forest" in model_results:
        rf_model = model_results["Random Forest"]["model"]
        importances = rf_model.feature_importances_
        series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        figs["Random Forest Feature Importance"] = go.Figure(
            [go.Bar(x=series.index, y=series.values)]
        ).update_layout(
            title="Random Forest Feature Importance",
            xaxis_title="Feature",
            yaxis_title="Importance",
        )

    # Gradient Boosting
    if "Gradient Boosting" in model_results:
        gb_model = model_results["Gradient Boosting"]["model"]
        importances = gb_model.feature_importances_
        series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        figs["Gradient Boosting Feature Importance"] = go.Figure(
            [go.Bar(x=series.index, y=series.values)]
        ).update_layout(
            title="Gradient Boosting Feature Importance",
            xaxis_title="Feature",
            yaxis_title="Importance",
        )

    return figs


def build_class_balance_figure(y):
    counts = y.value_counts().sort_index()
    labels = ["On Time (0)", "Late (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
            )
        ]
    )
    fig.update_layout(
        title="Class Balance: On-Time vs Late Deliveries",
        xaxis_title="Class",
        yaxis_title="Count",
    )
    return fig


def build_correlation_heatmap(df_with_target):
    corr = df_with_target.corr()
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorbar=dict(title="Correlation"),
            )
        ]
    )
    fig.update_layout(title="Feature Correlation Heatmap")
    return fig


def build_model_comparison_table_html(model_results):
    """
    Returns a Bootstrap-styled HTML table summarizing model metrics.
    """
    headers = ["Model", "AUC", "Accuracy", "Precision", "Recall", "F1", "Train Time (s)"]
    rows_html = ""
    for name, metrics in model_results.items():
        rows_html += f"""
        <tr>
          <td>{name}</td>
          <td>{metrics['auc']:.3f}</td>
          <td>{metrics['acc']:.3f}</td>
          <td>{metrics['precision']:.3f}</td>
          <td>{metrics['recall']:.3f}</td>
          <td>{metrics['f1']:.3f}</td>
          <td>{metrics['train_time']:.2f}</td>
        </tr>
        """

    table_html = f"""
    <div class="card shadow-sm mb-4">
      <div class="card-body">
        <h5 class="card-title mb-3">Model Performance Comparison</h5>
        <div class="table-responsive">
          <table class="table table-striped table-hover align-middle">
            <thead>
              <tr>
                {''.join(f'<th scope="col">{h}</th>' for h in headers)}
              </tr>
            </thead>
            <tbody>
              {rows_html}
            </tbody>
          </table>
        </div>
      </div>
    </div>
    """
    return table_html


# -------------------------------------------------------------------
# Model Saving
# -------------------------------------------------------------------

def save_models_and_scaler(model_results, scaler):
    """
    Saves all trained models and the scaler to output/models as .pkl files.
    """
    for name, metrics in model_results.items():
        model = metrics["model"]
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        path = os.path.join(MODELS_DIR, f"{safe_name}.pkl")
        joblib.dump(model, path)
        print(f"Saved model → {path}")

    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler → {scaler_path}")


# -------------------------------------------------------------------
# HTML Dashboard Builder
# -------------------------------------------------------------------

def build_ml_dashboard(
    model_results,
    roc_fig,
    cm_fig,
    feature_figs,
    class_balance_fig,
    corr_fig,
    df_with_target,
):
    """
    Builds the ML-focused HTML dashboard and saves it to LOCAL_DASHBOARD_PATH.
    """
    print(f"Building ML dashboard → {LOCAL_DASHBOARD_PATH}")

    # Decide best model by AUC
    best_model_name = max(model_results.items(), key=lambda kv: kv[1]["auc"])[0]
    best_model_auc = model_results[best_model_name]["auc"]
    best_model_acc = model_results[best_model_name]["acc"]

    y = df_with_target["is_late"]
    total_samples = len(df_with_target)
    late_ratio = y.mean() * 100 if total_samples > 0 else 0.0
    avg_delay = df_with_target["delay_days"].mean() if "delay_days" in df_with_target.columns else 0.0

    kpis = {
        "Total Samples": f"{total_samples:,}",
        "Late Delivery %": f"{late_ratio:.1f}%",
        "Avg Delay (days)": f"{avg_delay:.2f}",
        "Best Model (AUC)": best_model_name,
        "Best Model AUC": f"{best_model_auc:.3f}",
        "Best Model Accuracy": f"{best_model_acc:.3f}",
    }

    model_table_html = build_model_comparison_table_html(model_results)

    html_parts = [
        """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Olist ML Insights Dashboard</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body class="bg-light">
<div class="container my-4">
  <h1 class="text-center mb-3">Brazilian E-Commerce ML Insights</h1>
  <p class="text-center text-muted">Late Delivery Prediction • Model Evaluation • Feature Insights</p>
  <hr>
  <div class="row text-center mb-4">
"""
    ]

    # KPI cards
    for name, val in kpis.items():
        html_parts.append(
            f"""
        <div class="col-md-4 mb-3">
          <div class="card shadow-sm border-0">
            <div class="card-body">
              <h6 class="text-muted">{name}</h6>
              <h4 class="fw-bold">{val}</h4>
            </div>
          </div>
        </div>
        """
        )

    html_parts.append("</div>")  # end KPI row

    # Model comparison table
    html_parts.append(model_table_html)

    # ROC Curve
    html_parts.append(
        f"""
    <div class="card shadow-sm mb-4">
      <div class="card-body">
        <h5 class="card-title">ROC Curve Comparison</h5>
        {roc_fig.to_html(full_html=False, include_plotlyjs=False)}
      </div>
    </div>
    """
    )

    # Confusion matrices
    html_parts.append(
        f"""
    <div class="card shadow-sm mb-4">
      <div class="card-body">
        <h5 class="card-title">Confusion Matrices</h5>
        {cm_fig.to_html(full_html=False, include_plotlyjs=False)}
      </div>
    </div>
    """
    )

    # Class balance
    html_parts.append(
        f"""
    <div class="card shadow-sm mb-4">
      <div class="card-body">
        <h5 class="card-title">Class Balance</h5>
        {class_balance_fig.to_html(full_html=False, include_plotlyjs=False)}
      </div>
    </div>
    """
    )

    # Feature importance figs
    for title, fig in feature_figs.items():
        html_parts.append(
            f"""
    <div class="card shadow-sm mb-4">
      <div class="card-body">
        <h5 class="card-title">{title}</h5>
        {fig.to_html(full_html=False, include_plotlyjs=False)}
      </div>
    </div>
    """
        )

    # Correlation heatmap
    html_parts.append(
        f"""
    <div class="card shadow-sm mb-4">
      <div class="card-body">
        <h5 class="card-title">Correlation Heatmap</h5>
        {corr_fig.to_html(full_html=False, include_plotlyjs=False)}
      </div>
    </div>
    """
    )

    html_parts.append(
        """
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body></html>
"""
    )

    html = "\n".join(html_parts)
    with open(LOCAL_DASHBOARD_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"ML Insights Dashboard saved → {LOCAL_DASHBOARD_PATH}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("\n===============================================")
    print("🤖 Enhanced ML Pipeline: Late Delivery Prediction")
    print("   (LogReg + RF + GradientBoosting + SVC)")
    print("===============================================\n")

    # 1. Load data
    wrangled_df = load_or_fetch_data()

    # 2. Prepare features
    X, X_scaled, y, scaler, df_with_target = prepare_features(wrangled_df.copy())

    # 3. Train models
    model_results, _ = train_models(X_scaled, y)

    # 4. Save models & scaler
    save_models_and_scaler(model_results, scaler)

    # 5. Build visualizations
    roc_fig = build_roc_figure(model_results)
    cm_fig = build_confusion_matrix_figure(model_results)
    feature_figs = build_feature_importance_figures(model_results, X.columns)
    class_balance_fig = build_class_balance_figure(y)
    cols_for_corr = ["delay_days", "is_late"] + [c for c in X.columns if c in df_with_target.columns]
    corr_fig = build_correlation_heatmap(df_with_target[cols_for_corr])

    # 6. Build HTML dashboard
    build_ml_dashboard(
        model_results=model_results,
        roc_fig=roc_fig,
        cm_fig=cm_fig,
        feature_figs=feature_figs,
        class_balance_fig=class_balance_fig,
        corr_fig=corr_fig,
        df_with_target=df_with_target,
    )

    print("\n✅ Enhanced ML analysis completed.")
    print("📊 ML Dashboard → output/olist_ml_dashboard.html")
    print("📁 Models saved in → output/models/")
