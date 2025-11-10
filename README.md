## 💡 Project Summary

This project analyzes **Brazilian e-commerce sales data** to understand and predict **delivery performance**.  
It extracts valuable business insights such as:

- Sales trends by month, region, and category  
- Payment method distribution  
- Review sentiment & score patterns  
- Delivery delays and shipping times  
- Predictive modeling of **late deliveries**

The pipeline combines **data engineering (ETL)**, **data analytics (Athena EDA)**, and **machine learning (ML)**  
into a single automated workflow.


## 🧱 AWS Components Used

| Component              | Purpose                                                                 |
| ---------------------- | ----------------------------------------------------------------------- |
| **AWS S3**             | Data Lake storage for raw → normalized → preprocessed → wrangled layers |
| **AWS Glue (ETL)**     | PySpark-based normalization, cleaning, and wrangling jobs               |
| **AWS Glue Crawler**   | Auto-catalog S3 Parquet tables into Athena database                     |
| **AWS Athena**         | Serverless SQL analytics engine                                         |
| **AWS Boto3 SDK**      | Automate Glue, Athena, and S3 workflows                                 |
| **AWS Wrangler**       | Pandas ↔ Athena data transfer helper                                    |
| **Plotly + Bootstrap** | Interactive HTML dashboards                                             |
| **scikit-learn**       | Machine learning and model evaluation                                   |

## 🧰 Tech Stack

| Category | Technologies |
|-----------|---------------|
| **Language** | Python 3.12+ |
| **Data Engineering** | PySpark (AWS Glue), AWS Glue Crawler |
| **Data Storage** | Amazon S3 (Raw → Normalized → Preprocessed → Wrangled Layers) |
| **Data Analytics** | AWS Athena, AWS Wrangler |
| **Visualization** | Plotly, Bootstrap, Matplotlib, Seaborn |
| **Machine Learning** | scikit-learn, joblib |
| **AWS SDKs** | boto3, botocore |
| **Utilities** | pandas, numpy |
| **Deployment** | AWS Glue ETL Jobs, Athena Workgroup, S3 Dashboards |


⚙️ Setup Instructions
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/aws-etl-ecommerce-pipeline.git
cd aws-etl-ecommerce-pipeline

2️⃣ Create a Virtual Environment
python -m venv venv


Activate it:

Windows (PowerShell)

venv\Scripts\activate


macOS/Linux

source venv/bin/activate

3️⃣ Upgrade pip and Tooling
pip install --upgrade pip setuptools wheel

4️⃣ Install Dependencies
pip install -r requirements_resolved.txt


(Or use requirements.txt if requirements_resolved.txt is unavailable.)

5️⃣ Configure AWS Credentials

Ensure your environment has valid AWS credentials with access to:

The S3 bucket defined in config/config.json

AWS Glue and Athena services

IAM role AWSGlueServiceRole-ETL

Check your identity:

aws sts get-caller-identity

6️⃣ Run the Full Pipeline
python scripts/run_full_pipeline.py


This single command will:

Execute all AWS Glue ETL jobs sequentially

Catalog the wrangled data using AWS Glue Crawler

Run Athena queries for EDA

Generate the EDA dashboard

Train baseline and enhanced ML models

Produce two dashboards under output/



