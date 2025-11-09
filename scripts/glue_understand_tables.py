import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

# Adding support for handling runtime options
args = getResolvedOptions(sys.argv, ["JOB_NAME", "NORMALIZED_S3"])
NORMALIZED_S3_PATH = args["NORMALIZED_S3"].rstrip("/")

# Performing the configuration part
glueUnderstandSparkContextObject = SparkContext()
understandGlueContext = GlueContext(glueUnderstandSparkContextObject)
spark_session = understandGlueContext.spark_session
job = Job(understandGlueContext)
job.init(args["JOB_NAME"], args)

# Defining Tables to read data from
TABLE_NAMES = [
    "orders",
    "order_items",
    "order_payments",
    "order_reviews",
    "customers",
    "sellers",
    "products",
    "geolocation",
    "product_category_name_translation"
]

# Loop through every normalised Parquet table in s3
# Printing record count, schema, and first 10 rows
for table_name in TABLE_NAMES:
    table_path = f"{NORMALIZED_S3_PATH}/{table_name}"
    print(f"\n🔍 Inspecting table: {table_name} at {table_path}")
    try:
        # Read table data from S3
        table_df = spark_session.read.parquet(table_path)
        # Print row count
        print(f"   Rows: {table_df.count()}")
        # Print schema structure
        print("   Schema:")
        table_df.printSchema()
        # Show first few rows (non-truncated)
        print("   First 10 rows:")
        table_df.show(10, truncate=False)
    except Exception as e:
        print(f"Could not read {table_name}: {e}")

# Finally printing the complete status
print("\nUnderstanding Data job completed successfully.")
job.commit()
