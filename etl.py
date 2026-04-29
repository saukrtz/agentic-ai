import os
import argparse
import yaml
import logging
import re
import snowflake.connector
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_args():
    """Convert script into a CLI tool using argparse."""
    parser = argparse.ArgumentParser(description="Runtime ETL Pipeline (Snowflake + Groq Agent)")
    parser.add_argument("--config", type=str, help="Path to config.yaml file")
    parser.add_argument("--pipeline_name", type=str, help="Name of the pipeline/task")
    parser.add_argument("--source_table", type=str, help="Source table (e.g., RAW_DB.RAW_SCHEMA.ST_ORDERS)")
    parser.add_argument("--target_table", type=str, help="Target table (e.g., DWH_DB.DWH_SCHEMA.TT_ORDERS)")
    parser.add_argument("--warehouse", type=str, help="Snowflake warehouse")
    return parser.parse_args()

def load_config(config_path):
    """Refactored to use config.yaml for default settings."""
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_etl_sql(source_table, target_table):
    """Use Groq LLM to generate the SELECT statement."""
    logging.info("Generating ETL SQL using Groq LLM...")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY environment variable. Check GitHub Secrets.")

    # Build Groq client via LangChain
    llm = ChatGroq(
        temperature=0, 
        model_name="mixtral-8x7b-32768", 
        groq_api_key=api_key
    )

    template = """
    You are a Senior Data Engineer.
    Write a Snowflake SQL SELECT statement to extract, clean, and transform data from the source table {source_table} 
    so it matches the schema of {target_table}.
    
    Return ONLY a single valid Snowflake SELECT statement. 
    Do not include any prose, markdown, or explanations.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm

    response = chain.invoke({"source_table": source_table, "target_table": target_table})
    raw_sql = response.content
    
    logging.info(f"Raw LLM Output:\n{raw_sql}")
    return raw_sql

def sanitize_sql(raw_sql):
    """Clean and validate model output to avoid invalid SQL."""
    logging.info("Sanitizing LLM output...")
    
    # Strip markdown fences if the LLM ignored instructions
    cleaned_sql = re.sub(r'```sql\s*', '', raw_sql, flags=re.IGNORECASE)
    cleaned_sql = re.sub(r'```\s*', '', cleaned_sql)
    cleaned_sql = cleaned_sql.strip()

    # Validate that the final SQL begins with SELECT
    if not cleaned_sql.upper().startswith("SELECT"):
        raise ValueError("Sanitization failed: The generated model output does not begin with a SELECT statement.")

    logging.info(f"Final Cleaned SELECT:\n{cleaned_sql}")
    return cleaned_sql

def get_snowflake_conn(warehouse):
    """Establish Snowflake connection using Environment Variables."""
    logging.info("Connecting to Snowflake...")
    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        role=os.getenv("SNOWFLAKE_ROLE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        warehouse=warehouse
    )

def preview_data(conn, sql):
    """Run generated SELECT separately and print preview to Actions log."""
    logging.info("Executing preview of the query output (LIMIT 10)...")
    preview_sql = f"{sql}\nLIMIT 10"
    
    with conn.cursor() as cur:
        cur.execute(preview_sql)
        results = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        
        logging.info(f"Columns: {columns}")
        for row in results:
            logging.info(row)

def execute_snowflake_task(conn, pipeline_name, target_table, sql, warehouse):
    """Ensure target table exists, create/replace TASK dynamically, and execute it."""
    logging.info("Deploying ETL logic to Snowflake...")
    
    with conn.cursor() as cur:
        # Ensure the target table exists using SELECT ... WHERE 1=0
        create_table_sql = f"CREATE TABLE IF NOT EXISTS {target_table} AS {sql} WHERE 1=0;"
        cur.execute(create_table_sql)
        logging.info(f"Verified/Created target table schema: {target_table}")

        # Create or replace task dynamically
        task_name = f"{pipeline_name}_ETL_TASK"
        task_sql = f"""
        CREATE OR REPLACE TASK {task_name}
        WAREHOUSE = {warehouse}
        AS
        INSERT INTO {target_table}
        {sql};
        """
        cur.execute(task_sql)
        logging.info(f"Created/Replaced Snowflake TASK: {task_name}")

        # Execute task immediately
        cur.execute(f"EXECUTE TASK {task_name};")
        logging.info(f"Executed TASK {task_name} immediately.")

def run():
    """Main orchestration function with error handling."""
    args = parse_args()
    config = load_config(args.config)

    # Merge CLI arguments with Config file (CLI takes precedence)
    pipeline_name = args.pipeline_name or config.get("pipeline_name", "DEFAULT_PIPELINE")
    source_table = args.source_table or config.get("source_table")
    target_table = args.target_table or config.get("target_table")
    warehouse = args.warehouse or config.get("warehouse", "COMPUTE_WH")

    # Validate required parameters
    if not source_table or not target_table:
        logging.error("Validation Error: source_table and target_table are required.")
        return

    try:
        # Extract/Generate
        raw_sql = generate_etl_sql(source_table, target_table)
        
        # Validate/Sanitize
        cleaned_sql = sanitize_sql(raw_sql)
        
        # Connect & Load
        conn = get_snowflake_conn(warehouse)
        try:
            preview_data(conn, cleaned_sql)
            execute_snowflake_task(conn, pipeline_name, target_table, cleaned_sql, warehouse)
            logging.info("ETL Pipeline completed successfully.")
        finally:
            conn.close()
            logging.info("Snowflake connection closed.")
            
    except Exception as e:
        logging.error(f"ETL failed: {e}")

if __name__ == "__main__":
    run()