from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.utilities import SQLDatabase
import uvicorn
from typing import Optional
import os
import boto3
import json
import time
from botocore.config import Config
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# FastAPI configuration
app = FastAPI(
    title="🤖 Fast SQL Agent with SageMaker DeepSeek",
    description="Fast SQL Agent using DeepSeek on Amazon SageMaker",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres123")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5499")
DB_NAME = os.getenv("DB_NAME", "northwind")
DB_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# SageMaker configuration
SAGEMAKER_ENDPOINT = "deepseek-coder-endpoint"
AWS_REGION = "us-west-2"

print(f"🔗 Database: {DB_URI}")
print(f"🤖 SageMaker endpoint: {SAGEMAKER_ENDPOINT}")

# Pydantic models
class QueryRequest(BaseModel):
    pregunta: str
    verbose: Optional[bool] = True

# SageMaker + DB Client
class SimpleSageMakerClient:
    def __init__(self):
        cfg = Config(
            read_timeout=30,  # optimized for fast queries
            connect_timeout=5,
            retries={"max_attempts": 0}
        )
        self.endpoint_name = SAGEMAKER_ENDPOINT
        self.runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION, config=cfg)
        self.db = SQLDatabase.from_uri(DB_URI)

    def get_minimal_schema(self, relevant_tables):
        """Get only relevant tables and columns to reduce prompt size."""
        schema_info = {}
        with self.db._engine.connect() as conn:
            for table in relevant_tables:
                cols = conn.execute(f"""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = '{table}'
                    ORDER BY ordinal_position
                """).fetchall()
                schema_info[table] = [c[0] for c in cols]
        return "\n".join([f"{t}: {', '.join(cols)}" for t, cols in schema_info.items()])

    def call_sagemaker(self, prompt, max_new_tokens=64):
        """Call SageMaker endpoint."""
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0,
                    "do_sample": False,
                    "return_full_text": False
                }
            }
            response = self.runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload)
            )
            result = json.loads(response["Body"].read().decode())
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "").strip()
            elif isinstance(result, dict):
                return result.get("generated_text", "")
            return str(result)
        except Exception as e:
            raise RuntimeError(f"SageMaker invocation failed: {e}")

    def generate_sql_query(self, question):
        """Generate SQL query in English using minimal schema."""
        schema_info = self.get_minimal_schema(["customers", "orders", "order_details", "products"])
        prompt = f"""
You are an expert in PostgreSQL.
Only use the tables and columns below:

{schema_info}

Write ONLY the SQL query to answer the following question, no explanations, no backticks.

Question: {question}
"""
        sql_query = self.call_sagemaker(prompt)
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        if not sql_query.lower().startswith(("select", "with", "insert", "update", "delete")):
            raise ValueError(f"Invalid SQL generated: {sql_query}")
        return sql_query

    def execute_sql_query(self, sql_query):
        """Run SQL query against the database."""
        try:
            print(f"🔍 Executing SQL: {sql_query}")
            result = self.db.run(sql_query)
            return result
        except Exception as e:
            raise RuntimeError(f"Error executing SQL: {e}")

    def answer_question(self, question):
        """Generate SQL, execute it, and return results."""
        try:
            sql_query = self.generate_sql_query(question)
        except Exception as e:
            return f"❌ Could not generate SQL: {e}"

        try:
            sql_result = self.execute_sql_query(sql_query)
        except Exception as e:
            return f"❌ {e}"

        return {
            "sql": sql_query,
            "result": [dict(row) for row in sql_result]
        }

# Init client
print("🚀 Initializing configuration...")
try:
    sagemaker_client = SimpleSageMakerClient()
    print("🧪 Testing SageMaker connection...")
    test_response = sagemaker_client.call_sagemaker("SELECT 1;")
    print(f"✅ SageMaker test OK: {test_response[:50]}...")
    db_test = sagemaker_client.db.run("SELECT COUNT(*) FROM customers")
    print(f"✅ DB test OK: {db_test[0][0]} customers")
except Exception as e:
    print(f"❌ Critical error: {e}")
    sagemaker_client = None

# Static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("frontend/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        if not sagemaker_client:
            return {"status": "error", "detail": "Client not initialized"}
        start_time = time.time()
        sagemaker_client.call_sagemaker("Test")
        response_time = time.time() - start_time
        return {
            "status": "ok",
            "database": "connected",
            "llm_model": f"DeepSeek-Coder via SageMaker ({response_time:.2f}s)",
            "endpoint": SAGEMAKER_ENDPOINT,
            "region": AWS_REGION
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/simple-query")
async def consulta_simple(request: QueryRequest):
    """Process a question and return SQL + results."""
    try:
        if not sagemaker_client:
            raise HTTPException(status_code=500, detail="Client unavailable")
        print(f"🔍 Processing: {request.pregunta}")
        start_time = time.time()
        respuesta = sagemaker_client.answer_question(request.pregunta)
        response_time = time.time() - start_time
        return {
            "question": request.pregunta,
            "response": respuesta,
            "time": f"{response_time:.2f}s"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

@app.post("/query")
async def hacer_consulta(request: QueryRequest):
    """Alias for simple-query."""
    return await consulta_simple(request)

if __name__ == "__main__":
    print("🚀 Starting Fast SQL Agent...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
