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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Configuración FastAPI
app = FastAPI(
    title="🤖 SQL Agent with Bedrock DeepSeek",
    description="Agente SQL usando DeepSeek-R1 en Amazon Bedrock (sin EC2)",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de base de datos
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres123") 
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5499")
DB_NAME = os.getenv("DB_NAME", "northwind")
DB_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Configuración Bedrock (sin EC2!)
AWS_REGION = "us-east-1"  # Región confirmada que funciona
BEDROCK_MODEL_ID = "us.deepseek.r1-v1:0"  # ID del perfil de inferencia correcto

print(f"🔗 Base de datos: {DB_URI}")
print(f"🤖 Bedrock modelo: {BEDROCK_MODEL_ID}")
print(f"🌎 Región: {AWS_REGION}")

# Modelos Pydantic
class QueryRequest(BaseModel):
    pregunta: str
    verbose: Optional[bool] = True

# Cliente Bedrock para DeepSeek
class BedrockDeepSeekClient:
    def __init__(self):
        self.model_id = BEDROCK_MODEL_ID
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)
        self.db = SQLDatabase.from_uri(DB_URI)
    
    def call_bedrock(self, prompt, max_tokens=500):
        """Llamar a DeepSeek-R1 en Bedrock (completamente gestionado)"""
        try:
            # Formato específico para DeepSeek-R1
            formatted_prompt = f"<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜><think>\n"
            
            # Configuración para DeepSeek-R1
            body = {
                "prompt": formatted_prompt,
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "top_p": 0.9,
                "stop": ["<｜User｜>", "<｜end▁of▁sentence｜>"]
            }
            
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json'
            )
            
            result = json.loads(response['body'].read())
            # DeepSeek-R1 devuelve en formato 'choices'
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0].get('text', '').strip()
            else:
                return result.get('completion', str(result)).strip()
                
        except Exception as e:
            print(f"Error en Bedrock: {str(e)}")
            return f"Error: {str(e)}"
    
    def generate_sql_query(self, question):
        """Generate SQL query using DeepSeek-R1"""
        # Get database schema
        schema_info = self.db.get_table_info()
        
        # Optimized prompt for DeepSeek-R1 (English)
        prompt = f"""You are an expert SQL developer. Given the following Northwind database schema:

{schema_info}

User question: {question}

Generate ONLY the SQL query needed to answer this question. 
Do not include explanations, just clean and executable SQL.

SQL:"""
        
        sql_query = self.call_bedrock(prompt, max_tokens=200)
        
        # Clean response
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        if sql_query.startswith("SQL:"):
            sql_query = sql_query[4:].strip()
            
        return sql_query
    
    def execute_sql_query(self, sql_query):
        """Ejecutar consulta SQL en la base de datos"""
        try:
            print(f"🔍 Ejecutando SQL: {sql_query}")
            result = self.db.run(sql_query)
            return result
            
        except Exception as e:
            print(f"Error ejecutando SQL: {str(e)}")
            return f"Error ejecutando consulta: {str(e)}"
    
    def answer_question(self, question):
        """Answer complete question using DeepSeek-R1"""
        try:
            # Step 1: Generate SQL
            sql_query = self.generate_sql_query(question)
            
            # Step 2: Execute SQL
            sql_result = self.execute_sql_query(sql_query)
            
            # Step 3: Format response
            if "Error" in str(sql_result):
                return f"❌ {sql_result}"
            
            # Generate natural response with DeepSeek-R1
            response_prompt = f"""Question: {question}
SQL Query executed: {sql_query}
Results: {sql_result}

Provide a clear and natural response in the same language as the question, interpreting these results:"""
            
            natural_response = self.call_bedrock(response_prompt, max_tokens=300)
            
            return f"""
📊 **SQL Query:**
```sql
{sql_query}
```

📋 **Results:**
{sql_result}

✅ **Answer:**
{natural_response}
"""
            
        except Exception as e:
            return f"❌ Error processing question: {str(e)}"

# Inicializar cliente
print("🚀 Iniciando configuración con Bedrock...")
try:
    bedrock_client = BedrockDeepSeekClient()
    
    # Test initial
    print("🧪 Testing Bedrock connection...")
    test_response = bedrock_client.call_bedrock("Respond: OK")
    print(f"✅ Bedrock test successful: {test_response[:50]}...")
    
    # Test database
    db_test = bedrock_client.db.run("SELECT COUNT(*) FROM customers")
    print(f"✅ Database test successful: {db_test[0][0]} customers")
    
    print("🎉 System ready for English and Spanish queries!")
    
except Exception as e:
    print(f"❌ Error crítico: {e}")
    bedrock_client = None

# Servir archivos estáticos
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('frontend/index.html')

@app.get("/health")
async def health_check():
    """Health check"""
    try:
        if not bedrock_client:
            return {"status": "error", "detail": "Cliente no inicializado"}
        
        # Test rápido
        start_time = time.time()
        test_response = bedrock_client.call_bedrock("Test")
        response_time = time.time() - start_time
        
        return {
            "status": "ok",
            "database": "conectada",
            "llm_model": f"DeepSeek-R1 via Bedrock ({response_time:.2f}s)",
            "agent": "funcionando",
            "model_id": BEDROCK_MODEL_ID,
            "region": AWS_REGION,
            "infrastructure": "completamente gestionada (sin EC2)"
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/simple-query")
async def consulta_simple(request: QueryRequest):
    """Query using DeepSeek-R1 in Bedrock - Supports English and Spanish"""
    try:
        if not bedrock_client:
            raise HTTPException(status_code=500, detail="Client not available")
        
        print(f"🔍 Processing with Bedrock: {request.pregunta}")
        start_time = time.time()
        
        # Process question
        respuesta = bedrock_client.answer_question(request.pregunta)
        
        response_time = time.time() - start_time
        print(f"✅ Completed in {response_time:.2f}s")
        
        return {
            "question": request.pregunta,
            "answer": respuesta,
            "model": "DeepSeek-R1 (Bedrock)",
            "time": f"{response_time:.2f}s"
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/query")
async def hacer_consulta(request: QueryRequest):
    """Alias para compatibilidad"""
    return await consulta_simple(request)

if __name__ == "__main__":
    print("🚀 Iniciando SQL Agent con DeepSeek-R1 en Bedrock...")
    print("💰 SIN costos de EC2 - Servicio completamente gestionado")
    print(f"🌐 Acceso: http://35.94.145.5:8000")
    print("📖 Docs: http://35.94.145.5:8000/docs")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=False
    )