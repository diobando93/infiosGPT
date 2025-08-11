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
    title="🤖 SQL Agent with SageMaker DeepSeek",
    description="Agente SQL directo usando DeepSeek en Amazon SageMaker",
    version="2.0.0"
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

# Configuración SageMaker
SAGEMAKER_ENDPOINT = "deepseek-coder-endpoint"
AWS_REGION = "us-west-2"

print(f"🔗 Base de datos: {DB_URI}")
print(f"🤖 SageMaker endpoint: {SAGEMAKER_ENDPOINT}")

# Modelos Pydantic
class QueryRequest(BaseModel):
    pregunta: str
    verbose: Optional[bool] = True

# Cliente SageMaker simple
class SimpleSageMakerClient:
    def __init__(self):
        self.endpoint_name = SAGEMAKER_ENDPOINT
        self.runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
        self.db = SQLDatabase.from_uri(DB_URI)
    
    def call_sagemaker(self, prompt):
        """Llamar directamente a SageMaker"""
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 200,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            response = self.runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            
            result = json.loads(response['Body'].read().decode())
            
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '').strip()
            elif isinstance(result, dict):
                return result.get('generated_text', str(result))
            else:
                return str(result)
                
        except Exception as e:
            print(f"Error en SageMaker: {str(e)}")
            return f"Error: {str(e)}"
    
    def generate_sql_query(self, question):
        """Generar consulta SQL usando el modelo"""
        # Obtener esquema de la base de datos
        schema_info = self.db.get_table_info()
        
        # Crear prompt para generar SQL
        prompt = f"""
Eres un experto en SQL. Dada la siguiente información de la base de datos Northwind:

{schema_info}

Usuario pregunta: {question}

Genera SOLO la consulta SQL necesaria para responder la pregunta. No incluyas explicaciones adicionales.

SQL:"""
        
        sql_query = self.call_sagemaker(prompt)
        return sql_query.strip()
    
    def execute_sql_query(self, sql_query):
        """Ejecutar consulta SQL en la base de datos"""
        try:
            # Limpiar la consulta
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            if sql_query.startswith("SQL:"):
                sql_query = sql_query[4:].strip()
            
            print(f"🔍 Ejecutando SQL: {sql_query}")
            
            # Ejecutar consulta
            result = self.db.run(sql_query)
            return result
            
        except Exception as e:
            print(f"Error ejecutando SQL: {str(e)}")
            return f"Error ejecutando consulta: {str(e)}"
    
    def answer_question(self, question):
        """Responder pregunta completa"""
        try:
            # Paso 1: Generar SQL
            sql_query = self.generate_sql_query(question)
            
            # Paso 2: Ejecutar SQL
            sql_result = self.execute_sql_query(sql_query)
            
            # Paso 3: Formatear respuesta
            if "Error" in str(sql_result):
                return f"❌ {sql_result}"
            
            # Generar respuesta natural
            response_prompt = f"""
Pregunta del usuario: {question}
Consulta SQL ejecutada: {sql_query}
Resultado: {sql_result}

Proporciona una respuesta clara y natural en español sobre el resultado:"""
            
            natural_response = self.call_sagemaker(response_prompt)
            
            return f"""
📊 **Consulta SQL:**
```sql
{sql_query}
```

📋 **Resultado:**
{sql_result}

✅ **Respuesta:**
{natural_response}
"""
            
        except Exception as e:
            return f"❌ Error procesando pregunta: {str(e)}"

# Inicializar cliente
print("🚀 Iniciando configuración...")
try:
    sagemaker_client = SimpleSageMakerClient()
    
    # Test inicial
    print("🧪 Testeando conexión...")
    test_response = sagemaker_client.call_sagemaker("SELECT 1;")
    print(f"✅ Test SageMaker exitoso: {test_response[:50]}...")
    
    # Test base de datos
    db_test = sagemaker_client.db.run("SELECT COUNT(*) FROM customers")
    print(f"✅ Test BD exitoso: {db_test[0][0]} clientes")
    
    print("🎉 ¡Sistema listo!")
    
except Exception as e:
    print(f"❌ Error crítico: {e}")
    sagemaker_client = None

# Servir archivos estáticos
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('frontend/index.html')

@app.get("/health")
async def health_check():
    """Health check"""
    try:
        if not sagemaker_client:
            return {"status": "error", "detail": "Cliente no inicializado"}
        
        # Test rápido
        start_time = time.time()
        test_response = sagemaker_client.call_sagemaker("Test")
        response_time = time.time() - start_time
        
        return {
            "status": "ok",
            "database": "conectada",
            "llm_model": f"DeepSeek-Coder-6.7B via SageMaker ({response_time:.2f}s)",
            "agent": "funcionando",
            "endpoint": SAGEMAKER_ENDPOINT,
            "region": AWS_REGION
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/simple-query")
async def consulta_simple(request: QueryRequest):
    """Consulta simplificada directa"""
    try:
        if not sagemaker_client:
            raise HTTPException(status_code=500, detail="Cliente no disponible")
        
        print(f"🔍 Procesando: {request.pregunta}")
        start_time = time.time()
        
        # Procesar pregunta
        respuesta = sagemaker_client.answer_question(request.pregunta)
        
        response_time = time.time() - start_time
        print(f"✅ Completado en {response_time:.2f}s")
        
        return {
            "pregunta": request.pregunta,
            "respuesta": respuesta
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/query")
async def hacer_consulta(request: QueryRequest):
    """Alias para compatibilidad"""
    return await consulta_simple(request)

if __name__ == "__main__":
    print("🚀 Iniciando SQL Agent simple con SageMaker...")
    print(f"🌐 Acceso: http://35.94.145.5:8000")
    print("📖 Docs: http://35.94.145.5:8000/docs")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=False
    )