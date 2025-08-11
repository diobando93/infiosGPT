from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.llms.base import LLM
import uvicorn
from typing import Optional, List, Any
import os
import boto3
import json
import time
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Custom LLM class para SageMaker optimizado
class SageMakerDeepSeekLLM(LLM):
    endpoint_name: str
    region_name: str = "us-west-2"
    temperature: float = 0.1
    max_tokens: int = 200
    
    def __init__(self, endpoint_name: str, **kwargs):
        super().__init__(**kwargs)
        self.endpoint_name = endpoint_name
        self.runtime = boto3.client('sagemaker-runtime', region_name=self.region_name)
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Llamar al endpoint SageMaker con configuración optimizada"""
        try:
            # Preparar payload optimizado para DeepSeek
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False,  # Solo nueva generación
                    "pad_token_id": 50256,      # Para evitar warnings
                    "eos_token_id": 50256
                }
            }
            
            # Llamar al endpoint con timeout extendido
            response = self.runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            
            # Procesar respuesta
            result = json.loads(response['Body'].read().decode())
            
            # Extraer texto generado
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                # Limpiar el texto
                return generated_text.strip()
            elif isinstance(result, dict):
                return result.get('generated_text', result.get('outputs', str(result)))
            else:
                return str(result)
                
        except Exception as e:
            # Log del error pero continuar
            print(f"Error en SageMaker: {str(e)}")
            raise Exception(f"Error llamando SageMaker: {str(e)}")
    
    @property
    def _llm_type(self) -> str:
        return "sagemaker_deepseek"

# Configuración FastAPI
app = FastAPI(
    title="🤖 SQL Agent with SageMaker DeepSeek",
    description="Agente SQL ultra-rápido usando DeepSeek en Amazon SageMaker",
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

class QueryResponse(BaseModel):
    pregunta: str
    respuesta: dict
    status: str

# Configuración del agente SQL con SageMaker
class SageMakerSQLAgent:
    def __init__(self):
        self.db = None
        self.llm = None
        self.agent_executor = None
        self.initialize_agent()
    
    def initialize_agent(self):
        try:
            print("🔌 Conectando a PostgreSQL Northwind...")
            self.db = SQLDatabase.from_uri(DB_URI)
            
            print("🤖 Inicializando DeepSeek en SageMaker...")
            self.llm = SageMakerDeepSeekLLM(
                endpoint_name=SAGEMAKER_ENDPOINT,
                region_name=AWS_REGION,
                temperature=0.1,
                max_tokens=200
            )
            
            # Test de conexión inicial
            print("🧪 Testeando conexión SageMaker...")
            test_response = self.llm._call("SELECT 1;")
            print(f"✅ Test exitoso: {test_response[:50]}...")
            
            # Crear agente SQL
            toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
            self.agent_executor = create_sql_agent(
                llm=self.llm, 
                toolkit=toolkit, 
                verbose=True,
                max_iterations=3,
                early_stopping_method="generate"
            )
            
            print("✅ Agente SQL con SageMaker inicializado correctamente")
            
        except Exception as e:
            print(f"❌ Error inicializando agente: {str(e)}")
            raise e

# Inicializar agente
print("🚀 Iniciando configuración...")
try:
    sql_agent = SageMakerSQLAgent()
    print("🎉 ¡Sistema listo!")
except Exception as e:
    print(f"❌ Error crítico: {e}")
    sql_agent = None

# Servir archivos estáticos
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('frontend/index.html')

@app.get("/health")
async def health_check():
    """Health check con información de SageMaker"""
    try:
        db_status = "conectada" if sql_agent and sql_agent.db else "desconectada"
        
        # Test SageMaker
        sagemaker_status = "disponible"
        if sql_agent and sql_agent.llm:
            try:
                start_time = time.time()
                test_response = sql_agent.llm._call("Test connection")
                response_time = time.time() - start_time
                sagemaker_status = f"funcionando ({response_time:.2f}s)"
            except Exception as e:
                sagemaker_status = f"error: {str(e)[:50]}"
        else:
            sagemaker_status = "no inicializado"
        
        return {
            "status": "ok",
            "database": db_status,
            "llm_model": f"DeepSeek-Coder-6.7B via SageMaker ({sagemaker_status})",
            "agent": "inicializado" if sql_agent and sql_agent.agent_executor else "no inicializado",
            "endpoint": SAGEMAKER_ENDPOINT,
            "region": AWS_REGION
        }
    except Exception as e:
        return {
            "status": "error",
            "detail": str(e)
        }

@app.post("/query", response_model=QueryResponse)
async def hacer_consulta(request: QueryRequest):
    """Consulta usando SageMaker DeepSeek"""
    if not sql_agent or not sql_agent.agent_executor:
        raise HTTPException(
            status_code=500, 
            detail="Agente SQL no está inicializado"
        )
    
    try:
        print(f"🔍 Procesando con SageMaker: {request.pregunta}")
        start_time = time.time()
        
        respuesta = sql_agent.agent_executor.invoke(request.pregunta)
        
        response_time = time.time() - start_time
        print(f"✅ Consulta completada en {response_time:.2f}s")
        
        return QueryResponse(
            pregunta=request.pregunta,
            respuesta=respuesta,
            status="exitoso"
        )
        
    except Exception as e:
        print(f"❌ Error en consulta SageMaker: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error: {str(e)}"
        )

@app.post("/simple-query")
async def consulta_simple(request: QueryRequest):
    """Versión simplificada para el frontend"""
    try:
        if not sql_agent or not sql_agent.agent_executor:
            raise HTTPException(
                status_code=500, 
                detail="Agente no disponible"
            )
        
        print(f"🔍 Consulta simple: {request.pregunta}")
        
        respuesta = sql_agent.agent_executor.invoke(request.pregunta)
        respuesta_final = respuesta.get('output', respuesta) if isinstance(respuesta, dict) else respuesta
        
        return {
            "pregunta": request.pregunta,
            "respuesta": respuesta_final
        }
        
    except Exception as e:
        print(f"❌ Error en consulta simple: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error: {str(e)}"
        )

@app.get("/sagemaker-stats")
async def sagemaker_stats():
    """Estadísticas del endpoint SageMaker"""
    try:
        sagemaker = boto3.client('sagemaker', region_name=AWS_REGION)
        
        endpoint_info = sagemaker.describe_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT
        )
        
        return {
            "endpoint_name": SAGEMAKER_ENDPOINT,
            "status": endpoint_info['EndpointStatus'],
            "creation_time": str(endpoint_info['CreationTime']),
            "instance_type": "ml.g4dn.xlarge",
            "model": "deepseek-ai/deepseek-coder-6.7b-instruct"
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("🚀 Iniciando SQL Agent con SageMaker DeepSeek...")
    print(f"🌐 Acceso: http://35.94.145.5:8000")
    print("📖 Docs: http://35.94.145.5:8000/docs")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=False
    )