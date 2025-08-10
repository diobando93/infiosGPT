from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.llms import Ollama
import uvicorn
from typing import Optional
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse





# Inicializar FastAPI
app = FastAPI(
    title="SQL Agent API",
    description="API para hacer consultas a base de datos usando LangChain y Ollama",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos
    allow_headers=["*"],  # Permite todos los headers
)

# Configuración de base de datos desde variables de entorno o valores por defecto
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres123") 
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5499")
DB_NAME = os.getenv("DB_NAME", "northwind")
DB_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

print(f"Usando cadena de conexión: {DB_URI}")

# Modelo para la request
class QueryRequest(BaseModel):
    pregunta: str
    verbose: Optional[bool] = True

# Modelo para la response
class QueryResponse(BaseModel):
    pregunta: str
    respuesta: dict
    status: str

# Configuración global del agente
class SQLAgentConfig:
    def __init__(self):
        self.db = None
        self.llm = None
        self.agent_executor = None
        self.initialize_agent()
    
    def initialize_agent(self):
        try:
            # Conexión a la base de datos - EXACTAMENTE IGUAL QUE TU SCRIPT
            self.db = SQLDatabase.from_uri(DB_URI)
            
            # Modelo LLM
            """
            self.llm = Ollama(
                model="mistral",
                temperature=0,  # Más determinístico, menos "creatividad"
                num_predict=200,  # Limita tokens de salida
                top_k=10,  # Reduce opciones de tokens
                top_p=0.3,  # Más enfocado
                repeat_penalty=1.1,
                timeout=60  # Timeout de 60 segundos
          )
          
            
            self.llm = Ollama(
                model="deepseek-coder:33b", 
                 temperature=0.1,
                num_predict=300

            )
            self.llm = Ollama(
                model="deepseek-coder:33b",
                base_url="https://de60a716808f.ngrok-free.app",  # Tu URL de ngrok
                temperature=0.1,
                num_predict=300,
                timeout=120,
                # Agregar headers específicos para ngrok si es necesario
                headers={
                    "Content-Type": "application/json",
                    "ngrok-skip-browser-warning": "true"
                }
            )
            
"""
            self.llm = Ollama(
                model="deepseek-coder:33b", 
                temperature=0.1,
                num_predict=300,
                timeout=300,
            )
            
            # Toolkit y agente
            toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
            self.agent_executor = create_sql_agent(
                llm=self.llm, 
                toolkit=toolkit, 
                verbose=True
            )
            print("Agente SQL inicializado correctamente")
            
        except Exception as e:
            print(f"Error inicializando el agente: {str(e)}")
            raise e

# Instancia global del agente
sql_agent = SQLAgentConfig()


app.mount("/css", StaticFiles(directory="frontend/css"), name="css")
app.mount("/js", StaticFiles(directory="frontend/js"), name="js")
app.mount("/assets", StaticFiles(directory="frontend/assets"), name="assets")
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Agregar endpoint para servir el HTML principal
@app.get("/")
async def read_index():
    return FileResponse('frontend/index.html')

# Si tienes otros archivos HTML
@app.get("/app")
async def read_app():
    return FileResponse('frontend/index.html')



@app.get("/health")
async def health_check():
    """Verificar el estado de la API y conexiones"""
    try:
        # Verificar conexión a la base de datos
        db_status = "conectada" if sql_agent.db else "desconectada"
        
        # Verificar modelo LLM
        llm_status = "disponible" if sql_agent.llm else "no disponible"
        
        return {
            "status": "ok",
            "database": db_status,
            "llm_model": llm_status,
            "agent": "inicializado" if sql_agent.agent_executor else "no inicializado"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en health check: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def hacer_consulta(request: QueryRequest):
    """
    Realizar una consulta a la base de datos usando el agente SQL
    
    - **pregunta**: La pregunta en lenguaje natural sobre los datos
    - **verbose**: Si mostrar información detallada del proceso (opcional)
    """
    try:
        if not sql_agent.agent_executor:
            raise HTTPException(
                status_code=500, 
                detail="El agente SQL no está inicializado correctamente"
            )
        
        # Procesar la consulta
        print(f"Procesando pregunta: {request.pregunta}")
        respuesta = sql_agent.agent_executor.invoke(request.pregunta)
        
        return QueryResponse(
            pregunta=request.pregunta,
            respuesta=respuesta,
            status="exitoso"
        )
        
    except Exception as e:
        print(f"Error procesando consulta: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error procesando la consulta: {str(e)}"
        )

@app.post("/simple-query")
async def consulta_simple(request: QueryRequest):
    """
    Versión simplificada que solo devuelve la respuesta final
    """
    try:
        if not sql_agent.agent_executor:
            raise HTTPException(
                status_code=500, 
                detail="El agente SQL no está inicializado correctamente"
            )
        
        respuesta = sql_agent.agent_executor.invoke(request.pregunta)
        
        # Extraer solo la respuesta final si es posible
        respuesta_final = respuesta.get('output', respuesta) if isinstance(respuesta, dict) else respuesta
        
        return {
            "pregunta": request.pregunta,
            "respuesta": respuesta_final
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error: {str(e)}"
        )

# Ejecutar la API
if __name__ == "__main__":
    print("Iniciando API SQL Agent")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=False  # Cambiado a False para evitar conflictos
    )