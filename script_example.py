from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.llms import Ollama

db = SQLDatabase.from_uri("postgresql://postgres:123456@localhost:5499/postgres")

llm = Ollama(model="mistral")

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

pregunta = "¿Que ordenes tienen este order_date 1996-07-04?"
respuesta = agent_executor.invoke(pregunta)


print("\n Respuesta del agente:\n", respuesta)
