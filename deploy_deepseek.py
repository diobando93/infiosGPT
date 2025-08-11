#!/usr/bin/env python3
# deploy_deepseek.py
import boto3
import sagemaker
import json
import time
from sagemaker.huggingface import HuggingFaceModel
from sagemaker import get_execution_role

def main():
    print("🚀 Iniciando deployment de DeepSeek en SageMaker...")
    
    try:
        # Configuración inicial
        session = sagemaker.Session()
        region = session.boto_region_name
        
        print(f"📍 Región: {region}")
        
        # Obtener o crear rol de ejecución
        try:
            role = get_execution_role()
            print(f"✅ Usando rol existente: {role}")
        except Exception:
            print("⚠️  No se encontró rol de SageMaker. Creando rol...")
            role = create_sagemaker_role()
        
        # Configuración del modelo DeepSeek
        model_config = {
            "HF_MODEL_ID": "deepseek-ai/deepseek-coder-6.7b-instruct",  # Modelo más pequeño para empezar
            "HF_TASK": "text-generation",
            "MAX_INPUT_LENGTH": "2048",
            "MAX_TOTAL_TOKENS": "4096",
            "SM_NUM_GPUS": "1"
        }
        
        print(f"🤖 Configurando modelo: {model_config['HF_MODEL_ID']}")
        
        # Crear modelo HuggingFace
        huggingface_model = HuggingFaceModel(
            transformers_version="4.37.0",
            pytorch_version="2.1.0",
            py_version="py310",
            env=model_config,
            role=role
        )
        
        # Configurar endpoint
        endpoint_name = "deepseek-coder-endpoint"
        
        print(f"🏗️  Desplegando endpoint: {endpoint_name}")
        print("⏳ Esto puede tomar 5-10 minutos...")
        
        # Desplegar endpoint
        predictor = huggingface_model.deploy(
            instance_type="ml.g4dn.xlarge",
            initial_instance_count=1,
            endpoint_name=endpoint_name,
            wait=True
        )
        
        print(f"✅ ¡Endpoint desplegado exitosamente!")
        print(f"📝 Nombre del endpoint: {endpoint_name}")
        print(f"🌎 Región: {region}")
        
        # Probar el endpoint
        print("🧪 Probando el endpoint...")
        test_response = test_endpoint(predictor)
        print(f"✅ Test exitoso: {test_response[:100]}...")
        
        # Guardar configuración
        config = {
            "endpoint_name": endpoint_name,
            "region": region,
            "model_id": model_config["HF_MODEL_ID"]
        }
        
        with open("sagemaker_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Configuración guardada en: sagemaker_config.json")
        print("\n🎉 ¡Setup completo! Ahora puedes usar el endpoint.")
        print("\n📋 Próximos pasos:")
        print("1. Exportar variables de entorno:")
        print(f"   export SAGEMAKER_ENDPOINT={endpoint_name}")
        print(f"   export AWS_REGION={region}")
        print("2. Ejecutar el backend modificado:")
        print("   python index_sagemaker.py")
        
    except Exception as e:
        print(f"❌ Error durante el deployment: {str(e)}")
        print("\n🔧 Posibles soluciones:")
        print("1. Verificar credenciales AWS: aws sts get-caller-identity")
        print("2. Verificar permisos de SageMaker")
        print("3. Verificar límites de instancia en la región")
        raise

def create_sagemaker_role():
    """Crear rol de SageMaker si no existe"""
    print("🔑 Creando rol de SageMaker...")
    
    iam = boto3.client('iam')
    
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "sagemaker.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    role_name = "SageMakerExecutionRole-DeepSeek"
    
    try:
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Rol para ejecutar DeepSeek en SageMaker"
        )
        
        # Attachear políticas necesarias
        policies = [
            "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
        ]
        
        for policy in policies:
            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn=policy
            )
        
        # Esperar a que el rol se propague
        time.sleep(10)
        
        role_arn = response['Role']['Arn']
        print(f"✅ Rol creado: {role_arn}")
        return role_arn
        
    except iam.exceptions.EntityAlreadyExistsException:
        # El rol ya existe, obtenerlo
        response = iam.get_role(RoleName=role_name)
        role_arn = response['Role']['Arn']
        print(f"✅ Usando rol existente: {role_arn}")
        return role_arn

def test_endpoint(predictor):
    """Probar el endpoint desplegado"""
    test_prompt = "SELECT COUNT(*) FROM customers;"
    
    payload = {
        "inputs": test_prompt,
        "parameters": {
            "max_new_tokens": 50,
            "temperature": 0.1,
            "do_sample": True
        }
    }
    
    response = predictor.predict(payload)
    
    if isinstance(response, list) and len(response) > 0:
        return response[0].get('generated_text', str(response))
    return str(response)

if __name__ == "__main__":
    main()