// Configuración de la API
const API_URL = 'http://localhost:8000';

// Inicialización
window.onload = function() {
    checkConnection();
    setupEventListeners();
};

/**
 * Configurar event listeners
 */
function setupEventListeners() {
    // Permitir enviar con Enter (Ctrl+Enter para nueva línea)
    document.getElementById('questionInput').addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.ctrlKey) {
            e.preventDefault();
            sendQuery();
        }
    });
}

/**
 * Verificar conexión con la API
 */
async function checkConnection() {
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    
    statusText.textContent = 'Verificando...';
    statusDot.className = 'status-dot';
    
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        
        if (response.ok && data.status === 'ok') {
            statusDot.className = 'status-dot connected';
            statusText.textContent = `✅ Conectado - DB: ${data.database}, LLM: ${data.llm_model}`;
        } else {
            statusDot.className = 'status-dot';
            statusText.textContent = '❌ API responde pero hay problemas internos';
        }
    } catch (error) {
        statusDot.className = 'status-dot';
        statusText.textContent = '❌ No se puede conectar con la API';
        console.error('Error de conexión:', error);
    }
}

/**
 * Enviar consulta a la API
 */
async function sendQuery() {
    const questionInput = document.getElementById('questionInput');
    const responseBox = document.getElementById('responseBox');
    const sendBtn = document.getElementById('sendBtn');
    const sendIcon = document.getElementById('sendIcon');
    const sendText = document.getElementById('sendText');
    
    const question = questionInput.value.trim();
    
    if (!question) {
        alert('Por favor, escribe una consulta');
        return;
    }
    
    // Cambiar estado del botón a procesando
    updateButtonState(sendBtn, sendIcon, sendText, true);
    
    // Mostrar indicadores de progreso
    const progressInterval = showLoadingProgress(responseBox);
    
    try {
        const response = await fetch(`${API_URL}/simple-query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                pregunta: question,
                verbose: true
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showSuccessResponse(responseBox, data);
        } else {
            showErrorResponse(responseBox, data.detail || 'Error desconocido');
        }
        
    } catch (error) {
        showConnectionError(responseBox, error.message);
        console.error('Error en la consulta:', error);
    } finally {
        // Limpiar progreso y restaurar botón
        clearInterval(progressInterval);
        updateButtonState(sendBtn, sendIcon, sendText, false);
    }
}

/**
 * Actualizar estado del botón de envío
 */
function updateButtonState(btn, icon, text, isLoading) {
    if (isLoading) {
        btn.disabled = true;
        icon.innerHTML = '<div class="spinner"></div>';
        text.textContent = 'Procesando...';
    } else {
        btn.disabled = false;
        icon.textContent = '🚀';
        text.textContent = 'Ejecutar Consulta';
    }
}

/**
 * Mostrar progreso de carga
 */
function showLoadingProgress(responseBox) {
    responseBox.className = 'response-box loading';
    
    const loadingMessages = [
        '🔄 Analizando tu pregunta...',
        '🧠 Generando consulta SQL...',
        '📊 Ejecutando en la base de datos...',
        '✨ Procesando resultados...'
    ];
    
    let messageIndex = 0;
    responseBox.textContent = loadingMessages[0];
    
    return setInterval(() => {
        messageIndex = (messageIndex + 1) % loadingMessages.length;
        responseBox.textContent = loadingMessages[messageIndex];
    }, 3000);
}

/**
 * Mostrar respuesta exitosa
 */
function showSuccessResponse(responseBox, data) {
    responseBox.className = 'response-box success';
    responseBox.textContent = `Consulta: ${data.pregunta}\n\n✅ Resultado:\n${data.respuesta}`;
}

/**
 * Mostrar respuesta de error
 */
function showErrorResponse(responseBox, errorDetail) {
    responseBox.className = 'response-box error';
    responseBox.textContent = `❌ Error: ${errorDetail}`;
}

/**
 * Mostrar error de conexión
 */
function showConnectionError(responseBox, errorMessage) {
    responseBox.className = 'response-box error';
    responseBox.textContent = `❌ Error de conexión: ${errorMessage}\n\nVerifica que la API esté ejecutándose en ${API_URL}`;
}

/**
 * Establecer ejemplo de consulta
 */
function setExample(example) {
    const questionInput = document.getElementById('questionInput');
    questionInput.value = example;
    questionInput.focus();
}

/**
 * Limpiar formulario y respuesta
 */
function clearAll() {
    const questionInput = document.getElementById('questionInput');
    const responseBox = document.getElementById('responseBox');
    
    questionInput.value = '';
    responseBox.className = 'response-box';
    responseBox.textContent = 'Aquí aparecerá el resultado de tu consulta...';
    
    questionInput.focus();
}