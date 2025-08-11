// Configuración de la API
//const API_URL = 'http://localhost:8000';
const API_URL = 'http://35.94.145.5:8000';

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
    
    statusText.textContent = 'Checking...';
    statusDot.className = 'status-dot';
    
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        
        if (response.ok && data.status === 'ok') {
            statusDot.className = 'status-dot connected';
            statusText.textContent = `✅ Connected - DB: ${data.database}, LLM: ${data.llm_model}`;
        } else {
            statusDot.className = 'status-dot';
            statusText.textContent = '❌ API responds but has internal issues';
        }
    } catch (error) {
        statusDot.className = 'status-dot';
        statusText.textContent = '❌ Cannot connect to API';
        console.error('Connection error:', error);
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
        alert('Please write a query');
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
        console.error('Query error:', error);
    } finally {
        // Limpiar progreso y restaurar botón
        clearInterval(progressInterval);
        updateButtonState(sendBtn, sendIcon, sendText, false);
    }
}

/**
 * Parsear respuesta con formato markdown de DeepSeek-R1
 */
function parseMarkdownResponse(answer) {
    let parsed = {
        sql: '',
        results: '',
        interpretation: ''
    };
    
    // Extraer SQL query - buscar después de **SQL Query:**
    const sqlMatch = answer.match(/\*\*SQL Query:\*\*\n```sql\n(.*?)\n```/s);
    if (sqlMatch) {
        parsed.sql = sqlMatch[1].trim();
    }
    
    // Extraer Results - buscar después de **Results:**
    const resultsMatch = answer.match(/📋 \*\*Results:\*\*\n(.*?)(?=\n\n✅|$)/s);
    if (resultsMatch) {
        parsed.results = resultsMatch[1].trim();
    }
    
    // Extraer Answer - buscar después de **Answer:**
    const answerMatch = answer.match(/✅ \*\*Answer:\*\*\n(.*?)$/s);
    if (answerMatch) {
        parsed.interpretation = answerMatch[1].trim();
    }
    
    return parsed;
}

/**
 * Formatear respuesta para mostrar en el frontend
 */
function formatResponseText(data) {
    console.log('Data received:', data); // Debug log
    
    const question = data.question || data.pregunta || 'Unknown question';
    const time = data.time || '0s';
    const model = data.model || 'DeepSeek-R1';
    
    let formattedText = `❓ QUESTION:\n${question}\n\n`;
    formattedText += `⏱️ TIME: ${time} | 🤖 MODEL: ${model}\n\n`;
    
    // Si hay una respuesta formateada, parsearla
    const answer = data.answer || data.respuesta;
    if (answer) {
        console.log('Answer received:', answer); // Debug log
        const parsed = parseMarkdownResponse(answer);
        console.log('Parsed result:', parsed); // Debug log
        
        if (parsed.sql) {
            formattedText += `🔍 SQL QUERY:\n${parsed.sql}\n\n`;
        }
        
        if (parsed.results) {
            formattedText += `📋 RESULTS:\n${parsed.results}\n\n`;
        }
        
        if (parsed.interpretation) {
            formattedText += `✅ ANSWER:\n${parsed.interpretation}`;
        }
        
        // Si no se parseó correctamente, mostrar la respuesta completa
        if (!parsed.sql && !parsed.results && !parsed.interpretation) {
            formattedText += `📝 FULL RESPONSE:\n${answer}`;
        }
    } else {
        formattedText += `❌ No answer received from server`;
    }
    
    return formattedText;
}

/**
 * Actualizar estado del botón de envío
 */
function updateButtonState(btn, icon, text, isLoading) {
    if (isLoading) {
        btn.disabled = true;
        icon.innerHTML = '<div class="spinner"></div>';
        text.textContent = 'Processing...';
    } else {
        btn.disabled = false;
        icon.textContent = '🚀';
        text.textContent = 'Execute Query';
    }
}

/**
 * Mostrar progreso de carga
 */
function showLoadingProgress(responseBox) {
    responseBox.className = 'response-box loading';
    
    const loadingMessages = [
        '🔄 Analyzing your question with DeepSeek-R1...',
        '🧠 Generating SQL query...',
        '📊 Executing on database...',
        '✨ Processing results...'
    ];
    
    let messageIndex = 0;
    responseBox.textContent = loadingMessages[0];
    
    return setInterval(() => {
        messageIndex = (messageIndex + 1) % loadingMessages.length;
        responseBox.textContent = loadingMessages[messageIndex];
    }, 2000);
}

/**
 * Mostrar respuesta exitosa
 */
function showSuccessResponse(responseBox, data) {
    responseBox.className = 'response-box success';
    responseBox.textContent = formatResponseText(data);
}

/**
 * Mostrar respuesta de error
 */
function showErrorResponse(responseBox, errorDetail) {
    responseBox.className = 'response-box error';
    responseBox.textContent = `❌ ERROR: ${errorDetail}`;
}

/**
 * Mostrar error de conexión
 */
function showConnectionError(responseBox, errorMessage) {
    responseBox.className = 'response-box error';
    responseBox.textContent = `❌ CONNECTION ERROR: ${errorMessage}\n\nVerify that the API is running on ${API_URL}`;
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
    responseBox.textContent = 'The result of your query will appear here...';
    
    questionInput.focus();
}