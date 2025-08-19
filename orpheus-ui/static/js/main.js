// Orpheus TTS Enterprise UI - Main JavaScript

// Global variables
let currentSessionId = null;
let selectedVoice = 'leo';
let ws = null;
let audioContext = null;

// Sample texts
const sampleTexts = {
    professional: "Good morning, everyone. Today, we're excited to announce the launch of our new enterprise text-to-speech platform. This cutting-edge technology delivers natural, human-like voice synthesis with unprecedented quality.",
    conversational: "Hey there! <laugh> I just wanted to tell you about this amazing new feature we've been working on. It's pretty incredible how natural this sounds, right? <sigh> Technology these days is just mind-blowing!",
    narrative: "Once upon a time, in a world where machines could speak with the warmth and nuance of human voices, there lived a powerful AI named Orpheus. Its voice could convey emotions, tell stories, and connect with people in ways never before imagined.",
    technical: "The Orpheus TTS system utilizes a three billion parameter language model, fine-tuned specifically for speech synthesis. Operating at 24 kilohertz with 16-bit audio depth, it achieves real-time factor greater than 1.0x on modern GPUs."
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
    loadSessions();
    checkModelStatus();
});

function initializeApp() {
    // Initialize audio context
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    
    // Update text stats
    updateTextStats();
    
    // Set initial voice
    document.querySelector('.voice-card[data-voice="leo"]').classList.add('active');
}

function setupEventListeners() {
    // Text input
    document.getElementById('text-input').addEventListener('input', updateTextStats);
    
    // Voice selection
    document.querySelectorAll('.voice-card').forEach(card => {
        card.addEventListener('click', () => selectVoice(card.dataset.voice));
    });
    
    // Buttons
    document.getElementById('generate-btn').addEventListener('click', generateSpeech);
    document.getElementById('stream-btn').addEventListener('click', startStreaming);
    document.getElementById('clear-btn').addEventListener('click', clearText);
    document.getElementById('load-sample-btn').addEventListener('click', showSampleModal);
    
    // Settings
    document.getElementById('temperature').addEventListener('input', (e) => {
        document.getElementById('temp-value').textContent = e.target.value;
    });
    
    document.getElementById('top-p').addEventListener('input', (e) => {
        document.getElementById('top-p-value').textContent = e.target.value;
    });
}

function updateTextStats() {
    const text = document.getElementById('text-input').value;
    const charCount = text.length;
    const wordCount = text.trim().split(/\s+/).filter(word => word.length > 0).length;
    const estDuration = Math.max(1, Math.round(wordCount / 2.5)); // Rough estimate
    
    document.getElementById('char-count').textContent = charCount;
    document.getElementById('word-count').textContent = wordCount;
    document.getElementById('est-duration').textContent = `${estDuration}s`;
}

function selectVoice(voice) {
    selectedVoice = voice;
    document.querySelectorAll('.voice-card').forEach(card => {
        card.classList.toggle('active', card.dataset.voice === voice);
    });
}

async function generateSpeech() {
    const text = document.getElementById('text-input').value.trim();
    
    if (!text) {
        showToast('Please enter some text to generate speech', 'warning');
        return;
    }
    
    // Show progress
    showProgress(true, 'Initializing...');
    
    try {
        // Prepare request
        const requestData = {
            text: text,
            voice: selectedVoice,
            temperature: parseFloat(document.getElementById('temperature').value),
            top_p: parseFloat(document.getElementById('top-p').value),
            max_tokens: parseInt(document.getElementById('max-tokens').value)
        };
        
        // Update progress
        updateProgress(20, 'Sending request to server...');
        
        // Send request
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        updateProgress(50, 'Processing with Orpheus TTS...');
        
        const result = await response.json();
        
        updateProgress(80, 'Finalizing audio...');
        
        // Update current session
        currentSessionId = result.session_id;
        
        // Display audio
        displayAudio(result);
        
        updateProgress(100, 'Complete!');
        
        // Hide progress after delay
        setTimeout(() => {
            showProgress(false);
        }, 500);
        
        // Reload sessions
        loadSessions();
        
        showToast('Speech generated successfully!', 'success');
        
    } catch (error) {
        console.error('Error generating speech:', error);
        showToast(`Error: ${error.message}`, 'error');
        showProgress(false);
    }
}

function displayAudio(result) {
    const outputSection = document.getElementById('output-section');
    outputSection.style.display = 'block';
    
    // Set audio source
    const audioPlayer = document.getElementById('audio-player');
    audioPlayer.src = result.audio_url;
    
    // Update info
    document.getElementById('audio-duration').textContent = `${result.duration.toFixed(2)}s`;
    document.getElementById('audio-size').textContent = `${(result.file_size / 1024).toFixed(1)} KB`;
    document.getElementById('audio-voice').textContent = selectedVoice;
    
    // Setup download button
    document.getElementById('download-btn').onclick = () => downloadAudio(result.audio_url);
    
    // Setup share button
    document.getElementById('share-btn').onclick = () => shareAudio(result.session_id);
    
    // Scroll to output
    outputSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function startStreaming() {
    const text = document.getElementById('text-input').value.trim();
    
    if (!text) {
        showToast('Please enter some text to stream', 'warning');
        return;
    }
    
    // Connect WebSocket if not connected
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        await connectWebSocket();
    }
    
    // Send generation request
    ws.send(JSON.stringify({
        type: 'generate',
        text: text,
        voice: selectedVoice
    }));
    
    showProgress(true, 'Streaming audio...');
}

async function connectWebSocket() {
    return new Promise((resolve, reject) => {
        ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onopen = () => {
            console.log('WebSocket connected');
            resolve();
        };
        
        ws.onmessage = async (event) => {
            const data = JSON.parse(event.data);
            
            switch(data.type) {
                case 'status':
                    updateProgressMessage(data.message);
                    break;
                    
                case 'audio_chunk':
                    // Handle streaming audio chunk
                    await playAudioChunk(data.audio);
                    updateProgress(Math.min(90, 20 + data.chunk_index * 2), `Streaming chunk ${data.chunk_index}...`);
                    break;
                    
                case 'complete':
                    showToast(`Streaming complete! Duration: ${data.duration.toFixed(2)}s`, 'success');
                    showProgress(false);
                    loadSessions();
                    break;
                    
                case 'error':
                    showToast(`Streaming error: ${data.message}`, 'error');
                    showProgress(false);
                    break;
            }
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            reject(error);
        };
        
        ws.onclose = () => {
            console.log('WebSocket disconnected');
        };
    });
}

async function playAudioChunk(base64Audio) {
    // Decode base64 to array buffer
    const binaryString = atob(base64Audio);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    
    // Convert to audio buffer and play
    // Note: This is simplified - in production you'd want proper audio queuing
    const audioBuffer = await audioContext.decodeAudioData(bytes.buffer);
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);
    source.start();
}

async function loadSessions() {
    try {
        const response = await fetch('/api/sessions');
        const data = await response.json();
        
        const sessionList = document.getElementById('session-list');
        sessionList.innerHTML = '';
        
        data.sessions.slice(0, 10).forEach(session => {
            const sessionItem = createSessionElement(session);
            sessionList.appendChild(sessionItem);
        });
        
    } catch (error) {
        console.error('Error loading sessions:', error);
    }
}

function createSessionElement(session) {
    const div = document.createElement('div');
    div.className = 'session-item';
    div.onclick = () => loadSession(session.session_id);
    
    div.innerHTML = `
        <div class="session-item-header">
            <span class="session-voice">${session.voice}</span>
            <i class="fas fa-trash-alt" onclick="event.stopPropagation(); deleteSession('${session.session_id}')"></i>
        </div>
        <div class="session-text">${session.text}</div>
        <div class="session-time">${new Date(session.timestamp).toLocaleString()}</div>
    `;
    
    return div;
}

async function loadSession(sessionId) {
    try {
        const response = await fetch(`/api/session/${sessionId}`);
        const session = await response.json();
        
        // Load text
        document.getElementById('text-input').value = session.text;
        updateTextStats();
        
        // Select voice
        selectVoice(session.voice);
        
        // Display audio if available
        if (session.audio_url) {
            displayAudio({
                audio_url: session.audio_url,
                duration: session.duration,
                file_size: session.file_size,
                session_id: sessionId
            });
        }
        
        showToast('Session loaded', 'success');
        
    } catch (error) {
        console.error('Error loading session:', error);
        showToast('Failed to load session', 'error');
    }
}

async function deleteSession(sessionId) {
    if (!confirm('Delete this session?')) return;
    
    try {
        await fetch(`/api/session/${sessionId}`, {
            method: 'DELETE'
        });
        
        loadSessions();
        showToast('Session deleted', 'success');
        
    } catch (error) {
        console.error('Error deleting session:', error);
        showToast('Failed to delete session', 'error');
    }
}

async function checkModelStatus() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        const statusElement = document.getElementById('model-status');
        if (data.model_loaded) {
            statusElement.textContent = 'Model Ready';
            statusElement.style.background = 'var(--success-color)';
        } else {
            statusElement.textContent = 'Loading Model...';
            statusElement.style.background = 'var(--warning-color)';
            
            // Check again in 5 seconds
            setTimeout(checkModelStatus, 5000);
        }
        
    } catch (error) {
        console.error('Error checking model status:', error);
        document.getElementById('model-status').textContent = 'Offline';
        document.getElementById('model-status').style.background = 'var(--danger-color)';
    }
}

function clearText() {
    document.getElementById('text-input').value = '';
    updateTextStats();
}

function showSampleModal() {
    document.getElementById('sample-modal').style.display = 'flex';
}

function closeSampleModal() {
    document.getElementById('sample-modal').style.display = 'none';
}

function loadSample(sampleKey) {
    document.getElementById('text-input').value = sampleTexts[sampleKey];
    updateTextStats();
    closeSampleModal();
    showToast('Sample text loaded', 'success');
}

function downloadAudio(audioUrl) {
    const link = document.createElement('a');
    link.href = audioUrl;
    link.download = `orpheus_${selectedVoice}_${Date.now()}.wav`;
    link.click();
}

function shareAudio(sessionId) {
    const shareUrl = `${window.location.origin}/session/${sessionId}`;
    
    if (navigator.share) {
        navigator.share({
            title: 'Orpheus TTS Audio',
            text: 'Check out this generated speech',
            url: shareUrl
        });
    } else {
        // Copy to clipboard
        navigator.clipboard.writeText(shareUrl);
        showToast('Share link copied to clipboard', 'success');
    }
}

function showProgress(show, message = '') {
    const overlay = document.getElementById('progress-overlay');
    overlay.style.display = show ? 'flex' : 'none';
    
    if (message) {
        document.getElementById('progress-message').textContent = message;
    }
}

function updateProgress(percentage, message = '') {
    document.getElementById('progress-fill').style.width = `${percentage}%`;
    
    if (message) {
        updateProgressMessage(message);
    }
}

function updateProgressMessage(message) {
    document.getElementById('progress-message').textContent = message;
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    }[type] || 'fa-info-circle';
    
    toast.innerHTML = `
        <i class="fas ${icon}"></i>
        <span>${message}</span>
    `;
    
    container.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        toast.remove();
    }, 5000);
}

// Handle window close - cleanup WebSocket
window.addEventListener('beforeunload', () => {
    if (ws) {
        ws.close();
    }
});