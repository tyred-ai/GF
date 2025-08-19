// Orpheus TTS Enterprise UI - Main JavaScript

// Global variables
let currentSessionId = null;
let selectedVoice = 'tara';  // Changed default to tara
let audioContext = null;

// Sample texts with emotion tags
const sampleTexts = {
    professional: "Good morning, everyone. Today, we're excited to announce the launch of our new enterprise text-to-speech platform. This cutting-edge technology delivers natural, human-like voice synthesis with unprecedented quality.",
    conversational: "Hey there! I just wanted to tell you about this amazing new feature we've been working on. It's pretty incredible how natural this sounds, right? Technology these days is just mind-blowing!",
    narrative: "Once upon a time, in a world where machines could speak with the warmth and nuance of human voices, there lived a powerful AI named Orpheus. Its voice could convey emotions, tell stories, and connect with people in ways never before imagined.",
    technical: "The Orpheus TTS system utilizes a three billion parameter language model, fine-tuned specifically for speech synthesis. Operating at 24 kilohertz with 16-bit audio depth, it achieves real-time factor greater than 1.0x on modern GPUs.",
    emotional: "I can't believe it's finally happening! After all this time, we did it! I'm just so happy right now. Thank you all for believing in this project.",
    expressive: "Hmm, let me think about that for a moment. Oh, excuse me, it's been a long day. But seriously, this is an interesting question. Though I must admit, it's more complex than I initially thought."
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
    
    // Set initial voice to Tara
    document.querySelector('.voice-card[data-voice="tara"]').classList.add('active');
}

function setupEventListeners() {
    // Text input
    document.getElementById('text-input').addEventListener('input', updateTextStats);
    
    // Voice selection
    document.querySelectorAll('.voice-card').forEach(card => {
        card.addEventListener('click', () => selectVoice(card.dataset.voice));
    });
    
    // Emotion buttons
    document.querySelectorAll('.emotion-btn').forEach(btn => {
        btn.addEventListener('click', () => insertEmotionTag(btn.dataset.tag));
    });
    
    // Buttons
    document.getElementById('generate-btn').addEventListener('click', generateSpeech);
    document.getElementById('clear-btn').addEventListener('click', clearText);
    document.getElementById('load-sample-btn').addEventListener('click', showSampleModal);
    
    // Settings
    document.getElementById('temperature').addEventListener('input', (e) => {
        document.getElementById('temp-value').textContent = e.target.value;
    });
    
    document.getElementById('top-p').addEventListener('input', (e) => {
        document.getElementById('top-p-value').textContent = e.target.value;
    });
    
    document.getElementById('repetition-penalty').addEventListener('input', (e) => {
        document.getElementById('rep-penalty-value').textContent = e.target.value;
    });
    
    // Settings buttons
    document.getElementById('reset-settings-btn').addEventListener('click', resetSettings);
    document.getElementById('clear-all-sessions-btn').addEventListener('click', clearAllSessions);
    document.getElementById('set-unlimited-btn').addEventListener('click', setUnlimitedTokens);
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
    
    // Track start time for client-side metrics
    const startTime = Date.now();
    
    try {
        // Prepare request
        const requestData = {
            text: text,
            voice: selectedVoice,
            temperature: parseFloat(document.getElementById('temperature').value),
            top_p: parseFloat(document.getElementById('top-p').value),
            repetition_penalty: parseFloat(document.getElementById('repetition-penalty').value),
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
        
        // Add client-side timing if not provided by server
        if (!result.elapsed_ms) {
            result.elapsed_ms = Date.now() - startTime;
        }
        
        updateProgress(80, 'Finalizing audio...');
        
        // Update current session
        currentSessionId = result.session_id;
        
        // Display audio with stats
        displayAudio(result);
        
        // Update system stats
        updateSystemStats();
        
        updateProgress(100, 'Complete!');
        
        // Hide progress after delay
        setTimeout(() => {
            showProgress(false);
        }, 500);
        
        // Reload sessions
        loadSessions();
        
        // Show detailed success message
        const rtf = result.duration && (result.elapsed_ms/1000) ? (result.duration / (result.elapsed_ms/1000)).toFixed(2) : '?';
        showToast(`Speech generated in ${(result.elapsed_ms/1000).toFixed(2)}s (${rtf}x real-time)`, 'success');
        
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
    
    // Update basic info
    document.getElementById('audio-duration').textContent = `${result.duration.toFixed(2)}s`;
    document.getElementById('audio-size').textContent = `${(result.file_size / 1024).toFixed(1)} KB`;
    document.getElementById('audio-voice').textContent = selectedVoice;
    
    // Update advanced stats
    const genTimeMs = result.elapsed_ms || 0;
    const genTimeSec = genTimeMs / 1000;
    document.getElementById('gen-time').textContent = genTimeMs < 1000 ? `${genTimeMs}ms` : `${genTimeSec.toFixed(2)}s`;
    
    // Tokens and TPS
    document.getElementById('tokens-count').textContent = result.tokens || '0';
    document.getElementById('tps-rate').textContent = result.tps ? `${result.tps.toFixed(1)} t/s` : '-';
    
    // Real-time factor (how much faster than real-time)
    // RTF = audio_duration / generation_time
    const rtf = result.duration && genTimeSec ? (result.duration / genTimeSec).toFixed(2) : '-';
    const rtfElement = document.getElementById('rtf-ratio');
    rtfElement.textContent = rtf !== '-' ? `${rtf}x` : '-';
    
    // Color code RTF based on performance
    if (rtf !== '-') {
        const rtfValue = parseFloat(rtf);
        if (rtfValue >= 2.0) {
            rtfElement.className = 'stat-value good';
        } else if (rtfValue >= 1.0) {
            rtfElement.className = 'stat-value warning';
        } else {
            rtfElement.className = 'stat-value poor';
        }
    }
    
    // Setup download button
    document.getElementById('download-btn').onclick = () => downloadAudio(result.audio_url);
    
    // Setup share button
    document.getElementById('share-btn').onclick = () => shareAudio(result.session_id);
    
    // Scroll to output
    outputSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Removed streaming functions as they are not working
// The WebSocket streaming functionality has been disabled

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
        const statusElement2 = document.getElementById('model-status-2');
        
        if (data.model_loaded) {
            statusElement.textContent = 'Model Ready';
            statusElement.style.background = 'var(--success-color)';
            statusElement2.textContent = 'Ready';
            statusElement2.className = 'stat-value good';
            
            // Update GPU memory if available
            if (data.gpu_info && data.gpu_info.cuda_memory_allocated) {
                document.getElementById('gpu-memory').textContent = data.gpu_info.cuda_memory_allocated;
            }
        } else {
            statusElement.textContent = 'Loading Model...';
            statusElement.style.background = 'var(--warning-color)';
            statusElement2.textContent = 'Loading';
            statusElement2.className = 'stat-value warning';
            
            // Check again in 5 seconds
            setTimeout(checkModelStatus, 5000);
        }
        
        // Update session count
        document.getElementById('total-sessions').textContent = data.sessions_count || '0';
        
    } catch (error) {
        console.error('Error checking model status:', error);
        document.getElementById('model-status').textContent = 'Offline';
        document.getElementById('model-status').style.background = 'var(--danger-color)';
        document.getElementById('model-status-2').textContent = 'Offline';
        document.getElementById('model-status-2').className = 'stat-value poor';
    }
}

function clearText() {
    document.getElementById('text-input').value = '';
    updateTextStats();
}

function resetSettings() {
    // Reset to Orpheus recommended defaults
    document.getElementById('temperature').value = 0.4;
    document.getElementById('temp-value').textContent = '0.4';
    
    document.getElementById('top-p').value = 0.9;
    document.getElementById('top-p-value').textContent = '0.9';
    
    document.getElementById('repetition-penalty').value = 1.2;  // Higher to prevent repetition
    document.getElementById('rep-penalty-value').textContent = '1.2';
    
    document.getElementById('max-tokens').value = 4096;
    
    showToast('Settings reset to defaults', 'success');
}

function setUnlimitedTokens() {
    document.getElementById('max-tokens').value = 0;
    showToast('Max tokens set to unlimited', 'success');
}

async function clearAllSessions() {
    if (!confirm('Delete all saved audio sessions? This cannot be undone.')) return;
    
    try {
        const response = await fetch('/api/sessions');
        const data = await response.json();
        
        // Delete each session
        for (const session of data.sessions) {
            await fetch(`/api/session/${session.session_id}`, {
                method: 'DELETE'
            });
        }
        
        // Clear the output section
        document.getElementById('output-section').style.display = 'none';
        
        // Reload sessions list
        loadSessions();
        showToast('All sessions cleared', 'success');
    } catch (error) {
        console.error('Error clearing sessions:', error);
        showToast('Failed to clear sessions', 'error');
    }
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

// Update system stats
async function updateSystemStats() {
    try {
        const response = await fetch('/api/sessions');
        const data = await response.json();
        
        if (data.sessions && data.sessions.length > 0) {
            // Calculate average generation time and TPS
            let totalGenTime = 0;
            let totalTPS = 0;
            let validSessions = 0;
            
            data.sessions.forEach(session => {
                if (session.elapsed_ms) {
                    totalGenTime += session.elapsed_ms;
                    validSessions++;
                }
                if (session.tps) {
                    totalTPS += session.tps;
                }
            });
            
            if (validSessions > 0) {
                const avgGenTime = totalGenTime / validSessions;
                document.getElementById('avg-gen-time').textContent = 
                    avgGenTime < 1000 ? `${avgGenTime.toFixed(0)}ms` : `${(avgGenTime/1000).toFixed(2)}s`;
                
                const avgTPS = totalTPS / validSessions;
                document.getElementById('avg-tps').textContent = `${avgTPS.toFixed(1)} t/s`;
            }
        }
        
        // Check health for GPU info
        const healthResponse = await fetch('/api/health');
        const healthData = await healthResponse.json();
        
        if (healthData.gpu_info && healthData.gpu_info.cuda_memory_allocated) {
            document.getElementById('gpu-memory').textContent = healthData.gpu_info.cuda_memory_allocated;
        }
        
    } catch (error) {
        console.error('Error updating system stats:', error);
    }
}

// Insert emotion tag at cursor position
function insertEmotionTag(tag) {
    const textInput = document.getElementById('text-input');
    const cursorPos = textInput.selectionStart;
    const textBefore = textInput.value.substring(0, cursorPos);
    const textAfter = textInput.value.substring(cursorPos);
    
    // Add space before tag if needed
    const needsSpaceBefore = textBefore.length > 0 && !textBefore.endsWith(' ') && !textBefore.endsWith('\n');
    const spaceBefore = needsSpaceBefore ? ' ' : '';
    
    // Add space after tag if there's text after and it doesn't start with space or punctuation
    const needsSpaceAfter = textAfter.length > 0 && !textAfter.startsWith(' ') && !textAfter.startsWith('\n') && !/^[.,!?;:]/.test(textAfter);
    const spaceAfter = needsSpaceAfter ? ' ' : '';
    
    const emotionTag = `${spaceBefore}<${tag}>${spaceAfter}`;
    
    // Insert the tag
    textInput.value = textBefore + emotionTag + textAfter;
    
    // Move cursor to after the inserted tag
    const newCursorPos = cursorPos + emotionTag.length;
    textInput.setSelectionRange(newCursorPos, newCursorPos);
    
    // Focus back on text input
    textInput.focus();
    
    // Update stats
    updateTextStats();
    
    // Visual feedback
    const btn = document.querySelector(`[data-tag="${tag}"]`);
    btn.style.transform = 'scale(0.9)';
    setTimeout(() => {
        btn.style.transform = '';
    }, 150);
}

// Handle window close - cleanup audio context
window.addEventListener('beforeunload', () => {
    if (audioContext) {
        audioContext.close();
    }
});

// Update system stats periodically
setInterval(updateSystemStats, 30000); // Every 30 seconds