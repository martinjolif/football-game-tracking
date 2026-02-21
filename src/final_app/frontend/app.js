// Auto-detect API base: if running from PyCharm/IDE server (port 63342), use localhost:8080
const API_BASE = (window.location.port === '63342')
    ? 'http://localhost:8080'
    : window.location.origin;

let selectedFile = null;
let currentJobId = null;

// Elements
const uploadSection = document.getElementById('uploadSection');
const fileInput = document.getElementById('videoFile');
const selectedFileDiv = document.getElementById('selectedFile');
const fileNameSpan = document.getElementById('fileName');
const processBtn = document.getElementById('processBtn');
const progressSection = document.getElementById('progressSection');
const resultSection = document.getElementById('resultSection');
const errorMessage = document.getElementById('errorMessage');
const progressFill = document.getElementById('progressFill');
const statusMessage = document.getElementById('statusMessage');
const resultVideo = document.getElementById('resultVideo');
const downloadBtn = document.getElementById('downloadBtn');

// Upload section click
uploadSection.addEventListener('click', () => fileInput.click());

// Drag and drop
uploadSection.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadSection.classList.add('dragging');
});

uploadSection.addEventListener('dragleave', () => {
    uploadSection.classList.remove('dragging');
});

uploadSection.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadSection.classList.remove('dragging');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

// File input change
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    selectedFile = file;
    fileNameSpan.textContent = file.name;
    selectedFileDiv.classList.add('active');
    processBtn.disabled = false;
    hideError();
}

// Process button
processBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('video', selectedFile);
    formData.append('enable_radar', document.getElementById('enableRadar').checked);
    formData.append('enable_commentary', document.getElementById('enableCommentary').checked);
    formData.append('enable_tracking', document.getElementById('enableTracking').checked);
    formData.append('enable_team_clustering', document.getElementById('enableTeam').checked);
    formData.append('enable_tts', document.getElementById('enableTts').checked);

    const endFrame = document.getElementById('endFrame').value;
    if (endFrame) formData.append('end_frame', endFrame);

    formData.append('cluster_train_frames', document.getElementById('clusterFrames').value);

    try {
        processBtn.disabled = true;
        progressSection.classList.add('active');
        resultSection.classList.remove('active');

        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Upload failed');

        const data = await response.json();
        currentJobId = data.job_id;

        pollStatus(currentJobId);

    } catch (error) {
        showError('Error uploading video: ' + error.message);
        processBtn.disabled = false;
        progressSection.classList.remove('active');
    }
});

async function pollStatus(jobId) {
    try {
        const response = await fetch(`${API_BASE}/status/${jobId}`);
        const data = await response.json();

        progressFill.style.width = data.progress + '%';
        progressFill.textContent = data.progress + '%';
        statusMessage.textContent = data.message;

        if (data.status === 'completed') {
            progressSection.classList.remove('active');
            resultSection.classList.add('active');
            resultVideo.src = `${API_BASE}/outputs/${jobId}_output.mp4`;
            processBtn.disabled = false;
        } else if (data.status === 'failed') {
            showError('Processing failed: ' + data.message);
            progressSection.classList.remove('active');
            processBtn.disabled = false;
        } else {
            setTimeout(() => pollStatus(jobId), 1000);
        }
    } catch (error) {
        showError('Error checking status: ' + error.message);
        progressSection.classList.remove('active');
        processBtn.disabled = false;
    }
}

downloadBtn.addEventListener('click', () => {
    if (currentJobId) {
        window.open(`${API_BASE}/download/${currentJobId}`, '_blank');
    }
});

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.add('active');
}

function hideError() {
    errorMessage.classList.remove('active');
}