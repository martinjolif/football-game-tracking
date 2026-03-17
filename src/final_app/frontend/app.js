// Auto-detect API base: if running from PyCharm/IDE server (port 63342), use localhost:8080
const API_BASE = (window.location.port === '63342')
    ? 'http://localhost:8080'
    : window.location.origin;

document.addEventListener('DOMContentLoaded', () => {
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
    const demoBtn = document.getElementById('demoBtn');
    const stopBtn = document.getElementById('stopBtn');

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
        if (files.length > 0) handleFileSelect(files[0]);
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
    });

    function handleFileSelect(file) {
        selectedFile = file;
        fileNameSpan.textContent = file.name;
        selectedFileDiv.classList.add('active');
        processBtn.disabled = false;
        hideError();
    }

    function getFormOptions() {
        const formData = new FormData();
        formData.append('enable_radar', document.getElementById('enableRadar').checked);
        formData.append('enable_commentary', document.getElementById('enableCommentary').checked);
        formData.append('enable_tracking', document.getElementById('enableTracking').checked);
        formData.append('enable_team_clustering', document.getElementById('enableTeam').checked);
        formData.append('enable_tts', document.getElementById('enableTTS').checked);
        const endFrame = document.getElementById('endFrame').value;
        if (endFrame) formData.append('end_frame', endFrame);
        formData.append('cluster_train_frames', document.getElementById('clusterFrames').value);
        return formData;
    }

    // Process button
    processBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        const formData = getFormOptions();
        formData.append('video', selectedFile);

        try {
            processBtn.disabled = true;
            stopBtn.disabled = false;
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

    // Demo button
    demoBtn.addEventListener('click', async () => {
        try {
            demoBtn.disabled = true;
            demoBtn.textContent = 'Starting...';
            stopBtn.disabled = false;
            hideError();
            progressSection.classList.add('active');
            resultSection.classList.remove('active');

            const response = await fetch(`${API_BASE}/demo`, { method: 'POST' });
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Failed to start demo');
            }

            const data = await response.json();
            currentJobId = data.job_id;
            pollStatus(currentJobId);

        } catch (error) {
            showError('Error starting demo: ' + error.message);
            progressSection.classList.remove('active');
        } finally {
            demoBtn.disabled = false;
            demoBtn.textContent = '▶ Use Demo Video';
        }
    });

    // Stop button
    stopBtn.addEventListener('click', async () => {
        if (!currentJobId) return;
        stopBtn.disabled = true;
        try {
            await fetch(`${API_BASE}/cancel/${currentJobId}`, { method: 'POST' });
        } catch (e) {
            // Ignore — pollStatus will pick up the cancelled state
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
                stopBtn.disabled = true;
                resultSection.classList.add('active');
                resultVideo.src = `${API_BASE}/outputs/${jobId}_output.mp4`;
                processBtn.disabled = false;
                demoBtn.disabled = false;
                loadRecentVideos();
            } else if (data.status === 'cancelled') {
                progressSection.classList.remove('active');
                stopBtn.disabled = true;
                processBtn.disabled = false;
                demoBtn.disabled = false;
            } else if (data.status === 'failed') {
                showError('Processing failed: ' + data.message);
                progressSection.classList.remove('active');
                stopBtn.disabled = true;
                processBtn.disabled = false;
                demoBtn.disabled = false;
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

    // Recent videos
    async function loadRecentVideos() {
        try {
            const response = await fetch(`${API_BASE}/videos`);
            if (!response.ok) return;
            const videos = await response.json();
            renderRecentVideos(videos);
        } catch (e) {
            // Silently fail
        }
    }

    window.playRecentVideo = function playRecentVideo(jobId, streamUrl) {
        currentJobId = jobId;
        resultSection.classList.add('active');
        resultVideo.src = `${API_BASE}${streamUrl}`;
        resultVideo.load();
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }

    function renderRecentVideos(videos) {
        const list = document.getElementById('recentVideosList');
        if (!videos || videos.length === 0) {
            list.innerHTML = '<p class="no-videos-msg">No processed videos found.</p>';
            return;
        }

        list.innerHTML = videos.map(v => {
            const date = v.completed_at
                ? new Date(v.completed_at + 'Z').toLocaleString()
                : 'Unknown date';
            const name = v.original_filename || 'video.mp4';
            return `
            <div class="recent-video-item">
                <div class="recent-video-info">
                    <div class="recent-video-name">${escapeHtml(name)}</div>
                    <div class="recent-video-date">${date}</div>
                </div>
                <div class="recent-video-actions">
                    <button class="btn-small btn-play" onclick="playRecentVideo('${v.job_id}', '${v.stream_url}')">▶ Play</button>
                    <a class="btn-small btn-dl" href="${API_BASE}${v.download_url}" download>⬇ Download</a>
                </div>
            </div>`;
        }).join('');
    }

    function escapeHtml(str) {
        return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.add('active');
    }

    function hideError() {
        errorMessage.classList.remove('active');
    }

    loadRecentVideos();
});