// Global variables
let selectedFile = null;
let isAnalyzing = false;

// DOM Elements - Safely get elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('audioFile');
const filePreview = document.getElementById('filePreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingContainer = document.getElementById('loadingContainer');
const resultsContainer = document.getElementById('resultsContainer');
const errorContainer = document.getElementById('errorContainer');

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Check if required elements exist
    if (!uploadArea || !fileInput || !analyzeBtn) {
        console.error('Required DOM elements not found');
        return;
    }
    
    setupEventListeners();
    setupDragAndDrop();
    
    console.log('AI Ses Analizi uygulaması yüklendi');
}

function setupEventListeners() {
    // File input change
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    // Upload area click
    if (uploadArea) {
        uploadArea.addEventListener('click', () => {
            if (!isAnalyzing && fileInput) {
                fileInput.click();
            }
        });
    }
    
    // Analyze button
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', analyzeAudio);
    }
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const target = document.getElementById(targetId);
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

function setupDragAndDrop() {
    if (!uploadArea) return;
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    uploadArea.addEventListener('drop', handleDrop, false);
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight() {
    if (uploadArea) {
        uploadArea.classList.add('drag-over');
    }
}

function unhighlight() {
    if (uploadArea) {
        uploadArea.classList.remove('drag-over');
    }
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        const file = files[0];
        if (isValidAudioFile(file)) {
            selectedFile = file;
            showFilePreview(file);
        } else {
            showError('Lütfen geçerli bir ses dosyası seçin (.wav, .mp3, .flac)');
        }
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        if (isValidAudioFile(file)) {
            selectedFile = file;
            showFilePreview(file);
        } else {
            showError('Lütfen geçerli bir ses dosyası seçin (.wav, .mp3, .flac)');
        }
    }
}

function isValidAudioFile(file) {
    const validTypes = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/flac', 'audio/x-wav'];
    const validExtensions = ['.wav', '.mp3', '.flac'];
    
    const hasValidType = validTypes.includes(file.type);
    const hasValidExtension = validExtensions.some(ext => 
        file.name.toLowerCase().endsWith(ext)
    );
    
    const isValidSize = file.size <= 10 * 1024 * 1024; // 10MB
    
    if (!isValidSize) {
        showError('Dosya boyutu 10MB\'dan küçük olmalı!');
        return false;
    }
    
    return hasValidType || hasValidExtension;
}

function showFilePreview(file) {
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    
    if (fileName) fileName.textContent = file.name;
    if (fileSize) fileSize.textContent = formatFileSize(file.size);
    
    if (filePreview) {
        filePreview.style.display = 'block';
    }
    
    if (analyzeBtn) {
        analyzeBtn.disabled = false;
    }
    
    // Hide upload area, show preview
    if (uploadArea) {
        uploadArea.style.display = 'none';
    }
    
    trackFileUpload(file.size, file.type);
}

function removeFile() {
    selectedFile = null;
    
    if (fileInput) {
        fileInput.value = '';
    }
    
    if (filePreview) {
        filePreview.style.display = 'none';
    }
    
    if (uploadArea) {
        uploadArea.style.display = 'block';
    }
    
    if (analyzeBtn) {
        analyzeBtn.disabled = true;
    }
    
    hideResults();
    hideError();
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

async function analyzeAudio() {
    if (!selectedFile || isAnalyzing) return;
    
    isAnalyzing = true;
    
    if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analiz Ediliyor...';
    }
    
    showLoading();
    hideResults();
    hideError();
    
    trackAnalysisStart();
    const startTime = performance.now();
    
    const formData = new FormData();
    formData.append('audio', selectedFile);
    
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            showResults(result.data);
            trackAnalysisComplete(result.data.analysis.emotion, result.data.analysis.confidence);
        } else {
            showError(`Analiz hatası: ${result.error}`);
        }
        
    } catch (error) {
        console.error('Analiz hatası:', error);
        showError(`Bağlantı hatası: ${error.message}`);
    } finally {
        hideLoading();
        isAnalyzing = false;
        
        if (analyzeBtn) {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-brain"></i> Analiz Et';
        }
        
        logPerformance(startTime, 'Audio Analysis');
    }
}

function showLoading() {
    if (loadingContainer) {
        loadingContainer.style.display = 'block';
        
        // Progress animation
        const progressFill = loadingContainer.querySelector('.progress-fill');
        if (progressFill) {
            progressFill.style.animation = 'none';
            setTimeout(() => {
                progressFill.style.animation = 'progress 3s ease-in-out infinite';
            }, 100);
        }
    }
}

function hideLoading() {
    if (loadingContainer) {
        loadingContainer.style.display = 'none';
    }
}

function showResults(data) {
    const { analysis, visualization } = data;
    
    // Update result values safely
    const emotionValue = document.getElementById('emotionValue');
    const emotionConfidence = document.getElementById('emotionConfidence');
    const ageValue = document.getElementById('ageValue');
    const genderValue = document.getElementById('genderValue');
    const genderConfidence = document.getElementById('genderConfidence');
    
    if (emotionValue) {
        emotionValue.textContent = analysis.emotion;
    }
    
    if (emotionConfidence) {
        emotionConfidence.textContent = `Güven: %${(analysis.confidence * 100).toFixed(1)}`;
    }
    
    if (ageValue) {
        ageValue.textContent = `${analysis.age} yaş`;
    }
    
    if (genderValue) {
        genderValue.textContent = analysis.gender;
    }
    
    if (genderConfidence) {
        genderConfidence.textContent = `Güven: %${(analysis.gender_confidence * 100).toFixed(1)}`;
    }
    
    // Show visualization if available
    if (visualization) {
        const chartImg = document.getElementById('analysisChart');
        if (chartImg) {
            chartImg.src = `data:image/png;base64,${visualization}`;
            chartImg.style.display = 'block';
        }
    }
    
    // Show features
    if (analysis.features) {
        showFeatures(analysis.features);
    }
    
    // Show results container
    if (resultsContainer) {
        resultsContainer.style.display = 'block';
        
        // Scroll to results
        setTimeout(() => {
            resultsContainer.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });
        }, 300);
    }
}

function showFeatures(features) {
    const featuresGrid = document.getElementById('featuresGrid');
    if (!featuresGrid) return;
    
    const featureItems = [
        {
            name: 'Spektral Centroid',
            value: features.spectral_centroid ? features.spectral_centroid.toFixed(2) : 'N/A',
            unit: 'Hz',
            description: 'Ses frekansının merkezi'
        },
        {
            name: 'RMS Energy',
            value: features.rms_energy ? features.rms_energy.toFixed(4) : 'N/A',
            unit: '',
            description: 'Ses şiddeti'
        },
        {
            name: 'Zero Crossing Rate',
            value: features.zcr ? features.zcr.toFixed(4) : 'N/A',
            unit: '',
            description: 'Ses değişim hızı'
        },
        {
            name: 'Spektral Rolloff',
            value: features.rolloff ? features.rolloff.toFixed(2) : 'N/A',
            unit: 'Hz',
            description: 'Frekans dağılımı'
        }
    ];
    
    featuresGrid.innerHTML = featureItems.map(item => `
        <div class="feature-item">
            <strong>${item.name}:</strong> ${item.value}${item.unit}
            <br><small>${item.description}</small>
        </div>
    `).join('');
}

function hideResults() {
    if (resultsContainer) {
        resultsContainer.style.display = 'none';
    }
}

function showError(message) {
    const errorText = document.getElementById('errorText');
    
    if (errorText) {
        errorText.textContent = message;
    }
    
    if (errorContainer) {
        errorContainer.style.display = 'block';
        
        // Auto hide after 5 seconds
        setTimeout(hideError, 5000);
    }
    
    console.error('Error:', message);
}

function hideError() {
    if (errorContainer) {
        errorContainer.style.display = 'none';
    }
}

function resetAnalysis() {
    removeFile();
    hideResults();
    hideError();
    hideLoading();
    
    // Scroll back to upload area
    setTimeout(() => {
        const demoSection = document.getElementById('demo');
        if (demoSection) {
            demoSection.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });
        }
    }, 300);
}

// Window function for reset button (called from HTML)
window.removeFile = removeFile;
window.resetAnalysis = resetAnalysis;
window.analyzeAudio = analyzeAudio;

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Performance monitoring
function logPerformance(startTime, operation) {
    const endTime = performance.now();
    const duration = endTime - startTime;
    console.log(`${operation} took ${duration.toFixed(2)}ms`);
}

// Analytics functions (safe implementations)
function trackEvent(eventName, parameters = {}) {
    console.log(`Analytics: ${eventName}`, parameters);
    
    // Google Analytics 4 example (uncomment if GA is configured):
    // if (typeof gtag !== 'undefined') {
    //     gtag('event', eventName, parameters);
    // }
}

function trackFileUpload(fileSize, fileType) {
    trackEvent('file_upload', {
        file_size: fileSize,
        file_type: fileType,
        timestamp: new Date().toISOString()
    });
}

function trackAnalysisStart() {
    trackEvent('analysis_start', {
        timestamp: new Date().toISOString()
    });
}

function trackAnalysisComplete(emotion, confidence) {
    trackEvent('analysis_complete', {
        emotion: emotion,
        confidence: confidence,
        timestamp: new Date().toISOString()
    });
}

// Error tracking
window.addEventListener('error', function(e) {
    console.error('JavaScript Error:', e.error);
    trackEvent('javascript_error', {
        message: e.message,
        filename: e.filename,
        lineno: e.lineno
    });
});

// Unhandled promise rejection tracking
window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled Promise Rejection:', e.reason);
    trackEvent('unhandled_rejection', {
        reason: e.reason.toString()
    });
});

// Page visibility API for performance
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        console.log('Page hidden - pausing operations');
    } else {
        console.log('Page visible - resuming operations');
    }
});

// Service Worker registration (Progressive Web App)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/static/sw.js')
            .then(function(registration) {
                console.log('SW registered: ', registration);
            })
            .catch(function(registrationError) {
                console.log('SW registration failed: ', registrationError);
            });
    });
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl+U or Cmd+U to upload file
    if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
        e.preventDefault();
        if (!isAnalyzing && fileInput) {
            fileInput.click();
        }
    }
    
    // Enter to analyze when file is selected
    if (e.key === 'Enter' && selectedFile && !isAnalyzing) {
        analyzeAudio();
    }
    
    // Escape to reset
    if (e.key === 'Escape') {
        resetAnalysis();
    }
});

// Touch device optimizations
if ('ontouchstart' in window) {
    document.body.classList.add('touch-device');
}

// Connection quality detection (safe implementation)
if ('connection' in navigator && navigator.connection) {
    const connection = navigator.connection;
    
    function updateConnectionStatus() {
        const quality = connection.effectiveType;
        console.log(`Connection quality: ${quality}`);
        
        // Bağlantı kalitesine göre dosya boyutu uyarısı
        if (quality === 'slow-2g' || quality === '2g') {
            showError('Yavaş internet bağlantısı tespit edildi. Küçük dosyalar kullanın.');
        }
    }
    
    connection.addEventListener('change', updateConnectionStatus);
    updateConnectionStatus();
}

// Battery level monitoring (safe implementation)
if ('getBattery' in navigator) {
    navigator.getBattery().then(function(battery) {
        function updateBatteryStatus() {
            if (battery.level < 0.2 && !battery.charging) {
                console.warn('Low battery detected - consider reducing processing');
            }
        }
        
        battery.addEventListener('levelchange', updateBatteryStatus);
        battery.addEventListener('chargingchange', updateBatteryStatus);
        updateBatteryStatus();
    }).catch(function(error) {
        console.log('Battery API not available:', error);
    });
}

// Mobile device detection and optimizations
function isMobileDevice() {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
}

// Optimize for mobile devices
if (isMobileDevice()) {
    // Reduce animation complexity on mobile
    document.body.classList.add('mobile-device');
    
    // Disable some heavy animations
    const style = document.createElement('style');
    style.textContent = `
        .mobile-device .wave-bar {
            animation-duration: 3s !important;
        }
        .mobile-device .loading-spinner {
            animation-duration: 2s !important;
        }
    `;
    document.head.appendChild(style);
}

// Performance observer (if available)
if ('PerformanceObserver' in window) {
    const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
            if (entry.entryType === 'navigation') {
                console.log(`Page load time: ${entry.loadEventEnd - entry.loadEventStart}ms`);
            }
        }
    });
    
    observer.observe({ entryTypes: ['navigation'] });
}

// Initialize tooltips or help system if needed
function initializeTooltips() {
    const tooltipElements = document.querySelectorAll('[data-tooltip]');
    tooltipElements.forEach(element => {
        element.addEventListener('mouseenter', showTooltip);
        element.addEventListener('mouseleave', hideTooltip);
    });
}

function showTooltip(e) {
    const tooltip = e.target.getAttribute('data-tooltip');
    if (tooltip) {
        // Create and show tooltip
        const tooltipEl = document.createElement('div');
        tooltipEl.className = 'tooltip';
        tooltipEl.textContent = tooltip;
        document.body.appendChild(tooltipEl);
        
        // Position tooltip
        const rect = e.target.getBoundingClientRect();
        tooltipEl.style.position = 'absolute';
        tooltipEl.style.top = (rect.bottom + 5) + 'px';
        tooltipEl.style.left = rect.left + 'px';
    }
}

function hideTooltip() {
    const tooltip = document.querySelector('.tooltip');
    if (tooltip) {
        tooltip.remove();
    }
}

// Initialize tooltips after DOM load
document.addEventListener('DOMContentLoaded', initializeTooltips);

console.log('🎵 AI Ses Analizi JavaScript yüklendi!');