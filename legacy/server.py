import asyncio
import json
import socket
import ssl
import os
import math
import base64
import cv2
import numpy as np
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Warning: pyrealsense2 not available. RealSense streaming will be disabled.")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: mediapipe not available. Hand detection will be disabled.")

from aiohttp import web

def get_local_ip():
    """Get the local IP address of this computer"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def create_self_signed_cert():
    """Create a self-signed certificate for HTTPS"""
    try:
        from OpenSSL import crypto
        
        k = crypto.PKey()
        k.generate_key(crypto.TYPE_RSA, 2048)
        
        cert = crypto.X509()
        cert.get_subject().C = "US"
        cert.get_subject().ST = "State"
        cert.get_subject().L = "City"
        cert.get_subject().O = "OrientationApp"
        cert.get_subject().OU = "OrientationApp"
        cert.get_subject().CN = get_local_ip()
        cert.set_serial_number(1000)
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(365*24*60*60)
        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(k)
        cert.sign(k, 'sha256')
        
        with open("cert.pem", "wb") as f:
            f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
        
        with open("key.pem", "wb") as f:
            f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
        
        print("✓ SSL certificate created successfully")
        return True
    except ImportError:
        print("\n⚠ pyOpenSSL not installed. Installing it now...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'pyOpenSSL'])
        print("✓ pyOpenSSL installed. Please run the script again.")
        return False

# Madgwick Filter Implementation
class MadgwickFilter:
    def __init__(self, beta=0.1, sample_freq=50.0):
        """
        Initialize Madgwick filter
        beta: algorithm gain (lower = more stable but slower response)
        sample_freq: expected sample frequency in Hz
        """
        self.beta = beta
        self.sample_freq = sample_freq
        self.q = [1.0, 0.0, 0.0, 0.0]  # Quaternion [w, x, y, z]
    
    def update(self, gx, gy, gz, ax, ay, az):
        """
        Update filter with gyroscope (rad/s) and accelerometer (m/s²) data
        Returns roll, pitch, yaw in degrees
        """
        q1, q2, q3, q4 = self.q
        
        # Normalize accelerometer measurement
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm == 0:
            return self.to_euler()
        ax /= norm
        ay /= norm
        az /= norm
        
        # Gradient descent algorithm
        s1 = 2*q2*q4 - 2*q1*q3 - ax
        s2 = 2*q1*q2 + 2*q3*q4 - ay
        s3 = 1 - 2*q2*q2 - 2*q3*q3 - az
        
        # Normalise step magnitude
        norm = math.sqrt(s1*s1 + s2*s2 + s3*s3)
        if norm != 0:
            s1 /= norm
            s2 /= norm
            s3 /= norm
        
        # Compute rate of change of quaternion
        qDot1 = 0.5 * (-q2*gx - q3*gy - q4*gz) - self.beta * s1
        qDot2 = 0.5 * (q1*gx + q3*gz - q4*gy) - self.beta * s2
        qDot3 = 0.5 * (q1*gy - q2*gz + q4*gx) - self.beta * s3
        qDot4 = 0.5 * (q1*gz + q2*gy - q3*gx)
        
        # Integrate to yield quaternion
        dt = 1.0 / self.sample_freq
        q1 += qDot1 * dt
        q2 += qDot2 * dt
        q3 += qDot3 * dt
        q4 += qDot4 * dt
        
        # Normalise quaternion
        norm = math.sqrt(q1*q1 + q2*q2 + q3*q3 + q4*q4)
        self.q = [q1/norm, q2/norm, q3/norm, q4/norm]
        
        return self.to_euler()
    
    def to_euler(self):
        """Convert quaternion to Euler angles (roll, pitch, yaw) in degrees"""
        q1, q2, q3, q4 = self.q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q1 * q2 + q3 * q4)
        cosr_cosp = 1 - 2 * (q2 * q2 + q3 * q3)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (q1 * q3 - q4 * q2)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q1 * q4 + q2 * q3)
        cosy_cosp = 1 - 2 * (q3 * q3 + q4 * q4)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return (
            math.degrees(roll),
            math.degrees(pitch),
            math.degrees(yaw)
        )

# HTML page
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>XLeRobotHead - iPhone Camera & RealSense</title>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/control_utils/control_utils.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            max-width: 400px;
            width: 100%;
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 20px;
            font-size: 24px;
        }
        .mode-selector {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .mode-btn {
            flex: 1;
            padding: 10px;
            border: 2px solid #667eea;
            background: white;
            border-radius: 8px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .mode-btn.active {
            background: #667eea;
            color: white;
        }
        .status {
            text-align: center;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-weight: bold;
            font-size: 16px;
        }
        .status.disconnected {
            background: #fee;
            color: #c33;
        }
        .status.connected {
            background: #efe;
            color: #3c3;
        }
        .data-section {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .section-title {
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
            font-size: 14px;
        }
        .data-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #e0e0e0;
            font-size: 14px;
        }
        .data-row:last-child {
            border-bottom: none;
        }
        .label {
            font-weight: bold;
            color: #666;
        }
        .value {
            font-family: 'Courier New', monospace;
            color: #333;
        }
        button {
            width: 100%;
            padding: 15px;
            font-size: 18px;
            border: none;
            border-radius: 10px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-start {
            background: #667eea;
            color: white;
        }
        .btn-start:active {
            background: #5568d3;
        }
        .btn-stop {
            background: #f56565;
            color: white;
        }
        .btn-stop:active {
            background: #e05555;
        }
        .info {
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 15px;
        }
        .rate {
            text-align: center;
            color: #999;
            font-size: 11px;
            margin-top: 5px;
        }
        .video-container {
            width: 100%;
            margin-bottom: 15px;
            border-radius: 10px;
            overflow: hidden;
            background: #000;
        }
        .video-container video,
        .video-container canvas {
            width: 100%;
            display: block;
        }
        .video-label {
            background: #667eea;
            color: white;
            padding: 8px;
            font-size: 12px;
            font-weight: bold;
            text-align: center;
        }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .tab {
            flex: 1;
            padding: 10px;
            border: 2px solid #667eea;
            background: white;
            border-radius: 8px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .tab.active {
            background: #667eea;
            color: white;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .debug-console {
            margin-top: 15px;
            background: #1a1a1a;
            color: #00ff00;
            padding: 10px;
            border-radius: 10px;
            max-height: 200px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 11px;
            display: none;
        }
        .debug-console.active {
            display: block;
        }
        .debug-console .log-entry {
            margin: 2px 0;
            padding: 2px 5px;
            border-left: 2px solid #00ff00;
            padding-left: 10px;
        }
        .debug-console .log-error {
            color: #ff0000;
            border-left-color: #ff0000;
        }
        .debug-console .log-warn {
            color: #ffff00;
            border-left-color: #ffff00;
        }
        .debug-toggle {
            margin-top: 10px;
            padding: 8px;
            background: #444;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 12px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 XLeRobotHead</h1>
        
        <div class="tabs">
            <button class="tab active" onclick="switchTab('camera')">Camera</button>
            <button class="tab" onclick="switchTab('imu')">IMU</button>
        </div>
        
        <div id="cameraTab" class="tab-content active">
            <div class="video-container">
                <div class="video-label">📹 RealSense Camera</div>
                <img id="realsenseStream" src="/realsense_stream" alt="RealSense Stream" style="width: 100%; display: none;">
            </div>
            
            <div class="video-container">
                <div class="video-label">📱 iPhone Camera (Hand Detection)</div>
                <video id="input_video" autoplay playsinline style="width: 100%; display: none;"></video>
                <canvas id="output_canvas" style="width: 100%; display: none;"></canvas>
            </div>
            
            <div id="cameraStatus" class="status disconnected">
                Camera: Disconnected
            </div>
            
            <button id="cameraToggleBtn" class="btn-start" onclick="toggleCamera()">
                Start Camera
            </button>
            
            <button class="debug-toggle" onclick="toggleDebugConsole()">
                📋 Show/Hide Debug Console
            </button>
            
            <div id="debugConsole" class="debug-console">
                <div class="log-entry">Debug console ready. Click "Start Camera" to see logs.</div>
            </div>
        </div>
        
        <div id="imuTab" class="tab-content">
        <h2>📱 iPhone IMU Sender</h2>
        
        <div class="mode-selector">
            <button class="mode-btn active" onclick="setMode('fused')" id="fusedBtn">
                iOS Fused
            </button>
            <button class="mode-btn" onclick="setMode('raw')" id="rawBtn">
                Raw IMU
            </button>
        </div>
        
        <div id="imuStatus" class="status disconnected">
            Disconnected
        </div>
        
        <div class="data-section">
            <div class="section-title">Orientation</div>
            <div class="data-row">
                <span class="label">Roll:</span>
                <span class="value" id="roll">0.00°</span>
            </div>
            <div class="data-row">
                <span class="label">Pitch:</span>
                <span class="value" id="pitch">0.00°</span>
            </div>
            <div class="data-row">
                <span class="label">Yaw:</span>
                <span class="value" id="yaw">0.00°</span>
            </div>
        </div>
        
        <div class="data-section" id="rawDataSection" style="display: none;">
            <div class="section-title">Gyroscope (rad/s)</div>
            <div class="data-row">
                <span class="label">X:</span>
                <span class="value" id="gx">0.000</span>
            </div>
            <div class="data-row">
                <span class="label">Y:</span>
                <span class="value" id="gy">0.000</span>
            </div>
            <div class="data-row">
                <span class="label">Z:</span>
                <span class="value" id="gz">0.000</span>
            </div>
        </div>
        
        <div class="data-section" id="accelDataSection" style="display: none;">
            <div class="section-title">Accelerometer (m/s²)</div>
            <div class="data-row">
                <span class="label">X:</span>
                <span class="value" id="ax">0.000</span>
            </div>
            <div class="data-row">
                <span class="label">Y:</span>
                <span class="value" id="ay">0.000</span>
            </div>
            <div class="data-row">
                <span class="label">Z:</span>
                <span class="value" id="az">0.000</span>
            </div>
        </div>
        
            <button id="toggleBtn" class="btn-start" onclick="toggleStreaming()">
                Start Streaming
            </button>
            
            <div class="info">
                <span id="modeText">iOS sensor fusion</span>
            </div>
            <div class="rate" id="updateRate"></div>
        </div>
    </div>

    <script>
        // Debug Console Setup - Intercept console.log/warn/error
        let debugConsoleVisible = false;
        const debugConsole = document.getElementById('debugConsole');
        const maxLogEntries = 50;
        let logEntries = [];
        
        function addLogToConsole(message, type = 'log') {
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = { message: `${timestamp}: ${message}`, type: type };
            logEntries.push(logEntry);
            
            // Keep only last maxLogEntries
            if (logEntries.length > maxLogEntries) {
                logEntries.shift();
            }
            
            // Update console display if visible
            if (debugConsoleVisible) {
                updateDebugConsole();
            }
        }
        
        function updateDebugConsole() {
            if (!debugConsole) return;
            debugConsole.innerHTML = '';
            logEntries.forEach(entry => {
                const div = document.createElement('div');
                div.className = `log-entry log-${entry.type}`;
                div.textContent = entry.message;
                debugConsole.appendChild(div);
            });
            // Auto-scroll to bottom
            debugConsole.scrollTop = debugConsole.scrollHeight;
        }
        
        function toggleDebugConsole() {
            debugConsoleVisible = !debugConsoleVisible;
            if (debugConsoleVisible) {
                debugConsole.classList.add('active');
                updateDebugConsole();
            } else {
                debugConsole.classList.remove('active');
            }
        }
        
        // Intercept console methods
        const originalLog = console.log;
        const originalWarn = console.warn;
        const originalError = console.error;
        
        console.log = function(...args) {
            originalLog.apply(console, args);
            addLogToConsole(args.join(' '), 'log');
        };
        
        console.warn = function(...args) {
            originalWarn.apply(console, args);
            addLogToConsole(args.join(' '), 'warn');
        };
        
        console.error = function(...args) {
            originalError.apply(console, args);
            addLogToConsole(args.join(' '), 'error');
        };
        
        // MediaPipe Hands setup
        let camera = null;
        let hands = null;
        let isCameraActive = false;
        let lastImageSendTime = 0;
        const IMAGE_SEND_INTERVAL = 100; // Send image every 100ms
        
        // IMU WebSocket
        let ws = null;
        let isStreaming = false;
        let mode = 'fused';  // 'fused' or 'raw'
        let updateCount = 0;
        let lastRateUpdate = Date.now();
        
        // Tab switching
        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            if (tabName === 'camera') {
                event.target.classList.add('active');
                document.getElementById('cameraTab').classList.add('active');
            } else if (tabName === 'imu') {
                event.target.classList.add('active');
                document.getElementById('imuTab').classList.add('active');
            }
        }
        
        // MediaPipe Hands initialization
        function initializeMediaPipe() {
            hands = new Hands({
                locateFile: (file) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
                }
            });
            
            hands.setOptions({
                maxNumHands: 2,
                modelComplexity: 1,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });
            
            hands.onResults(onResults);
        }
        
        // Wait for MediaPipe to load
        window.addEventListener('load', () => {
            if (typeof Hands !== 'undefined') {
                // MediaPipe loaded
            }
        });
        
        // Process MediaPipe results
        function onResults(results) {
            if (!results || !results.image) {
                console.warn('onResults called without valid results/image');
                return;
            }
            
            const videoElement = document.getElementById('input_video');
            const canvasElement = document.getElementById('output_canvas');
            
            // Make videoElement globally available for sendImageToServer
            window.videoElement = videoElement;
            
            if (!canvasElement) {
                console.error('output_canvas element not found!');
                return;
            }
            
            // Ensure canvas size matches video
            if (canvasElement.width === 0 || canvasElement.height === 0) {
                if (videoElement.videoWidth > 0 && videoElement.videoHeight > 0) {
                    canvasElement.width = videoElement.videoWidth;
                    canvasElement.height = videoElement.videoHeight;
                } else {
                    console.warn('Video dimensions not ready yet');
                    return;
                }
            }
            
            const canvasCtx = canvasElement.getContext('2d');
            if (!canvasCtx) {
                console.error('Could not get canvas context!');
                return;
            }
            
            canvasCtx.save();
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
            canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
            
            if (results.multiHandLandmarks) {
                for (const landmarks of results.multiHandLandmarks) {
                    // Draw hand connections (if available)
                    if (typeof drawConnectors !== 'undefined' && typeof HAND_CONNECTIONS !== 'undefined') {
                        try {
                            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
                                color: '#00FF00',
                                lineWidth: 2
                            });
                        } catch (e) {
                            console.warn('Could not draw hand connections:', e);
                        }
                    }
                    // Draw hand landmarks (palms only - no armrests)
                    if (typeof drawLandmarks !== 'undefined') {
                        try {
                            drawLandmarks(canvasCtx, landmarks, {
                                color: '#FF0000',
                                lineWidth: 1,
                                radius: 3
                            });
                        } catch (e) {
                            console.warn('Could not draw hand landmarks:', e);
                        }
                    }
                }
            }
            canvasCtx.restore();
            
            
            // Send image to server periodically
            const now = Date.now();
            if (now - lastImageSendTime > IMAGE_SEND_INTERVAL && results.image) {
                sendImageToServer(results);
                lastImageSendTime = now;
            }
        }
        
        // Send image with hand detection to server
        async function sendImageToServer(results) {
            if (!results || !results.image) {
                console.warn('sendImageToServer called without valid results');
                return;
            }
            
            try {
                // Get videoElement (use global or get from DOM)
                const videoElement = window.videoElement || document.getElementById('input_video');
                
                // Use the output canvas that already has MediaPipe overlays drawn on it
                const canvasElement = document.getElementById('output_canvas');
                
                if (!canvasElement) {
                    return;
                }
                
                if (!canvasElement || canvasElement.width === 0 || canvasElement.height === 0) {
                
                // Fallback: create new canvas if output canvas not ready
                const canvas = document.createElement('canvas');
                canvas.width = results.image.width;
                canvas.height = results.image.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(results.image, 0, 0);
                
                // Draw hand landmarks if needed
                if (results.multiHandLandmarks && typeof drawConnectors !== 'undefined' && 
                    typeof drawLandmarks !== 'undefined' && typeof HAND_CONNECTIONS !== 'undefined') {
                    for (const landmarks of results.multiHandLandmarks) {
                        try {
                            drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
                                color: '#00FF00',
                                lineWidth: 2
                            });
                            drawLandmarks(ctx, landmarks, {
                                color: '#FF0000',
                                lineWidth: 1,
                                radius: 3
                            });
                        } catch (e) {
                            console.warn('Drawing failed:', e);
                        }
                    }
                }
                
                canvas.toBlob(async (blob) => {
                    if (!blob) return;
                    
                    const reader = new FileReader();
                    reader.onloadend = async () => {
                        const base64data = reader.result;
                        const handLandmarks = results.multiHandLandmarks || [];
                        
                        try {
                            const response = await fetch('/receive_image', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    image: base64data,
                                    handLandmarks: handLandmarks.map(landmarks => 
                                        landmarks.map(point => ({x: point.x, y: point.y, z: point.z}))
                                    )
                                })
                            });
                            
                            if (!response.ok) {
                                console.error('❌ Server error:', response.status, response.statusText);
                            } else {
                                const result = await response.json();
                                const now = Date.now();
                                if (!window.lastSendLog || now - window.lastSendLog > 5000) {
                                    console.log(`✓ Image sent to server (hands: ${handLandmarks.length})`);
                                    window.lastSendLog = now;
                                }
                            }
                        } catch (error) {
                            console.error('❌ Fetch error:', error);
                            console.error('Error details:', error.message, error.stack);
                        }
                    };
                    reader.readAsDataURL(blob);
                }, 'image/jpeg', 0.85);
                return;
            }
            
            // Use the output canvas directly (already has MediaPipe overlays)
            canvasElement.toBlob(async (blob) => {
                if (!blob) return;
                const reader = new FileReader();
                reader.onloadend = async () => {
                    const base64data = reader.result;
                    const handLandmarks = results.multiHandLandmarks || [];
                    
                    try {
                        await fetch('/receive_image', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                image: base64data,
                                handLandmarks: handLandmarks.map(landmarks => 
                                    landmarks.map(point => ({x: point.x, y: point.y, z: point.z}))
                                )
                            })
                        });
                    } catch (error) {
                        console.error('Error sending image:', error);
                    }
                };
                reader.readAsDataURL(blob);
            }, 'image/jpeg', 0.85);
            
            } catch (error) {
                console.error('Error in sendImageToServer:', error);
            }
        }
        
        // Toggle iPhone camera
        async function toggleCamera() {
            const btn = document.getElementById('cameraToggleBtn');
            const videoElement = document.getElementById('input_video');
            const canvasElement = document.getElementById('output_canvas');
            const realsenseStream = document.getElementById('realsenseStream');
            
            if (!isCameraActive) {
                // Initialize MediaPipe if not already done
                if (!hands) {
                    initializeMediaPipe();
                }
                
                // Request camera access - explicitly find back camera on iOS
                try {
                    // On iOS, we need to request permission first to get camera labels
                    // Step 1: Request any camera to get permission
                    let tempStream = null;
                    try {
                        tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
                        // Stop the temp stream immediately
                        tempStream.getTracks().forEach(track => track.stop());
                        console.log('Got camera permission');
                    } catch (permError) {
                        console.error('Camera permission error:', permError);
                        throw permError;
                    }
                    
                    // Step 2: Now enumerate devices (labels should be available now)
                    const devices = await navigator.mediaDevices.enumerateDevices();
                    const videoDevices = devices.filter(device => device.kind === 'videoinput');
                    
                    // Log camera count
                    if (videoDevices.length > 0) {
                        console.log(`Found ${videoDevices.length} camera(s)`);
                    }
                    
                    let backCameraId = null;
                    
                    // Find ultra-wide camera by label
                    // On iPhones, cameras are usually: Front Camera, Back Camera, Back Dual Wide Camera, Back Ultra Wide Camera
                    // We want specifically "Ultra Wide" not "Dual Wide" or regular "Back Camera"
                    
                    // Log all available cameras first
                    console.log('Available cameras:');
                    for (const device of videoDevices) {
                        console.log(`  - ${device.label}`);
                    }
                    
                    // First priority: Find "Ultra Wide" camera (explicitly exclude "Dual Wide")
                    for (const device of videoDevices) {
                        const label = device.label.toLowerCase();
                        // Look for "ultra" but NOT "dual" (to avoid selecting "Dual Wide")
                        if (label && label.includes('ultra') && !label.includes('dual')) {
                            backCameraId = device.deviceId;
                            console.log('✓ Found ultra-wide camera:', device.label);
                            break;
                        }
                    }
                    
                    // Second priority: Look for exact "ultra-wide" or "ultra wide" pattern
                    if (!backCameraId) {
                        for (const device of videoDevices) {
                            const label = device.label.toLowerCase();
                            if (label && (label.includes('ultra-wide') || label.match(/\bultra\s+wide\b/))) {
                                backCameraId = device.deviceId;
                                console.log('✓ Found ultra-wide camera (exact match):', device.label);
                                break;
                            }
                        }
                    }
                    
                    // Fallback: Any back camera (but log which one)
                    if (!backCameraId) {
                        for (const device of videoDevices) {
                            const label = device.label.toLowerCase();
                            if (label && (label.includes('back') || label.includes('rear') || 
                                label.includes('environment'))) {
                                backCameraId = device.deviceId;
                                console.log('⚠ Ultra-wide not found, using back camera:', device.label);
                                break;
                            }
                        }
                    }
                    
                    // If not found by label, try to get facingMode info
                    // Request a quick stream from each camera to check facingMode
                    if (!backCameraId && videoDevices.length > 1) {
                        console.log('⚠ Label not found, checking cameras for facingMode...');
                        for (let i = 0; i < videoDevices.length; i++) {
                            try {
                                const testStream = await navigator.mediaDevices.getUserMedia({
                                    video: { deviceId: { exact: videoDevices[i].deviceId } }
                                });
                                const testTrack = testStream.getVideoTracks()[0];
                                const testSettings = testTrack.getSettings();
                                testStream.getTracks().forEach(track => track.stop());
                                
                                if (testSettings.facingMode === 'environment') {
                                    backCameraId = videoDevices[i].deviceId;
                                    console.log(`✓ Found camera by facingMode:`, videoDevices[i].label);
                                    break;
                                }
                            } catch (e) {
                                console.warn(`Could not test camera ${i}:`, e);
                            }
                        }
                        
                        // If still not found, try the last camera (often the back camera on iPhones)
                        if (!backCameraId) {
                            backCameraId = videoDevices[videoDevices.length - 1].deviceId;
                        }
                        
                        // Also try the first camera as fallback (on some devices order might be reversed)
                        if (!backCameraId && videoDevices.length > 0) {
                            backCameraId = videoDevices[0].deviceId;
                        }
                    } else if (!backCameraId && videoDevices.length === 1) {
                        // Only one camera - might be iPad or device with single camera
                        backCameraId = videoDevices[0].deviceId;
                    }
                    
                    // Build constraints with specific camera ID or facingMode
                    let cameraConstraints = {
                        video: {
                            width: { ideal: 640 },
                            height: { ideal: 480 }
                        }
                    };
                    
                    if (backCameraId) {
                        // Use specific device ID (ultra-wide or back camera)
                        cameraConstraints.video.deviceId = { exact: backCameraId };
                        console.log('✓ Using camera with deviceId');
                    } else {
                        // Fallback: try facingMode
                        cameraConstraints.video.facingMode = { ideal: 'environment' };
                        console.log('⚠ Falling back to facingMode: environment');
                    }
                    
                    // Stop any existing camera/stream first
                    if (camera) {
                        camera.stop();
                        camera = null;
                        console.log('Stopped existing MediaPipe camera');
                    }
                    if (videoElement.srcObject) {
                        videoElement.srcObject.getTracks().forEach(track => track.stop());
                        videoElement.srcObject = null;
                        console.log('Stopped existing video stream');
                    }
                    
                    // Get the stream ourselves first to ensure we have the back camera
                    const stream = await navigator.mediaDevices.getUserMedia(cameraConstraints);
                    
                    // Verify we got the right camera
                    const videoTrack = stream.getVideoTracks()[0];
                    const settings = videoTrack.getSettings();
                    
                    if (settings.facingMode !== 'environment') {
                        console.error('❌ ERROR: Got wrong camera! facingMode:', settings.facingMode);
                        throw new Error('Failed to get back camera');
                    }
                    
                    // Assign stream to video element BEFORE MediaPipe starts
                    videoElement.srcObject = stream;
                    videoElement.style.display = 'block';
                    canvasElement.style.display = 'block';
                    realsenseStream.style.display = 'block';
                    
                    // Wait for video to be ready
                    await new Promise((resolve) => {
                        videoElement.onloadedmetadata = () => resolve();
                    });
                    
                    // IMPORTANT: Pass the video element WITH stream to MediaPipe Camera
                    // MediaPipe Camera should use the existing stream if videoElement.srcObject is set
                    // But we need to prevent MediaPipe from calling getUserMedia again
                    // Intercept getUserMedia to return our existing stream
                    const originalGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
                    let streamLocked = true;
                    
                    navigator.mediaDevices.getUserMedia = function(constraints) {
                        if (streamLocked && stream) {
                            return Promise.resolve(stream);
                        }
                        return originalGetUserMedia(constraints);
                    };
                    
                    // Start MediaPipe processing - it should use the existing stream
                    camera = new Camera(videoElement, {
                        onFrame: async () => {
                            await hands.send({image: videoElement});
                        },
                        width: 640,
                        height: 480
                    });
                    camera.start();
                    
                    // Verify after MediaPipe starts
                    setTimeout(() => {
                        if (videoElement.srcObject) {
                            const activeTrack = videoElement.srcObject.getVideoTracks()[0];
                            if (activeTrack) {
                                const finalSettings = activeTrack.getSettings();
                                
                                if (finalSettings.facingMode === 'environment') {
                                    streamLocked = false; // Allow normal operation now
                                    // Restore original getUserMedia
                                    navigator.mediaDevices.getUserMedia = originalGetUserMedia;
                                } else if (finalSettings.facingMode === 'user') {
                                    console.error('❌ ERROR: MediaPipe switched to FRONT camera!');
                                }
                            }
                        }
                    }, 1500);
                    
                    // Set canvas size
                    canvasElement.width = videoElement.videoWidth;
                    canvasElement.height = videoElement.videoHeight;
                    
                    // Start RealSense stream
                    realsenseStream.src = '/realsense_stream?' + new Date().getTime();
                    
                    isCameraActive = true;
                    btn.textContent = 'Stop Camera';
                    btn.className = 'btn-stop';
                    updateStatus(true);
                    
                } catch (error) {
                    alert('Error accessing camera: ' + error.message);
                    console.error('Camera error:', error);
                }
            } else {
                // Stop camera
                if (camera) {
                    camera.stop();
                    camera = null;
                }
                if (videoElement.srcObject) {
                    videoElement.srcObject.getTracks().forEach(track => track.stop());
                    videoElement.srcObject = null;
                }
                videoElement.style.display = 'none';
                canvasElement.style.display = 'none';
                realsenseStream.src = '';
                realsenseStream.style.display = 'none';
                
                isCameraActive = false;
                btn.textContent = 'Start Camera';
                btn.className = 'btn-start';
                updateStatus(false);
            }
        }
        
        function setMode(newMode) {
            mode = newMode;
            document.getElementById('fusedBtn').classList.toggle('active', mode === 'fused');
            document.getElementById('rawBtn').classList.toggle('active', mode === 'raw');
            document.getElementById('rawDataSection').style.display = mode === 'raw' ? 'block' : 'none';
            document.getElementById('accelDataSection').style.display = mode === 'raw' ? 'block' : 'none';
            document.getElementById('modeText').textContent = mode === 'fused' ? 'iOS sensor fusion' : 'Raw IMU data';
        }
        
        function updateStatus(connected) {
            const statusEl = document.getElementById('cameraStatus');
            if (statusEl) {
                statusEl.textContent = connected ? 'Camera: Connected ✓' : 'Camera: Disconnected';
                statusEl.className = 'status ' + (connected ? 'connected' : 'disconnected');
            }
        }
        
        function updateIMUStatus(connected) {
            const statusEl = document.getElementById('imuStatus');
            if (statusEl) {
                statusEl.textContent = connected ? 'Connected ✓' : 'Disconnected';
                statusEl.className = 'status ' + (connected ? 'connected' : 'disconnected');
            }
        }
        
        function updateRate() {
            const now = Date.now();
            const elapsed = (now - lastRateUpdate) / 1000;
            if (elapsed >= 1.0) {
                const rate = (updateCount / elapsed).toFixed(1);
                document.getElementById('updateRate').textContent = rate + ' Hz';
                updateCount = 0;
                lastRateUpdate = now;
            }
        }
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                console.log('WebSocket connected');
                updateIMUStatus(true);
            };
            
            ws.onclose = () => {
                console.log('WebSocket disconnected');
                updateIMUStatus(false);
                if (isStreaming) {
                    setTimeout(connectWebSocket, 2000);
                }
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }
        
        function handleOrientation(event) {
            const alpha = event.alpha || 0;
            const beta = event.beta || 0;
            const gamma = event.gamma || 0;
            
            document.getElementById('roll').textContent = gamma.toFixed(2) + '°';
            document.getElementById('pitch').textContent = beta.toFixed(2) + '°';
            document.getElementById('yaw').textContent = alpha.toFixed(2) + '°';
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    mode: 'fused',
                    roll: gamma,
                    pitch: beta,
                    yaw: alpha
                }));
            }
            
            updateCount++;
            updateRate();
        }
        
        function handleMotion(event) {
            const gx = event.rotationRate ? event.rotationRate.alpha * Math.PI / 180 : 0;
            const gy = event.rotationRate ? event.rotationRate.beta * Math.PI / 180 : 0;
            const gz = event.rotationRate ? event.rotationRate.gamma * Math.PI / 180 : 0;
            
            const ax = event.accelerationIncludingGravity ? event.accelerationIncludingGravity.x : 0;
            const ay = event.accelerationIncludingGravity ? event.accelerationIncludingGravity.y : 0;
            const az = event.accelerationIncludingGravity ? event.accelerationIncludingGravity.z : 0;
            
            document.getElementById('gx').textContent = gx.toFixed(3);
            document.getElementById('gy').textContent = gy.toFixed(3);
            document.getElementById('gz').textContent = gz.toFixed(3);
            document.getElementById('ax').textContent = ax.toFixed(3);
            document.getElementById('ay').textContent = ay.toFixed(3);
            document.getElementById('az').textContent = az.toFixed(3);
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    mode: 'raw',
                    gx: gx,
                    gy: gy,
                    gz: gz,
                    ax: ax,
                    ay: ay,
                    az: az,
                    interval: event.interval || 0.02
                }));
            }
            
            updateCount++;
            updateRate();
        }
        
        async function toggleStreaming() {
            const btn = document.getElementById('toggleBtn');
            
            if (!isStreaming) {
                // Request permissions
                if (typeof DeviceOrientationEvent !== 'undefined' && 
                    typeof DeviceOrientationEvent.requestPermission === 'function') {
                    try {
                        const permission = await DeviceOrientationEvent.requestPermission();
                        if (permission !== 'granted') {
                            alert('Permission denied for device orientation');
                            return;
                        }
                    } catch (error) {
                        alert('Error requesting permission: ' + error);
                        return;
                    }
                }
                
                if (typeof DeviceMotionEvent !== 'undefined' && 
                    typeof DeviceMotionEvent.requestPermission === 'function') {
                    try {
                        const permission = await DeviceMotionEvent.requestPermission();
                        if (permission !== 'granted') {
                            alert('Permission denied for device motion');
                            return;
                        }
                    } catch (error) {
                        alert('Error requesting permission: ' + error);
                        return;
                    }
                }
                
                connectWebSocket();
                
                if (mode === 'fused') {
                    window.addEventListener('deviceorientation', handleOrientation);
                } else {
                    window.addEventListener('devicemotion', handleMotion);
                }
                
                isStreaming = true;
                btn.textContent = 'Stop Streaming';
                btn.className = 'btn-stop';
                lastRateUpdate = Date.now();
                updateCount = 0;
            } else {
                window.removeEventListener('deviceorientation', handleOrientation);
                window.removeEventListener('devicemotion', handleMotion);
                if (ws) {
                    ws.close();
                }
                isStreaming = false;
                btn.textContent = 'Start Streaming';
                btn.className = 'btn-start';
                updateIMUStatus(false);
                document.getElementById('updateRate').textContent = '';
            }
        }
    </script>
</body>
</html>
"""

class RealSenseCamera:
    def __init__(self):
        self.pipeline = None
        self.config = None
        self.frame_data = None
        self.lock = asyncio.Lock()
        
        if REALSENSE_AVAILABLE:
            try:
                self.pipeline = rs.pipeline()
                self.config = rs.config()
                self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                self.pipeline.start(self.config)
                print("✓ RealSense camera initialized")
            except Exception as e:
                print(f"⚠ RealSense camera error: {e}")
                self.pipeline = None
    
    async def get_frame(self):
        """Get the latest frame from RealSense camera"""
        if not self.pipeline:
            return None
        
        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            
            frame = np.asanyarray(color_frame.get_data())
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        except Exception as e:
            print(f"RealSense frame error: {e}")
            return None
    
    def stop(self):
        if self.pipeline:
            try:
                self.pipeline.stop()
                print("✓ RealSense camera stopped")
            except:
                pass

class OrientationReceiver:
    def __init__(self):
        self.madgwick = MadgwickFilter(beta=0.1, sample_freq=50.0)
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.mode = 'fused'
        self.realsense = RealSenseCamera() if REALSENSE_AVAILABLE else None
        self.mediapipe_hands = None
        
        # Store latest iPhone camera image
        self.iphone_image = None
        self.iphone_image_time = None
        self.iphone_image_lock = asyncio.Lock()
        self.show_iphone_window = False  # Set to True to show OpenCV window
        
        # Initialize MediaPipe Hands if available
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_hands = mp.solutions.hands
                self.mp_drawing = mp.solutions.drawing_utils
                self.mediapipe_hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print("✓ MediaPipe Hands initialized")
            except Exception as e:
                print(f"⚠ MediaPipe error: {e}")
    
    async def websocket_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        print(f"\niPhone connected from {request.remote}")
        
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    self.mode = data.get('mode', 'fused')
                    
                    if self.mode == 'fused':
                        # Use iOS fused orientation
                        self.roll = data.get('roll', 0.0)
                        self.pitch = data.get('pitch', 0.0)
                        self.yaw = data.get('yaw', 0.0)
                        print(f"\r[iOS Fused] Roll: {self.roll:7.2f}° | Pitch: {self.pitch:7.2f}° | Yaw: {self.yaw:7.2f}°", end='')
                    
                    elif self.mode == 'raw':
                        # Use Madgwick filter with raw IMU data
                        gx = data.get('gx', 0.0)
                        gy = data.get('gy', 0.0)
                        gz = data.get('gz', 0.0)
                        ax = data.get('ax', 0.0)
                        ay = data.get('ay', 0.0)
                        az = data.get('az', 0.0)
                        
                        # Update sample frequency if provided
                        interval = data.get('interval', 0.02)
                        self.madgwick.sample_freq = 1.0 / interval if interval > 0 else 50.0
                        
                        # Run Madgwick filter
                        self.roll, self.pitch, self.yaw = self.madgwick.update(gx, gy, gz, ax, ay, az)
                        
                        print(f"\r[Madgwick] Roll: {self.roll:7.2f}° | Pitch: {self.pitch:7.2f}° | Yaw: {self.yaw:7.2f}°", end='')
                    
                    # Add your custom processing here
                    # You can access self.roll, self.pitch, self.yaw
                    
                except json.JSONDecodeError:
                    print("Invalid JSON received")
            elif msg.type == web.WSMsgType.ERROR:
                print(f'WebSocket error: {ws.exception()}')
        
        print("\niPhone disconnected")
        return ws
    
    async def index_handler(self, request):
        return web.Response(text=HTML_PAGE, content_type='text/html')
    
    async def pc_viewer_handler(self, request):
        """PC viewer page for iPhone camera stream"""
        pc_viewer_html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>iPhone Camera Viewer - PC</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            background: #1a1a1a;
            color: white;
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            overflow: hidden;
        }
        h1 {
            margin: 10px 0;
            font-size: 24px;
        }
        .video-container {
            border: 3px solid #667eea;
            border-radius: 10px;
            overflow: hidden;
            width: 100vw;
            height: calc(100vh - 100px);
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            background: #000;
        }
        #iphoneStream {
            display: block;
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .status {
            margin-top: 20px;
            padding: 10px 20px;
            background: #2a2a2a;
            border-radius: 5px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <h1>📱 iPhone Camera Stream (PC Viewer)</h1>
    <div class="video-container">
        <img id="iphoneStream" src="/iphone_stream" alt="iPhone Camera Stream">
    </div>
    <div class="status" id="status">Waiting for iPhone camera feed...</div>
    
    <script>
        const streamImg = document.getElementById('iphoneStream');
        const status = document.getElementById('status');
        
        streamImg.onload = () => {
            status.textContent = '✓ Receiving iPhone camera feed';
            status.style.color = '#00ff00';
        };
        
        streamImg.onerror = () => {
            status.textContent = '✗ No iPhone camera feed available. Start camera on iPhone first.';
            status.style.color = '#ff0000';
        };
        
        // Refresh status periodically
        setInterval(() => {
            if (streamImg.complete && streamImg.naturalWidth > 0) {
                status.textContent = '✓ Receiving iPhone camera feed';
                status.style.color = '#00ff00';
            }
        }, 2000);
    </script>
</body>
</html>
        """
        return web.Response(text=pc_viewer_html, content_type='text/html')
    
    async def realsense_stream_handler(self, request):
        """MJPEG stream handler for RealSense video"""
        response = web.StreamResponse()
        response.headers['Content-Type'] = 'multipart/x-mixed-replace; boundary=frame'
        await response.prepare(request)
        
        print(f"RealSense stream started for {request.remote}")
        
        while True:
            if self.realsense:
                frame = await self.realsense.get_frame()
                if frame is not None:
                    # Encode frame as JPEG
                    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 
                                            [cv2.IMWRITE_JPEG_QUALITY, 85])
                    
                    await response.write(b'--frame\r\n')
                    await response.write(b'Content-Type: image/jpeg\r\n\r\n')
                    await response.write(buffer.tobytes())
                    await response.write(b'\r\n')
            
            await asyncio.sleep(1/30)  # ~30 FPS
    
    async def receive_iphone_image_handler(self, request):
        """Receive iPhone camera image with MediaPipe hand detection"""
        try:
            data = await request.json()
            image_base64 = data.get('image', '')
            hand_landmarks = data.get('handLandmarks', [])
            
            if not image_base64:
                print("\n⚠ Received empty image data")
                return web.json_response({'status': 'error', 'message': 'No image data'})
            
            # Decode base64 image
            try:
                # Handle data URL format (data:image/jpeg;base64,...)
                if ',' in image_base64:
                    image_data = base64.b64decode(image_base64.split(',')[1])
                else:
                    image_data = base64.b64decode(image_base64)
                
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    print("\n⚠ Failed to decode image")
                    return web.json_response({'status': 'error', 'message': 'Failed to decode image'})
                
                # Store the image
                async with self.iphone_image_lock:
                    self.iphone_image = image.copy()
                    self.iphone_image_time = asyncio.get_event_loop().time()
                
                # Display in OpenCV window if enabled
                if self.show_iphone_window:
                    cv2.imshow('iPhone Camera (with Hand Detection)', image)
                    cv2.waitKey(1)
                
                # Log periodically to avoid spam
                import time
                current_time = time.time()
                if not hasattr(self, 'last_log_time') or current_time - self.last_log_time > 5:
                    print(f"\n✓ Received iPhone image: {image.shape}, Hands: {len(hand_landmarks)}")
                    self.last_log_time = current_time
                
                return web.json_response({'status': 'success', 'hands': len(hand_landmarks)})
                
            except Exception as decode_error:
                print(f"\n⚠ Error decoding image: {decode_error}")
                return web.json_response({'status': 'error', 'message': f'Decode error: {str(decode_error)}'})
            
        except Exception as e:
            import traceback
            print(f"\n❌ Error receiving iPhone image: {e}")
            print(traceback.format_exc())
            return web.json_response({'status': 'error', 'message': str(e)})
    
    async def iphone_stream_handler(self, request):
        """MJPEG stream handler for iPhone camera video"""
        response = web.StreamResponse()
        response.headers['Content-Type'] = 'multipart/x-mixed-replace; boundary=frame'
        await response.prepare(request)
        
        print(f"\niPhone camera stream started for {request.remote} (PC viewer)")
        
        # Create a placeholder image (black image with text)
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, 'Waiting for iPhone camera...', (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, placeholder_buffer = cv2.imencode('.jpg', placeholder)
        
        frame_count = 0
        no_image_count = 0
        
        while True:
            async with self.iphone_image_lock:
                image = self.iphone_image.copy() if self.iphone_image is not None else None
                image_time = self.iphone_image_time
            
            if image is not None:
                # Check if image is recent (within last 5 seconds)
                current_time = asyncio.get_event_loop().time()
                if image_time and (current_time - image_time) < 5:
                    # Encode frame as JPEG
                    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    
                    await response.write(b'--frame\r\n')
                    await response.write(b'Content-Type: image/jpeg\r\n\r\n')
                    await response.write(buffer.tobytes())
                    await response.write(b'\r\n')
                    
                    frame_count += 1
                    no_image_count = 0
                    if frame_count % 30 == 0:
                        print(f"\r✓ Streaming iPhone camera: {frame_count} frames sent", end='', flush=True)
                else:
                    # Image is too old, send placeholder
                    await response.write(b'--frame\r\n')
                    await response.write(b'Content-Type: image/jpeg\r\n\r\n')
                    await response.write(placeholder_buffer.tobytes())
                    await response.write(b'\r\n')
                    no_image_count += 1
                    if no_image_count == 1:
                        print(f"\n⚠ No recent iPhone images (last: {current_time - image_time:.1f}s ago)")
            else:
                # Send placeholder if no image yet
                await response.write(b'--frame\r\n')
                await response.write(b'Content-Type: image/jpeg\r\n\r\n')
                await response.write(placeholder_buffer.tobytes())
                await response.write(b'\r\n')
                no_image_count += 1
                if no_image_count == 30:  # Log every second if no images
                    print(f"\n⚠ Still waiting for iPhone images... (check if iPhone is sending)")
                    no_image_count = 0
            
            await asyncio.sleep(1/30)  # ~30 FPS
    
    def create_app(self):
        app = web.Application()
        app.router.add_get('/', self.index_handler)
        app.router.add_get('/pc_viewer', self.pc_viewer_handler)
        app.router.add_get('/ws', self.websocket_handler)
        app.router.add_get('/realsense_stream', self.realsense_stream_handler)
        app.router.add_get('/iphone_stream', self.iphone_stream_handler)
        app.router.add_post('/receive_image', self.receive_iphone_image_handler)
        return app
    
    def start_server(self, host='0.0.0.0', port=8443):
        if not os.path.exists('cert.pem') or not os.path.exists('key.pem'):
            print("Creating SSL certificate...")
            if not create_self_signed_cert():
                return
        
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain('cert.pem', 'key.pem')
        
        local_ip = get_local_ip()
        print("\n" + "=" * 70)
        print("XLeRobotHead - iPhone Camera & RealSense Server")
        print("=" * 70)
        print(f"✓ Server started with HTTPS support!")
        print(f"\n📱 iPhone: Open Safari and go to:")
        print(f"   https://{local_ip}:{port}\n")
        print(f"🖥️  PC Viewer: Open browser and go to:")
        print(f"   https://{local_ip}:{port}/pc_viewer\n")
        print(f"Features available:")
        print(f"  • iPhone Camera (main/back camera) with MediaPipe Hand Detection")
        print(f"  • RealSense Camera Streaming")
        print(f"  • iOS IMU/Orientation Data (Fused or Raw)")
        print(f"  • PC viewer for iPhone camera stream")
        if not REALSENSE_AVAILABLE:
            print(f"  ⚠ RealSense not available (install pyrealsense2)")
        if not MEDIAPIPE_AVAILABLE:
            print(f"  ⚠ MediaPipe not available (install mediapipe)")
        print("=" * 70)
        print("Waiting for connection...\n")
        
        app = self.create_app()
        web.run_app(app, host=host, port=port, ssl_context=ssl_context, print=None)

def main():
    receiver = OrientationReceiver()
    
    try:
        receiver.start_server()
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    finally:
        # Cleanup
        if receiver.realsense:
            receiver.realsense.stop()
        if receiver.mediapipe_hands:
            try:
                receiver.mediapipe_hands.close()
            except:
                pass

if __name__ == "__main__":
    main()