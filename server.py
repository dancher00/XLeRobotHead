import asyncio
import json
import socket
import ssl
import os
import math
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
    <title>iPhone IMU Sender</title>
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
    </style>
</head>
<body>
    <div class="container">
        <h1>📱 iPhone IMU Sender</h1>
        
        <div class="mode-selector">
            <button class="mode-btn active" onclick="setMode('fused')" id="fusedBtn">
                iOS Fused
            </button>
            <button class="mode-btn" onclick="setMode('raw')" id="rawBtn">
                Raw IMU
            </button>
        </div>
        
        <div id="status" class="status disconnected">
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

    <script>
        let ws = null;
        let isStreaming = false;
        let mode = 'fused';  // 'fused' or 'raw'
        let updateCount = 0;
        let lastRateUpdate = Date.now();
        
        function setMode(newMode) {
            mode = newMode;
            document.getElementById('fusedBtn').classList.toggle('active', mode === 'fused');
            document.getElementById('rawBtn').classList.toggle('active', mode === 'raw');
            document.getElementById('rawDataSection').style.display = mode === 'raw' ? 'block' : 'none';
            document.getElementById('accelDataSection').style.display = mode === 'raw' ? 'block' : 'none';
            document.getElementById('modeText').textContent = mode === 'fused' ? 'iOS sensor fusion' : 'Raw IMU data';
        }
        
        function updateStatus(connected) {
            const statusEl = document.getElementById('status');
            statusEl.textContent = connected ? 'Connected ✓' : 'Disconnected';
            statusEl.className = 'status ' + (connected ? 'connected' : 'disconnected');
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
                updateStatus(true);
            };
            
            ws.onclose = () => {
                console.log('WebSocket disconnected');
                updateStatus(false);
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
                updateStatus(false);
                document.getElementById('updateRate').textContent = '';
            }
        }
    </script>
</body>
</html>
"""

class OrientationReceiver:
    def __init__(self):
        self.madgwick = MadgwickFilter(beta=0.1, sample_freq=50.0)
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.mode = 'fused'
    
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
    
    def create_app(self):
        app = web.Application()
        app.router.add_get('/', self.index_handler)
        app.router.add_get('/ws', self.websocket_handler)
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
        print("iPhone IMU Receiver Server with Madgwick Filter")
        print("=" * 70)
        print(f"✓ Server started with HTTPS and Madgwick filter support!")
        print(f"\n📱 Open Safari on your iPhone and go to:")
        print(f"\n   https://{local_ip}:{port}\n")
        print(f"Two modes available:")
        print(f"  • iOS Fused: Use iPhone's built-in sensor fusion")
        print(f"  • Raw IMU: Get raw data + Madgwick filter on PC")
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

if __name__ == "__main__":
    main()