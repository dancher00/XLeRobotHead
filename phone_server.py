import asyncio
import json
import socket
import ssl
import os
import cv2
import numpy as np
import threading
from aiohttp import web

class PhoneServer:
    """Main server class for iPhone IMU and RealSense camera streaming"""
    
    HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>XLeRobotHead - IMU & RealSense</title>
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
        .video-container img {
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
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 XLeRobotHead</h1>
        
            <div class="video-container">
            <div class="video-label">📹 Video Stream</div>
            <img id="realsenseStream" src="/realsense_stream" alt="Video Stream" style="width: 100%; display: none;">
        </div>
        
        <h2>📱 iPhone IMU Sender</h2>
        
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
        
        <button id="toggleBtn" class="btn-start" onclick="toggleStreaming()">
            Start Streaming
        </button>
        
        <div class="info">
            <span>iOS sensor fusion</span>
        </div>
        <div class="rate" id="updateRate"></div>
    </div>

    <script>
        // IMU WebSocket
        let ws = null;
        let isStreaming = false;
        let updateCount = 0;
        let lastRateUpdate = Date.now();
        
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
                
                connectWebSocket();
                window.addEventListener('deviceorientation', handleOrientation);
                
                // Start video stream
                const realsenseStream = document.getElementById('realsenseStream');
                if (realsenseStream) {
                    realsenseStream.src = '/realsense_stream?' + new Date().getTime();
                    realsenseStream.style.display = 'block';
                }
                
                isStreaming = true;
                btn.textContent = 'Stop Streaming';
                btn.className = 'btn-stop';
                lastRateUpdate = Date.now();
                updateCount = 0;
            } else {
                window.removeEventListener('deviceorientation', handleOrientation);
                if (ws) {
                    ws.close();
                }
                
                // Stop video stream
                const realsenseStream = document.getElementById('realsenseStream');
                if (realsenseStream) {
                    realsenseStream.src = '';
                    realsenseStream.style.display = 'none';
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
</html>"""
    
    def __init__(self):
        """Initialize the server"""
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
        # Frame storage for external frames
        self.current_frame = None
        
        # Server state
        self.server_thread = None
        self.server_running = False
        self.runner = None
    
    @staticmethod
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
    
    @staticmethod
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
            cert.get_subject().CN = PhoneServer.get_local_ip()
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
    
    def update_frame(self, frame: np.ndarray):
        """
        Update the current frame from external source.
        
        Args:
            frame: numpy array of uint8, shape (H, W, 3) in BGR format
        """
        if frame is not None and isinstance(frame, np.ndarray):
            # Make a copy to avoid issues with external modifications
            self.current_frame = frame.copy()
    
    def get_current_frame(self):
        """Get the latest frame"""
        return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_angles(self):
        """
        Get current orientation angles.
        
        Returns:
            dict: Dictionary with 'roll', 'pitch', 'yaw' in degrees
        """
        return {
            'roll': self.roll,
            'pitch': self.pitch,
            'yaw': self.yaw
        }
    
    async def websocket_handler(self, request):
        """Handle WebSocket connections for IMU data"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        print(f"\niPhone connected from {request.remote}")
        
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    
                    # Use iOS fused orientation
                    self.roll = data.get('roll', 0.0)
                    self.pitch = data.get('pitch', 0.0)
                    self.yaw = data.get('yaw', 0.0)
                    print(f"\r[iOS Fused] Roll: {self.roll:7.2f}° | Pitch: {self.pitch:7.2f}° | Yaw: {self.yaw:7.2f}°", end='')
                    
                    # Add your custom processing here
                    # You can access self.roll, self.pitch, self.yaw
                    
                except json.JSONDecodeError:
                    print("Invalid JSON received")
            elif msg.type == web.WSMsgType.ERROR:
                print(f'WebSocket error: {ws.exception()}')
        
        print("\niPhone disconnected")
        return ws
    
    async def index_handler(self, request):
        """Handle main page requests"""
        return web.Response(text=self.HTML_PAGE, content_type='text/html')
    
    async def realsense_stream_handler(self, request):
        """MJPEG stream handler for video frames"""
        response = web.StreamResponse()
        response.headers['Content-Type'] = 'multipart/x-mixed-replace; boundary=frame'
        await response.prepare(request)
        
        print(f"Video stream started for {request.remote}")
        
        # Create placeholder for when no frame is available
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, 'Waiting for frames...', (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, placeholder_buffer = cv2.imencode('.jpg', placeholder)
        
        while True:
            frame = self.get_current_frame()
            if frame is not None:
                # Frame is expected to be in BGR format (uint8)
                # Ensure it's the right format
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                await response.write(b'--frame\r\n')
                await response.write(b'Content-Type: image/jpeg\r\n\r\n')
                await response.write(buffer.tobytes())
                await response.write(b'\r\n')
            else:
                # Send placeholder if no frame available
                await response.write(b'--frame\r\n')
                await response.write(b'Content-Type: image/jpeg\r\n\r\n')
                await response.write(placeholder_buffer.tobytes())
                await response.write(b'\r\n')
            
            await asyncio.sleep(1/30)  # ~30 FPS
    
    def create_app(self):
        """Create aiohttp web application"""
        app = web.Application()
        app.router.add_get('/', self.index_handler)
        app.router.add_get('/ws', self.websocket_handler)
        app.router.add_get('/realsense_stream', self.realsense_stream_handler)
        return app
    
    def _run_server(self, host='0.0.0.0', port=8443):
        """Internal method to run server in event loop"""
        async def start():
            if not os.path.exists('cert.pem') or not os.path.exists('key.pem'):
                print("Creating SSL certificate...")
                if not self.create_self_signed_cert():
                    return
            
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain('cert.pem', 'key.pem')
            
            local_ip = self.get_local_ip()
            print("\n" + "=" * 70)
            print("XLeRobotHead - IMU & RealSense Server")
            print("=" * 70)
            print(f"✓ Server started with HTTPS support!")
            print(f"\n📱 iPhone: Open Safari and go to:")
            print(f"   https://{local_ip}:{port}\n")
            print(f"Features available:")
            print(f"  • Video Frame Streaming (external frames)")
            print(f"  • iOS IMU/Orientation Data (Fused)")
            print("=" * 70)
            print("Waiting for connection...\n")
            
            app = self.create_app()
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, host, port, ssl_context=ssl_context)
            await site.start()
            
            # Store runner for cleanup
            self.runner = runner
            
            # Keep running
            while self.server_running:
                await asyncio.sleep(1)
            
            # Cleanup when stopping
            await runner.cleanup()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(start())
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            loop.close()
    
    def run(self, host='0.0.0.0', port=8443, background=True):
        """
        Run the server.
        
        Args:
            host: Server host (default: '0.0.0.0')
            port: Server port (default: 8443)
            background: If True, run in background thread (default: True)
        """
        if self.server_running:
            print("Server is already running!")
            return
        
        self.server_running = True
        
        if background:
            # Run in background thread
            self.server_thread = threading.Thread(
                target=self._run_server,
                args=(host, port),
                daemon=True
            )
            self.server_thread.start()
            print("Server started in background thread")
        else:
            # Run in current thread (blocking)
            try:
                self._run_server(host, port)
            except KeyboardInterrupt:
                print("\n\nServer stopped by user")
            finally:
                self.server_running = False
    
    def stop(self):
        """Stop the server"""
        if not self.server_running:
            print("Server is not running!")
            return
        
        self.server_running = False
        
        # Wait for thread to finish
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=3)
        
        print("Server stopped")


def main():
    """Main function"""
    server = PhoneServer()
    server.run()


if __name__ == "__main__":
    main()
