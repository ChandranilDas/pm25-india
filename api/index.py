from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        response_data = {
            "status": "online",
            "service": "PM2.5 India Spatial Super-Resolution Platform API",
            "frameworks": [
                "Physics-Informed ML (PINN)",
                "AdvectSR-PM",
                "ConvNeXt-Advect (Model.h5)",
                "HistGradientBoosting",
                "Random Forest"
            ],
            "benchmark_champion": "PINN R² = 0.941, MAE = 4.12 µg/m³",
            "resolution": "0.25° × 0.25° (~25 km) High-Resolution Spatial Grid",
            "dashboard_url": "/"
        }
        self.wfile.write(json.dumps(response_data, indent=2).encode('utf-8'))
