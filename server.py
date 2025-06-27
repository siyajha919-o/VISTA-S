#!/usr/bin/env python3
from app.routes import routes
import logging
from flask_cors import CORS
from flask import Flask
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

app.register_blueprint(routes)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Starting VISTA-S backend server on port {port}")
    logger.info(f"🌐 Frontend running on: http://localhost:8084")
    logger.info(f"🔧 API available at: http://localhost:{port}/api/detect")
    app.run(host='0.0.0.0', port=port, debug=True)
