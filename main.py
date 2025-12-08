# backend.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from routes.image.route import generate_image
from routes.health.health import health_check

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Image URLs
app.add_url_rule('/api/generate/image', 'generate_image', generate_image, methods=['POST'])

# Health Check
app.add_url_rule('/health', 'health_check', health_check, methods=['GET'])


if __name__ == '__main__':
    # Run on port 5000 (or whatever you want)
    app.run(host='0.0.0.0', port=5000, debug=True)