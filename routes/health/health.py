from flask import jsonify

def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Backend is running'})
