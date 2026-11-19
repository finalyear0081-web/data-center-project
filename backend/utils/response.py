"""
Standardized JSON API Response Helpers
"""
from flask import jsonify

def success_response(data=None, message="Operation successful", status_code=200, **extra):
    payload = {
        "success": True,
        "message": message,
        "data": data,
        **extra
    }
    return jsonify(payload), status_code

def error_response(message="Operation failed", errors=None, status_code=400, **extra):
    payload = {
        "success": False,
        "error": message,
        "details": errors,
        **extra
    }
    return jsonify(payload), status_code
