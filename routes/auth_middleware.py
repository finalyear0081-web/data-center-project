"""
Authentication and Role-Based Access Control Middleware
"""
from functools import wraps
from flask import request, jsonify, session
from services.firebase_service import get_firebase_service

def require_auth(admin_only: bool = False):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            fb = get_firebase_service()
            user = None
            
            # 1. Check Authorization Bearer header
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                user = fb.verify_token(token)
                
            # 2. Check Flask Session
            if not user and "user" in session:
                user = session.get("user")
                
            if not user:
                return jsonify({"error": "Unauthorized", "message": "Valid authentication token or session required."}), 401
                
            if admin_only and user.get("role") != "admin":
                return jsonify({"error": "Forbidden", "message": "Administrator privileges required for this action."}), 403
                
            request.user = user
            return f(*args, **kwargs)
        return decorated_function
    return decorator
