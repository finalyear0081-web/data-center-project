"""
Authentication Routes Blueprint
"""
from flask import Blueprint, request, jsonify, session
from services.firebase_service import get_firebase_service

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")

@auth_bp.route("/verify", methods=["POST"])
def verify_token_endpoint():
    data = request.get_json(silent=True) or {}
    token = data.get("id_token") or request.headers.get("Authorization", "").replace("Bearer ", "")
    
    if not token:
        return jsonify({"success": False, "error": "Token is required."}), 400
        
    fb = get_firebase_service()
    user = fb.verify_token(token)
    if not user:
        return jsonify({"success": False, "error": "Invalid or expired token."}), 401
        
    session["user"] = user
    return jsonify({"success": True, "user": user})

@auth_bp.route("/login", methods=["POST"])
def login_endpoint():
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip()
    role = data.get("role") or ("admin" if "admin" in email else "operator")
    
    if not email:
        return jsonify({"success": False, "error": "Email is required."}), 400
        
    fb = get_firebase_service()
    existing = fb.get_user(email)
    if existing:
        user = existing
    else:
        user = {
            "uid": f"user-{len(email)}",
            "email": email,
            "role": role,
            "display_name": email.split("@")[0].capitalize()
        }
        fb.save_record("users", user, doc_id=email)
        
    session["user"] = user
    return jsonify({"success": True, "user": user, "token": f"demo-token-{user['role']}"})

@auth_bp.route("/register", methods=["POST"])
def register_endpoint():
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip()
    name = data.get("name", "").strip()
    role = data.get("role", "operator")
    
    if not email:
        return jsonify({"success": False, "error": "Email is required."}), 400
        
    fb = get_firebase_service()
    user = {
        "uid": f"user-{abs(hash(email)) % 100000}",
        "email": email,
        "role": role if role in ["admin", "operator"] else "operator",
        "display_name": name or email.split("@")[0]
    }
    fb.save_record("users", user, doc_id=email)
    session["user"] = user
    return jsonify({"success": True, "user": user, "token": f"demo-token-{user['role']}"})

@auth_bp.route("/logout", methods=["POST"])
def logout_endpoint():
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully."})

@auth_bp.route("/me", methods=["GET"])
def get_current_user():
    user = session.get("user")
    if not user:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            fb = get_firebase_service()
            user = fb.verify_token(token)
            
    if not user:
        return jsonify({"authenticated": False}), 200
        
    return jsonify({"authenticated": True, "user": user})
