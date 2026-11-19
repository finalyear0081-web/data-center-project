"""
Root Entry Point for Production Servers (Gunicorn / Render / Local)
"""
from backend.app import app

if __name__ == "__main__":
    from backend.config import Config
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
