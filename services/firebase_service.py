"""
Firebase Authentication & Firestore Service Module
Provides token verification, role-based access control, and Firestore document persistence.
Includes local fallback for offline development and testing.
"""
import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

# Check for Firebase Admin SDK availability
FIREBASE_AVAILABLE = False
try:
    import firebase_admin
    from firebase_admin import credentials, auth, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

LOCAL_DATA_DIR = "services/local_storage"

class FirestoreLocalFallback:
    """Local JSON-backed database mimicking Firestore collections for offline execution."""
    def __init__(self, storage_dir: str = LOCAL_DATA_DIR):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        
    def _get_coll_path(self, collection_name: str) -> str:
        return os.path.join(self.storage_dir, f"{collection_name}.json")
        
    def _read_collection(self, collection_name: str) -> Dict[str, Any]:
        p = self._get_coll_path(collection_name)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
        
    def _write_collection(self, collection_name: str, data: Dict[str, Any]):
        p = self._get_coll_path(collection_name)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def set_document(self, collection_name: str, doc_id: str, data: Dict[str, Any]):
        coll = self._read_collection(collection_name)
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        coll[doc_id] = data
        self._write_collection(collection_name, coll)

    def add_document(self, collection_name: str, data: Dict[str, Any]) -> str:
        import uuid
        doc_id = str(uuid.uuid4())[:12]
        data["id"] = doc_id
        data["created_at"] = datetime.now(timezone.utc).isoformat()
        self.set_document(collection_name, doc_id, data)
        return doc_id

    def get_document(self, collection_name: str, doc_id: str) -> Optional[Dict[str, Any]]:
        coll = self._read_collection(collection_name)
        return coll.get(doc_id)

    def list_documents(self, collection_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        coll = self._read_collection(collection_name)
        docs = list(coll.values())
        return docs[-limit:]

class FirebaseService:
    def __init__(self):
        self.app = None
        self.db = None
        self.is_connected = False
        self.local_db = FirestoreLocalFallback()
        self._init_firebase()
        
    def _init_firebase(self):
        cert_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY")
        if FIREBASE_AVAILABLE and cert_path and os.path.exists(cert_path):
            try:
                cred = credentials.Certificate(cert_path)
                self.app = firebase_admin.initialize_app(cred)
                self.db = firestore.client()
                self.is_connected = True
                print("Firebase Admin SDK successfully connected.")
            except Exception as e:
                print(f"Firebase initialization notice: {e}. Running in local storage mode.")
        else:
            # Seed default admin user in local store if not present
            if not self.local_db.get_document("users", "admin@datacenter.internal"):
                self.local_db.set_document("users", "admin@datacenter.internal", {
                    "uid": "admin-uid-001",
                    "email": "admin@datacenter.internal",
                    "role": "admin",
                    "display_name": "Chief Operations Engineer"
                })
            if not self.local_db.get_document("users", "operator@datacenter.internal"):
                self.local_db.set_document("users", "operator@datacenter.internal", {
                    "uid": "operator-uid-002",
                    "email": "operator@datacenter.internal",
                    "role": "operator",
                    "display_name": "Data Center Technician"
                })

    def verify_token(self, id_token: str) -> Optional[Dict[str, Any]]:
        """
        Verifies Firebase ID token or authenticates local development token.
        """
        if not id_token:
            return None
            
        # Support developer bypass / demo token
        if id_token.startswith("demo-token-"):
            role = "admin" if "admin" in id_token else "operator"
            email = "admin@datacenter.internal" if role == "admin" else "operator@datacenter.internal"
            return {
                "uid": f"demo-{role}-uid",
                "email": email,
                "role": role,
                "name": "Demo " + role.capitalize()
            }
            
        if self.is_connected:
            try:
                decoded = auth.verify_id_token(id_token)
                email = decoded.get("email", "")
                user_record = self.get_user(email)
                role = user_record.get("role", "operator") if user_record else "operator"
                decoded["role"] = role
                return decoded
            except Exception as e:
                print(f"Token verification error: {e}")
                return None
        return None

    def get_user(self, email: str) -> Optional[Dict[str, Any]]:
        if self.is_connected and self.db:
            try:
                doc = self.db.collection("users").document(email).get()
                return doc.to_dict() if doc.exists else None
            except Exception:
                pass
        return self.local_db.get_document("users", email)

    def save_record(self, collection: str, data: Dict[str, Any], doc_id: Optional[str] = None) -> str:
        """Saves a document to Firestore or local fallback."""
        if self.is_connected and self.db:
            try:
                coll_ref = self.db.collection(collection)
                if doc_id:
                    coll_ref.document(doc_id).set(data)
                    return doc_id
                else:
                    _, doc_ref = coll_ref.add(data)
                    return doc_ref.id
            except Exception as e:
                print(f"Firestore write error: {e}. Writing to local fallback.")
                
        if doc_id:
            self.local_db.set_document(collection, doc_id, data)
            return doc_id
        return self.local_db.add_document(collection, data)

    def get_records(self, collection: str, limit: int = 50) -> List[Dict[str, Any]]:
        if self.is_connected and self.db:
            try:
                docs = self.db.collection(collection).limit(limit).stream()
                return [d.to_dict() for d in docs]
            except Exception:
                pass
        return self.local_db.list_documents(collection, limit=limit)

_firebase_service = None

def get_firebase_service() -> FirebaseService:
    global _firebase_service
    if _firebase_service is None:
        _firebase_service = FirebaseService()
    return _firebase_service
