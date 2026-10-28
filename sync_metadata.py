import json
import os
import unicodedata
import firebase_admin
from firebase_admin import credentials, firestore

CREDENTIALS_FILE = "firebase_credentials.json"
METADATA_FILE = "metadatos.json"

def normalizar_texto(texto: str) -> str:
    if not texto:
        return ""
    texto = unicodedata.normalize("NFD", str(texto).lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn").strip()

def sincronizar_metadatos_desde_firebase():
    if not os.path.exists(CREDENTIALS_FILE):
        print(f"Error: No se encontró el archivo de credenciales {CREDENTIALS_FILE}")
        return

    if not firebase_admin._apps:
        cred = credentials.Certificate(CREDENTIALS_FILE)
        firebase_admin.initialize_app(cred)

    db = firestore.client()
    perros_ref = db.collection("perros")
    docs = list(perros_ref.stream())

    print(f"\n--- SINCRONIZANDO FIRESTORE ({len(docs)} documentos encontrados) ---")
    nuevos_metadatos = {}

    for doc in docs:
        data = doc.to_dict() or {}
        doc_id = doc.id

        # Soporta 'nombre', 'Nombre', 'Name', etc.
        nombre = data.get("nombre") or data.get("Nombre") or ""
        
        # Si no tiene nombre definido, muestra aviso en consola para diagnosticar
        if not nombre:
            print(f"[!] Documento {doc_id} descartado porque no tiene campo 'nombre'. Campos presentes: {list(data.keys())}")
            continue

        clave = normalizar_texto(nombre)
        sexo = data.get("sexo") or data.get("Sexo") or "Desconocido"
        
        # Busca imagen_asset, fotoUrl o imagen
        foto_url = (
            data.get("imagen_asset") 
            or data.get("fotoUrl") 
            or data.get("foto") 
            or ""
        )
        
        raza = data.get("raza") or data.get("Raza") or "Mestizo"
        edad = data.get("edad") or data.get("Edad") or ""
        subtitulo = f"{raza} · {edad}".strip(" ·")

        nuevos_metadatos[clave] = {
            "id_firestore": doc_id,
            "nombre_mostrado": str(nombre).strip(),
            "sexo": str(sexo).strip(),
            "foto": foto_url,
            "subtitulo": subtitulo if subtitulo else "Perro registrado"
        }
        print(f"  [+] Sincronizado: {nombre} (clave: '{clave}') -> ID: {doc_id}")

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(nuevos_metadatos, f, ensure_ascii=False, indent=2)

    print(f"¡Sincronización exitosa! Se guardaron {len(nuevos_metadatos)} perros en {METADATA_FILE}\n")

if __name__ == "__main__":
    sincronizar_metadatos_desde_firebase()