# Criterio de evaluación: 
# Calculamos la diferencia: {diferencia} = {score}_1 - {score}_2
# Match claro (30$ de ventaja): Se declara ganador absoluto (status: "MATCH").
# Duda entre dos (10% < {diferencia} < 30%): Retorna ambos candidatos (status: "AMBIGUOUS").
# Diferencia < 10% o similitud base muy baja (< 35%): Se descarta por falta de certeza (status: "NO_MATCH").

import io
import os
import json
import pickle
import traceback
import unicodedata
from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO

app = FastAPI(title="GUAUrdados Dog Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detector = YOLO("yolov8n.pt")

transform = transforms.Compose([
    transforms.Resize((236, 236)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
])

feature_extractor = models.convnext_small(
    weights=models.ConvNeXt_Small_Weights.DEFAULT
)
feature_extractor.classifier[2] = torch.nn.Identity()
feature_extractor = feature_extractor.to(device).eval()

def normalizar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")

def corregir_orientacion(img: Image.Image) -> Image.Image:
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img

def recortar_perro_desde_pil(img: Image.Image) -> Optional[Image.Image]:
    img = corregir_orientacion(img).convert("RGB")
    results = detector(img, verbose=False)[0]
    dog_boxes = [box for box in results.boxes if int(box.cls[0]) == 16]
    
    if not dog_boxes:
        return None
        
    best_box = max(dog_boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
    
    # Añadir un 10% de margen para no cortar orejas/hocico
    ancho, alto = img.size
    dx = int((x2 - x1) * 0.1)
    dy = int((y2 - y1) * 0.1)
    
    nx1 = max(0, x1 - dx)
    ny1 = max(0, y1 - dy)
    nx2 = min(ancho, x2 + dx)
    ny2 = min(alto, y2 + dy)
    
    return img.crop((nx1, ny1, nx2, ny2))

def obtener_embedding(pil_img: Image.Image) -> np.ndarray:
    pil_img = corregir_orientacion(pil_img).convert("RGB")
    tensor_img = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = feature_extractor(tensor_img)
    embedding = embedding / torch.norm(embedding, p=2, dim=1, keepdim=True)
    return embedding.cpu().numpy().flatten()

BASE_DATOS: dict = {}
CACHE_EMBEDDINGS = "embeddings.pkl"
DB_DIR = "db_perros"
METADATA_FILE = "metadatos.json"

@app.on_event("startup")
def inicializar_base_datos():
    global BASE_DATOS
    
    if os.path.exists(CACHE_EMBEDDINGS):
        print(f"Cargando embeddings desde {CACHE_EMBEDDINGS}...")
        with open(CACHE_EMBEDDINGS, "rb") as f:
            BASE_DATOS = pickle.load(f)
        print(f"Base de datos cargada: {len(BASE_DATOS)} perros en memoria.")
        return

    if not os.path.exists(DB_DIR):
        print(f"AVISO: No se encontró la carpeta {DB_DIR}.")
        return

    metadatos = {}
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadatos = json.load(f)

    print("Indexando fotos de db_perros/ con corrección de orientación...")
    
    for perro_folder in os.listdir(DB_DIR):
        perro_path = os.path.join(DB_DIR, perro_folder)
        if not os.path.isdir(perro_path):
            continue

        vectores_perro = []
        for archivo in os.listdir(perro_path):
            if not archivo.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            
            foto_path = os.path.join(perro_path, archivo)
            try:
                img = Image.open(foto_path)
                recorte = recortar_perro_desde_pil(img)
                recorte_final = recorte if recorte is not None else corregir_orientacion(img).convert("RGB")
                emb = obtener_embedding(recorte_final)
                vectores_perro.append(emb)
            except Exception as e:
                print(f"Error procesando {foto_path}: {e}")

        if vectores_perro:
            # Busca ignorando si la carpeta tiene mayúsculas o minúsculas
            clave_perro = perro_folder.lower().strip()
            info_meta = metadatos.get(clave_perro, {})
            nombre_mostrar = info_meta.get("nombre_mostrado", perro_folder.capitalize())
            sexo = info_meta.get("sexo", "Desconocido")
            
            BASE_DATOS[perro_folder] = {
                "nombre": nombre_mostrar,
                "sexo": sexo,
                "vectores": np.array(vectores_perro)
            }
            print(f"  -> {nombre_mostrar} ({sexo}): {len(vectores_perro)} fotos indexadas.")

    with open(CACHE_EMBEDDINGS, "wb") as f:
        pickle.dump(BASE_DATOS, f)
    print(f"Indexación completada. Guardado en {CACHE_EMBEDDINGS}.")


@app.get("/")
def home():
    return {"status": "ok", "perros_registrados": len(BASE_DATOS)}


@app.post("/identificar")
async def identificar(
    file: UploadFile = File(...),
    sexo_filtro: Optional[str] = Form(None)
):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        img = corregir_orientacion(img).convert("RGB")
        recorte = recortar_perro_desde_pil(img)

        img_eval = recorte if recorte is not None else img
        query_vec = obtener_embedding(img_eval).reshape(1, -1)

        if not BASE_DATOS:
            return {
                "status": "NO_MATCH",
                "mensaje": "No hay perros registrados en la base de datos."
            }

        candidatos = []
        for dog_id, datos in BASE_DATOS.items():
            if sexo_filtro and normalizar_texto(datos["sexo"]) != normalizar_texto(sexo_filtro):
                continue

            similitudes = cosine_similarity(query_vec, datos["vectores"])[0]
            max_score = float(np.max(similitudes))
            
            candidatos.append({
                "id": dog_id,
                "nombre": datos["nombre"],
                "sexo": datos["sexo"],
                "score": max_score
            })

        if not candidatos:
            return {
                "status": "NO_MATCH",
                "mensaje": "No hay perros que coincidan con el criterio seleccionado."
            }

        # Ordenar de mayor a menor coincidencia
        candidatos.sort(key=lambda x: x["score"], reverse=True)
        top1 = candidatos[0]
        score1 = top1["score"]

        # Si el mejor score es menor al 35%, la imagen ni siquiera se parece a los registros
        if score1 < 0.35:
            return {
                "status": "NO_MATCH",
                "mensaje": "No se encontró coincidencia con los perros registrados."
            }

        # Caso donde solo hay 1 candidato disponible
        if len(candidatos) == 1:
            return {
                "status": "MATCH",
                "perro": {
                    "id": top1["id"],
                    "nombre": top1["nombre"],
                    "sexo": top1["sexo"],
                    "coincidencia": f"{round(score1 * 100, 1)}%"
                }
            }

        top2 = candidatos[1]
        score2 = top2["score"]
        diferencia = (score1 - score2) * 100  # Margen en puntos porcentuales

        print(f"Top 1: {top1['nombre']} ({round(score1*100, 1)}%) vs Top 2: {top2['nombre']} ({round(score2*100, 1)}%) | Margen: {round(diferencia, 1)}%")

        # 1. Match definitivo si le saca 30% o más de ventaja al segundo
        if diferencia >= 30.0:
            return {
                "status": "MATCH",
                "perro": {
                    "id": top1["id"],
                    "nombre": top1["nombre"],
                    "sexo": top1["sexo"],
                    "coincidencia": f"{round(score1 * 100, 1)}%"
                }
            }

        # 2. Duda razonable si la diferencia está entre 10% y 30%
        if 10.0 <= diferencia < 30.0:
            return {
                "status": "AMBIGUOUS",
                "candidato_1": {
                    "id": top1["id"],
                    "nombre": top1["nombre"],
                    "sexo": top1["sexo"],
                    "coincidencia": f"{round(score1 * 100, 1)}%"
                },
                "candidato_2": {
                    "id": top2["id"],
                    "nombre": top2["nombre"],
                    "sexo": top2["sexo"],
                    "coincidencia": f"{round(score2 * 100, 1)}%"
                }
            }

        # 3. Diferencia menor al 10% (empate técnico sin certeza suficiente)
        return {
            "status": "NO_MATCH",
            "mensaje": f"No hay certeza suficiente entre {top1['nombre']} ({round(score1*100, 1)}%) y {top2['nombre']} ({round(score2*100, 1)}%)."
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "ERROR", "mensaje": f"Error interno: {str(e)}"}
        )