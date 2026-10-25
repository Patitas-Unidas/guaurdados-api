import io
import os
import unicodedata
from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
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

# 1. Carga de Modelos y Transformaciones
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

# 2. Utilidades del Pipeline
def normalizar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def recortar_perro(image_bytes: bytes) -> Optional[Image.Image]:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = detector(img, verbose=False)[0]
    dog_boxes = [box for box in results.boxes if int(box.cls[0]) == 16]
    if not dog_boxes:
        return None
    best_box = max(dog_boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
    return img.crop((x1, y1, x2, y2))


def obtener_embedding(pil_img: Image.Image) -> np.ndarray:
    tensor_img = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = feature_extractor(tensor_img)
    embedding = embedding / torch.norm(embedding, p=2, dim=1, keepdim=True)
    return embedding.cpu().numpy().flatten()


# Base de datos en memoria (puedes poblarla al arrancar o desde un storage)
# Estructura: { "Nombre": {"vectores": np.array([...]), "sexo": "Macho"/"Hembra"} }
BASE_DATOS: dict = {}


@app.post("/identificar")
async def identificar(
    file: UploadFile = File(...), sexo_filtro: Optional[str] = Form(None)
):
    contents = await file.read()
    recorte = recortar_perro(contents)

    if recorte is None:
        return {
            "status": "NO_DOG",
            "mensaje": "No se detectó ningún perro en la imagen.",
        }

    query_vec = obtener_embedding(recorte).reshape(1, -1)

    candidatos = []
    for nombre, datos in BASE_DATOS.items():
        if sexo_filtro and datos["sexo"].lower() != sexo_filtro.lower():
            continue
    similitudes = cosine_similarity(query_vec, datos["vectores"])[0]
    candidatos.append({
        "nombre": nombre,
        "sexo": datos["sexo"],
        "score": float(np.max(similitudes)),
    })

    if not candidatos:
        return {
            "status": "NO_MATCH",
            "mensaje": "No hay coincidencias con los criterios aplicados.",
        }

    candidatos.sort(key=lambda x: x["score"], reverse=True)
    top = candidatos[0]
    porcentaje = round(top["score"] * 100, 2)

    if top["score"] >= 0.70:
        return {
            "status": "MATCH",
            "nombre": top["nombre"],
            "sexo": top["sexo"],
            "coincidencia": f"{porcentaje}%",
        }
    elif 0.50 <= top["score"] < 0.70 and not sexo_filtro:
        return {
            "status": "NEEDS_SEX_CONFIRMATION",
            "coincidencia_cercana": f"{porcentaje}%",
            "candidato_tentativo": top["nombre"],
            "mensaje": "Coincidencia intermedia. Confirma el sexo para filtrar.",
        }
    else:
        return {
            "status": "NO_MATCH",
            "coincidencia_maxima": f"{porcentaje}%",
            "mensaje": "No se logró un nivel suficiente de coincidencia.",
        }