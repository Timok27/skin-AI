import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import torchvision.transforms as transforms
import uvicorn
from azure.data.tables import TableServiceClient, TableEntity
import logging
from typing import Dict
import torch
import torch.nn as nn
from torchvision.models import googlenet, GoogLeNet_Weights
import asyncio
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

# Инициализация приложения
app = FastAPI()

# Настройка Azure Tables для кэширования
#connection_string = os.getenv("TABLE_CONNECTION_STRING")
#if not connection_string:
#    raise ValueError("Необходимо задать переменную окружения TABLE_CONNECTION_STRING")

#table_service = TableServiceClient.from_connection_string(conn_str=connection_string)
#table_client = table_service.get_table_client("SkinCache")

#try:
#    table_client.create_table()
#    logging.info("Таблица SkinCache создана.")
#except Exception as e:
#    logging.warning(f"Таблица уже существует или возникла ошибка: {str(e)}")

#Инициализация ThreadPoolExecutor для параллельной обработки изображений
executor = ThreadPoolExecutor(max_workers=4)

class GoogleNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(GoogleNetClassifier, self).__init__()
        self.googlenet = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
        self.googlenet.fc = nn.Linear(self.googlenet.fc.in_features, num_classes)

    def forward(self, x):
        return self.googlenet(x)

# Логирование
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ],
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.getcwd(), "googlenet_classifier.pth")
num_classes = 5
model = GoogleNetClassifier(num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

async def preprocess_image(file: bytes) -> torch.Tensor:
    """Обрабатывает изображение для анализа асинхронно."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _preprocess_image, file)

def _preprocess_image(file: bytes) -> torch.Tensor:
    """Вспомогательная синхронная функция для обработки изображения."""
    try:
        np_image = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            raise HTTPException(status_code=400, detail="На изображении не обнаружено лицо. Попробуйте другое фото.")
        (x, y, w, h) = faces[0]
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return transform(face).unsqueeze(0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка обработки изображения: {str(e)}")

async def analyze_skin_tone(image: torch.Tensor):
    """Анализирует тон кожи с использованием модели асинхронно."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _analyze_skin_tone, image)

def map_skin_tone_to_features(fitzpatrick_label: int) -> Dict[str, str]:
    """Маппинг тона кожи на его характеристики."""
    tone_mapping = {0: "cool", 1: "warm", 2: "warm", 3: "cool", 4: "cool"}
    undertone_mapping = {0: "cool", 1: "warm", 2: "warm", 3: "olive", 4: "neutral"}
    season_mapping = {0: "summer", 1: "spring", 2: "autumn", 3: "winter", 4: "winter"}
    skin_type_mapping = {
        0: "sensitive",
        1: "normal",
        2: "combination",
        3: "oily",
        4: "oily"
    }

    return {
        "tone": tone_mapping[fitzpatrick_label],
        "undertone": undertone_mapping[fitzpatrick_label],
        "season": season_mapping[fitzpatrick_label],
        "skin_type": skin_type_mapping[fitzpatrick_label]
    }

def _analyze_skin_tone(image: torch.Tensor):
    """Вспомогательная синхронная функция для анализа тона кожи."""
    try:
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            tone_class = torch.argmax(probabilities).item()
            confidence = probabilities[tone_class].item()
            return {
                "class": tone_class,
                "confidence": confidence
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа тона кожи: {str(e)}")

#async def get_from_cache(key: str):
#    """Пытаемся получить результат из Azure Tables."""
#    try:
#        entity = table_client.get_entity(partition_key="cache", row_key=key)
#        return eval(entity["value"])
#    except:
#        return None

#async def set_to_cache(key: str, value: Dict):
    """Кэшируем результат в Azure Tables."""
    entity = TableEntity(
        partition_key="cache",
        row_key=key,
        value=str(value)
    )
    table_client.upsert_entity(entity)

# Модели данных
class StatusResponse(BaseModel):
    status: str

class AnalysisResult(BaseModel):
    class_id: int
    confidence: float
    features: Dict[str, str]

# API endpoints
@app.post("/analyze", response_model=AnalysisResult)
async def analyze_image(file: UploadFile = File(..., max_size=5 * 1024 * 1024)):
    """Загружает изображение и возвращает анализ с кэшированием."""
    try:
        contents = await file.read()
        cache_key = f"image_{hash(contents)}"
        #cached_result = await get_from_cache(cache_key)
        #if cached_result:
            #return JSONResponse(content=cached_result)

        image = await preprocess_image(contents)
        result = await analyze_skin_tone(image)
        features = map_skin_tone_to_features(result["class"])
        response = {
            "class_id": result["class"],
            "confidence": result["confidence"],
            "features": features
        }

        #await set_to_cache(cache_key, response)
        logger.info("Успешный анализ изображения")
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"Ошибка анализа: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")

@app.get("/status", response_model=StatusResponse)
def get_status():
    """Возвращает статус сервера."""
    return StatusResponse(status="Сервер работает")

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000)