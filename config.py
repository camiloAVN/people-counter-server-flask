"""Configuración global y constantes de la aplicación.

Este módulo centraliza todos los parámetros para evitar valores mágicos
dispersos en el código. Modifica este archivo para ajustar el comportamiento
del sistema sin tocar la lógica principal.
"""

# ---------------------------------------------------------------------------
# Detección (YOLO)
# ---------------------------------------------------------------------------
PERSON_CLASS_ID = 0

DEFAULT_MODEL = "yolov8s.pt"
AVAILABLE_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]

DEFAULT_CONFIDENCE = 0.3
DEFAULT_INFER_SIZE = 640

# Dispositivo de inferencia: 0 = CUDA GPU 0, "cpu" = solo CPU
INFERENCE_DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------
AREA_THRESHOLD = 0.30           # Porcentaje mínimo del bbox para registrar cruce
TRACKER_RESET_INTERVAL_S = 30 * 60  # Reset automático del tracker (30 min)

# ---------------------------------------------------------------------------
# Líneas de conteo (posición relativa 0.0–1.0 del frame)
# ---------------------------------------------------------------------------
DEFAULT_LINE_POSITION_H = 0.5
DEFAULT_LINE_POSITION_V = 0.5

# ROI por defecto (x1, y1, x2, y2) en coordenadas normalizadas
DEFAULT_ROI = (0.0, 0.0, 1.0, 1.0)

# ---------------------------------------------------------------------------
# Estadísticas
# ---------------------------------------------------------------------------
INTERVAL_SECONDS = 15 * 60     # Período de agregación de datos (15 min)
AUTOSAVE_INTERVAL_S = 5 * 60   # Auto-guardado periódico (5 min)

# ---------------------------------------------------------------------------
# Cámara
# ---------------------------------------------------------------------------
# Opciones para CAMERA_SOURCE:
#   Entero (0, 1, 2…)         → webcam USB/V4L2 por índice
#   "rtsp://user:pass@ip/…"   → stream RTSP
#   String GStreamer           → pipeline personalizado (Jetson CSI, etc.)
CAMERA_SOURCE = 0

# Resolución solicitada a la cámara (puede ser ignorada por el driver)
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS    = 30

# Pipeline GStreamer para cámara CSI en Jetson Orin Nano.
# Descomenta y ajusta si usas una cámara CSI en lugar de USB.
# CAMERA_SOURCE = (
#     "nvarguscamerasrc ! "
#     "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
#     "nvvidconv flip-method=0 ! "
#     "video/x-raw,format=BGRx ! videoconvert ! appsink"
# )

# Calidad de compresión JPEG para el stream de video (0–100)
MJPEG_QUALITY = 80

# ---------------------------------------------------------------------------
# Servidor web
# ---------------------------------------------------------------------------
SERVER_HOST = "localhost"        # Escuchar en todas las interfaces
SERVER_PORT = 8000

# Intervalo de broadcast de estadísticas por WebSocket (segundos)
WS_BROADCAST_INTERVAL = 1.0