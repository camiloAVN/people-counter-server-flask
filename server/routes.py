"""Registro de rutas Flask y función de construcción de estadísticas.

Rutas registradas por ``register_routes()``:
    GET  /                  → Sirve el dashboard (index.html)
    GET  /video_feed        → Stream MJPEG de la cámara en vivo
    GET  /api/stats         → Estadísticas actuales en JSON
    POST /api/reset         → Reinicia todos los contadores
    GET  /api/export/csv    → Descarga el informe de conteo como CSV
"""

import logging
import os
import tempfile
import time
from datetime import datetime

from flask import Response, after_this_request, jsonify, request, send_file, send_from_directory

logger = logging.getLogger("ContadorPersonas")


def register_routes(app, counter, camera) -> None:
    """Registra todas las rutas HTTP en la aplicación Flask.

    Se usa el patrón de closures para que cada handler tenga acceso
    directo a ``counter`` y ``camera`` sin depender del contexto de request.

    Args:
        app:     Instancia de ``Flask``.
        counter: Motor de detección compartido.
        camera:  Captura de cámara compartida.
    """

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    @app.route("/")
    def index():
        """Sirve el dashboard web principal."""
        return send_from_directory(app.static_folder, "index.html")

    # ------------------------------------------------------------------
    # Stream MJPEG
    # ------------------------------------------------------------------

    def _mjpeg_generator():
        """Generador síncrono que produce frames MJPEG continuamente."""
        while True:
            frame = camera.get_jpeg_frame()
            if frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
            time.sleep(1 / 30)   # tope de 30 fps en el stream

    @app.route("/video_feed")
    def video_feed():
        """Devuelve el video de la cámara como stream MJPEG continuo.

        Compatible con la etiqueta ``<img src="/video_feed">`` en cualquier
        navegador moderno.
        """
        return Response(
            _mjpeg_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/snapshot")
    def snapshot():
        """Devuelve el frame actual de la cámara como imagen JPEG única."""
        frame = camera.get_jpeg_frame()
        if frame is None:
            return Response("Sin frame disponible", status=503)
        return Response(
            frame,
            mimetype="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    # ------------------------------------------------------------------
    # Estadísticas
    # ------------------------------------------------------------------

    @app.route("/api/stats")
    def get_stats():
        """Retorna las métricas de conteo actuales en formato JSON."""
        return jsonify(build_stats(counter))

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    @app.route("/api/config", methods=["POST"])
    def update_config():
        """Reconfigura el modo de detección y sus parámetros en tiempo real."""
        data = request.get_json(force=True, silent=True) or {}

        if "counting_mode" in data:
            mode = data["counting_mode"]
            if mode in ("line", "roi", "fov"):
                counter.set_mode(mode)

        if "confidence" in data:
            try:
                counter.update_confidence(float(data["confidence"]))
            except (TypeError, ValueError):
                pass

        # Líneas
        line_kwargs = {}
        if "line_position" in data:
            try:
                line_kwargs["position"] = float(data["line_position"])
            except (TypeError, ValueError):
                pass
        if "line_position_vertical" in data:
            try:
                line_kwargs["position_vertical"] = float(data["line_position_vertical"])
            except (TypeError, ValueError):
                pass
        if line_kwargs:
            counter.update_line(**line_kwargs)

        h_enabled = data.get("use_horizontal_line")
        v_enabled = data.get("use_vertical_line")
        if h_enabled is not None or v_enabled is not None:
            counter.set_line_enabled(
                horizontal=bool(h_enabled) if h_enabled is not None else None,
                vertical=bool(v_enabled) if v_enabled is not None else None,
            )

        # ROI
        roi_keys = ("roi_x1", "roi_y1", "roi_x2", "roi_y2")
        if all(k in data for k in roi_keys):
            try:
                counter.set_roi(
                    float(data["roi_x1"]), float(data["roi_y1"]),
                    float(data["roi_x2"]), float(data["roi_y2"]),
                )
            except (TypeError, ValueError):
                pass

        logger.info("Configuración actualizada vía API: %s", data)
        return jsonify({"status": "ok"})

    @app.route("/api/reset", methods=["POST"])
    def reset_counters():
        """Pone todos los contadores a cero y reinicia el tracker."""
        counter.reset_counters()
        logger.info("Contadores reiniciados vía API.")
        return jsonify({"status": "ok", "message": "Contadores reiniciados."})

    # ------------------------------------------------------------------
    # Exportación CSV
    # ------------------------------------------------------------------

    @app.route("/api/export/csv")
    def export_csv():
        """Genera y descarga el informe de conteo como archivo CSV.

        El archivo temporal se elimina automáticamente tras el envío.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filename = f"conteo_{timestamp}.csv"

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
        )
        tmp.close()
        counter.export_csv(tmp.name)
        logger.info("CSV exportado vía API: %s", filename)

        @after_this_request
        def _cleanup(response):
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            return response

        return send_file(
            tmp.name,
            as_attachment=True,
            download_name=filename,
            mimetype="text/csv",
        )


# ---------------------------------------------------------------------------
# Función de utilidad compartida con el broadcaster WebSocket
# ---------------------------------------------------------------------------

def build_stats(counter) -> dict:
    """Construye el diccionario de estadísticas a partir del estado del contador.

    Este dict es serializado a JSON tanto para el endpoint ``/api/stats``
    como para el broadcast periódico por WebSocket.
    """
    hourly: dict = counter.hourly_entries

    peak_hour: int | None = max(hourly, key=hourly.get) if hourly else None
    peak_count: int = hourly[peak_hour] if peak_hour is not None else 0

    total_hours = len(hourly)
    hourly_avg = round(sum(hourly.values()) / total_hours, 1) if total_hours > 0 else 0.0

    total_today = (
        counter.fov_count
        if counter.counting_mode == "fov"
        else counter.in_count
    )

    return {
        "in_count":                counter.in_count,
        "out_count":               counter.out_count,
        "fov_count":               counter.fov_count,
        "persons_in_frame":        counter.persons_in_frame,
        "net_flow":                counter.in_count - counter.out_count,
        "fps":                     round(counter.fps, 1),
        "counting_mode":           counter.counting_mode,
        "hourly_entries":          {str(k): v for k, v in hourly.items()},
        "peak_hour":               peak_hour,
        "peak_count":              peak_count,
        "hourly_average":          hourly_avg,
        "total_today":             total_today,
        # Estado de configuración (para sincronizar el panel de control)
        "line_position":           counter.line_position,
        "line_position_vertical":  counter.line_position_vertical,
        "use_horizontal_line":     counter.use_horizontal_line,
        "use_vertical_line":       counter.use_vertical_line,
        "roi_x1":                  counter.roi_x1,
        "roi_y1":                  counter.roi_y1,
        "roi_x2":                  counter.roi_x2,
        "roi_y2":                  counter.roi_y2,
        "confidence":              counter.confidence,
    }
