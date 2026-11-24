"""NeuroEngine: REST API that classifies brain MRI images into four Alzheimer's stages.

Academic research project. It is not a medical device and must not be used for diagnosis.
"""

from __future__ import annotations

from flask import Flask, jsonify, request
from PIL import Image

from neuroengine.config import Settings
from neuroengine.images import ImageInputError, image_from_upload, image_from_url
from neuroengine.predictor import Predictor

__version__ = "1.0.0"


def create_app(settings: Settings | None = None, predictor: Predictor | None = None) -> Flask:
    """Application factory. Pass a `predictor` to inject a fake one in tests."""
    settings = settings or Settings()
    if predictor is None:
        from neuroengine.predictor import KerasPredictor

        predictor = KerasPredictor(settings.model_path)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = settings.max_image_bytes

    def read_image() -> Image.Image:
        """Multipart file `image`, or form field `image` with an HTTPS URL from an allowed host."""
        if "image" in request.files:
            return image_from_upload(request.files["image"], settings.max_image_bytes)
        if request.form.get("image"):
            return image_from_url(
                request.form["image"], settings.allowed_image_hosts, settings.max_image_bytes, settings.download_timeout
            )
        raise ImageInputError("Send an image file or an image URL in the `image` field.")

    @app.errorhandler(ImageInputError)
    def bad_image(exc: ImageInputError):
        return jsonify(error=str(exc)), 400

    @app.get("/")
    @app.get("/health")
    def health():
        return jsonify(status="ok", service="neuroengine", version=__version__)

    @app.post("/api")
    def classify():
        """Most likely class. Response shape kept compatible with the NeuroImage Insight app."""
        top = predictor.predict(read_image())[0]
        return jsonify({top.label: f"{top.probability:.2f}%"})

    @app.post("/api/v2/classify")
    def classify_all():
        """Every class with its probability, sorted from most to least likely."""
        predictions = predictor.predict(read_image())
        return jsonify(predictions=[{"label": p.label, "probability": round(p.probability, 2)} for p in predictions])

    return app
