"""WSGI entry point: `gunicorn wsgi:app` or `flask --app wsgi run`."""

from dotenv import load_dotenv

load_dotenv()

from neuroengine import create_app  # noqa: E402

app = create_app()
