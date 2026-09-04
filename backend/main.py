import logging
import os
import socket
from contextlib import asynccontextmanager

from fastapi import FastAPI
from routes import health

logger = logging.getLogger("roadvision")


def check_tcp_connection(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    chroma_host = os.getenv("CHROMA_HOST", "chromadb")
    chroma_port = int(os.getenv("CHROMA_PORT", 8000))

    app.state.service_status = {
        "chromadb": "ok" if check_tcp_connection(chroma_host, chroma_port) else "unreachable"
    }

    for name, status in app.state.service_status.items():
        (logger.info if status == "ok" else logger.warning)(f"{name}: {status}")

    yield


app = FastAPI(lifespan=lifespan)
app.include_router(health.router)
