import os
import warnings

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 1. Silencia avisos do TensorFlow e logs C++ (absl / oneDNN)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "2"

# 3. (Opcional) Oculta a barra de progresso do download ("Fetching 12 files...")
# os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# 4. Silencia os UserWarnings do Python (como o do pyfaidx)
warnings.filterwarnings("ignore", category=UserWarning, module="pyfaidx")

from fastapi import FastAPI  # noqa: E402
from src.router import router  # noqa: E402
from src.config import settings  # noqa: E402

app = FastAPI(
    title="AlphaGenome API",
    description="FastAPI wrapper for AlphaGenome — genomic variant effect prediction",
    version="0.2.0",
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "name": "AlphaGenome API",
        "version": "0.2.0",
        "docs": "/docs",
        "health": "/v1/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL,
    )
