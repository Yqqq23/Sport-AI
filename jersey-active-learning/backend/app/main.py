import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings
from app.db.session import Base, engine, ensure_sqlite_schema
from app.services.storage import ensure_data_dirs


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

Base.metadata.create_all(bind=engine)
ensure_sqlite_schema()
ensure_data_dirs()

app = FastAPI(
    title="Jersey Active Learning API",
    description="用于篮球视频中球衣号码识别结果复查、人工修正和数据集导出的后端 API。普通使用者通常只需要按顺序使用：上传视频 → 运行识别 → 查看候选帧 → 保存标注 → 下载数据集。",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static/frames", StaticFiles(directory=str((settings.data_dir / "frames").resolve())), name="frames")

app.include_router(router)


@app.get(
    "/health",
    summary="健康检查",
    description="确认后端服务是否正常运行。",
    tags=["系统检查"],
)
def health_check() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}
