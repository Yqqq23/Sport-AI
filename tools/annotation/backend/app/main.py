import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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


# --- 托管前端静态文件 -------------------------------------------------
#
# main.py 位于 tools/annotation/backend/app/，前端构建产物在
# tools/annotation/frontend/dist/（`npm run build` 之后生成）。
#
# 这段必须放在文件最后：下面的 "/{full_path:path}" 是兜底路由，会匹配
# 任何未被上面 API 路由（router、/health、/static/frames 挂载）命中的
# GET 请求。FastAPI/Starlette 按注册顺序匹配路由，只要它排在最后，就
# 不会抢在真正的 API 路由前面；同名路径永远优先命中前面注册的 API。
#
# 行为：
#   - 请求路径在 dist 目录下能找到对应文件（如 /assets/index-xxx.js）
#     → 直接返回该文件；
#   - 否则（包括 /videos、/basketball-classes 这类前端路由路径）
#     → 回退返回 index.html，交给前端路由处理（SPA fallback）。
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"

if FRONTEND_INDEX.is_file():

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_frontend(full_path: str) -> FileResponse:
        requested = (FRONTEND_DIST / full_path).resolve()
        if requested.is_file() and FRONTEND_DIST in requested.parents:
            return FileResponse(requested)
        return FileResponse(FRONTEND_INDEX)

else:
    logging.getLogger(__name__).warning(
        "未找到前端构建产物：%s。请先在 tools/annotation/frontend 下执行 `npm run build`，"
        "否则除 API 外的路径都不会返回页面。",
        FRONTEND_DIST,
    )
