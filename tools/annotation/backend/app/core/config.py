import json
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="jersey-active-learning", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    database_url: str = Field(default="sqlite:///./jersey_active_learning.db", alias="DATABASE_URL")
    cors_origins_raw: str = Field(default="http://127.0.0.1:5173,http://localhost:5173", alias="CORS_ORIGINS")
    data_dir: Path = Field(default=Path("../data"), alias="DATA_DIR")
    real_recognition_enabled: bool = Field(default=True, alias="REAL_RECOGNITION_ENABLED")
    sport_cv_root: Path | None = Field(default=None, alias="SPORT_CV_ROOT")
    player_detector_weights: str = Field(default="", alias="PLAYER_DETECTOR_WEIGHTS")
    jersey_ocr_weights: str = Field(default="", alias="JERSEY_OCR_WEIGHTS")
    detector_confidence: float = Field(default=0.2, alias="DETECTOR_CONFIDENCE")
    detector_iou: float = Field(default=0.45, alias="DETECTOR_IOU")
    recognition_frame_stride: int = Field(default=5, alias="RECOGNITION_FRAME_STRIDE")

    @property
    def cors_origins(self) -> list[str]:
        return self._parse_cors(self.cors_origins_raw)

    @staticmethod
    def _parse_cors(value: str) -> list[str]:
        if not value:
            return ["http://127.0.0.1:5173", "http://localhost:5173"]
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if item]
        except (json.JSONDecodeError, TypeError):
            pass
        return [item.strip() for item in value.split(",") if item.strip()]

    @field_validator("sport_cv_root", mode="before")
    @classmethod
    def parse_sport_cv_root(cls, value):
        if value in (None, "", "null"):
            return None
        if isinstance(value, str):
            return Path(value)
        return value


settings = Settings()
