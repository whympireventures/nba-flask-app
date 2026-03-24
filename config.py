import os
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


_load_dotenv()


def _default_season_start_year() -> int:
    today = datetime.now()
    return today.year if today.month >= 10 else today.year - 1


@dataclass(frozen=True)
class Settings:
    rapidapi_key: str = os.getenv("RAPIDAPI_KEY", "")
    rapidapi_host: str = os.getenv("RAPIDAPI_HOST", "api-nba-v1.p.rapidapi.com")
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "15"))
    season_start_year: int = int(os.getenv("NBA_SEASON_START_YEAR", str(_default_season_start_year())))
    model_dir: Path = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent))
    tracking_file: Path = Path(os.getenv("TRACKING_FILE", Path(__file__).resolve().parent / "data" / "prediction_tracking.csv"))
    prizepicks_provider: str = os.getenv("PRIZEPICKS_PROVIDER", "prop_professor")
    prizepicks_api_base: str = os.getenv("PRIZEPICKS_API_BASE", "https://api.prizepicks.com")
    prizepicks_nba_league_id: str = os.getenv("PRIZEPICKS_NBA_LEAGUE_ID", "7")
    prizepicks_ncaab_league_id: str = os.getenv("PRIZEPICKS_NCAAB_LEAGUE_ID", "33")
    prop_professor_api_key: str = os.getenv("PROP_PROFESSOR_API_KEY", "")
    prop_professor_base_url: str = os.getenv("PROP_PROFESSOR_BASE_URL", "https://api.propprofessor.com")
    underdog_api_base: str = os.getenv("UNDERDOG_API_BASE", "https://api.underdogfantasy.com")
    underdog_search_path: str = os.getenv("UNDERDOG_SEARCH_PATH", "/v2/pickem_search/search_results")
    underdog_lines_path: str = os.getenv("UNDERDOG_LINES_PATH", "/v1/lobbies/content/lines")
    underdog_product: str = os.getenv("UNDERDOG_PRODUCT", "fantasy")
    underdog_product_experience_id: str = os.getenv("UNDERDOG_PRODUCT_EXPERIENCE_ID", "018e1234-5678-9abc-def0-123456789002")
    underdog_sport_id: str = os.getenv("UNDERDOG_SPORT_ID", "NBA")
    underdog_ncaab_sport_id: str = os.getenv("UNDERDOG_NCAAB_SPORT_ID", "NCAAB")
    underdog_state_config_id: str = os.getenv("UNDERDOG_STATE_CONFIG_ID", "60267d4a-6aeb-4ccc-b5c5-344b9a5466db")
    underdog_include_live: bool = os.getenv("UNDERDOG_INCLUDE_LIVE", "true").lower() == "true"
    underdog_show_mass_option_markets: bool = os.getenv("UNDERDOG_SHOW_MASS_OPTION_MARKETS", "false").lower() == "true"
    underdog_user_agent: str = os.getenv("UNDERDOG_USER_AGENT", "Mozilla/5.0")
    parlayplay_api_base: str = os.getenv("PARLAYPLAY_API_BASE", "https://parlayplay.io")
    parlayplay_search_path: str = os.getenv("PARLAYPLAY_SEARCH_PATH", "/api/v1/crossgame/search/")
    parlayplay_period: str = os.getenv("PARLAYPLAY_PERIOD", "FG")
    parlayplay_origin: str = os.getenv("PARLAYPLAY_ORIGIN", "https://parlayplay.io")
    parlayplay_referer: str = os.getenv("PARLAYPLAY_REFERER", "https://parlayplay.io/")
    parlayplay_user_agent: str = os.getenv("PARLAYPLAY_USER_AGENT", "Whympire-NBA-Sports-Predictor/1.0")
    parlayplay_accept_language: str = os.getenv("PARLAYPLAY_ACCEPT_LANGUAGE", "en-US,en;q=0.9")
    parlayplay_cookie: str = os.getenv("PARLAYPLAY_COOKIE", "")
    odds_api_key: str = os.getenv("ODDS_API_KEY", "")
    flask_debug: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    flask_port: int = int(os.getenv("PORT", "5001"))


settings = Settings()
