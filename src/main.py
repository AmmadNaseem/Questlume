"""Phase 4 CLI. Run from the project root: python -m src.main."""

import argparse
import sys

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import ValidationError
from pydantic_settings import SettingsError

from src.chains.smoke_test import build_smoke_test_chain
from src.core.config import Settings
from src.core.llm_factory import AllProvidersUnavailableError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 4 LCEL smoke test.")
    parser.add_argument("--topic", required=True, help="A short technical topic.")
    parser.add_argument(
        "--offline", action="store_true",
        help="Use a fixed fake response; no credentials or network required.",
    )
    args = parser.parse_args(argv)

    try:
        if args.offline:
            # Explicit overrides isolate offline mode from real .env credentials.
            settings = Settings(
                _env_file=None,
                llm_provider="groq",
                llm_fallback_providers=(),
                groq_model="offline-test-model",
                groq_api_key="offline-test-key",
            )
            model = FakeListChatModel(
                responses=["Offline smoke test succeeded. This fixed response verifies "
                           "LCEL wiring; it is not a generated explanation."]
            )
            chain = build_smoke_test_chain(settings, llm=model)
        else:
            chain = build_smoke_test_chain()
        result = chain.invoke({"topic": args.topic})
    except AllProvidersUnavailableError:
        print("All providers are unavailable. Retry later.", file=sys.stderr)
        return 2
    except (ValidationError, SettingsError):
        print("Invalid configuration. Check .env keys, model IDs, and limits.", file=sys.stderr)
        return 2
    except ValueError:
        print("Invalid topic or provider configuration.", file=sys.stderr)
        return 2
    except ImportError:
        print("Missing dependency. Install requirements.txt into .venv.", file=sys.stderr)
        return 2
    except Exception:
        # SDK errors may contain request data; keep raw payloads out of CLI output.
        print("Model request failed. Check provider authentication and model compatibility.", file=sys.stderr)
        return 1

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
