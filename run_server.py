"""Codespaces-friendly runner for the FastAPI server.

Binds to 0.0.0.0 so GitHub Codespaces port forwarding can expose it,
uses port 8000 by default (or the PORT environment variable), and
enables reload mode by default for fast iteration.

Usage:
    python run_server.py
"""

import os

import uvicorn


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_enabled = os.getenv("RELOAD", "true").lower() in ("1", "true", "yes")

    print(f"Starting CrewAI Local Agents API on http://{host}:{port}")
    print(f"Swagger docs available at http://localhost:{port}/docs")

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=reload_enabled,
    )


if __name__ == "__main__":
    main()
