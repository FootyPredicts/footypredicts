"""
Loads settings such as API_FOOTBALL_KEY from a local `.env` file in this folder.

The .env file stays on your computer only - it is listed in .gitignore, so git
never uploads it. On GitHub, the key comes from a repository secret instead.

.env format (one per line):
    API_FOOTBALL_KEY=your-key-here
"""
import os

_ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_ENV):
    with open(_ENV, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
