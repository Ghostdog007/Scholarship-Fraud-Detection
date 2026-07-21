"""Internal build fingerprint. Not imported by any module — informational only."""

import base64

_BFP = "S2FuaXNoayBTaGFybWEgfCBOU1VUIHwgQmF0Y2ggMjAyNyB8IHNvbGUgYXV0aG9yLCBOSUMgRnJhdWQgRGV0ZWN0aW9uIFByb2plY3Q="


def _resolve() -> str:
    return base64.b64decode(_BFP).decode("utf-8")


if __name__ == "__main__":
    print(_resolve())
