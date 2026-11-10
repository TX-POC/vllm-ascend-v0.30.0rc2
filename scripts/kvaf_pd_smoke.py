# SPDX-License-Identifier: Apache-2.0
"""Send deterministic non-streaming requests through the native PD proxy."""

import argparse
import json
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="glm")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if min(args.rounds, args.max_tokens, args.timeout) <= 0:
        parser.error("rounds, max-tokens and timeout must be positive")
    prompt = args.prompt_file.read_text(encoding="utf-8")
    if not prompt.strip():
        parser.error("prompt-file must contain text")
    payload = json.dumps(
        dict(model=args.model, prompt=prompt, max_tokens=args.max_tokens, temperature=0, stream=False)
    ).encode()
    # Cluster endpoints must not be sent through installation proxies.
    opener = build_opener(ProxyHandler({}))
    for index in range(args.rounds):
        request = Request(
            args.base_url.rstrip("/") + "/v1/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with opener.open(request, timeout=args.timeout) as response:
            result = json.load(response)
        if not result.get("choices") or "text" not in result["choices"][0] or result.get("error"):
            raise RuntimeError(f"Invalid completion response: {result}")
        print(f"PD_HTTP_PASS round={index} usage={json.dumps(result.get('usage', {}))}")


if __name__ == "__main__":
    main()
