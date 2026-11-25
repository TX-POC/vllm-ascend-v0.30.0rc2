# SPDX-License-Identifier: Apache-2.0
"""Launch the PD proxy with optional request-phase capture."""

import importlib.util
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path


def load_proxy(*, capture=True):
    path = (
        Path(__file__).resolve().parents[1]
        / "examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py"
    )
    spec = importlib.util.spec_from_file_location("kvaf_observed_proxy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not capture:
        return module
    native_lifespan = module.app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        async with native_lifespan(app):
            observer = None
            try:
                from kvaf_pd_performance import load_config

                config = load_config()
                if config.active:
                    from kvaf_pd_proxy_observer import ProxyObserver

                    observer = ProxyObserver(config)
                    module.proxy_state.request_observer = observer
            except Exception as error:
                logging.getLogger(__name__).warning("Proxy capture disabled; native routing retained: %s", error)
            try:
                yield
            finally:
                if observer is not None:
                    observer.capture.close()

    module.app.router.lifespan_context = lifespan
    return module


def main():
    import uvicorn

    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        os.environ.pop(name, None)
    argv = sys.argv[1:]
    capture = argv[:1] != ["--no-capture"]  # Used by the plain serving launcher.
    proxy = load_proxy(capture=capture)
    proxy.global_args = proxy.parse_args(argv if capture else argv[1:])
    uvicorn.run(proxy.app, host=proxy.global_args.host, port=proxy.global_args.port)


if __name__ == "__main__":
    main()
