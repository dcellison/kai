# Prefix for venv binaries. Override with `make BIN= test` in CI
# where tools are installed globally (no .venv).
BIN = .venv/bin/

.PHONY: run lint format check typecheck test setup config install install-status tts-model refresh-models

run:
	$(BIN)python -m kai

lint:
	$(BIN)ruff check .

format:
	$(BIN)ruff format .

check: lint
	$(BIN)ruff format --check .

typecheck:
	$(BIN)pyright --pythonpath "$$($(BIN)python -c 'import sys; print(sys.executable)')"

test:
	$(BIN)python -m pytest tests/ -v

# Development environment setup (editable install with dev tools)
setup:
	$(BIN)pip install -e '.[dev]'

# Protected installation targets
config:
	$(BIN)python -m kai install config

# Any non-empty DRY_RUN becomes an explicit CLI flag inside the root
# process. Do not rely on sudo environment propagation for this safety gate.
install:
	sudo $(BIN)python -m kai install apply $(if $(strip $(DRY_RUN)),--dry-run,)

install-status:
	$(BIN)python -m kai install status

# Refresh helper for PROVIDER_MODELS. Queries each curated provider's
# /v1/models endpoint and prints a diff against the in-tree list;
# operator hand-edits src/kai/config.py after review. Pass
# `ARGS=--write-snippet` to emit a paste-able Python fragment.
refresh-models:
	$(BIN)python -m kai.refresh_models $(ARGS)

models/ggml-base.en.bin:
	mkdir -p models
	curl -L -o $@ https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

PIPER_URL = https://huggingface.co/rhasspy/piper-voices/resolve/main

tts-model:
	mkdir -p models/piper
	curl -L -o models/piper/en_GB-cori-medium.onnx      $(PIPER_URL)/en/en_GB/cori/medium/en_GB-cori-medium.onnx
	curl -L -o models/piper/en_GB-cori-medium.onnx.json  $(PIPER_URL)/en/en_GB/cori/medium/en_GB-cori-medium.onnx.json
	curl -L -o models/piper/en_GB-alba-medium.onnx       $(PIPER_URL)/en/en_GB/alba/medium/en_GB-alba-medium.onnx
	curl -L -o models/piper/en_GB-alba-medium.onnx.json  $(PIPER_URL)/en/en_GB/alba/medium/en_GB-alba-medium.onnx.json
	curl -L -o models/piper/en_GB-jenny_dioco-medium.onnx      $(PIPER_URL)/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx
	curl -L -o models/piper/en_GB-jenny_dioco-medium.onnx.json  $(PIPER_URL)/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx.json
	curl -L -o models/piper/en_GB-alan-medium.onnx       $(PIPER_URL)/en/en_GB/alan/medium/en_GB-alan-medium.onnx
	curl -L -o models/piper/en_GB-alan-medium.onnx.json  $(PIPER_URL)/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json
	curl -L -o models/piper/en_US-amy-medium.onnx        $(PIPER_URL)/en/en_US/amy/medium/en_US-amy-medium.onnx
	curl -L -o models/piper/en_US-amy-medium.onnx.json   $(PIPER_URL)/en/en_US/amy/medium/en_US-amy-medium.onnx.json
	curl -L -o models/piper/en_US-lessac-medium.onnx     $(PIPER_URL)/en/en_US/lessac/medium/en_US-lessac-medium.onnx
	curl -L -o models/piper/en_US-lessac-medium.onnx.json $(PIPER_URL)/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
	curl -L -o models/piper/en_US-ryan-medium.onnx       $(PIPER_URL)/en/en_US/ryan/medium/en_US-ryan-medium.onnx
	curl -L -o models/piper/en_US-ryan-medium.onnx.json  $(PIPER_URL)/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json
	curl -L -o models/piper/en_US-joe-medium.onnx        $(PIPER_URL)/en/en_US/joe/medium/en_US-joe-medium.onnx
	curl -L -o models/piper/en_US-joe-medium.onnx.json   $(PIPER_URL)/en/en_US/joe/medium/en_US-joe-medium.onnx.json
