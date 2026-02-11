.PHONY: run lint format check test install tts-model

run:
	.venv/bin/python -m kai

lint:
	.venv/bin/ruff check .

format:
	.venv/bin/ruff format .

check: lint
	.venv/bin/ruff format --check .

test:
	.venv/bin/python -m pytest tests/ -v

install:
	.venv/bin/pip install -e '.[dev]'

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
