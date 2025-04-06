.PHONY: install run

install:
	pip install -r requirements.txt

run:
	uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
