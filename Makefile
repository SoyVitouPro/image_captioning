.PHONY: create_env install train run

IMAGE_CAPTION_ENV = /home/vitoupro/code/image_captioning/image_env
PROJECT_DIR = /home/vitoupro/code/image_captioning

create_env:
	@echo "Creating Python virtual environment..."
	python3 -m venv $(IMAGE_CAPTION_ENV)

install:
	@echo "Installing dependencies..."
	$(IMAGE_CAPTION_ENV)/bin/pip install -r $(PROJECT_DIR)/requirements.txt

train:
	@echo "Running the training script..."
	. $(IMAGE_CAPTION_ENV)/bin/activate; cd $(PROJECT_DIR); python3 -m trainer.final.train_model

run:
	@echo "Running the application..."
	. $(IMAGE_CAPTION_ENV)/bin/activate; cd $(PROJECT_DIR); uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
