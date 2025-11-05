.PHONY: sync setup install dev-backend dev-frontend dev clean docker-build docker-run docker-stop docker-clean docker-logs

# Sync all dependencies (Python + Node.js)
sync:
	@echo "Syncing Python dependencies..."
	uv sync --all-extras
	@echo "Installing frontend dependencies..."
	cd cua2-front && npm install
	@echo "✓ All dependencies synced!"

setup: sync

install-frontend:
	cd cua2-front && npm install

# Start backend development server
dev-backend:
	cd cua2-core && uv run uvicorn cua2_core.main:app --reload --host 0.0.0.0 --port 8000

# Start frontend development server
dev-frontend:
	cd cua2-front && npm run dev

pre-commit:
	uv run pre-commit run --all-files --show-diff-on-failure
	make test

# Run tests
test:
	cd cua2-core && uv run pytest tests/ -v

test-coverage:
	cd cua2-core && uv run pytest tests/ -v --cov=cua2_core --cov-report=html --cov-report=term

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	cd cua2-front && rm -rf node_modules dist 2>/dev/null || true
	@echo "✓ Cleaned!"

# Docker commands
docker-build:
	@echo "Building Docker image..."
	make docker-stop
	docker build -t cua2:latest .
	@echo "✓ Docker image built successfully!"

docker-run:
	@echo "Starting CUA2 container..."
	@if [ -z "$$E2B_API_KEY" ]; then \
		echo "Error: E2B_API_KEY environment variable is not set"; \
		echo "Please set it with: export E2B_API_KEY=your-key"; \
		exit 1; \
	fi
	@if [ -z "$$HF_TOKEN" ]; then \
		echo "Error: HF_TOKEN environment variable is not set"; \
		echo "Please set it with: export HF_TOKEN=your-token"; \
		exit 1; \
	fi
	docker run -d --name cua2-app -p 7860:7860 \
		-e E2B_API_KEY="$$E2B_API_KEY" \
		-e HF_TOKEN="$$HF_TOKEN" \
		cua2:latest
	@echo "✓ Container started! Access at http://localhost:7860"

docker-stop:
	@echo "Stopping CUA2 container..."
	docker stop cua2-app || true
	docker rm cua2-app || true
	@echo "✓ Container stopped!"

docker-clean:
	@echo "Removing CUA2 Docker images..."
	docker rmi cua2:latest || true
	@echo "✓ Docker images removed!"

docker-logs:
	docker logs -f cua2-app
