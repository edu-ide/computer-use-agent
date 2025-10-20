.PHONY: sync setup install dev-backend dev-frontend dev clean

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

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	cd cua2-front && rm -rf node_modules dist 2>/dev/null || true
	@echo "✓ Cleaned!"
