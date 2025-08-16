.PHONY: api dev

api:
	cd apps/api && uvicorn app.main:app --reload --port 8080

dev:
	echo "Add frontend dev after Next.js is initialized"
