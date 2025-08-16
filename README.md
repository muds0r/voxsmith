# Voxsmith

Turn conversations into insights.

This is a modular monorepo skeleton to build Voxsmith sustainably.

## Structure
- `apps/api` – Python FastAPI service for analysis orchestration.
- `apps/frontend` – Web app (Next.js) to be added later.
- `packages/config` – Shared static configs (e.g., profiles).
- `packages/schemas` – Shared JSON Schemas & types.
- `infra/local` – Local dev (docker-compose).
- `.github/workflows` – CI (to be added later).

## Quick Start
1. Create a virtualenv and install `apps/api` requirements.
2. Run the API locally: `uvicorn app.main:app --reload --port 8080`.
3. Later, add the frontend (Next.js) under `apps/frontend`.

## Conventions
- Python: Black + Ruff, type hints, FastAPI.
- JS/TS: TypeScript, ESLint + Prettier (when frontend is added).
- Config-first: JSON/YAML in `packages/config` consumed by services.
