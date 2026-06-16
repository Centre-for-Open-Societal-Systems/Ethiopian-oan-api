#!/usr/bin/env bash
# Create an ISOLATED virtualenv for the RAGAS retrieval eval (tests/rag_ragas_eval.py).
#
# Why isolated: ragas 0.2.x pulls langchain-openai, which pins `openai<2`, but the app's
# pydantic-ai needs `openai>=2`. Installing ragas into the app environment breaks the app.
# This venv keeps ragas + its langchain/openai-1.x stack completely separate; the eval
# script never imports app modules — it only calls the app over HTTP and reads the CSV.
#
# Run inside the app container (the venv lives in the container FS, not the repo):
#   docker exec oan_app bash /app/tests/setup_ragas_venv.sh
#
# It is ephemeral — re-run this after a container rebuild (`docker compose up -d app`).
set -euo pipefail

VENV="${RAGAS_VENV:-/opt/ragas-venv}"

echo "Creating ragas venv at ${VENV} ..."
python3 -m venv "${VENV}"
"${VENV}/bin/pip" install --quiet --upgrade pip

# Pin the langchain stack to <0.4: ragas 0.2.15 imports
# langchain_community.chat_models.vertexai, a path removed in langchain 1.x.
"${VENV}/bin/pip" install --quiet \
    "ragas==0.2.15" \
    "langchain<0.4" \
    "langchain-core<0.4" \
    "langchain-community<0.4" \
    "langchain-openai<0.3"

echo -n "Done. "
"${VENV}/bin/python" -c "import ragas; print('ragas', ragas.__version__, 'ready at ${VENV}')"
