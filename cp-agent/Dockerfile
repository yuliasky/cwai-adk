# cp-agent-prototype container image

FROM python:3.12-slim

WORKDIR /app

# Install the SDK from the bundled wheel.
COPY cwai_adk-2.0.0rc2.dev0-py3-none-any.whl ./
RUN pip install --no-cache-dir cwai_adk-2.0.0rc2.dev0-py3-none-any.whl && rm -f cwai_adk-2.0.0rc2.dev0-py3-none-any.whl

# Agent source.
COPY cp-agent/main.py cp-agent/README.md ./

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]