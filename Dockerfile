FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
 && git clone https://github.com/tobymao/sqlglot /sqlglot \
 && cd /sqlglot && git checkout eb11aec4be8cb2122afe713a4816662c3f0fb2fa \
 && rm -rf .git && apt-get purge -y git && rm -rf /var/lib/apt/lists/*
COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r /app/requirements.lock
COPY selector.py /app/selector.py
COPY maps /app/maps
COPY tools/bigrun.py tools/bigchecks.py tools/runtests.py tools/plan.py /app/tools/
RUN chmod -R a+rX /app /sqlglot
ENV PYTHONPATH=/sqlglot SKIP_INTEGRATION=0 PYTHONDONTWRITEBYTECODE=1
CMD ["python", "/app/selector.py"]
