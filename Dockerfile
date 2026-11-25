# Current submission image (contest_harness workspace.v2).
# Build from repo root:  docker build --network none -t zhixun-harness:sdk14 -f Dockerfile .
# History: Dockerfile.sdk5 / Dockerfile.sdk7 are archived, do NOT use.
FROM python:3.11-slim

COPY vendor/wheels/ /wheels/
RUN pip install --no-cache-dir --no-index --find-links=/wheels numpy==1.26.4 \
    && rm -rf /wheels

COPY harness_sdk/app/ /app/

WORKDIR /app

# The platform calls exactly these three verbs.
ENTRYPOINT ["python", "/app/harness.py"]
CMD ["describe"]
