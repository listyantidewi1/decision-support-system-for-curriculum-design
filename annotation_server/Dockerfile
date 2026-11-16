# Collector image. Mount a PERSISTENT volume at /data — the whole point of this
# service is that the annotations survive a container restart.
FROM python:3.12-slim
WORKDIR /app
COPY annotation_server/requirements.txt /app/annotation_server/requirements.txt
RUN pip install --no-cache-dir -r /app/annotation_server/requirements.txt
COPY annotation_server /app/annotation_server
# Bake the sample sentences in. DATA/gold_standard is gitignored, so the
# Streamlit UI deployed from GitHub has no sentence data; the collector
# serves them instead, gated by the same per-annotator token. These are the
# small CSVs (~50 KB), never the 128 MB sentence pool.
COPY DATA/gold_standard/*.csv /app/samples/
ENV GS_DB_PATH=/data/annotations.sqlite
ENV GS_SAMPLES_DIR=/app/samples
VOLUME ["/data"]
EXPOSE 8000
CMD ["uvicorn", "annotation_server.api:app", "--host", "0.0.0.0", "--port", "8000"]
