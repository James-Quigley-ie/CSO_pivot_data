FROM python:3.11.6-slim

WORKDIR /app

COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


ENV SENTENCE_TRANSFORMERS_HOME="/app/models"
RUN mkdir -p $SENTENCE_TRANSFORMERS_HOME
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2', \
               local_dir='$SENTENCE_TRANSFORMERS_HOME/all-MiniLM-L6-v2')"

COPY . .

EXPOSE 5000

CMD ["python", "server.py"]