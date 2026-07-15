# nginx front door image — serves frontend/ and proxies the API.
# Build context is the REPO ROOT (so it can COPY frontend/):
#   docker build -f deploy/nginx/Dockerfile -t nic-fraud-nginx:latest .
# The docker-compose `nginx` service sets this context automatically.
FROM nginx:1.27-alpine

# Reverse-proxy + static config (replaces the stock default server block).
COPY deploy/nginx/default.conf /etc/nginx/conf.d/default.conf

# Bake the frontend into the image so k8s needs no shared volume for static
# assets. In local dev, docker-compose.override.yml shadows this with a live
# bind mount for hot-editing.
COPY frontend/ /usr/share/nginx/html/
