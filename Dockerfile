# =============================================================================
# Email Agent - Production Dockerfile
# Multi-stage build for optimized container with comprehensive configuration
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies and compile
# -----------------------------------------------------------------------------
FROM python:3.9-slim as builder

# Set build-time environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better cache utilization
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Production - Minimal runtime image
# -----------------------------------------------------------------------------
FROM python:3.9-slim as production

# Labels for container metadata
LABEL maintainer="Pompeii3 Engineering Team" \
      version="1.0.0" \
      description="Email Agent - AI-powered customer service automation" \
      org.opencontainers.image.source="https://github.com/pompeii3/email-agent"

# Runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    # Application settings
    APP_HOME=/app \
    APP_USER=emailagent \
    APP_GROUP=emailagent \
    # Default port for Cloud Run
    PORT=8080 \
    # Python path
    PATH="/opt/venv/bin:$PATH" \
    # Timezone
    TZ=UTC

# Set working directory
WORKDIR ${APP_HOME}

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN groupadd --gid 1000 ${APP_GROUP} && \
    useradd --uid 1000 --gid ${APP_GROUP} --shell /bin/bash --create-home ${APP_USER}

# Copy application code
COPY --chown=${APP_USER}:${APP_GROUP} . .

# Remove unnecessary files from container
RUN rm -rf \
    .git \
    .gitignore \
    .env \
    .env.* \
    *.md \
    __pycache__ \
    .pytest_cache \
    tests \
    .github \
    *.log \
    2>/dev/null || true

# Set proper permissions
RUN chmod -R 755 ${APP_HOME} && \
    chown -R ${APP_USER}:${APP_GROUP} ${APP_HOME}

# Switch to non-root user
USER ${APP_USER}

# Expose port
EXPOSE ${PORT}

# Health check - verify the application is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Run the application with gunicorn for production
# Falls back to direct Python execution if gunicorn not available
CMD ["sh", "-c", "if command -v gunicorn >/dev/null 2>&1; then gunicorn --bind :${PORT} --workers 2 --threads 4 --timeout 120 --access-logfile - --error-logfile - 'main:app'; else python main.py; fi"]

# -----------------------------------------------------------------------------
# Stage 3: Development - For local development with hot reload
# -----------------------------------------------------------------------------
FROM production as development

USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install \
    pytest \
    pytest-cov \
    flake8 \
    black \
    mypy \
    ipython

USER ${APP_USER}

# Override CMD for development (with hot reload if using Flask)
CMD ["python", "main.py"]
