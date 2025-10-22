FROM python:3.12-slim

# Configuración Python para imágenes ligeras y logs inmediatos
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Establece el directorio de trabajo
WORKDIR /app

# Instala dependencias de sistema necesarias para librerías (xgboost necesita libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copia y instala dependencias
COPY ./backend/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia el resto del código
COPY ./backend ./backend
COPY ./resources ./resources

# Crear usuario no-root
RUN useradd -m appuser && chown -R appuser:appuser /app

USER appuser

# Expone el puerto por defecto de Uvicorn
EXPOSE 8000

# Comando por defecto al iniciar el contenedor
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
