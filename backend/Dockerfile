# Usa una imagen oficial de Python como base (sin soporte GPU)
FROM python:3.12-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia solo el archivo de requerimientos para aprovechar el cache
COPY backend/requirements.txt .

# Instala las dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código fuente al contenedor
COPY backend/ backend/
COPY resources/ resources/
COPY common/ common/

# Expone el puerto que usará FastAPI
EXPOSE 8000

# Comando por defecto para iniciar la aplicación
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
