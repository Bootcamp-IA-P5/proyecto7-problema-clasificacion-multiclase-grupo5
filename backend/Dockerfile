# Utiliza la imagen oficial de Python como base
FROM python:3.12-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instala build tools (si los necesitas para xgboost)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Usuario sin privilegios
RUN adduser --disabled-password --gecos "" appuser

# Copia el archivo de requerimientos del backend y los instala
# Se copian primero para aprovechar el cache de Docker si no cambian
COPY ./backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r ./backend/requirements.txt
RUN pip install xgboost --no-deps

# Copia el código del backend y los recursos compartidos a sus respectivas carpetas
COPY ./backend ./
COPY ./resources ./resources
COPY ./common ./common

# Cambiar dueño de los archivos al nuevo usuario
RUN chown -R appuser:appuser /app

# Cambiar al usuario sin privilegios
USER appuser

# Expone el puerto que usa FastAPI
EXPOSE 8000

# Comando por defecto para ejecutar la aplicación de FastAPI
# Este comando se ejecutará si no se especifica otro en docker-compose.yml
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]