# Utiliza la imagen oficial de Python como base
FROM python:3.12-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de requerimientos del backend y los instala
# Se copian primero para aprovechar el cache de Docker si no cambian
COPY ./backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r ./backend/requirements.txt
RUN pip install xgboost --no-deps

# Copia el código del backend y los recursos compartidos a sus respectivas carpetas
COPY ./backend ./backend
COPY ./resources ./resources
COPY ./common ./common

# Expone el puerto que usa FastAPI
EXPOSE 8000

# Comando por defecto para ejecutar la aplicación de FastAPI
# Este comando se ejecutará si no se especifica otro en docker-compose.yml
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]