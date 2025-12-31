# Dockerfile para el frontend React + Vite
FROM node:18-alpine

WORKDIR /app

# Copiar package files
COPY package*.json ./

# Instalar dependencias
RUN npm install

# Copiar código fuente
COPY . .

# Exponer puerto de Vite
EXPOSE 5174

# Comando por defecto (se sobrescribe en docker-compose)
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
