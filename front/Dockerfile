# Imagen base de Node
FROM node:20

# Directorio de trabajo en el contenedor
WORKDIR /usr/src/app

# Copiamos package.json y package-lock.json si existe
COPY package*.json ./

# Instalamos dependencias
RUN npm install

# Copiamos el resto del proyecto Angular
COPY . .

# Puerto del dev server de Angular
EXPOSE 4200

# Comando: levantar Angular
CMD ["npm", "run", "start", "--", "--host", "0.0.0.0", "--port", "4200"]
