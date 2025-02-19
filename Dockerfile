# Basis-Image wählen (Python 3.9 als Beispiel)
FROM python:3.12-slim

# Arbeitsverzeichnis festlegen
WORKDIR /docker-app

# Alle Dateien in das Arbeitsverzeichnis kopieren
COPY . /docker-app

# install nodejs
RUN apt-get update
RUN apt-get install -y nodejs npm
RUN npm install -g yarn

# Go to frontend and run yarn install
WORKDIR /docker-app/frontend
RUN yarn install
RUN yarn build

# Go back to root directory
WORKDIR /docker-app

# Abhängigkeiten installieren
RUN pip install --no-cache-dir -r requirements.txt

# Falls dein Webserver auf einem bestimmten Port (z. B. 5000) lauscht:
EXPOSE 8070

# Kommando zum Starten der Anwendung
CMD ["python", "run.py", "--setup"]