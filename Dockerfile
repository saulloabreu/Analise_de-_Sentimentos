# Usa uma imagem Python leve
FROM python:3.9-slim

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos para dentro do contêiner
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código
COPY . /app/

# Expõe a porta
EXPOSE 8080

# Comando para rodar o app
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:server"]
