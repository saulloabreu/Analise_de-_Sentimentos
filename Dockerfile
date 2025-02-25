# Usar uma imagem menor baseada em Alpine
FROM python:3.9-slim

# Definir diretório de trabalho
WORKDIR /app

# Copiar apenas os arquivos essenciais para instalar dependências primeiro
COPY requirements.txt /app/

# Atualizar pip e instalar dependências sem cache
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar o restante dos arquivos
COPY . /app/

# Expor a porta onde o app será executado
EXPOSE 8150

# Definir comando para rodar o app
CMD ["gunicorn", "-b", "0.0.0.0:8150", "index:server"]
