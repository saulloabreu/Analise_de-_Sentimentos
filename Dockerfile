# Usar uma imagem menor para reduzir o tamanho do container
FROM python:3.9-alpine

# Definir diretório de trabalho
WORKDIR /app

# Copiar os arquivos do projeto
COPY . /app/

# Atualizar pip e instalar dependências sem cache
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Expor a porta em que o app rodará
EXPOSE 8150

# Adicionar um healthcheck para verificar se o app está rodando
HEALTHCHECK CMD curl --fail http://localhost:8150 || exit 1


# Definir o comando para rodar o app
CMD ["python", "index.py"]
