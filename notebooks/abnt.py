import json
from datetime import datetime

# Carrega os dados do arquivo JSON
with open('data/json/metadados.json', 'r') as file:
    data = json.load(file)

# Define a data de acesso
data_acesso = datetime(2024, 5, 29).strftime('%d de %B de %Y')

# Função para formatar a referência bibliográfica no estilo ABNT
def formatar_referencia_abnt(fundo):
    nome_fundo = fundo.get("Fundo", "Nome do Fundo Não Informado")
    url = fundo.get("URL do Fundo", "URL Não Informada")
    referencia = f"{nome_fundo}. Disponível em: <{url}>. Acesso em: {data_acesso}."
    return referencia

# Itera sobre cada fundo e gera as referências
referencias = [formatar_referencia_abnt(fundo) for fundo in data]

# Exibe as referências
for referencia in referencias:
    print(referencia)
