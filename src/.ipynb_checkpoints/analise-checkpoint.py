#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo que contém funções necessárias à análise de dados dos regulamentos pela API da OpenAI.

"""

from openai import OpenAI
import json
import getpass

client = OpenAI(
api_key=getpass.getpass("Digite sua senha: "))

def analisar_esg(fundo, prompt):
    """
    Analisa a metodologia e os objetivos ESG de um fundo com base em seus dados, estruturando a resposta em um dicionário específico.
    
    Argumentos:
        fundo (dict): Um dicionário contendo os dados do fundo.
        prompt (str): As instruções para a análise pela API da OpenAI.
        
    Retorno:
        dict: Um dicionário com os resultados da análise estruturados conforme especificado.
    """
    
    resultados_analise = {'Origem da metodologia ASG': [], 'Objetivo ASG': [], 'Método ASG': [], 'Referência ASG': [], 'URL referência ASG': [], 'Relatório ASG': [], 'Entidade certificadora ASG': [], 'Norma ASG citada': []}
    texto_analise = f"{fundo.get('Metodologia', '')} {fundo.get('Objetivos', '')} {fundo.get('Diligência', '')}"
    mensagem_input_usuario = {
        "role": "user",
        "content": f"Analise este texto: {texto_analise}, segundo estas instruções: {prompt}. Coloque os resultados neste formato: {resultados_analise}. Lembre-se, se não encontrar resposta, preencha com 'n/d'."
    }
    
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "Este é um sistema de análise de regulamentos de fundos ASG/ESG, que busca fazer um perfil dos critérios empregados para definir o fundo como ESG/ASG."},
            mensagem_input_usuario
        ],
        temperature=0.2,
        max_tokens=300
    )
    
    return response.choices[0].message.content
    
def confere(metadados):
    """Confere se todos os fundos foram analisados."""
    counter_dict = 0
    counter_str = 0
    strlist = []
    for item in metadados:
        if type(item["Análise"]) == str:
            counter_str += 1
            strlist.append(item["Análise"])
        else:
            counter_dict += 1
    print(f"Número de dicionários: {counter_dict}. \nNúmero de strings: {counter_str}. \nTotal: {counter_str + counter_dict}.")
     

def integra_dados():
    """Integra os resultados de análise em chaves próprias do dicionário principal."""
    
    with open('../data/json/metadados.json', 'r', encoding='utf-8') as f:
        dados = json.load(f)
    dados_modificados = []
    for item in dados:
        # Extrai o dicionário 'Análise', se existir
        analise = item.pop('Análise', {})
        # Para cada chave em 'Análise', move essa chave-valor para o nível superior
        for chave, valor in analise.items():
            item[chave] = valor
        # Adiciona o item modificado à nova lista
        dados_modificados.append(item)
    chaves_para_renomear = ['origem', 'objetivo', 'método', 'referência', 'URL', 'relatório', 'entidade certificadora', 'norma']
    # Processa cada dicionário na lista
    for dicionario in dados_modificados:
        for chave in list(dicionario.keys()):
            if chave in chaves_para_renomear:
                # Renomeia a chave
                dicionario[f'Metodologia ESG - {chave}'] = dicionario.pop(chave)
    with open('dados.json', 'w', encoding='utf-8') as f: # Salva o progresso em dados.json.
                    json.dump(dados_modificados, f, ensure_ascii=False, indent=4)
                    
