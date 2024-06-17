#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo que contém funções necessárias à análise de dados dos regulamentos pela API da OpenAI.

"""

from openai import OpenAI
import json
import getpass
import requests

def analisar_esg(fundo, prompt):
    """
    Analisa a metodologia e os objetivos ESG de um fundo com base em seus dados, estruturando a resposta em um dicionário específico.
    
    Argumentos:
        fundo (dict): Um dicionário contendo os dados do fundo.
        prompt (str): As instruções para a análise pela API da OpenAI.
        
    Retorno:
        dict: Um dicionário com os resultados da análise estruturados conforme especificado.
    """
    client = OpenAI()
    resultados_analise = {'Origem metodologia ASG': '', 'Objetivo ASG': '', 'Objetivo Geral/Específico': '', 'Método ASG': '', 'Índice': '', 'Referência ASG': '', 'URL referência ASG': '', 'Relatório ASG': ''}
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


def analisar_esg_gemma(fundo, prompt):
    """
    Analisa a metodologia e os objetivos ESG de um fundo com base em seus dados, utilizando a API da Gemma,
    estruturando a resposta em um dicionário específico.

    Argumentos:
        fundo (dict): Um dicionário contendo os dados do fundo.
        prompt (str): As instruções para a análise pela API da Gemma.

    Retorno:
        dict: Um dicionário com os resultados da análise estruturados conforme especificado.
    """

    api_key = getpass.getpass("Digite sua API KEY da Gemma: ")

    # Definição do endpoint da API da Gemma e chave de autenticação
    api_url = 'https://api.gemma.ai/v1/generate'
    headers = {'Authorization': f'Bearer {api_key}'}  # Substitua YOUR_API_KEY pela sua chave de API

    # Construção do prompt para a Gemma
    texto_analise = f"{fundo.get('Metodologia', '')} {fundo.get('Objetivos', '')} {fundo.get('Diligência', '')}"
    prompt_completo = f"{prompt} {texto_analise}"

    # Dados para a requisição POST
    data = {
        "prompt": prompt_completo,
        "max_tokens": 300,
        "temperature": 0.2
    }

    # Fazendo a chamada para a API da Gemma
    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code == 200:
        # Processando a resposta
        resposta = response.json()['choices'][0]['text']
    else:
        resposta = "Erro na chamada da API da Gemma"

    # Processamento da resposta para extrair informações estruturadas
    resultados_analise = {'Origem metodologia ASG': '', 'Objetivo ASG': '', 'Objetivo Geral/Específico': '',
                          'Método ASG': '', 'Índice': '', 'Referência ASG': '', 'URL referência ASG': '',
                          'Relatório ASG': ''}

    # Aqui você pode adicionar um código para analisar a resposta gerada e preencher o dicionário de resultados
    # Este é um placeholder para o processo de parseamento
    # resultados_analise = parse_response_to_dict(resposta)

    return resultados_analise