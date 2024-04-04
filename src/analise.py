#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo que contém funções necessárias à análise de dados dos regulamentos pela API da OpenAI.

"""

from openai import OpenAI
import json
import getpass

client = OpenAI(
api_key=getpass.getpass("Digite sua API KEY da OpenAI: "))

def analisar_esg(fundo, prompt):
    """
    Analisa a metodologia e os objetivos ESG de um fundo com base em seus dados, estruturando a resposta em um dicionário específico.
    
    Argumentos:
        fundo (dict): Um dicionário contendo os dados do fundo.
        prompt (str): As instruções para a análise pela API da OpenAI.
        
    Retorno:
        dict: Um dicionário com os resultados da análise estruturados conforme especificado.
    """
    
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
    
