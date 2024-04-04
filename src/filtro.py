#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Filtra os arquivos txt segundo palavras-chave relacionadas a metodologia e objetivo ESG.

"""
import re
import os
import json

# Configurações Iniciais
diretorio_entrada = '../data/txt/'

def metodologias(fundo):
    """Faz a pesquisa de metodologias e objetivos ESG segundo listas de palavras-chave."""
    
    # Palavras-chave
    palavras_metodologia = ["metodologia", "critérios", "método", "métodos", "abordagem", "avaliação", "índice", "índices", "filtro", "parâmetros", "princípio", "princípios", "diretrizes", "engajamento", "ações continuadas", "métrica", "métricas", "indicadores", "estratégia", "escore", "escore", "certificado", "certificação"]
    palavras_objetivo = ["objetivo do fundo", "objetivos do fundo", "objetivos de investimento", "objetivo de investimento", "compromisso", "política de investimento", "benefício", "benefícios"]
    palavras_relatorio = ["relatório","relatórios", "periódicos", "periodicidade", "parecer", "resultados", "transparência", "demonstrativo", "due diligence", "monitoramento", "performance"]
    palavras_auxiliares = ["verde", "verdes", "responsável", "melhores", "práticas", "esg", "asg", "positiva", "raça", "diversidade", "positivas", "governança", "ambiente", "ambiental", "ambientais", "equidade", "sociais", "gênero", "sustentável", "sustentabilidade", "transição energética", "carbono", "água", "energia", "trabalho escravo", "trabalho análogo ao escravo", "trabalho infantil", "mudanças climáticas", "mudança climática", "biodiversidade", "desmatamento", "eficiência energética", "recursos naturais", "limpa", "poluição", "inclusão", "direitos humanos", "ética", "pegada de carbono"]
    
    # Expressões Regulares
    regex_metodologia = re.compile('|'.join([f"\\b{palavra}\\b" for palavra in palavras_metodologia]), re.IGNORECASE)
    regex_objetivo = re.compile('|'.join([f"\\b{palavra}\\b" for palavra in palavras_objetivo]), re.IGNORECASE)
    regex_relatorio = re.compile('|'.join([f"\\b{palavra}\\b" for palavra in palavras_relatorio]), re.IGNORECASE)
    regex_auxiliar = re.compile('|'.join([f"\\b{palavra}\\b" for palavra in palavras_auxiliares]), re.IGNORECASE)
    regex_frase = re.compile(r'(?<!\w\.\w)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s')
    
    
    def extrair_informacoes_e_organizar(fundo, diretorio_entrada):
        """Faz a pesquisa de fato."""
        nome_do_arquivo = fundo["Arquivo"] + '.txt'
        caminho_completo = os.path.join(diretorio_entrada, nome_do_arquivo)   
        fundo["Metodologia"] = []
        fundo["Objetivos"] = []
        fundo["Diligência"] = []
        urls_encontradas = set()
        with open(caminho_completo, 'r', encoding='utf-8') as f:
            try:
                texto = f.read()
            except Exception as e:
                print(f"Erro no processamento do arquivo {nome_do_arquivo}: {e}")
        frases = regex_frase.split(texto)
        for frase in frases:
            if regex_metodologia.search(frase) and regex_auxiliar.search(frase):
                fundo["Metodologia"].append(frase)
            if regex_relatorio.search(frase):
                fundo["Diligência"].append(frase)
            if regex_objetivo.search(frase) and regex_auxiliar.search(frase):
                fundo["Objetivos"].append(frase)
        print(f"Arquivo processado: {nome_do_arquivo}")
        return fundo
    
    # No contexto da função principal, chamada as auxiliares e faz a tarefa
    extrair_informacoes_e_organizar(fundo, diretorio_entrada)