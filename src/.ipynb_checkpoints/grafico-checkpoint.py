#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:43:30 2024

@author: pedblan
"""
import pandas as pd
import matplotlib as plt

import matplotlib.pyplot as plt

def elaborar_grafico(dados, tipo='bar', tamanho=(10, 6), titulo='', x_label='', y_label='Percentagem', x_ticks_rotation=45, tight_layout=True, cores=None, explode=None, autopct=None, startangle=None, legend=False):
    """
    Função para elaborar gráficos com matplotlib.

    Args:
        dados (DataFrame or Series): Os dados a serem plotados.
        tipo (str): Tipo de gráfico, por exemplo, 'bar', 'line', 'pie', etc.
        tamanho (tuple): Tamanho da figura do gráfico.
        titulo (str): Título do gráfico.
        x_label (str): Rótulo do eixo X. Não aplicável a gráficos de pizza.
        y_label (str): Rótulo do eixo Y. Não aplicável a gráficos de pizza.
        x_ticks_rotation (int): Rotação dos rótulos do eixo X. Não aplicável a gráficos de pizza.
        tight_layout (bool): Se deve usar layout apertado.
        cores (list): Lista de cores para o gráfico.
        explode (list): Distância de cada parte do gráfico de pizza do centro. Apenas para gráficos de pizza.
        autopct (str): String de formatação para os rótulos das porções no gráfico de pizza.
        startangle (int): Rotação inicial do gráfico de pizza.
        legend (bool): Se verdadeiro, exibe a legenda do gráfico.
    """
    plt.figure(figsize=tamanho)
    
    if tipo == 'pie':
        dados.plot(kind=tipo, colors=cores, explode=explode, autopct=autopct, startangle=startangle)
        plt.title(titulo)
        if legend:
            plt.legend(dados.index)
    else:
        dados.plot(kind=tipo, color=cores)
        plt.title(titulo)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(rotation=x_ticks_rotation)
    
    if tight_layout:
        plt.tight_layout()

    plt.show()
