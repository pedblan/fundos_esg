# atualizar stop words, limpar texto de stop words, dividir em quartis, fazer word cloud, tf idf, extrair ner, bm 25,  tokenizar, clustering, análise de tópicos, word clouds, redes semânticas

import spacy
nlp = spacy.load("pt_core_news_lg")
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
from rank_bm25 import BM25Okapi
import numpy as np
import json

stop_words = nlp.Defaults.stop_words


def listar_textos(diretorio, extensao='.txt'):
    """
    Lê e retorna o conteúdo de todos os arquivos com a extensão especificada em um diretório,
    armazenando-os em um dicionário com o nome do arquivo como chave.

    Args:
        diretorio (str): Caminho do diretório a ser explorado.
        extensao (str): Extensão dos arquivos a serem lidos.

    Returns:
        dict: Dicionário com o nome dos arquivos (sem extensão) como chaves e o conteúdo dos arquivos como valores.
    """
    textos = {}
    for arquivo in os.listdir(diretorio):
        if arquivo.endswith(extensao):
            nome_arquivo = os.path.splitext(arquivo)[0]  # Remove a extensão do arquivo para usar como chave
            caminho_completo = os.path.join(diretorio, arquivo)
            with open(caminho_completo, 'r', encoding='utf-8') as f:
                textos[nome_arquivo] = f.read()
    return textos




def tokenizar(texto):
    return nlp(texto)

def remover_pontuacao_espacos(tokens):
    return [token for token in tokens if not token.is_punct and not token.is_space]

def remover_stop_words(tokens, stop_words):
    return [token for token in tokens if token.text.lower() not in stop_words]

def lematizar(tokens):
    return [token.lemma_.lower() for token in tokens]


def limpar_texto(texto, stop_words, substantivos_adjetivos=False):
    tokens = tokenizar(texto)
    tokens = remover_pontuacao_espacos(tokens)
    tokens = remover_stop_words(tokens, stop_words)
    lemas = lematizar(tokens)
    

    if substantivos_adjetivos:
        texto_retokenizado = nlp(" ".join(lemas))
        tokens_processados = [token for token in texto_retokenizado if token.pos_ in ["NOUN", "ADJ"]]
    else:
        tokens_processados = lemas

    return " ".join(tokens_processados)

def converter_em_numeros(valor):
    """Limpa e converte o valor para numérico."""
    if isinstance(valor, str):
        valor_limpo = valor.replace('R$', '').replace('.', '').replace(',', '.').str.replace('R$', '')
        return pd.to_numeric(valor_limpo, errors='coerce')
    return pd.to_numeric(valor, errors='coerce')

def dividir_em_quartis(df, criterio):
    """Cria nova coluna em um DataFrame com os quartis baseados no critério escolhido.
    
    Args:
        df (DataFrame): DataFrame de entrada.
        criterio (str): Coluna do DataFrame que será usada para o cálculo dos quartis.
    
    Returns:
        DataFrame: Novo DataFrame com a coluna de quartis adicionada.
    
    Raises:
        ValueError: Se o critério não corresponder a uma coluna no DataFrame.
    """
    if criterio not in df.columns:
        raise ValueError(f'O critério {criterio} não é uma coluna do DataFrame.')

    coluna_convertida = df[criterio].apply(limpar_e_converter_valor_em_reais)
    df[f'Quartis {criterio}'] = pd.qcut(coluna_convertida, 4, labels=['1º Quartil', '2º Quartil', '3º Quartil', '4º Quartil'])
    return df

def gerar_word_cloud(texto, largura=800, altura=400, cor_fundo='white'):
    """
    Gera e exibe uma nuvem de palavras a partir de um texto.

    Args:
        texto (str): Texto do qual a nuvem de palavras será gerada.
        largura (int, opcional): Largura da imagem da nuvem de palavras. Padrão é 800.
        altura (int, opcional): Altura da imagem da nuvem de palavras. Padrão é 400.
        cor_fundo (str, opcional): Cor de fundo da imagem da nuvem de palavras. Padrão é 'white'.

    Retorna:
        None
    """
    # Geração da word cloud
    wordcloud = WordCloud(width=largura, height=altura, background_color=cor_fundo).generate(texto)

    # Exibição da word cloud
    plt.figure(figsize=(largura / 100, altura / 100))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Não mostra os eixos
    plt.show()

def tfidf(documentos, max_features=1000, min_df=0.001, max_df=0.8, ngram_range=(1, 2)):
    """
    Calcula o TF-IDF para um conjunto de documentos e retorna um DataFrame com os resultados.

    Args:
        documentos (list of str): Lista contendo os documentos para os quais calcular o TF-IDF.

    Returns:
        DataFrame: DataFrame do pandas com os scores TF-IDF para cada termo em cada documento.
    """
    # Inicializa o calculador TF-IDF
    tfidf_vectorizer = TfidfVectorizer()

    # Calcula o TF-IDF
    tfidf_matrix = tfidf_vectorizer.fit_transform(documentos)

    # Cria um DataFrame com os resultados
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    return tfidf_df


def calcular_bm25(documentos, argumento_de_pesquisa):
    """
    Calcula as pontuações BM25 para uma consulta em um conjunto de documentos.

    Args:
        documentos (list of str): Lista contendo os documentos.
        consulta (str): A consulta de pesquisa para a qual calcular as pontuações BM25.

    Returns:
        list: Lista de pontuações BM25 para cada documento.
    """
    # Tokeniza os documentos e a consulta
    tokenized_docs = [tokenizar(doc.lower()) for doc in documentos]
    tokenized_query = tokenizar(argumento_de_pesquisa.lower())

    # Inicializa o objeto BM25
    bm25 = BM25Okapi(tokenized_docs)

    # Calcula as pontuações BM25 para a consulta
    scores = bm25.get_scores(tokenized_query)

    # Parear os documentos com seus scores
    documentos_scores = zip(documentos, scores)
    
    # Ordenar pela pontuação, do maior para o menor
    documentos_ordenados = sorted(documentos_scores, key=lambda x: x[1], reverse=True)

    return documentos_ordenados


def resultados_bm25(documentos_ordenados):
    """Informa resultados da análise com o algoritmo BM25, com gráfico."""
    # Exibir os documentos e seus scores
    for doc, score in documentos_ordenados:
        print(f"Score: {score:.3f} - Documento: {doc}")
   
    # Criar índices para os documentos
    indices = range(len(documentos_ordenados))
    
    # Criar um gráfico de barras
    plt.bar(indices, scores, align='center', alpha=0.7)
    plt.xticks(indices, [f"Doc {i+1}" for i in indices], rotation=45)
    plt.ylabel('Score BM25')
    plt.title('Scores BM25 por Documento')
    
    plt.show()

    

def extrair_ner(texto):
    """Extrai entidades de um texto específico."""
    doc = nlp(texto)
    return [ent.text for ent in doc.ents]

def obter_embeddings(documentos):
    """
    Informa vetores de embeddings dos documentos fornecidos.
    
    Args:
        documentos (list of str): Lista contendo os documentos.

    Returns:
        dict: Dicionário com índices dos documentos como chaves e vetores de embeddings como valores.
    """
    vetores = {}
    for i, doc in enumerate(documentos):
        processed_doc = nlp(doc)
        if processed_doc.has_vector:
            vetores[i] = processed_doc.vector
        else:
            print(f"Documento {i} não possui vetor de embeddings.")
    return vetores

def clusterizar_textos(vetores, n_clusters=5):
    """
    Clusteriza textos usando embeddings e K-means.

    Args:
        vetores (dict): Dicionário com índices dos documentos como chaves e vetores de embeddings como valores.
        n_clusters (int): Número de clusters desejado.

    Returns:
        clusters
    """

    # Checar se temos vetores suficientes para clusterizar
    if len(vetores) == 0:
        raise ValueError("Nenhum documento possui um vetor associado.")

    # Usar K-means para clusterizar os vetores
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(list(vetores.values()))

    return clusters

def grafico_clusters(vetores, clusters):
    # Assumindo que vetores é um dicionário com índices como chaves e vetores de embeddings como valores
    X = list(vetores.values())  # Convertendo os vetores de embeddings em uma lista para PCA
    
    # Aplicar PCA nos vetores de embeddings
    pca = PCA(n_components=2)
    reduced_documentos = pca.fit_transform(X)
    
    # Plotagem dos clusters
    plt.figure(figsize=(10, 6))
    for i in range(5):
        cluster_points = reduced_documentos[clusters == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')
    
    plt.title('Clusters de Documentos')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend()
    plt.show()

def palavras_distintivas_por_cluster(documentos_dict, clusters_dict, n_palavras=10):
    """
    Identifica e exibe as palavras distintivas para cada cluster.

    Args:
        documentos_dict (dict): Dicionário com nomes de arquivos como chaves e textos como valores.
        clusters_dict (dict): Dicionário com índices de clusters e listas de índices de documentos.
        n_palavras (int): Número de palavras distintivas a serem exibidas por cluster.

    Returns:
        dict: Dicionário com clusters como chaves e listas de palavras distintivas como valores.
    """
    # Assegurar que os nomes dos arquivos em clusters_dict são válidos em documentos_dict
    nomes_arquivos = list(documentos_dict.keys())
    
    # Concatenar textos de cada cluster para formar um documento por cluster
    textos_por_cluster = {
        cluster: " ".join([documentos_dict[nomes_arquivos[i]] for i in indices])
        for cluster, indices in clusters_dict.items()
    }

    # Calcular TF-IDF para cada cluster
    tfidf_vectorizer = TfidfVectorizer()
    corpus = list(textos_por_cluster.values())
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    # Identificar palavras distintivas para cada cluster
    palavras_distintivas = {}
    feature_array = np.array(tfidf_vectorizer.get_feature_names_out())
    for cluster, _ in textos_por_cluster.items():
        row = list(textos_por_cluster.keys()).index(cluster)
        tfidf_sorting = np.argsort(tfidf_matrix[row].toarray()[0])[::-1]

        top_n = feature_array[tfidf_sorting][:n_palavras]
        palavras_distintivas[cluster] = top_n

    return palavras_distintivas



