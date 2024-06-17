# README
Este pacote apresenta scripts, cadernos e dados empregados em pesquisa sobre regulamentos de fundos de investimento ESG.
- O scraping foi feito em 01/04/2024. Os arquivos de texto, de dados e metadados constam do programa, para que outros pesquisadores façam suas próprias análises.
- Este é meu primeiro trabalho do gênero. Correções, sugestões e aprimoramentos serão bem-vindos.
  - Os resultados da análise baseiam-se em filtros dos textos dos regulamentos e no processamento destes mediante a API da OpenAI. Gostaria de incentivar outros pesquisadores a reproduzirem  a pesquisa, editando as palavras-chave dos filtros (filtro.py) e a orientação ("prompt", em 4_pesquisa_esg.ipynb) à API.
## Estrutura
### /notebooks
- contém os cadernos Jupyter com cada um dos passos da pesquisa.
  - 1_coleta.ipynb: faz a coleta de dados
  - 2_converte_txt.ipynb: converte os regulamentos de pdf para txt
  - 3_aed.ipynb: faz análise exploratória dos dados obtidos pelo scraping de fundo
  - 4_pesquisa_esg.ipynb: comunica-se com a API da OpenAI para uma análise mais aprofundada dos regulamentos
    - neste caderno se encontra o prompt dirigido à API da OpenAI. Gostaria de incentivar interessados a editar o prompt e apresentar as respostas obtidas.
  - 5_analise_esg.ipynb: faz análise NLP de objetivos, metodologia, referências e relatório da cada fundo
  
### /data
#### /json/
- apresenta dados empregados em cada etapa da pesquisa
- json
  - total_links.json: resultado do scraping inicial, dos resultados de pesquisa na página da Anbima. 
  - metadados.json: resultado do scraping da página de cada fundo
  - respostas.json: respostas da API OpenAI
  - dados.json: combinação de metadados.json e respostas.json
#### /pdf/
- arquivos pdf dos regulamentos dos fundos, descarregados na página da Anbima ou buscados manualmente, em caso de erro 
#### /txt/
- versão em txt dos arquivos pdf, para posterior análise textual

### /src
- analise.py: função destinada a comunicar-se com o modelo OpenAI. Requer pass-key, obtida mediante cadastro no site respectivo: https://www.openai.com
- analise_nlp.py: diversas funções para NLP do material obtido
  - emprega biblioteca spaCY, que requer download de corpus da língua portuguesa. Empreguei pt_core_news_lg, disponível em: https://spacy.io/models/pt 
- converte_txt.py: função para extrair texto dos arquivos pdf e gravá-lo em arquivos txt.
  - Emprega o módulo PyMuPDF, que é importado sob o nome de "fitz".
    - Este foi o único módulo que conseguiu extrair a informação de caixas de texto, formato comum a vários regulamentos pesquisados
    - O módulo PyMuPDF/fitz tem um problema conhecido: RuntimeError(f“Directory '{directory}' does not exist”)
RuntimeError: Directory 'static/' does not exist from import fitz. Consegui corrigir isso seguindo as orientações deste link: https://stackoverflow.com/questions/67982760/raise-runtimeerrorfdirectory-directory-does-not-exist-runtimeerror-dire
- filtro.py: funções para filtrar os arquivos txt, o que dá mais foco à pesquisa e reduz o preço da consulta à API OpenAI.
  - os filtros selecionam frases que tem uma das palavras-chave de metodologia/relatório/objetivo
E uma das palavras-chave temáticas. Gostaria de incentivar interessados a reproduzirem o experimento, editando as
palavras-chave dos filtros.
- grafico.py: funções para fazer gráficos apresentando correlações de dados
- scraping.py: função para fazer o scraping no sítio da Anbima.
  - biblioteca empregada: Selenium
  - webdriver empregado: Chromedriver, que faz uso automático do navegador Chrome
    - requer que se descarregue o webdriver em https://chromedriver.chromium.org/downloads
    - o Chromedriver deve ser descarregado em alguma pasta da variável de ambiente PATH (recurso do sistema operacional que confere onde procurar arquivos executáveis).
    - Num Mac, a lista destas pastas inclui /usr/local/bin. O Windows e o Linux seguem procedimentos parecidos, é preciso descobrir a pasta certa.