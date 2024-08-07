# README
Este pacote apresenta scripts, cadernos e dados empregados em pesquisa sobre regulamentos de fundos de investimento ESG.
- O scraping foi feito em 29/05/2024. Os arquivos de texto, de dados e metadados constam do programa (v. /data/), para que outros pesquisadores façam suas próprias análises.
- Este é meu primeiro trabalho do gênero. Correções, sugestões e aprimoramentos serão bem-vindos.
  - Os resultados da análise baseiam-se em filtros dos textos dos regulamentos e no processamento dos trechos filtrados mediante a API da OpenAI. Gostaria de incentivar outros pesquisadores a reproduzirem  a pesquisa, editando as palavras-chave dos filtros (src/filtro.py) e a orientação ("prompt", em notebooks/4_pesquisa_esg.ipynb) à API.
  - Para reproduzir a pesquisa, é necessário instalar as bibliotecas listadas em requirements.txt, por meio do comando pip necessário, e descarregar manualmente o ChromeDriver (webdriver que opera o navegador Google Chrome para a biblioteca Selenium), bem como o corpus da língua portuguesa pt_core_news_lg, que subsidia a biblioteca spaCy. Também é preciso configurar uma chave para a API da OpenAI. Mais informações nos respectivos sites e documentos de ajuda.
- Contato: http://pedblan.wordpress.com
## Estrutura
### /notebooks
- Contém os cadernos Jupyter com cada um dos passos da pesquisa.
  - 1_coleta.ipynb: faz a coleta de dados
  - 2_converte_txt.ipynb: converte os regulamentos de pdf para txt
  - 3_aed.ipynb: faz análise exploratória dos dados obtidos pelo scraping de fundo
  - 4_pesquisa_esg.ipynb: comunica-se com a API da OpenAI para uma análise mais aprofundada dos regulamentos
    - neste caderno se encontra o prompt dirigido à API da OpenAI. Gostaria de incentivar interessados a editar o prompt e apresentar as respostas obtidas.
  - 5_analise_esg.ipynb: faz análise NLP de objetivos, metodologia, referências e relatório da cada fundo
  
### /data
#### /json/ - Apresenta dados empregados em cada etapa da pesquisa
  - total_links.json: resultado do scraping inicial, dos resultados de pesquisa na página da Anbima. 
  - metadados.json: resultado do scraping da página de cada fundo
  - respostas.json: respostas da API OpenAI
  - dados.json: combinação de metadados.json e respostas.json
#### /pdf/
- arquivos pdf dos regulamentos dos fundos, descarregados na página da Anbima ou buscados manualmente, em caso de erro 
#### /txt/
- versão em txt dos arquivos pdf, para posterior análise textual
### /src - Contém scripts usados nos cadernos Jupyter
- analise.py: função destinada a comunicar-se com o modelo OpenAI. Requer pass-key, obtida mediante cadastro no site respectivo: https://www.openai.com
- analise_nlp.py: diversas funções para NLP do material obtido
  - emprega biblioteca spaCY, que requer download de corpus da língua portuguesa. Empreguei pt_core_news_lg, que não se encontra neste pacote, por causa do tamanho do arquivo. Está disponível para download em: https://spacy.io/models/pt 
- converte_txt.py: função para extrair texto dos arquivos pdf e gravá-lo em arquivos txt.
  - Emprega o módulo PyMuPDF, que é importado sob o nome de "fitz".
    - Este foi o único módulo que conseguiu extrair a informação de caixas de texto, formato comum a vários regulamentos pesquisados
    - O módulo PyMuPDF/fitz tem um problema conhecido: RuntimeError(f“Directory '{directory}' does not exist”)
RuntimeError: Directory 'static/' does not exist from import fitz. Consegui corrigir isso seguindo as orientações deste link: https://stackoverflow.com/questions/67982760/raise-runtimeerrorfdirectory-directory-does-not-exist-runtimeerror-dire
- filtro.py: funções para filtrar os arquivos txt, o que dá mais foco à pesquisa e reduz o preço da consulta à API OpenAI.
  - os filtros selecionam frases que tem tanto uma das palavras-chave de método/relatório/referência quanto uma das palavras-chave temáticas. A divisão por frases se justifica pela convenção estilística que caracteriza documentos jurídico-empresariais, e.g., contratos: frases longas, com alguma complexidade sintática.
  - Gostaria de incentivar interessados a reproduzirem o experimento, editando as palavras-chave dos filtros.
- grafico.py: funções para fazer gráficos apresentando correlações de dados
- scraping.py: função para fazer o scraping no sítio da Anbima.
  - biblioteca empregada: Selenium
  - webdriver empregado: Chromedriver, que faz uso automático do navegador Chrome
    - requer que se descarregue o webdriver em https://chromedriver.chromium.org/downloads
    - o Chromedriver deve ser descarregado em alguma pasta da variável de ambiente PATH (recurso do sistema operacional que verifica onde procurar arquivos executáveis).
    - Num Mac, a lista destas pastas inclui /usr/local/bin. O Windows e o Linux seguem procedimentos parecidos, é preciso descobrir a pasta certa.
    - a variável PATH pode ser o diretório principal do programa.
