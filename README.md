# PROJEÇÃO DE DEMANDA ELÉTRICA COM ALGORITMOS DE APRENDIZADO DE MÁQUINA
Repositório do TCC para projetar a demanda elétrica de curto prazo (diária) na região sul do Brasil com modelos de aprendizado de máquina e compará-los com as metodologias tradicionais de Holt-Winters e SARIMA.
<br>
* Scripts para [web scraping](https://github.com/matheus-vizzotto/projecao_demanda_eletrica/blob/main/data/1_scrape_elec_load.py) e [análise exploratória](https://github.com/matheus-vizzotto/projecao_demanda_eletrica/blob/main/data/2_eda_load.ipynb) dos dados de demanda elétrica;
* Script para [web scraping](https://github.com/matheus-vizzotto/projecao_demanda_eletrica/blob/main/data/dados_inmet/1_scraper_temp.ipynb), [tratamento](https://github.com/matheus-vizzotto/projecao_demanda_eletrica/blob/main/data/dados_inmet/2_temp_wrangling.ipynb) e [análise exploratória](https://github.com/matheus-vizzotto/projecao_demanda_eletrica/blob/main/data/3_eda_weather.ipynb) dos dados climáticos;
* [Modelos utilizados](https://github.com/matheus-vizzotto/projecao_demanda_eletrica/tree/main/models/forecasts);
* [Comparação de resultados](https://github.com/matheus-vizzotto/projecao_demanda_eletrica/blob/main/models/forecasts/12_compare_fcs.ipynb);
* O diretório [lab](https://github.com/matheus-vizzotto/projecao_demanda_eletrica/tree/main/lab) contém o código dos modelos utilizados em produção, mas em formato Jupyter Notebook, para facilitar a visualização;
* O documento do trabalho pode ser visto [aqui](https://github.com/matheus-vizzotto/projecao_demanda_eletrica/blob/main/doc.pdf). A apresentação do trabalho também se encontra em formato [PPT](https://docs.google.com/presentation/d/1q7WB4qba9i__uf1zBRZLcRWGmj5rYZ49zIcowuaBX8c/edit?usp=sharing).
