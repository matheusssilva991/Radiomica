# Radiomica

## 📋 Sobre o Projeto

Projeto de Iniciação Científica focado no desenvolvimento de técnicas de armazenamento de dados, processamento de imagens e inteligência artificial aplicadas no auxílio ao diagnóstico do câncer de mama.

**Estado:** Em desenvolvimento

---

## 🎯 Objetivos

Este projeto tem como objetivo desenvolver um sistema completo de análise de imagens médicas de mama utilizando:

- **Processamento de Imagens:** Extração e análise de características radiômicas
- **Inteligência Artificial:** Modelos de Machine Learning para classificação e diagnóstico
- **Armazenamento de Dados:** Organização e catalogação de metadados DICOM
- **Análise Exploratória:** Visualização e interpretação de dados médicos

---

## 📁 Estrutura do Projeto

```
Radiomica/
├── src/                          # Código fonte
│   ├── mamografia/               # Análise de mamografias
│   │   ├── catalogar_imagens.ipynb
│   │   ├── eda_metadados.ipynb
│   │   ├── extrair_features.ipynb
│   │   ├── model.ipynb
│   │   └── apresentacao.ipynb
│   ├── termografia/              # Análise de termografias
│   ├── ultrassom/                # Análise de ultrassons
│   ├── mongoDB/                  # Scripts para banco de dados
│   └── utils/                    # Funções utilitárias
├── outputs/                      # Resultados e metadados extraídos
│   ├── mamografia/
│   ├── termografia/
│   ├── ultrassom/
│   └── inserts_mongodb/
├── environment.yml               # Ambiente Conda
├── requirements.txt              # Dependências Python
└── README.md                     # Este arquivo
```

---

## 🗂️ Bases de Dados Utilizadas

### Mamografia

- **CBIS-DDSM:** Curated Breast Imaging Subset of DDSM
- **CMMD:** Chinese Mammography Database
- **INbreast:** Instituto Portugês de Oncologia
- **MIAS:** Mammographic Image Analysis Society

### Termografia

- **DMR-IR:** Database for Mastology Research with Infrared Image
- **Mammotherm:** Base de termografia mamária

### Ultrassom

- **BUSI:** Breast Ultrasound Images Dataset
- **HMSS:** Hospital Marília de Sorocaba e Santos
- **OASBUD:** Open Access Series of Breast Ultrasound Dataset
- **Thammasat:** Thammasat University Hospital Dataset

---

## 🚀 Quick Start

### Pré-requisitos

- Python 3.8+
- Conda ou Miniconda instalado
- Git

### Instalação

1. **Clone o repositório:**

```bash
git clone https://github.com/matheusssilva991/Radiomica.git
cd Radiomica
```

1. **Crie o ambiente Conda:**

```bash
conda env create -f environment.yml
```

1. **Ative o ambiente:**

```bash
conda activate radiomica
```

1. **Ou instale via pip (alternativo):**

```bash
pip install -r requirements.txt
```

### Como Usar

#### 1. Catalogação de Imagens

Organize e extraia metadados das imagens médicas:

```bash
# Navegue até a pasta da modalidade desejada
cd src/mamografia/  # ou termografia, ultrassom

# Abra o Jupyter Notebook
jupyter notebook catalogar_imagens.ipynb
```

**Importante:** Ajuste os caminhos das bases de dados nos notebooks antes de executar.

#### 2. Análise Exploratória de Dados (EDA)

Explore estatísticas e visualizações dos metadados:

```bash
jupyter notebook eda_metadados.ipynb
```

#### 3. Extração de Features

Extraia características radiômicas das imagens:

```bash
jupyter notebook extrair_features.ipynb
```

#### 4. Modelagem

Treine e avalie modelos de Machine Learning:

```bash
jupyter notebook model.ipynb
```

---

## 🔬 Etapas do Projeto

### ✅ 1ª Etapa - Catalogação das Imagens

#### Extração de Metadados

- Extração de metadados de arquivos CSV/TXT
- Leitura de cabeçalhos DICOM/PGM
- Catalogação das bases CMMD, CBIS-DDSM, INbreast e MIAS

#### Dicionário de Tags DICOM

- Mapeamento completo de tags DICOM presentes nas bases
- Contagem de frequência de uso de cada tag
- Unificação de dicionários entre diferentes bases

### 🔄 2ª Etapa - Processamento de Imagens

- Redimensionamento e normalização
- Extração de ROI (Region of Interest)
- Aplicação de máscaras de segmentação
- Cálculo de features de primeira ordem e GLCM

### 🤖 3ª Etapa - Modelagem e IA

- Seleção e engenharia de features
- Treinamento de modelos de classificação
- Validação cruzada e avaliação de métricas
- Otimização de hiperparâmetros

### 💾 4ª Etapa - Armazenamento

- Organização de metadados em MongoDB
- Estruturação de schemas para diferentes modalidades
- Geração de scripts de inserção

---

## 🛠️ Principais Funções Utilitárias

O módulo `utils.py` contém funções essenciais para o projeto:

- **Manipulação de DICOM:** `get_dicom_meta()`, `extract_image_dicom()`
- **Processamento de Imagens:** `resize_image()`, `extract_roi()`
- **Extração de Features:** `get_fo_features()` (First Order Features)
- **Máscaras:** `load_inbreast_mask()` para ROI
- **Visualização:** `plot_history()`, `draw_image_mias()`
- **Dados:** `load_json()`, `save_json()`, `create_df()`

---

## 📊 Features Extraídas

### First Order Features

- Média (Mean)
- Variância (Variance)
- Desvio padrão (Standard Deviation)
- Suavidade (Smoothness)
- Terceiro momento (Third Moment)
- Uniformidade (Uniformity)
- Entropia (Entropy)

### GLCM Features

- Características de textura baseadas em Gray Level Co-occurrence Matrix

---

## 📝 Formato dos Dados de Saída

Os metadados e features extraídas são salvos em formato CSV e JSON:

```
outputs/
├── mamografia/
│   ├── cbis-ddsm/
│   │   ├── metadata_*.csv
│   │   ├── first_order_features_*.csv
│   │   └── glcm_features_*.csv
│   ├── cmmd/
│   ├── inbreast/
│   └── mias/
```

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona NovaFeature'`)
4. Push para a branch (`git push origin feature/NovaFeature`)
5. Abra um Pull Request

---

## 📄 Licença

Este é um projeto acadêmico de Iniciação Científica. Para uso dos dados, consulte as licenças específicas de cada base de dados utilizada.

---

## 👥 Autor

**Matheus Silva**

- GitHub: [@matheusssilva991](https://github.com/matheusssilva991)

---

## 📚 Referências

As bases de dados utilizadas neste projeto são disponibilizadas publicamente para fins de pesquisa. Consulte a documentação oficial de cada base para mais informações sobre citação e uso adequado.

---

## ⚙️ Desativação do Ambiente

Para desativar o ambiente Conda:

```bash
conda deactivate
```

---

## 🐛 Problemas Conhecidos

- Alguns notebooks podem requerer ajuste manual dos caminhos das bases de dados
- Certifique-se de ter espaço em disco suficiente para as bases de imagens médicas

---

## 📮 Contato

Para dúvidas ou sugestões, abra uma issue no repositório do GitHub.
