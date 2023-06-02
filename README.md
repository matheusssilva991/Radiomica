# **Radiomica**  

## **Projeto de iniciação ciêntifica**  

**Tema:** Técnicas de armazenamento de dados, processamento de imagens e inteligência artificial aplicadas no aúxilio ao diagnóstico do câncer de mama.

**Estado:** Em desenvolvimento

---

### **1ª Etapa - Catalogação das Imagens**  

#### **Extração de metadados nos arquivos CSV e Dicom**

- Foram extraidos os metadados do arquivo CSV/TXT e dos cabeçalhos dos arquivos Dicom/PGM das bases de dados CMMD, DDSM e MIAS.

#### **Dicionário de tags Dicom**

- Foram extraidos todas as tags presentes nos cabeçalhos dos arquivos dicom/pgm, contendo o nome da tag e a quantidade de vezes que a tag foi utilizada em cada base de dados.
- Foram feitos um dicionário de tags para cada base e um sendo a junção dos dicionários de tags apenas das bases no padrão Dicom

---

#### **Quick Start**

##### Criar ambiente

- ```conda env create -f environment.yml```

##### Ativar/Desativar ambiente

- ```conda activate prov```
- ```conda deactivate```

##### Executar  

- Abrir os notebooks catalogar_dicom_imagens, dicionario_meta e visualizar_dicionario na pasta src/catalogar_imagens
- Mudar path das bases DDSM, CMMD e MIAS nos notebooks
- Rodar os notebooks
