# <strong>Radiomica</strong> 

## <strong>Projeto de iniciação ciêntifica</strong> 
<strong>Tema:</strong> Técnicas de armazenamento de dados, processamento de imagens e inteligência artificial aplicadas no aúxilio ao diagnóstico do câncer de mama.<br>
<strong>Estado:</strong> Em desenvolvimento

<hr>

### <strong>1ª Etapa - Catalogação das Imagens</strong> 

#### <strong>Extração de metadados nos arquivos CSV e Dicom</strong>
- Foram extraidos os metadados do arquivo CSV/TXT e dos cabeçalhos dos arquivos Dicom/PGM das bases de dados CMMD, DDSM e MIAS.

#### <strong>Dicionário de tags Dicom</strong>
- Foram extraidos todas as tags presentes nos cabeçalhos dos arquivos dicom/pgm, contendo o nome da tag e a quantidade de vezes que a tag foi utilizada em cada base de dados.
- Foram feitos um dicionário de tags para cada base e um sendo a junção dos dicionários de tags apenas das bases no padrão Dicom

<hr>

#### <strong>Quick Start - Executar Catalogação</strong> 
- Abrir os notebooks catalogar_dicom_imagens, dicionario_meta e visualizar_dicionario na pasta src/catalogar_imagens
- Mudar path das bases DDSM, CMMD e MIAS nos notebooks
- Rodar os notebooks


