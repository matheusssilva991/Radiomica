# <strong>Radiomica</strong> 

## <strong>Projeto de iniciação ciêntifica</strong> 
<strong>Tema:</strong> Técnicas de armazenamento de dados, processamento de imagens e inteligência artificial aplicadas no aúxilio ao diagnóstico do câncer de mama.<br>
<strong>Estado:</strong> Em desenvolvimento

<hr>

### <strong>1ª Etapa - Catalogação das Imagens</strong> 

#### <strong>Extração de metadados nos arquivos CSV e Dicom</strong>
- Foram extraidos os metadados dos arquivos CSV e dos cabeçalhos dos arquivos Dicom das bases de dados CMMD e DDSM.
- Os metadados foram salvos em quatro arquivos para a base DDSM e em um arquivo para a base CMMD, localizados nas pastas metadata/DDSM/metadata_csv_and_dicom e metadata/CMMD.

#### <strong>Dicionário de tags Dicom</strong>
- Foram extraidos todas as tags presentes nos cabeçalhos dos arquivos dicom, contendo o nome da tag e a quantidade de vezes que a tag foi utilizada em cada base de dados.
- Foram feitos três dicionários de tags, sendo um deles para a base CMMD, um para a base DDSM e o último sendo a junção dos dois primeiros, e estão localizados nas pastas metadata/DDSM, metadata/CMMD e metadata, respectivamente.

<hr>

#### <strong>Quick Start - Executar Catalogação</strong> 
- Abrir os notebooks catalogar_dicom_imagens e dicionario_meta na pasta src/catalogar_imagens
- Mudar path das bases DDSM e CMMD nos notebooks
- Rodar os notebooks


