# Carregar metadados no json
import os
import json

# Carregar arquivos em JSON
def load_json(object_name: str, path = None) -> None | object:
    if path is None:
        path = os.getcwd().replace("\\", "/")   
    path = path + f"/{object_name}"
    
    try:
        with open(path, 'r') as json_file:
            return json.load(json_file)  
    except json.decoder.JSONDecodeError:
        return None
    except FileNotFoundError:
        with open(path, 'w', encoding='utf-8') as json_file:
            return None
        
# Salvar arquivos em JSON
def save_json(object_name: str, data: list | dict, path = None) -> None:
    if path is None:
        path = os.getcwd().replace("\\", "/")
        
    with open(path + f"/{object_name}.json", 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=3)
 
# Extrair metadados em cabeçalhos dicom       
def get_dicom_meta(dicom_file: object, drop=False) -> dict:
    dictionary = {}

    for data_element in dicom_file:
        if data_element.description() in ["Pixel Array", "Pixel Data"]:
            continue
        if drop and data_element.value == "":
            continue
    
        tag = data_element.tag
        tag_name = data_element.description()
        tag_name = tag_name.replace(" ", "_").lower()
        
        if tag_name in ["patient's_name", "referring_physician's_name"]:
            value = "^".join(data_element.value.components)
        else:
            value = data_element.value
            
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        
        dictionary[f"{tag_name} {tag}"] = value
    
    return dictionary
        
def preprocessing_path(path: str) -> str:
    path = path.split("/")
    path = path[0]
    
    return path