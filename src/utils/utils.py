# Carregar metadados no json
import os
import json
from pydicom.valuerep import PersonName
from pydicom.sequence import Sequence
from pydicom.multival import MultiValue

def load_json(object_name: str, path = None) -> None | object:
    """Carregar arquivos em JSON"""
    
    if path is None:
        path = os.getcwd().replace("\\", "/")   
    path = path + f"/{object_name}.json"
    
    try:
        with open(path, 'r') as json_file:
            return json.load(json_file)  
    except json.decoder.JSONDecodeError:
        print("error")
        return None
    except FileNotFoundError:
        #with open(path, 'w', encoding='utf-8') as json_file:
            return None
        

def save_json(object_name: str, data: list | dict, path = None) -> None:
    """Salvar arquivos em JSON"""
    
    if path is None:
        path = os.getcwd().replace("\\", "/")
        
    with open(path + f"/{object_name}.json", 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=3)
 

def get_dicom_meta_seq(seq_element):
    """Extrai metadados de elementos sequencia dicom"""
    
    elements = []
    
    for element in seq_element:
        dict_temp = {}
        
        for key, value in element.to_json_dict().items():
            new_key = f"({key[:4:]}, {key[4::]})"
            dict_temp[new_key] = value['Value'][0] if len(value['Value']) >= 1 else ""
            
        elements.append(dict_temp)
    return elements
        

def get_dicom_meta(dicom_file: object, drop=False) -> dict:
    """Extrair metadados em cabeçalhos dicom """  
    
    dictionary = {}

    for data_element in dicom_file:
        if data_element.description() in ["Pixel Array", "Pixel Data"]:
            continue
        elif drop and data_element.value == "":
            continue
    
        tag = data_element.tag
        tag_name = data_element.description()
        tag_name = tag_name.replace(" ", "_").lower()
        
        if isinstance(data_element.value, PersonName):
            value = "^".join(data_element.value.components)
        elif isinstance(data_element.value, Sequence):
            value = get_dicom_meta_seq(data_element.value)
        elif isinstance(data_element.value, MultiValue):
            value = []
            
            for element in data_element.value:
                value.append(str(element))
            value = " , ".join(value)
        else:
            value = data_element.value
            
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        
        dictionary[f"{tag_name} {tag}"] = value
    
    return dictionary


def study_factory(study_name: str, metadata_csv: dict, metadata_dicom_files: list) -> dict:
    """Gerar dicionario de estudo"""
    
    return {'study_name': study_name,
            'metadata_csv': metadata_csv,
            'metadata_dicom_files':metadata_dicom_files
            }
    

def update_count_tag(list_tags: list | set, dictionary_tags: dict) -> None:
    """Itera sobre uma lista de tags e adiciona no dicionário se não estiver
    nele ou aumentar o contador se estiver nele"""
    
    for key in list_tags:
        if key in dictionary_tags.keys():
            dictionary_tags[key] += 1
        else:
            dictionary_tags[key] = 1
        
def preprocessing_path(path: str) -> str:
    path = path.split("/")
    path = path[0]
    
    return path