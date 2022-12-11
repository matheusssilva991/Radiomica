# Carregar metadados no json
import os
import json
from pydicom.valuerep import PersonName
from pydicom.sequence import Sequence
from pydicom.multival import MultiValue
import re
import numpy as np
import pandas as pd

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
    dict_tags_seq = {"(0008, 0100)": "code_value",
                     "(0008, 0102)": "coding_scheme_designator",
                     "(0008, 0104)": "code_meaning",
                     "(0054, 0222)": "View Modifier Code Sequence"}
    elements = []
    
    for element in seq_element:
        dict_temp = {}
        
        for key, value in element.to_json_dict().items():
            new_key = f"({key[:4:]}, {key[4::]})"
            
            if new_key in dict_tags_seq.keys():
                new_key = f"{dict_tags_seq[new_key]} {new_key}"
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
        elif isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            value = float(value)
        elif value is None:
            value = ""
        
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



def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width))), bytes.decode(header)


def merge_dictionary(left_dict:dict, right_dict: dict) -> dict:
    """Junta dois dicionários de tags"""
    
    list_keys = list(left_dict.keys()) + list(right_dict.keys())
    dictionary_merged = {}

    for key in list_keys:
        if key in set(left_dict.keys()) & set(right_dict.keys()): # key contido (left^right)
            dictionary_merged[key] = left_dict[key] + right_dict[key]
        elif key in left_dict.keys(): # key contido em left
            dictionary_merged[key] = left_dict[key]
        else: # key contido em right
            dictionary_merged[key] = right_dict[key]
            
    return dictionary_merged

def get_bits_allocated(value: int) -> int:
    value = int(value)
    
    if 0 <= value < 256:
        return 8
    elif 256 <= value < 4096:
        return 12
    elif 4096 <= value < 65536:
        return 16
    
def create_dict_meta(metadata: list, type:str) -> dict:
    """Retorna um dicionário de tags contidas em uma lista de estudos
    
       metadata: lista contendo os estudos
       type: csv or txt
    """
    dictionary_metadata = {}
    dict_keys_to_rename = {
        "ID1": "Patient_id",
        "LeftRight": "Left or right breast",
        "abnormality": "Abnormality_type",
        "classification": "Pathology",
        "reference_number": "Patient_id",
        "laterality": "Left or right breast",
        "view": "Image View",
        'assessment': "Bi-Rads"
    }

    for current_meta in metadata: # Iterar sobre os estudos
        meta_csv_files = current_meta[f'metadata_{type}']
        
        for key, value in meta_csv_files.items():
            if key in dict_keys_to_rename.keys():
                key = dict_keys_to_rename[key]
                
            if key == 'image_path' or key == 'cropped_image_path' or key == 'original_image_path' or key == 'file_name':
                continue
            
            key = key.lower()
            
            if key not in dictionary_metadata.keys():
                if value in ['NaN', '']:
                    dictionary_metadata[key] = 0
                else:
                    dictionary_metadata[key] = 1
            else:
                if value not in ['NaN', '']:
                    dictionary_metadata[key] += 1
                    
    return dictionary_metadata

def buscar_tags(df: pd.DataFrame, freq: int) -> pd.DataFrame:
    """Retorna um DataFrame com as tags que contém a frequência informada"""
    return df.loc[df['frequencia'] == freq].copy(deep=True).reset_index(drop=True)

def create_df(dictionary: dict, x_label: str) -> pd.DataFrame: 
    """Cria um dataframe da frequência por tag/atribute"""
    keys = [key for key in dictionary.keys()]
    values = [value for value in dictionary.values()]

    df = pd.DataFrame({x_label: keys, "frequencia": values})
    return df