# Carregar metadados no json
import os
import json
from pydicom.valuerep import PersonName
from pydicom.sequence import Sequence
from pydicom.multival import MultiValue
import re
import numpy

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
    return numpy.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width))), bytes.decode(header)
