# Carregar metadados no json
import json
import os
import re
from pathlib import Path
import sys  # noqa: 

import cv2
import numpy as np
import pandas as pd
from pydicom import dcmread
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence
from pydicom.valuerep import PersonName
from copy import deepcopy
from skimage.feature import graycomatrix, graycoprops


def load_json(object_name: str, path=None) -> None | object:
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
        # with open(path, 'w', encoding='utf-8') as json_file:
        return None


def save_json(object_name: str, data: list | dict, path=None) -> None:
    """Salvar arquivos em JSON"""

    if path is None:
        path = os.getcwd().replace("\\", "/")

    with open(path + f"/{object_name}.json", 'w', encoding='utf-8') as json_file:  # noqa: E501
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
            dict_temp[new_key] = value['Value'][0] if len(value['Value']) >= 1 else ""  # noqa: E501

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


def study_factory(study_name: str, metadata_csv: dict, metadata_dicom_files: list) -> dict:  # noqa: E501
    """Gerar dicionario de estudo"""

    return {'study_name': study_name,
            'metadata_csv': metadata_csv,
            'metadata_dicom_files': metadata_dicom_files
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
            b"(^P5\s(?:\s*#.*[\r\n])*"  # noqa: W605
            b"(\d+)\s(?:\s*#.*[\r\n])*"  # noqa: W605
            b"(\d+)\s(?:\s*#.*[\r\n])*"  # noqa: W605
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()  # noqa: W605
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                         dtype='u1' if int(maxval) < 256 else byteorder+'u2',  # noqa: E501
                         count=int(width)*int(height),
                         offset=len(header)
                         ).reshape((int(height), int(width))), bytes.decode(header)  # noqa: E501


def merge_dictionary(left_dict: dict, right_dict: dict) -> dict:
    """Junta dois dicionários de tags"""

    list_keys = list(left_dict.keys()) + list(right_dict.keys())
    dictionary_merged = {}

    for key in list_keys:
        if key in set(left_dict.keys()) & set(right_dict.keys()):  # key contido (left^right)  # noqa: E501
            dictionary_merged[key] = left_dict[key] + right_dict[key]
        elif key in left_dict.keys():  # key contido em left
            dictionary_merged[key] = left_dict[key]
        else:  # key contido em right
            dictionary_merged[key] = right_dict[key]

    return dictionary_merged


def get_bits_allocated(value: int) -> int:
    value = int(value)

    if 0 <= value < 256:
        return 8
    elif 256 <= value < 4096:
        return 12
    elif 4096 <= value < 16.383:
        return 14
    elif 16.383 <= value < 65536:
        return 16


def create_dict_meta(metadata: list, type: str) -> dict:
    """Retorna um dicionário de tags contidas em uma lista de estudos

       metadata: lista contendo os estudos
       type: csv or txt
    """
    dictionary_metadata = {}
    dict_keys_to_rename = {
        "id1": "patient_id",
        "leftright": "left_or_right_breast",
        "abnormality": "abnormality_type",
        "classification": "pathology",
        "reference_number": "patient_id",
        "laterality": "left_or_right breast",
        "view": "image_view",
        "assessment": "bi-rads",
        "age": "patient_age",
        "acr": "breast_density"
    }

    for current_meta in metadata:  # Iterar sobre os estudos
        meta_csv_files = current_meta[f'metadata_{type}']

        for key, value in meta_csv_files.items():
            key = key.lower().replace(" ", "_")

            if key in dict_keys_to_rename.keys():
                key = dict_keys_to_rename[key]

            if key == 'image_path' or key == 'cropped_image_path' or key == 'original_image_path' or key == 'file_name':  # noqa: E501
                continue

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
    return df.loc[df['frequencia'] == freq].copy(deep=True).reset_index(drop=True)  # noqa: E501


def create_df(dictionary: dict, x_label: str) -> pd.DataFrame: 
    """Cria um dataframe da frequência por tag/atribute"""
    keys = [key for key in dictionary.keys()]
    values = [value for value in dictionary.values()]

    df = pd.DataFrame({x_label: keys, "frequencia": values})
    return df


def get_images_size(path: str, image_type: str = "", multiple=False) -> float | int:  # noqa: E501
    """ Retorna o tamanho da imagem em MegaBytes (MB)"""

    if multiple:
        directory = Path(path)
        paths_images = list(directory.rglob(f"*.{image_type}*"))
    else:
        paths_images = [path]

    images_size = []
    for path_image in paths_images:
        try:
            image = []
            if image_type.lower() == "dcm":
                image = dcmread(path_image).pixel_array
            else:
                image = cv2.imread(str(path_image))

            images_size.append(round(image.nbytes / 1000000, 2))
        except FileNotFoundError:
            return None
        except AttributeError:
            return None

    if multiple:
        return images_size
    else:
        return images_size[0]


def rename_keys(dictionary):
    dict_keys_to_rename = {
        "id1": "patientId",
        "id": "patientId",
        "image_name": "patientId",
        "abnormality_id": "abnormalityId",
        "assessment": "biRads",
        "age": "patientAge",
        "acr": "breastDensity",
        "leftright": "leftOrRightBreast",
        "abnormality": "abnormalityType",
        "classification": "pathology",
        "reference_number": "patientId",
        "laterality": "leftOrRightBreast",
        "view": "orientation",
        "patient_id": "patientId",
        "left_or_right_breast": "leftOrRightBreast",
        "abnormality_type": "abnormalityType",
        "image_view": "orientation",
        "bi_rads": "biRads",
        "bi-rads": "biRads",
        "patient_age": "patientAge",
        "breast_density": "breastDensity",
        "calc_type": "calcificationType",
        "calc_distribution": "calcificationDistribution",
        "mass_shape": "massShape",
        "mass_margins": "massMargins",
        "cropped_image_path": "croppedImagePath",
        "cropped_path": "croppedImagePath",
        "image_path": "imagePath",
        "findings_notes": "findingsNotes",
        "acquisition_date": "acquisitionDate",
        "file_name": "fileName",
        "x_centre_abnormality": "xCentreAbnormality",
        "y_centre_abnormality": "yCentreAbnormality",
        'background_tissue': 'backgroundTissue',
        "image_size_mb": "imageSizeMb",
    }

    keys_order = list(dictionary.keys())

    for i, key in enumerate(dictionary.keys()):
        key_lower = deepcopy(key)
        key_lower = key_lower.lower().replace(" ", "_")

        if key_lower in dict_keys_to_rename.keys():
            keys_order[i] = dict_keys_to_rename[key_lower]  # noqa: E501

    new_dict = {}
    for key_order, key in zip(keys_order, dictionary.keys()):
        new_dict[key_order] = dictionary[key]  # noqa: E501

    return new_dict


def get_angles_labels(angles):
    labels = []

    for angle in angles:
        if angle == 0:
            labels.append('0')
        elif angle == np.pi/4:
            labels.append('45')
        elif angle == np.pi/2:
            labels.append('90')
        elif angle == 3*np.pi/4:
            labels.append('135')
    return labels


def get_glcm_features(image, distances, angles, levels, symmetric, normed, properties):  # noqa: E501

    glcm = graycomatrix(image, distances=distances, angles=angles, levels=levels,  # noqa: E501
                        symmetric=symmetric, normed=normed)

    glcm_features = []
    # manual_properties = ['mean', 'skewness', 'kurtosis']
    # tmp_properties = set(properties) - set(manual_properties)

    # glcm_props = [propery for name in tmp_properties for propery in graycoprops(glcm, name)]  # noqa: E501
    glcm_props = [propery for name in properties for propery in graycoprops(glcm, name)]  # noqa: E501

    for glcm_props_distance in glcm_props:
        for item in glcm_props_distance:
            glcm_features.append(item)

    """ for i in range(glcm.shape[2]):
        for j in range(glcm.shape[3]):
            if 'mean' in properties:
                mean = np.mean(glcm[::, ::, i, j])
                glcm_features.append(mean)

            std = np.std(glcm[::, ::, i, j])

            if 'standard deviation' in properties:
                glcm_features.append(std)

            if 'skewness' in properties:
                skewness = np.mean((glcm[::, ::, i, j] - mean) ** 3) / (std ** 3)  # noqa: E501
                glcm_features.append(skewness)

            if 'kurtosis' in properties:
                kurtosis = np.mean((glcm[::, ::, i, j] - mean) ** 4) / (std ** 4) - 3  # noqa: E501
                glcm_features.append(kurtosis) """

    return glcm_features
