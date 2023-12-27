# Carregar metadados no json
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence
from pydicom.valuerep import PersonName
from copy import deepcopy
from skimage.feature import graycomatrix, graycoprops
from skimage.draw import polygon
import plistlib
import matplotlib.pyplot as plt


def load_json(path: str) -> object:
    """Carregar arquivos em JSON"""
    try:
        with open(path, 'r') as json_file:
            return json.load(json_file)
    except json.decoder.JSONDecodeError:
        raise json.decoder.JSONDecodeError(f"Arquivo {path} não é um JSON válido")  # noqa: E501
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo {path} não encontrado")


def save_json(path: str, data: list | dict) -> None:
    """Salvar arquivos em JSON"""
    try:
        with open(path, 'w', encoding='utf-8') as json_file:  # noqa: E501
            json.dump(data, json_file, ensure_ascii=False, indent=3)
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo {path} não encontrado")


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
        paths_images.sort()
    else:
        paths_images = [path]

    images_size = []
    for path_image in paths_images:
        try:
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
        "view": "imageView",
        "patient_id": "patientId",
        "left_or_right_breast": "leftOrRightBreast",
        "abnormality_type": "abnormalityType",
        "image_view": "imageView",
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
    glcm_props = [propery for name in properties for propery in graycoprops(glcm, name)]  # noqa: E501

    for glcm_props_distance in glcm_props:
        for item in glcm_props_distance:
            glcm_features.append(item)

    return glcm_features


def load_inbreast_mask(mask_path, imshape=(4084, 3328)):
    """
    This function loads a osirix xml region as a binary numpy array for
    INBREAST dataset
    @mask_path : Path to the xml file
    @imshape : The shape of the image as an array e.g. [4084, 3328]
    return: numpy array where positions in the roi are assigned a value of 1.
    """

    mask = np.zeros(imshape)
    with open(mask_path, 'rb') as mask_file:
        plist_dict = plistlib.load(mask_file, fmt=plistlib.FMT_XML)['Images'][0]  # noqa: E501
        numRois = plist_dict['NumberOfROIs']
        rois = plist_dict['ROIs']
        assert len(rois) == numRois
        for roi in rois:
            numPoints = roi['NumberOfPoints']
            points = roi['Point_px']
            assert numPoints == len(points)
            points = [eval(point) for point in points]
            if len(points) <= 2:
                for point in points:
                    mask[int(point[1]), int(point[0])] = 1
            else:
                x, y = zip(*points)
                # x coord is the column coord in an image and y is the row
                col, row = np.array(x), np.array(y)
                poly_x, poly_y = polygon(row, col, shape=imshape)
                mask[poly_x, poly_y] = 1
    return mask


def get_first_order_features(image: np.ndarray) -> list:  # noqa: E501
    """Retorna as features de primeira ordem de uma imagem"""
    features = []

    # Mean
    mean = np.mean(image)
    features.append(mean)

    # Variance
    variance = np.var(image)
    features.append(variance)

    # Standard deviation
    stddev = np.std(image)
    features.append(stddev)
    # Skewness
    # Check for invalid values before calculating skewness
    if stddev != 0:
        skewness = (np.mean((image - mean)**3) / stddev**3)
    else:
        skewness = 0  # Assign a default value if stddev is zero
    features.append(skewness)

    # Check for invalid values before calculating kurtosis
    if stddev != 0:
        kurtosis = (np.mean((image - mean)**4) / stddev**4) - 3
    else:
        kurtosis = 0  # Assign a default value if stddev is zero
    features.append(kurtosis)

    return features


def draw_image_mias(df: pd.DataFrame, idx: int) -> None:
    ''' draw image with location of center of abnormality '''
    img = cv2.imread(df['image_path'][idx])
    plt.imshow(img, cmap='gray')

    #  account for horizontal flip of some images
    if idx % 2 == 0:
        x_loc = df.x_center_abnormality[idx]
    else:
        x_loc = 1024 - df.x_center_abnormality[idx]
    plt.plot([x_loc], [1024-df.y_center_abnormality[idx]], 'ro')
    radius = str(df.radius[idx]) if df.radius[idx] == 'nan' else "N/A"
    plt.title("Radius:" + radius)
    plt.show()
