import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence
from pydicom.valuerep import PersonName
from skimage.draw import polygon
import plistlib
import matplotlib.pyplot as plt
from pydicom import dcmread
from cv2 import imread, imwrite, COLOR_BGR2GRAY, INTER_AREA, cvtColor, resize


def load_json(path: str) -> object:
    """Carregar arquivos em JSON"""
    try:
        with open(path, "r") as json_file:
            return json.load(json_file)
    except json.decoder.JSONDecodeError:
        raise json.decoder.JSONDecodeError(
            f"Arquivo {path} não é um JSON válido"
        )  # noqa: E501
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo {path} não encontrado")


def save_json(path: str, data: list | dict) -> None:
    """Salvar arquivos em JSON"""
    try:
        with open(path, "w", encoding="utf-8") as json_file:  # noqa: E501
            json.dump(data, json_file, ensure_ascii=False, indent=3)
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo {path} não encontrado")


def get_dicom_meta_seq(seq_element):
    """Extrai metadados de elementos sequencia dicom"""
    dict_tags_seq = {
        "(0008, 0100)": "code_value",
        "(0008, 0102)": "coding_scheme_designator",
        "(0008, 0104)": "code_meaning",
        "(0054, 0222)": "View Modifier Code Sequence",
    }
    elements = []

    for element in seq_element:
        dict_temp = {}

        for key, value in element.to_json_dict().items():
            new_key = f"({key[:4:]}, {key[4::]})"

            if new_key in dict_tags_seq.keys():
                new_key = f"{dict_tags_seq[new_key]} {new_key}"
            dict_temp[new_key] = (
                value["Value"][0] if len(value["Value"]) >= 1 else ""
            )  # noqa: E501

        elements.append(dict_temp)
    return elements


def get_dicom_meta(dicom_file: object, drop=False) -> dict:
    """Extrair metadados em cabeçalhos dicom"""

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
    return (
        df.loc[df["frequencia"] == freq].copy(deep=True).reset_index(drop=True)
    )  # noqa: E501


def create_df(dictionary: dict, x_label: str) -> pd.DataFrame:
    """Cria um dataframe da frequência por tag/atribute"""
    keys = [key for key in dictionary.keys()]
    values = [value for value in dictionary.values()]

    df = pd.DataFrame({x_label: keys, "frequencia": values})
    return df


def get_images_size(
    path: str, image_type: str = "", multiple=False
) -> float | int:  # noqa: E501
    """Retorna o tamanho da imagem em MegaBytes (MB)"""

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


def get_angles_labels(angles):
    labels = []

    for angle in angles:
        if angle == 0:
            labels.append("0")
        elif angle == np.pi / 4:
            labels.append("45")
        elif angle == np.pi / 2:
            labels.append("90")
        elif angle == 3 * np.pi / 4:
            labels.append("135")
    return labels


def load_inbreast_mask(mask_path, imshape=(4084, 3328)):
    """
    This function loads a osirix xml region as a binary numpy array for
    INBREAST dataset
    @mask_path : Path to the xml file
    @imshape : The shape of the image as an array e.g. [4084, 3328]
    return: numpy array where positions in the roi are assigned a value of 1.
    """

    mask = np.zeros(imshape)
    with open(mask_path, "rb") as mask_file:
        plist_dict = plistlib.load(mask_file, fmt=plistlib.FMT_XML)["Images"][
            0
        ]  # noqa: E501
        numRois = plist_dict["NumberOfROIs"]
        rois = plist_dict["ROIs"]
        assert len(rois) == numRois
        for roi in rois:
            numPoints = roi["NumberOfPoints"]
            points = roi["Point_px"]
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


def get_fo_features(
    image: np.ndarray,
    mask: np.ndarray = None,
    features: list = [
        "mean",
        "variance",
        "std",
        "smoothness",
        "third_moment",
        "uniformity",
        "entropy"
    ],
) -> list:  # noqa: E501
    """Retorna as features de primeira ordem de uma imagem"""
    hist = None

    if mask is not None:
        if image.shape != mask.shape:
            raise ValueError("Dimensões da imagem e máscara são diferentes")
        hist = cv2.calcHist([image], [0], mask, [256], [0, 256])
    else:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    p_hist = hist / hist.sum()

    try:
        mean = sum(i * p for i, p in enumerate(p_hist))
        variance = sum(((i - mean) ** 2) * p for i, p in enumerate(p_hist))
        std = np.sqrt(variance)
        smoothness = 1 - 1 / (1 + variance)
        third_moment = sum(((i - mean) ** 3) * p for i, p in enumerate(p_hist))
        uniformity = sum(p_hist ** 2)
        entropy = -sum(p * np.log2(p) for p in p_hist if p != 0)

        dict_features = {
            "mean": mean[0],
            "variance": variance[0],  # noqa: E501
            "std": std[0],
            "smoothness": smoothness[0],
            "third_moment": third_moment[0],
            "uniformity": uniformity[0],
            "entropy": entropy[0]
        }

        return {feature: dict_features[feature] for feature in features}
    except ZeroDivisionError:
        raise ZeroDivisionError("Divisão por zero")
    except TypeError:
        raise TypeError("Tipo de dado inválido")
    except KeyError:
        raise KeyError("Feature inválida")


def draw_image_mias(df: pd.DataFrame, idx: int) -> None:
    """draw image with location of center of abnormality"""
    img = cv2.imread(df["image_path"][idx])
    plt.imshow(img, cmap="gray")

    #  account for horizontal flip of some images
    if idx % 2 == 0:
        x_loc = df.x_center_abnormality[idx]
    else:
        x_loc = 1024 - df.x_center_abnormality[idx]
    plt.plot([x_loc], [1024 - df.y_center_abnormality[idx]], "ro")
    radius = str(df.radius[idx]) if df.radius[idx] != "nan" else "N/A"
    plt.title("Radius:" + radius)
    plt.show()


def extract_image_dicom(image_path, save=False, path=None, image_type=None):
    try:
        # Read the dicom file
        dicom_file = dcmread(image_path)

        # Extract the image from dicom file
        image = dicom_file.pixel_array

        if save:
            image_type = image_type or "png"
            path = path or image_path.split("/")[-1].replace("dcm", image_type)
            imwrite(path, image)

        return image
    except FileNotFoundError:
        raise Exception("Arquivo não encontrado")


def resize_image(image: np.array, dim, save=False, path=None, image_type=None):
    try:
        if isinstance(image, str):
            image = imread(image)
            image = cvtColor(image, COLOR_BGR2GRAY)

        # resize image
        resized = resize(image, dim, interpolation=INTER_AREA)

        if save:
            image_type = image_type or "png"
            path = path or f"./resized_image.{image_type}"
            imwrite(path, resized)

        return resized
    except FileNotFoundError:
        raise Exception("Arquivo não encontrado")
    except Exception as e:
        raise Exception(e)


def extract_roi(image: np.array, mask: np.array) -> np.array:
    """
    Extrai a região de interesse de uma imagem
    @image: Imagem original
    @mask: Máscara da região de interesse
    return: Imagem com a região de interesse
    """

    if image.shape != mask.shape:
        raise ValueError("Dimensões da imagem e máscara são diferentes")

    result = np.copy(image)
    result[mask == 0] = 0

    return result
