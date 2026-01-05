"""
Módulo de funções utilitárias para processamento de imagens médicas e análise radiômica.

Este módulo contém funções para:
- Manipulação de arquivos DICOM e extração de metadados
- Processamento e transformação de imagens médicas
- Extração de features radiômicas (first-order, GLCM)
- Manipulação de máscaras e ROI (Region of Interest)
- Visualização de dados e resultados
- Operações com arquivos JSON e DataFrames

Autor: Matheus Silva
Projeto: Radiomica - Análise de Imagens Médicas para Diagnóstico de Câncer de Mama
"""

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
    """
    Carrega e deserializa um arquivo JSON.

    Args:
        path (str): Caminho absoluto ou relativo do arquivo JSON.

    Returns:
        object: Conteúdo do arquivo JSON (dict ou list).

    Raises:
        json.decoder.JSONDecodeError: Se o arquivo não for um JSON válido.
        FileNotFoundError: Se o arquivo não for encontrado no caminho especificado.
    """
    try:
        with open(path, "r") as json_file:
            return json.load(json_file)
    except json.decoder.JSONDecodeError:
        raise json.decoder.JSONDecodeError(f"Arquivo {path} não é um JSON válido")  # noqa: E501
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo {path} não encontrado")


def save_json(path: str, data: list | dict) -> None:
    """
    Serializa e salva dados em um arquivo JSON.

    Args:
        path (str): Caminho onde o arquivo JSON será salvo.
        data (list | dict): Dados a serem salvos (lista ou dicionário).

    Raises:
        FileNotFoundError: Se o diretório do caminho não existir.

    Note:
        O arquivo é salvo com encoding UTF-8 e indentação de 3 espaços.
    """
    try:
        with open(path, "w", encoding="utf-8") as json_file:  # noqa: E501
            json.dump(data, json_file, ensure_ascii=False, indent=3)
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo {path} não encontrado")


def get_dicom_meta_seq(seq_element):
    """
    Extrai metadados de elementos de sequência DICOM.

    Processa elementos de sequência DICOM e converte suas tags e valores
    em um formato de dicionário legível.

    Args:
        seq_element: Elemento de sequência DICOM (pydicom.sequence.Sequence).

    Returns:
        list: Lista de dicionários contendo os metadados de cada elemento
              da sequência com tags formatadas e valores extraídos.

    Note:
        Tags conhecidas são mapeadas para nomes descritivos.
        Tags desconhecidas são mantidas no formato (xxxx, xxxx).
    """
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
            dict_temp[new_key] = value["Value"][0] if len(value["Value"]) >= 1 else ""  # noqa: E501

        elements.append(dict_temp)
    return elements


def get_dicom_meta(dicom_file: object, drop=False) -> dict:
    """
    Extrai todos os metadados de um arquivo DICOM.

    Lê o cabeçalho DICOM e extrai tags, nomes e valores, convertendo
    tipos especiais (PersonName, Sequence, MultiValue) para formatos Python.

    Args:
        dicom_file (object): Objeto DICOM lido com pydicom.dcmread().
        drop (bool, optional): Se True, remove tags com valores vazios.
                               Padrão é False.

    Returns:
        dict: Dicionário com metadados no formato {"nome_tag (xxxx, xxxx)": valor}.

    Note:
        - Pixel Data não é incluído nos metadados
        - Valores de bytes são decodificados para UTF-8
        - Valores numéricos numpy são convertidos para int/float Python
    """

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
    """
    Determina o número de bits alocados para um pixel com base em seu valor.

    Args:
        value (int): Valor do pixel.

    Returns:
        int: Número de bits necessários (8, 12, 14 ou 16).

    Note:
        Usado para determinar a profundidade de bits em imagens médicas.
    """
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
    return df.loc[df["frequencia"] == freq].copy(deep=True).reset_index(drop=True)  # noqa: E501


def create_df(dictionary: dict, x_label: str) -> pd.DataFrame:
    """
    Cria DataFrame de frequência a partir de um dicionário.

    Converte um dicionário de contagens/frequências em um DataFrame
    formatado para análise e visualização.

    Args:
        dictionary (dict): Dicionário onde chaves são itens e valores são frequências.
        x_label (str): Nome da coluna para as chaves do dicionário.

    Returns:
        pd.DataFrame: DataFrame com duas colunas: x_label e 'frequencia'.

    Example:
        >>> tags_count = {'tag1': 10, 'tag2': 5}
        >>> df = create_df(tags_count, 'tag_name')
    """
    keys = [key for key in dictionary.keys()]
    values = [value for value in dictionary.values()]

    df = pd.DataFrame({x_label: keys, "frequencia": values})
    return df


def get_images_size(path: str, image_type: str = "", multiple=False) -> float | int:  # noqa: E501
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
    """
    Converte ângulos em radianos para rótulos de graus.

    Args:
        angles (list): Lista de ângulos em radianos (0, π/4, π/2, 3π/4).

    Returns:
        list: Lista de strings com ângulos em graus ('0', '45', '90', '135').

    Note:
        Usado para rotular features GLCM calculadas em diferentes direções.
    """
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
    Carrega máscara de ROI do dataset INbreast a partir de arquivo XML OsiriX.

    Lê anotações XML do OsiriX e converte regiões de interesse (ROI) em
    uma máscara binária numpy.

    Args:
        mask_path (str): Caminho para o arquivo XML de anotação.
        imshape (tuple, optional): Dimensões da imagem (altura, largura).
                                   Padrão é (4084, 3328) para INbreast.

    Returns:
        np.ndarray: Máscara binária onde pixels da ROI têm valor 1.

    Note:
        - Suporta múltiplas ROIs em um mesmo arquivo
        - Para ROIs com ≤2 pontos, marca pixels individuais
        - Para ROIs com >2 pontos, preenche polígono
        - Específico para o formato XML do INbreast dataset

    Example:
        >>> mask = load_inbreast_mask('anotacao.xml', imshape=(4084, 3328))
    """

    mask = np.zeros(imshape)
    with open(mask_path, "rb") as mask_file:
        plist_dict = plistlib.load(mask_file, fmt=plistlib.FMT_XML)["Images"][0]  # noqa: E501
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
                # Coordenada x é a coluna e y é a linha na imagem
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
        "entropy",
    ],
) -> list:  # noqa: E501
    """
    Calcula features radiômicas de primeira ordem (First-Order Features).

    Extrai características estatísticas do histograma de intensidade de pixels,
    opcionalmente aplicando uma máscara para calcular apenas em uma ROI.

    Args:
        image (np.ndarray): Imagem em escala de cinza (2D numpy array).
        mask (np.ndarray, optional): Máscara binária para delimitar ROI.
                                     Deve ter mesmas dimensões da imagem.
        features (list, optional): Lista de features a extrair. Opções:
            - 'mean': Intensidade média
            - 'variance': Variância da intensidade
            - 'std': Desvio padrão
            - 'smoothness': Suavidade da textura
            - 'third_moment': Terceiro momento (assimetria)
            - 'uniformity': Uniformidade da distribuição
            - 'entropy': Entropia (complexidade da textura)

    Returns:
        dict: Dicionário com features calculadas {nome_feature: valor}.

    Raises:
        ValueError: Se dimensões de imagem e máscara forem diferentes.
        ZeroDivisionError: Se o histograma for inválido.
        TypeError: Se tipos de dados forem inválidos.
        KeyError: Se feature solicitada não existir.

    Example:
        >>> features = get_fo_features(image, mask, ['mean', 'entropy'])
        >>> print(features['mean'])  # 127.5
    """
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
        uniformity = sum(p_hist**2)
        entropy = -sum(p * np.log2(p) for p in p_hist if p != 0)

        dict_features = {
            "mean": mean[0],
            "variance": variance[0],  # noqa: E501
            "std": std[0],
            "smoothness": smoothness[0],
            "third_moment": third_moment[0],
            "uniformity": uniformity[0],
            "entropy": entropy[0],
        }

        return {feature: dict_features[feature] for feature in features}
    except ZeroDivisionError:
        raise ZeroDivisionError("Divisão por zero")
    except TypeError:
        raise TypeError("Tipo de dado inválido")
    except KeyError:
        raise KeyError("Feature inválida")


def draw_image_mias(df: pd.DataFrame, idx: int) -> None:
    """
    Visualiza imagem MIAS com marcação do centro de anormalidade.

    Plota imagem do dataset MIAS marcando o centro da lesão com um ponto
    vermelho e exibindo o raio da anormalidade no título.

    Args:
        df (pd.DataFrame): DataFrame MIAS com colunas:
            - 'image_path': caminho da imagem
            - 'x_center_abnormality': coordenada x do centro
            - 'y_center_abnormality': coordenada y do centro
            - 'radius': raio da anormalidade
        idx (int): Índice da linha no DataFrame a visualizar.

    Note:
        - Corrige inversão horizontal de imagens ímpares
        - Exibe 'N/A' se raio não estiver disponível
        - Usa matplotlib.pyplot.show() para exibir

    Example:
        >>> draw_image_mias(df_mias, idx=42)
    """
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
    """
    Extrai array de pixels de um arquivo DICOM.

    Lê arquivo DICOM e extrai a imagem como numpy array, opcionalmente
    salvando em formato de imagem comum (PNG, JPG, etc.).

    Args:
        image_path (str): Caminho do arquivo DICOM.
        save (bool, optional): Se True, salva a imagem extraída. Padrão é False.
        path (str, optional): Caminho para salvar a imagem. Se None, usa nome do DICOM.
        image_type (str, optional): Formato da imagem ('png', 'jpg', etc.).
                                    Padrão é 'png'.

    Returns:
        np.ndarray: Array 2D com os pixels da imagem.

    Raises:
        FileNotFoundError: Se arquivo DICOM não for encontrado.

    Example:
        >>> img = extract_image_dicom('imagem.dcm', save=True, image_type='png')
    """
    try:
        # Lê o arquivo DICOM
        dicom_file = dcmread(image_path)

        # Extrai o array de pixels do arquivo DICOM
        image = dicom_file.pixel_array

        if save:
            image_type = image_type or "png"
            path = path or image_path.split("/")[-1].replace("dcm", image_type)
            imwrite(path, image)

        return image
    except FileNotFoundError:
        raise Exception("Arquivo não encontrado")


def resize_image(image: np.array, dim, save=False, path=None, image_type=None):
    """
    Redimensiona imagem para dimensões especificadas.

    Args:
        image (np.array | str): Array numpy da imagem ou caminho do arquivo.
        dim (tuple): Dimensões de saída (largura, altura).
        save (bool, optional): Se True, salva a imagem redimensionada.
        path (str, optional): Caminho para salvar. Padrão é './resized_image.png'.
        image_type (str, optional): Formato do arquivo ('png', 'jpg'). Padrão é 'png'.

    Returns:
        np.ndarray: Imagem redimensionada.

    Raises:
        FileNotFoundError: Se caminho da imagem não for encontrado.
        Exception: Para outros erros durante o processamento.

    Note:
        Usa interpolação INTER_AREA, ideal para redução de tamanho.
        Se image for string, converte automaticamente para escala de cinza.

    Example:
        >>> img_resized = resize_image('imagem.png', (512, 512), save=True)
    """
    try:
        if isinstance(image, str):
            # Carrega imagem do caminho e converte para escala de cinza
            image = imread(image)
            image = cvtColor(image, COLOR_BGR2GRAY)

        # Redimensiona imagem usando interpolação por área
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


def plot_history(history):
    """
    Visualiza histórico de treinamento de modelo de Machine Learning.

    Plota gráficos de acurácia e perda (loss) para conjunto de treino
    e validação ao longo das épocas.

    Args:
        history: Objeto History retornado por model.fit() do Keras/TensorFlow.
                 Deve conter: 'accuracy', 'val_accuracy', 'loss', 'val_loss'.

    Returns:
        None: Exibe gráficos usando matplotlib.pyplot.show().

    Note:
        - Subplot 1: Acurácia (treino vs validação)
        - Subplot 2: Erro/Loss (treino vs validação)
        - Útil para detectar overfitting e avaliar convergência

    Example:
        >>> history = model.fit(X_train, y_train, validation_data=(X_val, y_val))
        >>> plot_history(history)
    """

    _, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Cria subplot de acurácia
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="valid accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # Cria subplot de erro/perda
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="valid error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()
