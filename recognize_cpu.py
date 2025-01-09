import cv2
import os
import shutil # copying data
import numpy as np
import json
import torch
import torch.nn as nn

from datasets import Dataset, DatasetDict, Image # working with datasets
from torchvision.transforms import ColorJitter, Resize, Compose, PILToTensor # randomly modify brightness, contrast and colors of image
from transformers import SegformerImageProcessor # function from Hugging Face to modify images at the beggining for Segformer model
from functools import partial

from transformers import SegformerForSemanticSegmentation
from transformers import SegformerFeatureExtractor
from transformers import TrainingArguments, Trainer
from torchvision.transforms.functional import to_tensor

# Test function to display an image using OpenCV.
def show_image(image_path):
    image = cv2.imread(image_path)
    tgt_height = 300
    height, width = image.shape[:2]
    scale_factor = tgt_height / height
    tgt_width = int(width * scale_factor)
    resized_image = cv2.resize(image, (tgt_width, tgt_height))
    cv2.imshow("Image Viewer", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to copy files
def copy_files(source_path, file_list, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)  # Create the directory if it does not exist
    for file_name in file_list:
        src_path = os.path.join(source_path, file_name)
        dest_path = os.path.join(destination_dir, file_name)
        if not os.path.exists(dest_path):  # Check if file already exists
            shutil.copy(src_path, dest_path)

# Function to clear a directory
def clear_directory(directory_path):
    if os.path.exists(directory_path):
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove the file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove the folder
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

# Function to split photos between folders
def split_data(train_ratio, source_path, train_path, valid_path):
    # Clear existing data in train and valid directories
    clear_directory(train_path)
    clear_directory(valid_path)

    # Getting list of files
    image_files = [f for f in os.listdir(source_path) if f.endswith('.jpg')]

    # Counts of images after split
    total_images = len(image_files)
    train_count = int(total_images * train_ratio)
    valid_count = total_images - train_count

    # Split image paths
    train_files = image_files[:train_count]
    valid_files = image_files[train_count:train_count + valid_count]

    # Copy images
    copy_files(source_path, train_files, train_path)
    copy_files(source_path, valid_files, valid_path)

    # Show parameters
    print(f"Total images: {total_images}")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(valid_files)}")

# Function to convert polygons from JSON to jpg
def json_to_mask(json_folder, output_folder, image_shape):
    # Create the directory if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  

    # Clear existing data in train and valid directories
    clear_directory(output_folder)
    
    # List of all files
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

    # Iterate through JSONs
    for json_file in json_files:
        json_path = os.path.join(json_folder, json_file)

        # Open JSONs
        with open(json_path, 'r') as file:
            data = json.load(file)

        # Extract polygons
        # Format like that - {'objects': [{'points': {'exterior': [[x1, y1], [x2, y2], ...]}}, ...]}
        coordinates = []
        for obj in data.get('objects', []):
            if 'points' in obj and 'exterior' in obj['points']:
                coordinates.append(np.array(obj['points']['exterior'], dtype=np.int32))

        # White starting mask
        mask = np.ones(image_shape, dtype=np.uint8) * 255

        # Draw polygons on mask
        for polygon in coordinates:
            cv2.fillPoly(mask, [polygon], color=(0)) # Fulfill inside of polygon with black
            cv2.polylines(mask, [polygon], isClosed=True, color=(255), thickness=3)  # Draw white contur

        # Reverse colors
        mask = cv2.bitwise_not(mask)

        # Save masks
        mask_filename = os.path.splitext(os.path.splitext(json_file)[0])[0]
        mask_path = os.path.join(output_folder, mask_filename + ".jpg")  # Adding .jpg
        cv2.imwrite(mask_path, mask)

# Function to give list of file paths
def get_file_list(data_root_path, source_path): # idk if needed
    img_paths = sorted(os.listdir(os.path.join(data_root_path, source_path))) 
    img_paths = [os.path.join(data_root_path, source_path,img) for img in img_paths]
    return img_paths

# Function to give list of image and mask paths
def get_image_mask_paths(image_dir, mask_dir):
    image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
    mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(image_dir)])  # Maski nazywają się tak samo jak zdjęcia
    return image_paths, mask_paths

# Function to crate datasets using paths
def create_dataset(image_paths, mask_paths):
    dataset = Dataset.from_dict({"pixel_values": sorted(image_paths),
                                 "label": sorted(mask_paths)})
    dataset = dataset.cast_column("pixel_values", Image())
    dataset = dataset.cast_column("label", Image())
    return dataset

#  Definiowanie funkcji konwersji obrazów
def convert_to_rgb(image): # convert to rgb
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image
def convert_to_grayscale(image): # convert to grayyyyyscale
    if image.mode != 'L':
        return image.convert('L')
    return image
def convert_to_black_white(image, threshold=10): # convert to black/white with threshhold
    image = image.convert('L')
    bw = image.point(lambda x: 0 if x < threshold else 255, '1')
    return bw

# Funkcje to transform datasets
def train_transforms(example_batch):  
    feature_extractor = SegformerImageProcessor()
    jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1) # Narzędzie do augmentacji obrazów, które losowo modyfikuje jasność, kontrast, nasycenie i barwę obrazów. Służy do wzbogacania danych treningowych poprzez tworzenie większej różnorodności w danych
    images = [convert_to_rgb(jitter(x)) for x in example_batch["pixel_values"]] # conversion to RGB and augmentation
    labels = [convert_to_black_white(x) for x in example_batch["label"]] # to black&white with threshhold
    inputs = feature_extractor(images=images, segmentation_maps=labels)  # changes pictures to a model needed format
    return inputs

def val_transforms(example_batch):  # just like up, but whitout augmetation
    feature_extractor = SegformerImageProcessor()
    images = [convert_to_rgb(x) for x in example_batch["pixel_values"]]
    labels = [convert_to_black_white(x) for x in example_batch["label"]]
    inputs = feature_extractor(images=images, segmentation_maps=labels)  
    return inputs

'''
- preds: Przewidywane etykiety segmentacji (wyjścia modelu).
- labels: Prawdziwe etykiety segmentacyjne (maska segmentacji).
- num_labels: Liczba klas (etykiet), które są segmentowane (np. "tło", "zęby").
- ignore_index: Indeks klasy, którą należy zignorować podczas obliczeń (zwykle np. 255, jeśli oznacza brak danych w masce segmentacyjnej).
'''
# Funkcja do obliczenia wyniku Dice'a (2*TP w liczniku, 2*TP+FP+FN w mianowniku)
def dice_score(preds, labels, num_labels, ignore_index):
    dice_scores = []
    for label in range(num_labels):
        if label == ignore_index:
            continue
        pred_label = preds == label
        true_label = labels == label

        if true_label.sum() == 0 and pred_label.sum() == 0:
            dice_scores.append(0)
        else:
            dice = (2.0 * (pred_label & true_label).sum()) / (pred_label.sum() + true_label.sum())
            dice_scores.append(dice)

    return np.mean(dice_scores)

'''
- eval_pred: To wynik predykcji modelu, zazwyczaj para (logity, etykiety) używane podczas ewaluacji.
Logity to surowe wyjścia modelu (przed softmaxem), które zawierają wartości
prawdopodobieństwa dla każdej klasy, a etykiety to prawdziwe maski segmentacyjne.
'''


def main():
    # Check if GPU is available (i still dont use it i think)
    print(torch.cuda.is_available())  # Should print True if GPU is available
    print(torch.backends.cudnn.enabled)  # Should also print True

    # Call test function to display the image
    # show_image(r"C:/mgr/data/kakashi.png")
    
    # Splitting ratio
    train_ratio = 0.85
    valid_ratio = 0.15

    # Defining data folders
    data_root_path = r'C:/mgr/data'
    source_path = data_root_path + r'/Teeth Segmentation PNG/d2/img'
    train_path = data_root_path + r'/TRAIN_IMAGES'
    valid_path = data_root_path + r'/VALID_IMAGES'
    
    # Defining mask data
    masks_path = data_root_path + r'/MASKS'
    json_path = data_root_path + r'/Teeth Segmentation PNG/d2/ann'
    image_shape = (1024, 2041)
    
    # Split data into train and valid data
    #split_data(train_ratio, source_path, train_path, valid_path) # not needed to run everytime rn

    # Create masks
    #json_to_mask(json_path, masks_path, image_shape) # not needed to run everytime rn

    # List of files used to train
    image_paths = get_file_list(data_root_path, source_path)
    label_paths = get_file_list(data_root_path, masks_path)

    # Number of files used to train
    train_idx = int(len(image_paths)*train_ratio) # idk if needed - its possible to do that differenyly

    # Dictionary assigingin classes
    class_dict = {"background": 0, "teeth": 1} # background is 0, teeth has 1
    id2label = {idx: key for idx, key in enumerate(list(class_dict.keys()))} # reverse of class_dict (for model)
    label2id = {v: k for k, v in id2label.items()} # for model
    num_labels = len(id2label) # needed for compute_metrics
    
    # Splitting files to train and valid
    train_image_paths, train_mask_paths = get_image_mask_paths(train_path, masks_path)
    valid_image_paths, valid_mask_paths = get_image_mask_paths(valid_path, masks_path)

    # Creating datasets
    train_dataset = create_dataset(train_image_paths, train_mask_paths) 
    validation_dataset = create_dataset(valid_image_paths, valid_mask_paths)

    # Full dataset (train + validation)
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
    })

    # Helping variable to get to data faster
    train_ds = dataset["train"]
    valid_ds = dataset["validation"]

    # Define transformation (augmentation)
    feature_extractor = SegformerImageProcessor()  # ekstraktor cech (feature extractor) używany do przekształcania obrazów w format, który może być wprowadzony do modelu Segformer

    # Bind the feature extractor to transformations
    train_ds.set_transform(train_transforms)
    valid_ds.set_transform(val_transforms)

    # Setting transformation to datasets
    #train_ds.set_transform(train_transforms)
    #valid_ds.set_transform(val_transforms)

    # Wstępnie wytrenowany model, który chcemy załadować. W tym przypadku jest to model Segformer w wersji "nvidia/mit-b4", który jest dostępny w bibliotece Hugging Face
    # Oparty na architekturze MiT-B4 (Mixture of Transformers, wersja B4), który jest dobrze dostosowany do zadań segmentacji semantycznej
    pretrained_model_name = "nvidia/mit-b4"

    """
    pretrained_model_name: Nazwa modelu, który chcemy załadować, czyli "nvidia/mit-b4".
    id2label: To słownik mapujący numery ID na nazwy klas. Wcześniej w kodzie zdefiniowaliśmy ten słownik, np. {0: 'background', 1: 'teeth'}. Mówi on modelowi, co oznaczają poszczególne etykiety wyjściowe w zadaniu segmentacji (tło, ząb itp.).
    label2id: To odwrotny słownik, który mapuje nazwy klas na ich ID, np. {'background': 0, 'teeth': 1}. Pozwala modelowi na odpowiednią interpretację wyników.
    """

    # Załadowanie wstępnie wytrenowanego model Segformer do segmentacji semantycznej, dostosowany do konkretnego zadania poprzez mapowanie etykiet
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name,
        id2label=id2label,
        label2id=label2id
    )

    # Ścieżka do zapisywania tego co się nauczy - sama sie tworzy
    save_path = data_root_path + r'/segformer-teeth_segment_10ep_b4'

    # Wyłączenie integracji z Weights & Biases (W&B) (bez tego chciało ode mnie jakiś klucz czy coś)
    os.environ["WANDB_DISABLED"] = "true"

    # Definiowanie liczby epok
    n_epochs = 10

    """
    - output_dir=save_path: Miejsce, w którym będą zapisywane wyniki treningu (np. checkpointy, wytrenowany model). Wartość save_path to ścieżka zdefiniowana wcześniej (np. "segformer-teeth_segment_10ep_b4").
    - learning_rate=0.0001: Ustawia wartość współczynnika uczenia modelu (ang. learning rate). Mniejszy współczynnik uczenia pozwala modelowi uczyć się wolniej, ale bardziej precyzyjnie.
    - num_train_epochs=n_epochs: Liczba epok trenowania. Tutaj jest to 10 epok.
    - per_device_train_batch_size=2: Liczba próbek danych używanych na każde urządzenie (np. GPU) w jednej iteracji trenowania. Tutaj model używa 2 próbek na urządzenie.
    - per_device_eval_batch_size=2: Liczba próbek danych używanych podczas walidacji na każde urządzenie. Podobnie jak w trenowaniu, również tutaj wartość to 2.
    - save_total_limit=3: Limit liczby zapisanych checkpointów (modeli). Starsze checkpointy zostaną usunięte, aby oszczędzić miejsce.
    - evaluation_strategy="steps": Ewaluacja modelu odbywa się co pewną liczbę kroków (zamiast na końcu każdej epoki). To oznacza, że ewaluacja będzie przeprowadzana co eval_steps kroków.
    - save_strategy="steps": Podobnie jak ewaluacja, zapisywanie modeli (checkpointów) odbywa się co save_steps kroków.
    - save_steps=20: Model będzie zapisywany co 20 kroków podczas trenowania.
    - eval_steps=20: Model będzie ewaluowany co 20 kroków podczas trenowania.
    - logging_steps=10: Wyniki (logi) będą zapisywane co 10 krok, co zapewnia częste logowanie podczas trenowania.
    - eval_accumulation_steps=5: Wyniki ewaluacji są akumulowane przez 5 kroków przed wykonaniem obliczeń, co zmniejsza zużycie pamięci GPU.
    - remove_unused_columns=False: Ustawione na False, co oznacza, że kolumny z danych, które nie są bezpośrednio używane w trenowaniu, nie zostaną usunięte.
    - push_to_hub=False: Model nie zostanie automatycznie opublikowany na platformie Hugging Face Hub.
    - load_best_model_at_end=True: Po zakończeniu trenowania model zostanie załadowany z najlepszych checkpointów na podstawie wyników ewaluacji.
    """

    # Klasa, która pozwala na konfigurację parametrów trenowania modelu. Oto omówienie kluczowych argumentów:
    training_args = TrainingArguments(
        output_dir= save_path,
        learning_rate=0.0001,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        save_total_limit=3,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=20,
        eval_steps=20,
        logging_steps=10,
        eval_accumulation_steps=5,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        no_cuda=True,  # Force CPU if needed
    )

    '''
    - model=model: Model, który ma być trenowany. W tym przypadku jest to wcześniej załadowany Segformer.
    - args=training_args: Przekazuje argumenty trenowania, które zostały zdefiniowane wcześniej (np. liczba epok, współczynnik uczenia, częstotliwość zapisywania modelu).
    - train_dataset=train_ds: Zbiór danych treningowych (np. obrazy rentgenowskie zębów i ich maski).
    - eval_dataset=valid_ds: Zbiór danych walidacyjnych, na którym model będzie ewaluowany co określoną liczbę kroków.
    - compute_metrics=compute_metrics: Funkcja służąca do obliczania metryk ewaluacyjnych (np. wynik Dice’a). Funkcja compute_metrics była wcześniej omówiona, a jej zadaniem jest ocena jakości segmentacji modelu.
    '''
    # TROCHE NIE ROZUMIEM O CO CHODZI Z TĄ FUNKCJĄ NA DOLE
    # Ona jest wstawiona tu bo nie wiem jak przekazac num_labels inaczej
    # Funkcja, która ocenia model na danych walidacyjnych/testowych i oblicza metryki (w tym przypadku wynik Dice'a)
    def compute_metrics(eval_pred):
        with torch.no_grad():
            logits, labels = eval_pred
            logits_tensor = torch.from_numpy(logits)
            logits_tensor = nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)

            pred_labels = logits_tensor.detach().cpu().numpy()
            dice_scores = dice_score(pred_labels, labels, num_labels, ignore_index=255)

            return {"dice_score": dice_scores}
    
    # Klasa, która automatyzuje proces trenowania modelu w Hugging Face.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
    )

    # Uruchomienie procesu trenowania
    trainer.train()

    # Zapisanie ekstraktora cech
    feature_extractor.save_pretrained(save_path)
    model.save_pretrained(save_path)


if __name__ == "__main__":
    main()