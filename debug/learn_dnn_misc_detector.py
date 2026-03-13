import copy
import os
import sys

import pandas as pd
from torchvision import transforms

from GenericDataset import *
from plmodels import *

from sprout.SPROUTObject import SPROUTObject
from sprout.classifiers.Classifier import get_classifier_name
from sprout.utils.sprout_utils import build_SPROUT_dataset
from generate_SCC_outputs import generate_scc_outputs

# Needed by SPROUT combined/multicombined in your environment
sys.path.append('/home/fahadk/anaconda3/envs/SPROUT/lib/python3.8/site-packages/confens/')

MODELS_FOLDER = "../models/"
TMP_FOLDER = "tmp"
TRAIN_DATA_FOLDER = "/home/users/muhammad.atif/tt100k_50_30_20"

NUM_CLASSES = 9
MAX_EPOCHS = 50


def get_dnn_classifiers():
    models = []
    model_name = ['EfficientNet_B0', 'VGG11', 'DenseNet121', 'GoogLeNet', 'Inception_V3', 'ResNet50', 'AlexNet', 'ConvNeXt_Base', 'MobileNet_V3_Large', 'ShuffleNet_V2_X1_0']
    for m in model_name:
        models.append(ImageClassifier(m, num_classes=NUM_CLASSES, learning_rate=1e-4, max_epochs=MAX_EPOCHS))
    return models


def get_list_del_classifiers():
    models = []
    model_name = ['VGG11', 'DenseNet121', 'GoogLeNet', 'Inception_V3', 'ResNet50', 'AlexNet']
    for m in model_name:
        models.append(ImageClassifier(model_name=m, num_classes=NUM_CLASSES, learning_rate=1e-4, max_epochs=MAX_EPOCHS))
    return models


def read_image_dataset(dataset_file):
    """
    Returns:
      train_loader, test_loader, val_loader,
      y_test, y_val,
      label_tags
    """
    train_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    custom_data = GenericDatasetLoader(dataset_name=dataset_file, root_dir=TRAIN_DATA_FOLDER, batch_size=32)

    train_loader = custom_data.create_dataloader(split='train', transform=train_transform, shuffle=True)
    val_loader = custom_data.create_dataloader(split='val', transform=eval_transform, shuffle=False)
    test_loader = custom_data.create_dataloader(split='test', transform=eval_transform, shuffle=False)

    y_val = custom_data.extract_labels(val_loader)
    y_test = custom_data.extract_labels(test_loader)

    label_tags = train_loader.dataset.classes
    global NUM_CLASSES
    NUM_CLASSES = len(label_tags)

    return train_loader, test_loader, val_loader, y_test, y_val, label_tags


def list_image_datasets():
    return ['CUSTOM']


def build_supervised_object(x_train, y_train, val_train, label_tags):
    sp_obj = SPROUTObject(models_folder=MODELS_FOLDER)
    classifier = get_list_del_classifiers()

    sp_obj.add_calculator_maxprob()
    sp_obj.add_calculator_entropy(n_classes=len(label_tags))
    sp_obj.add_calculator_recloss(x_train=x_train, val_train=val_train, num_classes=len(label_tags))

    sp_obj.add_calculator_combined(
        classifier=classifier[0],
        x_train=x_train,
        y_train=y_train,
        val_train=val_train,
        n_classes=len(label_tags)
    )

    sp_obj.add_calculator_multicombined(
        clf_set=classifier,
        x_train=x_train,
        y_train=y_train,
        val_train=val_train,
        n_classes=len(label_tags)
    )

    return sp_obj


def compute_datasets_uncertainties():
    for dataset_file in list_image_datasets():
        if (dataset_file is None) or len(dataset_file) == 0:
            print("Error while processing the dataset")
            continue

        print("Processing Dataset " + dataset_file + "'")

        train_loader, test_loader, val_loader, y_test, y_val, label_tags = read_image_dataset(dataset_file)

        print("Preparing Uncertainty Calculators...")
        sp_obj = build_supervised_object(train_loader, test_loader, val_loader, label_tags)

        for classifier in get_dnn_classifiers():
            sprout_obj = copy.deepcopy(sp_obj)

            # Train model (train + val)
            classifier.fit(train_dataloader=train_loader, val_dataloader=val_loader)

            # ---------------- VAL CSV ----------------
            y_proba_val = classifier.predict_proba(val_loader)
            y_pred_val = classifier.predict(val_loader)

            out_df_val = build_SPROUT_dataset(y_proba_val, y_pred_val, y_val, label_tags)
            q_df_val = sprout_obj.compute_set_trust(data_set=val_loader, classifier=classifier, y_proba=y_proba_val)
            out_df_val = pd.concat([out_df_val, q_df_val], axis=1)

            file_out_val = os.path.join(
                TMP_FOLDER, dataset_file + "_" + get_classifier_name(classifier.model) + "_VAL.csv"
            )
            os.makedirs(os.path.dirname(file_out_val), exist_ok=True)
            out_df_val.to_csv(file_out_val, index=False)
            print("File '" + file_out_val + "' Printed")

            # ---------------- TEST CSV ----------------
            y_proba_test = classifier.predict_proba(test_loader)
            y_pred_test = classifier.predict(test_loader)

            out_df_test = build_SPROUT_dataset(y_proba_test, y_pred_test, y_test, label_tags)
            q_df_test = sprout_obj.compute_set_trust(data_set=test_loader, classifier=classifier, y_proba=y_proba_test)
            out_df_test = pd.concat([out_df_test, q_df_test], axis=1)

            file_out_test = os.path.join(
                TMP_FOLDER, dataset_file + "_" + get_classifier_name(classifier.model) + "_TEST.csv"
            )
            out_df_test.to_csv(file_out_test, index=False)
            print("File '" + file_out_test + "' Printed")
def combine_val_test_files():
    """
    Combine *_VAL.csv and *_TEST.csv into one file per classifier.
    VAL rows first, then TEST rows.
    """

    print("\nCombining VAL and TEST CSVs...")

    for file in os.listdir(TMP_FOLDER):

        if file.endswith("_VAL.csv"):

            base = file.replace("_VAL.csv", "")
            val_path = os.path.join(TMP_FOLDER, base + "_VAL.csv")
            test_path = os.path.join(TMP_FOLDER, base + "_TEST.csv")

            if not os.path.exists(test_path):
                print("Missing TEST file for", base)
                continue

            out_path = os.path.join(TMP_FOLDER, base + ".csv")

            df_val = pd.read_csv(val_path)
            df_test = pd.read_csv(test_path)

            df_all = pd.concat([df_val, df_test], ignore_index=True)

            df_all.to_csv(out_path, index=False)

            print("Combined file created:", out_path)

if __name__ == '__main__':
    os.makedirs(TMP_FOLDER, exist_ok=True)
    compute_datasets_uncertainties()
    #combine_val_test_files()

    #the following function call need to be updated according to new generate_SCC_outputs.py
    #generate_scc_outputs(
    #   tmp_folder=TMP_FOLDER,
    #   output_folder="SCC_outputs",
    #   alr_list=[0.08, 0.008, 0.0008, 0.00008, 0.02, 0.002, 0.0002, 0.00002, 0.03, 0.003, 0.0003, 0.00003, 0.04, 0.004, 0.0004, 0.00004, 0.05, 0.005, 0.0005, 0.00005 ],  # <-- your ALRs

    #)