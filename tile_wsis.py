import urllib
urllib.request.http.client
import slideflow as sf
from slideflow.mil import mil_config
import slideflow.mil as mil
from slideflow.stats.metrics import ClassifierMetrics
from sklearn.metrics import balanced_accuracy_score
from slideflow.model.extractors._factory_torch import TorchFeatureExtractor
from ssl import MAE, BarlowTwins, DINO, SimCLR
from slideflow.model.extractors import register_torch
from torchvision import transforms


import itertools
import pandas as pd
import os
import re
import argparse
import json
import torch
from tqdm import tqdm
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from ssl import train_ssl_model

parser = argparse.ArgumentParser()
#Global arguments
parser.add_argument('-p', '--project_directory',
                    help="Directory to store the project")
parser.add_argument('-s', '--slide_directory',
                    help="Directory where slides are located")
parser.add_argument('-a', '--annotation_file',
                    help="CSV file having the slide id's, labels and patient id's. It should, at least, contain a'slide' and 'patient' column.")
parser.add_argument('-tf', '--test_fraction', type=float, default=0.2,
                    help="Fraction of dataset to hold apart for testing.")
parser.add_argument('-k', '--k_fold', type=int, default=5,
                    help="number of folds to use for k-fold cross-validation")
parser.add_argument('-ts', '--tile_size', type=int, default=256,
                    help="Size of tiles to use in pixels")
parser.add_argument('-mg', '--magnification', choices=['40x', '20x', '10x', '5x'], default='40x',
                    help="Magnification level to use")
parser.add_argument('-ag', '--augmentation', type=str, default='xyjrbn',
                    help="augmentation methods to use. Can be any combination of x: random x-flip, y: random y-flip, r: random cardinal rotation,\
                     j: random JPEG compression, b: random gaussian blur, n: stain augmentation. e.g. 'xyjn'")
parser.add_argument('-l', '--aggregation_level', choices=['patient', 'slide'],
                    help="Level of bag aggregation to use, can be 'patient' or 'slide'.")
parser.add_argument('-b', '--training_balance', choices=['tile', 'slide', 'patient', 'category'],
                    help="Balances batches for training. tile causes each tile to be sampled with equal probability,\
                    slide causes batches to be sampled on the same slide with equal probability,\
                    patient causes batches to be sampled from the same patient with equal probability,\
                    category makes sure the categorical outcome labeled are sampled with equal probability")

#Set feature extractor
parser.add_argument('-f', '--feature_extractor', choices=['CTransPath', 'RetCCL', 'HistoSSL', 'PLIP', 'resnet50_imagenet'],
                    help="Pretrained feature extractors to use", default="RetCCL")

#Set MIL model
parser.add_argument('-m', '--mil_model', choices=['Attention_MIL', 'CLAM_SB', 'CLAM_MB', 'MIL_fc', 'MIL_fc_mc', 'TransMIL'],
                    help="MIL model to use", default="Attention MIL")

parser.add_argument('-sl', '--ssl_model', choices=["SimCLR", "DINO", "MAE", "BarlowTwins", "None"],
                    default="None", help="Self supervised pretraining model to use for extracting features. Note that this will overwrite the feature extractor parameters.")
#Set normalization parameters
parser.add_argument('-n', '--normalization', choices=['macenko', 'vahadane', 'reinhard', 'cyclegan', 'None'], default="macenko",
                    help="Normalization method to use, the parameter preset is set using --stain_norm_preset ")

parser.add_argument('-sp', '--stain_norm_preset', choices=['v1', 'v2', 'v3'], default='v3',
                    help="Stain normalization preset parameter sets to use.")

parser.add_argument('-j', '--json_file', default=None,
                    help="JSON file to load for defining experiments with multiple models/extractors/normalization steps. Overrides other parsed arguments.")
args = parser.parse_args()

#Print chosen arguments
print(args)
#Print available feature extraction methods
print("Available feature extractors: ", sf.model.list_extractors())

# Check if GPU is available
if torch.cuda.is_available():
    # Get the current GPU device
    device = torch.cuda.current_device()

    # Get the name of the GPU
    gpu_name = torch.cuda.get_device_name(device)

    print(f'Using GPU: {gpu_name}')
else:
    print('GPU not available. Using CPU.')

#Check whether JSON file should be used
if args.json_file != None:
    #Load parameters from JSON file
    with open(args.json_file, "r") as params_file:
        params = json.load(params_file)


@register_torch
class SimCLRExtractor(TorchFeatureExtractor):
    tag = "simclr"
    def __init__(self, pretrained_model, backbone_type):
        super().__init__()

        self.model = pretrained_model
        self.model.to(device)
        self.model.eval()

        if backbone_type == 'resnet18':
            self.num_features = 512
        elif backbone_type == 'vit32':
            self.num_features = 512
        elif backbone_type == 'vit16':
            backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
            self.num_features = backbone.embed_dim

        # Image preprocessing.
        self.transform = transforms.Compose([
            transforms.Resize(args.tile_size),
            transforms.Lambda(lambda x: x / 255.),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.preprocess_kwargs = {'standardize': False}



    def dump_config(self):
        return {
            'class': tag,
            'kwargs': {}
        }



@register_torch
class DINOExtractor(TorchFeatureExtractor):
    tag = "dino"
    def __init__(self, pretrained_model, backbone_type):
        super().__init__()

        self.model = pretrained_model
        self.model.to(device)
        self.model.eval()

        if backbone_type == 'resnet18':
            self.num_features = 512
        elif backbone_type == 'vit32':
            self.num_features = 512
        elif backbone_type == 'vit16':
            backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
            self.num_features = backbone.embed_dim

        # Image preprocessing.
        self.transform = transforms.Compose([
            transforms.Resize(args.tile_size),
            transforms.Lambda(lambda x: x / 255.),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.preprocess_kwargs = {'standardize': False}



    def dump_config(self):
        return {
            'class': tag,
            'kwargs': {}
        }


@register_torch
class BarlowTwinsExtractor(TorchFeatureExtractor):
    tag = "barlowtwins"
    def __init__(self, pretrained_model, backbone_type):
        super().__init__()

        self.model = pretrained_model
        self.model.to(device)
        self.model.eval()

        if backbone_type == 'resnet18':
            self.num_features = 512
        elif backbone_type == 'vit32':
            self.num_features = 512
        elif backbone_type == 'vit16':
            backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
            self.num_features = backbone.embed_dim

        # Image preprocessing.
        self.transform = transforms.Compose([
            transforms.Resize(args.tile_size),
            transforms.Lambda(lambda x: x / 255.),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.preprocess_kwargs = {'standardize': False}



    def dump_config(self):
        return {
            'class': tag,
            'kwargs': {}
        }


@register_torch
class MAEFeatureExtractor(TorchFeatureExtractor):
    tag = "mae"
    def __init__(self, pretrained_model, backbone_type):
        super().__init__()

        self.model = pretrained_model
        self.model.to(device)
        self.model.eval()

        if backbone_type == 'resnet18':
            self.num_features = 512
        elif backbone_type == 'vit32':
            self.num_features = 512
        elif backbone_type == 'vit16':
            backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
            self.num_features = backbone.embed_dim

        # Image preprocessing.
        self.transform = transforms.Compose([
            transforms.Resize(args.tile_size),
            transforms.Lambda(lambda x: x / 255.),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.preprocess_kwargs = {'standardize': False}



    def dump_config(self):
        return {
            'class': tag,
            'kwargs': {}
        }

def process_annotation_file(original_path : str):
    """
    Parameters:
        -original_path: Path to the annotation file to be processed.

    This function renames the columns 'case_id' and 'slide_id' to 'patient' and 'slide'
    in the annotation file, in order to make it suitable for slideflow.

    """
    df = pd.read_csv(original_path)
    df.rename(columns={'case_id' : 'patient', 'slide_id' : 'slide'}, inplace=True)
    df.to_csv(f"{os.path.basename(original_path).strip('.csv')}_slideflow.csv", index=False)


def get_highest_numbered_filename(directory_path : str):
    """
    Parameters:
        -directory_path: Path to the directory where the filenames should be checked.

    This helper function returns the filename in a given directory with the highest
    numeric value in its name, including leadining zeros. This is useful when
    accessing the last trained model for example.

    Returns:
        -highest_number_part: The filename of the highest numbered file.
    """
    # List all files in the directory
    files = os.listdir(directory_path)

    # Initialize variables to keep track of the highest number and corresponding filename
    highest_number = float('-inf')
    highest_number_filename = None

    # Iterate over each file
    for filename in files:
        # Get the part before the first '-'
        first_part = filename.split('-')[0]

        # Try to convert the first part to a number
        try:
            number = int(first_part)
            # If the converted number is higher than the current highest, update the variables
            if number > highest_number:
                highest_number = number
                highest_number_part = first_part
        except ValueError:
            pass  # Ignore non-numeric parts

    return highest_number_part

"""
import torch.nn as nn

class Aggregator(nn.Module):
    def __init__(self, input_features_size: int = 2048):
        super(Aggregator, self).__init__()
        self.projector = nn.Linear(in_features=input_features_size, out_features=16)
        self.selu = torch.nn.SELU()
        self.dropout_projector = torch.nn.Dropout(0.5)
        self.weighing_layer = nn.Linear(in_features=16, out_features=1, bias=False)
        self.classifier = nn.Linear(in_features=16, out_features=3)
        self.dropout_classifier = torch.nn.Dropout(0.5)
        self.init_weights()

    def forward(self, h):
        h = self.selu(self.projector(self.dropout_projector(h)))
        weights = torch.relu(self.weighing_layer(h))
        weights = weights / (weights.sum() + 1e-8)
        weighted_sum = torch.mm(weights.T, h)
        logits = self.classifier(self.dropout_classifier(weighted_sum))
        return logits, weights

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



def linear_probe_model(image_shape, **kwargs):
    model = Aggregator(image_shape)
    return model
"""

def tile_wsis(dataset : sf.Dataset):
    """
    Parameters:
        -dataset: Slideflow dataset to extract tiles from

    Extracts and then saves tiles from a slideflow dataset objects.

    Returns:
        -dataset: Slideflow dataset with extracted tiles
    """
    dataset.extract_tiles(
    qc='both', #Both use Otsu Thresholding and Blur detection
    save_tiles=True,
    img_format='png',
    enable_downsample=False
    )
    return dataset


def split_dataset(dataset : sf.Dataset, test_fraction : float = 0.2):
    """
    Parameters:
        -dataset: Slideflow dataset to split
        -test_fraction: Fraction of data to use as test set

    This function splits a slideflow dataset into a training and testing set.
    It then balances the training dataset based on the specified training
    balance hyperparameter.

    Returns:
        -train: training dataset
        -test: test dataset
    """
    train, test = dataset.split(
    model_type="categorical",
    labels="label",
    val_strategy='fixed',
    val_fraction=test_fraction
    )

    train = train.balance(headers='label', strategy=args.training_balance)

    return train, test


def extract_features(extractor : str, normalizer : str, dataset : sf.Dataset, project: sf.Project):
    """
    Extractor:



    """
    feature_extractor = sf.model.build_feature_extractor(extractor.lower(), tile_px=args.tile_size)
    bag_directory = project.generate_feature_bags(feature_extractor,
                                                  dataset,
                                                  outdir=f"{args.project_directory}/bags/{extractor.lower()}_{normalizer.lower()}",
                                                  normalizer=normalizer,
                                                  normalizer_source=args.stain_norm_preset,
                                                  augment=args.augmentation)

def calculate_combinations(parameters):
    param_values = list(parameters.values())
    param_names = list(parameters.keys())
    max_length = len(parameters)

    all_combinations = list(itertools.product(*param_values))

    result_combinations = [dict(zip(param_names, combo)) for combo in all_combinations]

    return result_combinations

def train_mil_model(train, val, test, model, extractor, normalizer, project, config):
    project.train_mil(
    config=config,
    outcomes="label",
    train_dataset=train,
    val_dataset=val,
    bags=f"{args.project_directory}/bags/{extractor.lower()}_{normalizer.lower()}",
    #attention_heatmaps=True,
    #cmap="coolwarm",
    exp_label=f"{model.lower()}_{extractor.lower()}_{normalizer.lower()}"
    )

    current_highest_exp_number = get_highest_numbered_filename(f"{args.project_directory}mil/")

    result_frame = mil.eval_mil(
    weights=f"{args.project_directory}mil/{current_highest_exp_number}-{model.lower()}_{extractor.lower()}_{normalizer.lower()}",
    outcomes="label",
    dataset=test,
    bags=f"{args.project_directory}/bags/{extractor.lower()}_{normalizer.lower()}",
    config=config,
    outdir=f"{args.project_directory}/mil_eval/{current_highest_exp_number}_{model.lower()}_{extractor.lower()}_{normalizer.lower()}",
    #attention_heatmaps=True,
    #cmap="coolwarm"
    )

    print(result_frame)

    y_pred_cols = [c for c in result_frame.columns if c.startswith('y_pred')]
    for idx in range(len(y_pred_cols)):
        m = ClassifierMetrics(
            y_true=(result_frame.y_true.values == idx).astype(int),
            y_pred=result_frame[f'y_pred{idx}'].values
        )

        fpr, tpr, auroc, threshold = m.fpr, m.tpr, m.auroc, m.threshold
        optimal_idx = np.argmax(tpr-fpr)
        optimal_threshold = threshold[optimal_idx]
        y_pred_binary = (result_frame[f'y_pred{idx}'].values > optimal_threshold).astype(int)
        balanced_accuracy = balanced_accuracy_score((result_frame.y_true.values == idx).astype(int), y_pred_binary)

    print(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="orange",
        lw=lw,
        label=f"ROC curve (area = %0.2f)" % auroc,
        )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC")
    plt.legend(loc="lower right")
    plt.savefig(f"{args.project_directory}/mil_eval/{current_highest_exp_number}_{model.lower()}_{extractor.lower()}_{normalizer.lower()}/roc_auc_test.png")
    #result_frame = pd.read_parquet(f"{args.project_directory}/mil/{current_highest_exp_number}-{model.lower()}_{extractor.lower()}_{normalizer.lower()}/predictions.parquet", engine='pyarrow')
    return result_frame, balanced_accuracy, auroc

def train_ssl(ssl_model_name, backbone, path_to_images):
    # Check if trained_ssl_models directory exists, if not, create it
    if not os.path.exists("trained_ssl_models"):
        os.makedirs("trained_ssl_models")

    # Check if the ssl_model_name already exists in trained_ssl_models directory
    if os.path.exists(os.path.join("trained_ssl_models", ssl_model_name)):
        print("Found pre-trained model. Loading weights...")
        # Load the pre-trained model
        backbone_state_dict = torch.load(os.path.join("trained_ssl_models", ssl_model_name))
        # Load the state dictionary into the backbone
        backbone.load_state_dict(backbone_state_dict)
    else:
        print("Pre-trained model not found. Training...")
        # Train the SSL model
        train_ssl_model(ssl_model_name, backbone, path_to_images, os.path.join("trained_ssl_models", ssl_model_name))


def cross_validate_combination(possible_parameters, parameter_combinations, dataset, project, train, test):
    result_df = pd.DataFrame(columns=list(possible_parameters.keys()) + ['split', 'balanced_accuracy', 'auc'])

    for comb in tqdm(parameter_combinations):
        print(comb)
        # Initialize comb_dict with values from possible_parameters
        comb_dict = {param: possible_parameters[param][0] if len(possible_parameters[param]) == 1 else None for param in possible_parameters}

        # Update comb_dict with values from parameter_combinations
        for k, v in comb_dict.items():
            if v == None:
                comb_dict[k] = comb[k]

        print(comb_dict)

        if comb_dict.get('ssl_model') != 'None':
            train_ssl_and_extract_features(comb_dict.get('ssl_model'), comb_dict.get('normalization'),
                                           project, train, dataset)
        else:
            # Extract features and other preprocessing based on comb_dict values
            extract_features(comb_dict.get('feature_extractor'), comb_dict.get('normalization'), dataset, project)

        # Construct config based on comb_dict values
        config = mil_config(comb_dict.get('mil_model', '').lower(), aggregation_level=args.aggregation_level)

        # Split using specified k-fold
        splits = train.kfold_split(
            k=args.k_fold,
            labels="label",
        )

        split_index = 0
        for train, val in splits:
            result_frame, balanced_accuracy, roc_auc = train_mil_model(train, val, test,
                                                                       comb_dict.get('mil_model'),
                                                                       comb_dict.get('feature_extractor'),
                                                                       comb_dict.get('normalization'),
                                                                       project, config)

            data = {param: [comb_dict[param]] if param in comb else comb_dict[param] for param in possible_parameters}
            data.update({
                'split': [split_index],
                'balanced_accuracy': [balanced_accuracy],
                'auc': [roc_auc]
            })

            result_df = result_df.append(pd.DataFrame(data), ignore_index=True)
            split_index += 1

    return result_df

def main():
    #Create project directory
    if not os.path.exists(args.project_directory):
        project = sf.create_project(
        root = args.project_directory,
        annotations = args.annotation_file,
        slides = args.slide_directory,
        )

    else:
        project = sf.load_project(args.project_directory)

    dataset = project.dataset(tile_px=args.tile_size, tile_um=args.magnification)
    print(dataset.summary())

    dataset = tile_wsis(dataset)
    train, test = split_dataset(dataset, test_fraction=args.test_fraction)

    possible_parameters = {
    'augmentation' : [],
    'normalization' : [],
    'feature_extractor' : [],
    'ssl_model' : [],
    'mil_model' : [],
    'stain_norm_preset' : [],
    }

    if args.json_file == 'None':
        args.json_file = None

    if args.json_file != None:
        config = json.load(open(args.json_file,))
        multi_value_params = [param for param, values in config.items() if isinstance(values, list)]

        for param in possible_parameters.keys():
            if param not in multi_value_params:
                possible_parameters[param] = [getattr(args, param, None)]
            else:
                possible_parameters[param] = config.get(param, None)

        print(possible_parameters)

        parameter_combinations = calculate_combinations(possible_parameters)

        result_df = cross_validate_combination(possible_parameters, parameter_combinations,
                                                dataset, project, train, test)

        grouped_df = result_df.groupby(list(possible_parameters.keys()))
        final_df = grouped_df.agg({
        'augmentation' : 'first',
        'normalization' : 'first',
        'feature_extractor' : 'first',
        'ssl_model' : 'first',
        'mil_model' : 'first',
        'stain_norm_preset' : 'first',
        'balanced_accuracy' : ['mean', 'std'],
        'auc' : ['mean', 'std']
        })

    else:
        columns = ['augmentation', 'normalization', 'feature_extractor',
                    'ssl_model', 'mil_model', 'stain_norm_preset', 'split', 'balanced_accuracy', 'auc']

        result_df = pd.DataFrame(columns=columns)

        if args.ssl_model != "None":
            #If SSL model should be used
            if args.ssl_model == 'DINO' or args.ssl_model == 'MAE':
                backbone = 'vit16'
            else:
                backbone = 'resnet18'

            #Train SSL model on image tiles
            train_ssl_model(args.ssl_model, backbone, f"{args.tile_size}px_{args.magnification}")
            #Extract features using the trained SSL model
            args.feature_extractor = args.ssl_model

        extract_features(args.feature_extractor, args.normalization, dataset, project)
        config = mil_config(args.mil_model.lower(), aggregation_level=args.aggregation_level)
        #Split using specified k-fold
        splits = train.kfold_split(
        k=args.k_fold,
        labels="label",
        )

        split_index = 0
        for train, val in splits:
            result_frame, balanced_accuracy, roc_auc = train_mil_model(train, val, test,
                                                       args.mil_model, args.feature_extractor,
                                                       args.normalization, project, config)

            print("Balanced Accuracy: ", balanced_accuracy)

            data = {
            'tile_size' : args.tile_size,
            'magnification' : args.magnification,
            'augmentation' : args.augmentation,
            'normalization' : args.normalization,
            'feature_extractor' : args.feature_extractor,
            'ssl_model' : args.ssl_model,
            'mil_model' : args.mil_model,
            'stain_norm_preset' : args.stain_norm_preset,
            }

            data.update({
            'split': split_index,
            'balanced_accuracy' : balanced_accuracy,
            'auc' : roc_auc
            })

            result_df = result_df.append(data, ignore_index=True)
            split_index += 1

        #Summarize over splits
        grouped_df = result_df.groupby(['tile_size', 'magnification', 'augmentation', 'normalization', 'feature_extractor',
                                        'ssl_model', 'mil_model', 'stain_norm_preset'])


        final_df = grouped_df.agg({
        'tile_size' : 'first',
        'magnification' : 'first',
        'augmentation' : 'first',
        'normalization' : 'first',
        'feature_extractor' : 'first',
        'ssl_model' : 'first',
        'mil_model' : 'first',
        'stain_norm_preset' : 'first',
        'balanced_accuracy' : ['mean', 'std'],
        'auc' : ['mean', 'std']
        })


    date = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
    final_df.to_csv(f"{args.project_directory}/results_{date}.csv", index=False)



if __name__ == "__main__":
    annotations = "../../train_list_definitive.csv"
    if not os.path.exists(f"{os.path.basename(annotations).strip('.csv')}_slideflow.csv"):
        process_annotation_file(annotations)
    main()
