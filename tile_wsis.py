import slideflow as sf
from slideflow.mil import mil_config
import pandas as pd
import os
import argparse
import json
import torch
from tqdm import tqdm

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
parser.add_argument('-k', '--k_fold', type=int, default=5
                    help="number of folds to use for k-fold cross-validation")
parser.add_argument('-ts', '--tile_size', type=int, default=256,
                    help="Size of tiles to use in pixels")
parser.add_argument('-mg', '--magnification', choices=['40x', '20x', '10x', '5x'], default='40x',
                    help="Magnification level to use")
parser.add_argument('-ag', '--augmentation', type='str', defualt='xyjrbn',
                    help="augmentation methods to use. Can be any combination of x: random x-flip, y: random y-flip, r: random cardinal rotation,\
                     j: random JPEG compression, b: random gaussian blur, n: stain augmentation. e.g. 'xyjn'")

#Set feature extractor
parser.add_argument('-f', '--feature_extractor', choices=['CTransPath', 'RetCCL', 'HistoSSL', 'PLIP', 'SimCLR', 'DinoV2', 'resnet50_imagenet'],
                    help="Pretrained feature extractors to use", default="RetCCL")

#Set MIL model
parser.add_argument('-m', '--model', choices=['Attention_MIL', 'CLAM_SB', 'CLAM_MB', 'MIL_fc', 'MIL_fc_mc', 'TransMIL'],
                    help="MIL model to use", default="Attention MIL")

#Set normalization parameters
parser.add_argument('-n', '--normalization', choices=['macenko', 'vahadane', 'reinhard', 'cyclegan', 'None'], default="macenko",
                    help="Normalization method to use, the parameter preset is set using --stain_norm_preset ")
parser.add_argument('-sp', '--stain_norm_preset', choices=['v1', 'v2', 'v3'], default='v3',
                    help="Stain normalization preset parameter sets to use.")


parser.add_argument('-j', '--json_file', default=None,
                    help="JSON file to load for defining experiments with multiple models/extractors/normalization steps. Overrides other parsed arguments.")
args = parser.parse_args()
print(args)

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

if args.json_file != None:
    with open(args.json_file, "r") as params_file:
        params = json.load(params_file)

def process_annotation_file(original_path):
    df = pd.read_csv(original_path)
    df.rename(columns={'case_id' : 'patient', 'slide_id' : 'slide'}, inplace=True)
    df.to_csv(f"{os.path.basename(original_path).strip('.csv')}_slideflow.csv", index=False)

def tile_wsis(dataset):
    if normalization.lower() == 'none':
        norm = None
    else:
        norm = normalization.lower()

    dataset.extract_tiles(
    qc='both', #Both use Otsu Thresholding and Blur detection
    save_tiles=True,
    img_format='png',
    enable_downsample=False
    )

    return dataset


def split_dataset(dataset, test_fraction=0.2):
    train, test = dataset.split(
    model_type="categorical",
    labels="label",
    val_strategy='fixed',
    val_fraction=test_fraction
    )
    return train, test


def extract_features(extractor, normalizer, dataset, project):
    feature_extractor = sf.model.build_feature_extractor(extractor.lower(), tile_px=args.tile_size)
    bag_directory = project.generate_feature_bags(feature_extractor,
                                                  dataset,
                                                  outdir=f"{args.project_directory}/bags/{extractor.lower()}_{normalizer.lower()}",
                                                  normalizer=normalizer,
                                                  normalizer_source=args.stain_norm_preset,
                                                  augment=args.augmentation)


def train_mil_model(train, val, test, model, extractor, normalizer, project, config):
    project.train_mil(
    config=config,
    outcomes="label",
    train_dataset=train,
    val_dataset=val,
    bags=f"{args.project_directory}/bags/{extractor.lower()}",
    attention_heatmaps=True,
    outdir=f"{args.project_directory}/model/{model.lower()}_{extractor.lower()}_{normalizer.lower()}"
    )

    project.evaluate_mil(
    model=f"{args.project_directory}/model/{model.lower()}_{extractor.lower()}_{normalizer.lower()}",
    outcomes="label",
    dataset=test,
    bags=f"{args.project_directory}/bags/{extractor.lower()}_{normalizer.lower()}",
    config=config,
    attention_heatmaps=True
    )


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


    if args.json_file != None:
        normalizers = params['normalization_methods']
        extractors = params['feature_extractors']
        models = params['mil_models']
        if params['train_using_ssl']:
            ssl_models = params['ssl_params']['ssl_models']

    else:
        normalizers = [args.normalization]
        extractors = [args.feature_extractor]
        models = [args.model]


    dataset = project.dataset(tile_px=args.tile_size, tile_um=args.magnification)
    print(dataset.summary())

    dataset = tile_wsis(dataset)
    train, test = split_dataset(dataset, test_fraction=args.test_fraction)

    for extractor in tqdm(extractors, desc="Outer extractor loop"):
        for normalization in tqdm(normalizers, desc="Middle normalizer loop"):
            for model in tqdm(models desc="Inner model loop"):
                extract_features(extractor, normalizer, dataset, project)

                config = mil_config(args.model.lower())

                splits = train.kfold_split(
                k=args.k_fold,
                labels="label",
                )

                for train, val in splits:
                    train_mil_model(train, val, test, model, extractor, normalizer, project, config)


if __name__ == "__main__":
    annotations = "../../train_list_definitive.csv"
    if not os.path.exists(f"{os.path.basename(annotations).strip('.csv')}_slideflow.csv"):
        process_annotation_file(annotations)
    main()
