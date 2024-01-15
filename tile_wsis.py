import slideflow as sf
import pandas as pd
import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--project_directory',
                    help="Directory to store the project")
parser.add_argument('-s', '--slide_directory',
                    help="Directory where slides are located")
parser.add_argument('-a', '--annotation_file',
                    help="CSV file having the slide id's, labels and patient id's. It should, at least, contain a'slide' and 'patient' column.")
parser.add_argument('-f', '--feature_extractor', choices=['CTransPath', 'RetCCL', 'HistoSSL', 'PLIP', 'SimCLR', 'DinoV2', 'resnet50_imagenet'],
                    help="Pretrained feature extractors to use", default="RetCCL")
parser.add_argument('-m', '--model', choices=['Attention_MIL', 'CLAM_SB', 'CLAM_MB', 'MIL_fc', 'MIL_fc_mc', 'TransMIL'],
                    help="MIL model to use", default="Attention MIL")
parser.add_argument('-n', '--normalization', choices=['macenko', 'vahadane', 'reinhard', 'cyclegan'])
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

    dataset.extract_tiles(
    qc='both', #Both use Otsu Thresholding and Blur detection
    normalizer=args.normalization.lower(),
    save_tiles=True,
    img_format='png',
    enable_downsample=False
    )

    train, test = dataset.split(
    model_type="categorical",
    labels="label",
    val_strategy='fixed',
    val_fraction=0.2
    )

    return train, test

def extract_features(extractor, dataset, project):
    feature_extractor = sf.model.build_feature_extractor(extractor.lower(), tile_px=512)
    bag_directory = project.generate_feature_bags(feature_extractor,
                                                  dataset,
                                                  outdir=f"bags/{extractor.lower()}")


def train_mil_model(train, val, test, model, extractor, project, config):
    project.train_mil(
    config=config,
    outcomes="label",
    train_dataset=train,
    val_dataset=val,
    bags=f"bags/{extractor.lower()}",
    attention_heatmaps=True,
    outdir=f"/model/{model.lower()}"
    )

    project.evaluate_mil(
    model=f"/model/{model.lower()}",
    outcomes="label",
    dataset=test,
    bags=f"bags/{extractor.lower()}",
    config=config,
    attention_heatmaps=True
    )


def main():
    if not os.path.exists(args.project_directory):
        project = sf.create_project(
        root = args.project_directory,
        annotations = args.annotation_file,
        slides = args.slide_directory,
        )

    else:
        project = sf.load_project(args.project_directory)

    dataset = project.dataset(tile_px=512, tile_um=128)
    print(dataset.summary())

    train, test = tile_wsis(dataset)

    if args.json_file != None:
        for extractor in params['feature_extractors']:
            extract_features(extractor, dataset, project)
    else:
        extract_features(args.feature_extractor, dataset, project)

    config = sf.mil.mil_config(args.model.lower())

    splits = train.kfold_split(
    k=3,
    labels="label",
    )

    if args.json_file != None:
        for extractor in params['feature_extractors']:
            for model in params['mil_models']:
                for train, val in splits:
                    train_mil_model(train, vall, test, model, extractor, project, config)

    else:
        for train, val in splits:
            train_mil_model(train, val, test, args.model, args.feature_extractor, project, config)


if __name__ == "__main__":
    annotations = "../../train_list_definitive.csv"
    if not os.path.exists(f"{os.path.basename(annotations).strip('.csv')}_slideflow.csv"):
        process_annotation_file(annotations)
    main()
