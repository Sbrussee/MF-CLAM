import slideflow as sf
import pandas as pd
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p' '--project_directory',
                    help="Directory to store the project")
parser.add_argument('-s', '--slide_directory',
                    help="Directory where slides are located")
parser.add_argument('-a', '--annotation_file',
                    help="CSV file having the slide id's, labels and patient id's. It should contain the 'slide' and 'patient' columns.")
parser.add_argument('-f', '--feature_extractor', choices=['CTransPath', 'RetCCL', 'HistoSSL', 'PLIP', 'SimCLR', 'DinoV2', 'resnet50_imagenet'],
                    help="Pretrained feature extractors to use", default="RetCCL")
parser.add_argument('-m', '--model', choices=['Attention_MIL', 'CLAM_SB', 'CLAM_MB', 'MIL_fc', 'MIL_fc_mc', 'TransMIL'],
                    help="MIL model to use", default="Attention MIL")
args = parser.parse_args()


print("Available feature extractors: ", sf.model.list_extractors())
def process_annotation_file(original_path):

    df = pd.read_csv(original_path)
    df.rename(columns={'case_id' : 'patient', 'slide_id' : 'slide'}, inplace=True)
    df.to_csv(f"{os.path.basename(original_path).strip('.csv')}_slideflow.csv", index=False)

def tile_wsis(dataset):

    dataset.extract_tiles(
    qc='both', #Both use Otsu Thresholding and Blur detection
    normalizer="macenko",
    save_tiles=True,
    whitespace_fraction=0.75,
    img_format='png',
    )

    train, test = dataset.split(
    model_type="categorical",
    labels="label",
    val_strategy='fixed'
    )

    return train, test

def main():
    if not os.path.exists("./mf"):
        project = sf.create_project(
        root = "./mf/",
        annotations = "train_list_definitive_slideflow.csv",
        slides = "../../MF_AI_dataset_cropped",
        )

    else:
        project = sf.load_project("./mf/")

    dataset = project.dataset(tile_px=512, tile_um="40x")
    print(dataset.summary())

    train, test = tile_wsis(dataset)

    feature_extractor = sf.model.build_feature_extractor(args.feature_extractor.lower(), tile_px=512)
    bag_directory = project.generate_feature_bags(feature_extractor, dataset)

    config = sf.mil.mil_config(args.model.lower())

    splits = train.kfold_split(
    k=3,
    labels="label",
    )

    for train, val in splits:
        project.train_mil(
        config=config,
        outcomes="label",
        train_dataset=train,
        val_dataset=val,
        bags=bag_directory,
        attention_heatmaps=True,
        cmap="magma",
        interpolation=None
        )



    hp = sf.ModelParams(
    tile_px=512,
    tile_um='40x',
    model='xception',
    batch_size=32,
    epochs=[3,5,10]
    )

    result = project.train(
    'mf_vs_bid',
    dataset=train,
    params=hp,
    val_strategy='k-fold',
    val_k_fold=5
    )

    test_result = project.evaluate(
    model="mf/models/mf_vs_bd",
    outcomes='label',
    dataset=test
    )

if __name__ == "__main__":
    annotations = "../../train_list_definitive.csv"
    if not os.path.exists(f"{os.path.basename(annotations).strip('.csv')}_slideflow.csv"):
        process_annotation_file(annotations)
    main()
