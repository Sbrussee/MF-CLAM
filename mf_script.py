import slideflow as sf
from slideflow.mil import mil_config
import slideflow.mil as mil
from slideflow.stats.metrics import ClassifierMetrics
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
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
parser.add_argument('-f', '--feature_extractor', choices=['CTransPath', 'RetCCL', 'HistoSSL', 'PLIP', 'SimCLR', 'DinoV2', 'resnet50_imagenet'],
                    help="Pretrained feature extractors to use", default="RetCCL")

#Set MIL model
parser.add_argument('-m', '--model', choices=['Attention_MIL', 'CLAM_SB', 'CLAM_MB', 'MIL_fc', 'MIL_fc_mc', 'TransMIL'],
                    help="MIL model to use", default="Attention MIL")

#Set normalization parameters
parser.add_argument('-n', '--normalization', choices=['macenko', 'vahadane_sklearn', 'reinhard', 'cyclegan', 'None'], default="macenko",
                    help="Normalization method to use, the parameter preset is set using --stain_norm_preset ")

parser.add_argument('-sp', '--stain_norm_preset', choices=['v1', 'v2', 'v3'], default='v3',
                    help="Stain normalization preset parameter sets to use.")

parser.add_argument('-j', '--json_file', default=None,
                    help="JSON file to load for defining experiments with multiple models/extractors/normalization steps. Overrides other parsed arguments.")
args = parser.parse_args()
#Print chosen arguments
print(args)
#Print available feature extractors
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

#Check if JSON file should be used
if args.json_file != None:
    with open(args.json_file, "r") as params_file:
        params = json.load(params_file)

def process_annotation_file(original_path):
    df = pd.read_csv(original_path, header=0, sep=";")
    df.rename(columns={'case_id' : 'patient', 'slide_id' : 'slide'}, inplace=True)
    #df['slide'] = df['slide'].apply(lambda x:x + '.tiff')
    df.drop_duplicates(inplace=True, subset="slide")
    rows_to_drop = df[(df['patient'].isin([3787608, 8839385, 9476237, 1093978])) & (df['category'] == 'BID')].index
    df = df.drop(rows_to_drop)
    print("Processed annotation file: ", df)
    df.to_csv(f"{os.path.basename(original_path).strip('.csv')}_slideflow.csv", index=False,sep=",")

def get_highest_numbered_filename(directory_path):
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

    print(highest_number_part)
    return highest_number_part

def split_dataset(dataset, test_fraction=0.2):
    train, test = dataset.split(
    model_type="categorical",
    val_strategy='fixed',
    labels="category",
    val_fraction=test_fraction
    )

    train = train.balance(headers='category', strategy=args.training_balance)

    return train, test


def extract_features(extractor, normalizer, dataset, project):
    feature_extractor = sf.model.build_feature_extractor(extractor.lower(), tile_px=args.tile_size)
    bag_directory = project.generate_feature_bags(feature_extractor,
                                                  dataset,
                                                  outdir=f"{args.project_directory}/bags/{extractor.lower()}_{normalizer.lower()}",
                                                  normalizer=normalizer,
                                                  normalizer_source=args.stain_norm_preset,
                                                  augment=args.augmentation)

def read_easy_set():
    easy_df = pd.read_csv("../../train_list_easy_only.csv")
    easy_df.replace(';;', '', inplace=True)
    easy_df.rename(columns={'case_id' : 'patient', 'slide_id' : 'slide', 'label;;' : 'label'}, inplace=True)
    easy_df['label'] = easy_df['label'].str.replace(';;', '')
    easy_slides = easy_df['slide'].tolist()
    print("Processed easy annotation file: ", easy_df)
    easy_df.to_csv("train_list_easy_only_slideflow.csv")

    easy_set = sf.Dataset(
    slides = '../../MF_AI_dataset_cropped',
    annotations="train_list_easy_only_slideflow.csv",
    tiles=f"{args.project_directory}/tiles/easy_set",
    tfrecords=f"{args.project_directory}/tfrecords/easy_set",
    tile_px = args.tile_size,
    tile_um = args.magnification
    )

    test_df = pd.read_csv("train_list_definitive_slideflow.csv")
    #Only retain the slides not in the easy set
    test_df = test_df[~test_df['slide'].isin(easy_slides)]

    test_df.to_csv("test_list_easy_only_slideflow.csv")

    test_set = sf.Dataset(
    slides = '../../MF_AI_dataset_cropped',
    annotations="test_list_easy_only_slideflow.csv",
    tiles=f"{args.project_directory}/tiles/easy_test",
    tfrecords=f"{args.project_directory}/tfrecords/easy_test",
    tile_px = args.tile_size,
    tile_um = args.magnification
    )

    easy_set = tile_wsis(easy_set)

    test_set = tile_wsis(test_set)



    return easy_set, test_set

def read_validation_set(project):
    #process_annotation_file("../../Thom_Doeleman/annotations.csv")

    project.add_source(
    name="ext_set",
    slides="../../Thom_Doeleman/CLAM_validate_cropped",
    roi="../../Thom_Doeleman/CLAM_validate_cropped/rois",
    tfrecords=f"{args.project_directory}/tfrecords/ext_set",
    tiles=f"{args.project_directory}/tiles/ext_set"
    )
    test_set = project.dataset(
    sources=['ext_set'],
    filters = {'dataset' : 'validate'},
    tile_px=args.tile_size,
    tile_um=args.magnification
    )

    project.extract_tiles(
            qc='both', #Both use Otsu Thresholding and Blur detection
            source='ext_set',
            tile_px=args.tile_size,
            tile_um=args.magnification,
            save_tiles=True,
            img_format='png',
            enable_downsample=False
            )

    return project, test_set


def train_mil_model(train, val, test, model, extractor, normalizer, project, config):

    if args.aggregation_level == 'patient':
        project.train_mil(
        config=config,
        outcomes="category",
        train_dataset=train,
        val_dataset=val,
        bags=f"{args.project_directory}/bags/{extractor.lower()}_{normalizer.lower()}",
        #attention_heatmaps=True,
        #cmap="coolwarm",
        exp_label=f"{model.lower()}_{extractor.lower()}_{normalizer.lower()}"
        )

        current_highest_exp_number = get_highest_numbered_filename(f"{args.project_directory}/mil/")

        result_frame = mil.eval_mil(
        weights=f"{args.project_directory}/mil/{current_highest_exp_number}-{model.lower()}_{extractor.lower()}_{normalizer.lower()}",
        outcomes="category",
        dataset=test,
        bags=f"{args.project_directory}/bags/{extractor.lower()}_{normalizer.lower()}",
        config=config,
        outdir=f"{args.project_directory}/mil_eval/{current_highest_exp_number}_{model.lower()}_{extractor.lower()}_{normalizer.lower()}",
        #attention_heatmaps=True,
        #cmap="coolwarm"
        )

    elif args.aggregation_level == 'slide':
        project.train_mil(
        config=config,
        outcomes="category",
        train_dataset=train,
        val_dataset=val,
        bags=f"{args.project_directory}/bags/{extractor.lower()}_{normalizer.lower()}",
        attention_heatmaps=True,
        cmap="coolwarm",
        exp_label=f"{model.lower()}_{extractor.lower()}_{normalizer.lower()}"
        )

        current_highest_exp_number = get_highest_numbered_filename(f"{args.project_directory}/mil/")

        result_frame = mil.eval_mil(
        weights=f"{args.project_directory}/mil/{current_highest_exp_number}-{model.lower()}_{extractor.lower()}_{normalizer.lower()}",
        outcomes="category",
        dataset=test,
        bags=f"{args.project_directory}/bags/{extractor.lower()}_{normalizer.lower()}",
        config=config,
        outdir=f"{args.project_directory}/mil_eval/{current_highest_exp_number}_{model.lower()}_{extractor.lower()}_{normalizer.lower()}",
        attention_heatmaps=True,
        cmap="coolwarm"
        )
    return result_frame


def visualize_results(result_frame, model, extractor, normalizer, ext_set=False):


    current_highest_exp_number = get_highest_numbered_filename(f"{args.project_directory}/mil/")
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
        conf_mat = confusion_matrix(y_true=(result_frame.y_true.values == idx).astype(int), y_pred=y_pred_binary, normalize='all', labels=np.array([0,1]))
        display = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=['MF', 'BID'])
        display.plot()
        if ext_set:
            plt.savefig(f"{args.project_directory}/mil_eval/{current_highest_exp_number}_{model.lower()}_{extractor.lower()}_{normalizer.lower()}_ext_set/conf_mat_test.png")
        else:
            plt.savefig(f"{args.project_directory}/mil_eval/{current_highest_exp_number}_{model.lower()}_{extractor.lower()}_{normalizer.lower()}/conf_mat_test.png")
        balanced_accuracy = balanced_accuracy_score((result_frame.y_true.values == idx).astype(int), y_pred_binary)
        print(f"BA cat #{idx}: {balanced_accuracy}")


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
    if ext_set:
        plt.savefig(f"{args.project_directory}/mil_eval/{current_highest_exp_number}_{model.lower()}_{extractor.lower()}_{normalizer.lower()}_ext_set/roc_auc_test.png")
    else:
        plt.savefig(f"{args.project_directory}/mil_eval/{current_highest_exp_number}_{model.lower()}_{extractor.lower()}_{normalizer.lower()}/roc_auc_test.png")
    #result_frame = pd.read_parquet(f"{args.project_directory}/mil/{current_highest_exp_number}-{model.lower()}_{extractor.lower()}_{normalizer.lower()}/predictions.parquet", engine='pyarrow')
    return result_frame, balanced_accuracy, auroc

def main(easy=False, validation=False):

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
        normalizers = params['normalization']
        extractors = params['feature_extractor']
        models = params['mil_model']

    else:
        normalizers = [args.normalization]
        extractors = [args.feature_extractor]
        models = [args.model]


    print(args.tile_size, args.magnification)
    dataset = project.dataset(tile_px=args.tile_size, tile_um=args.magnification, filters={'dataset' : 'train'})
    print(dataset)

    print("Tiling...")
    project.extract_tiles(
        qc='both', #Both use Otsu Thresholding and Blur detection
        save_tiles=True,
        tile_px=args.tile_size,
        tile_um=args.magnification,
        img_format='png',
        enable_downsample=False
        )
    print("Splitting...")
    train, test = split_dataset(dataset, test_fraction=args.test_fraction)

    if easy:
        train, test = read_easy_set()
        print("Easy training set: ", train)
        print("Test set: ", test)
        train.balance(headers='category', strategy=args.training_balance)

    #overwrite test with external validation set
    if validation:
        project, ext_test = read_validation_set(project)

    results = {}

    columns = ['normalization', 'feature_extractor', 'mil_model', 'split', 'balanced_accuracy', 'auc']
    df = pd.DataFrame(columns=columns)
    if validation:
        ext_df = pd.DataFrame(columns=columns)
    for extractor in tqdm(extractors, desc="Outer extractor loop"):
        for normalizer in tqdm(normalizers, desc="Middle normalizer loop"):
            if normalizer.lower() == 'none':
                normalizer = None
            for model in tqdm(models, desc="Inner model loop"):
                extract_features(extractor, normalizer, dataset, project)
                #Set model configuration
                config = mil_config(args.model.lower(),
                aggregation_level=args.aggregation_level,
                wd=1e-4)
                #Split using specified k-fold
                splits = train.kfold_split(
                k=args.k_fold,
                labels="category",
                )
                split_index = 0
                best_roc_auc = 0
                for train, val in splits:
                    if easy:
                        feature_extractor = sf.model.build_feature_extractor(extractor.lower(), tile_px=args.tile_size)
                        bag_directory = project.generate_feature_bags(feature_extractor,
                                                                      train,
                                                                      outdir=f"{args.project_directory}/bags/{extractor.lower()}_{normalizer.lower()}_easy",
                                                                      normalizer=normalizer,
                                                                      normalizer_source=args.stain_norm_preset,
                                                                      augment=args.augmentation)

                        bag_directory = project.generate_feature_bags(feature_extractor,
                                                                      test,
                                                                      outdir=f"{args.project_directory}/bags/{extractor.lower()}_{normalizer.lower()}_easy",
                                                                      normalizer=normalizer,
                                                                      normalizer_source=args.stain_norm_preset,
                                                                      augment=args.augmentation)

                    result_frame = train_mil_model(train, val, test, model, extractor, normalizer, project, config)
                    result_frame, balanced_accuracy, roc_auc  = visualize_results(result_frame, model, extractor, normalizer)

                    #print(extractor, normalizer, model, result_frame)
                    results["_".join([extractor, normalizer, model, str(split_index)])] = balanced_accuracy
                    print("Balanced Accuracy: ", balanced_accuracy)
                    data = {
                    'normalization' : normalizer,
                    'feature_extractor' : extractor,
                    'mil_model' : model,
                    'split': split_index,
                    'balanced_accuracy' : balanced_accuracy,
                    'auc' : roc_auc
                    }
                    df = df.append(data, ignore_index=True)
                    print(df)
                    print("Validating...")
                    if validation:
                        feature_extractor = sf.model.build_feature_extractor(extractor.lower(), tile_px=args.tile_size)
                        bag_directory = project.generate_feature_bags(feature_extractor,
                                                                      ext_test,
                                                                      outdir=f"{args.project_directory}/bags/{extractor.lower()}_{normalizer.lower()}_ext_set",
                                                                      normalizer=normalizer,
                                                                      normalizer_source=args.stain_norm_preset,
                                                                      augment=args.augmentation)

                        current_highest_exp_number = get_highest_numbered_filename(f"{args.project_directory}/mil/")

                        result_frame = mil.eval_mil(
                        weights=f"{args.project_directory}/mil/{current_highest_exp_number}-{model.lower()}_{extractor.lower()}_{normalizer.lower()}",
                        outcomes="category",
                        dataset=ext_test,
                        bags=f"{args.project_directory}/bags/{extractor.lower()}_{normalizer.lower()}_ext_set",
                        config=config,
                        outdir=f"{args.project_directory}/mil_eval/{current_highest_exp_number}_{model.lower()}_{extractor.lower()}_{normalizer.lower()}_ext_set",
                        #attention_heatmaps=True,
                        #cmap="coolwarm"
                        )

                        result_frame, balanced_accuracy, roc_auc = visualize_results(result_frame, model, extractor, normalizer, ext_set=True)
                        data = {
                        'normalization' : normalizer,
                        'feature_extractor' : extractor,
                        'mil_model' : model,
                        'split': split_index,
                        'balanced_accuracy' : balanced_accuracy,
                        'auc' : roc_auc
                        }
                        ext_df = ext_df.append(data, ignore_index=True)
                        print(ext_df)

                    print("One loop done...")

                    split_index += 1


    #Summarize over splits
    grouped_df = df.groupby(['normalization', 'feature_extractor', 'mil_model'])
    if validation:
        ext_grouped_df = ext_df.groupby(['normalization', 'feature_extractor', 'mil_model'])

    result_df = grouped_df.agg({
    'normalization' : 'first',
    'feature_extractor' : 'first',
    'mil_model' : 'first',
    'balanced_accuracy' : ['mean', 'std'],
    'auc' : ['mean', 'std']
    })

    if validation:
        ext_result_df = ext_grouped_df.agg({
        'normalization' : 'first',
        'feature_extractor' : 'first',
        'mil_model' : 'first',
        'balanced_accuracy' : ['mean', 'std'],
        'auc' : ['mean', 'std']
        })


    date = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")

    df.to_csv(f"{args.project_directory}/results_{date}_full.csv", index=False)
    result_df.to_csv(f"{args.project_directory}/results_{date}.csv", index=False)

    if validation:
        ext_df.to_csv(f"{args.project_directory}/ext_set_results_{date}_full.csv", index=False)
        ext_result_df.to_csv(f"{args.project_directory}/ext_set_results_{date}.csv", index=False)

    with open("test_run.pkl", 'wb') as f:
        pickle.dump(results, f)




if __name__ == "__main__":
    annotations = "../../Thom_Doeleman/annotations.csv"
    if not os.path.exists(f"{os.path.basename(annotations).strip('.csv')}_slideflow.csv"):
        process_annotation_file(annotations)
    main(easy=False, validation=True)
