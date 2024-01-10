import slideflow as sf
import pandas as pd
import os

def process_annotation_file(original_path):
    df = pd.read_csv(original_path)
    df.rename(columns={'case_id' : 'patient', 'slide_id' : 'slide'}, inplace=True)
    df.to_csv(f"{os.path.basename(original_path)}_slideflow.csv", index=False)

def tile_wsis(dataset):
    dataset.extract_tiles(
    qc='both', #Both use Otsu Thresholding and Blur detection
    normalizer="macenko",
    whitespace_fraction="0.75"
    )




def main():

    project = sf.create_project(
    root = "./mf/",
    annotations = "../../train_list_definitive_slideflow.csv",
    slides = "../../MF_AI_dataset_cropped",
    )

    dataset = project.dataset(tile_px=512, tile_um="40x")
    print(dataset.summary())

    tile_wsis(dataset)

if __name__ == "__main__":
    annotations = "../../train_list_definitive.csv"
    if not os.path.exists(f"{os.path.basename(annotations)}_slideflow.csv"):
        process_annotation_file(annotations)
    main()
