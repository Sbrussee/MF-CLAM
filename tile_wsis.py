import slideflow as sf


def tile_wsis(dataset):
    dataset.extract_tiles(
    qc='both', #Both use Otsu Thresholding and Blur detection
    normalizer="macenko",
    whitespace_fraction="0.75"
    )












def main():
    project = sf.create_project(
    root="./mf/"
    annotations = "../../train_list_definitive.csv",
    slides = "../../MF_AI_dataset_cropped"
    )

    dataset = project.dataset(tile_px=512, tile_um="40x")
    print(dataset.summary())

    tile_wsis(dataset)

if __name__ == "__main__":
    main()
