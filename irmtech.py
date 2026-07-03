import os
import SimpleITK as sitk
import numpy as np
import csv
from pathlib import Path
from radiomics import featureextractor

import irmtech_lib

RADIOMICS_CONFIG = {
    "imageType": {"Original": {}},
    "setting": {
        "binWidth": 1,
        "resampledPixelSpacing": None,
        "interpolator": "sitkBSpline",
        "normalize": False,
        "normalizeScale": 1,
        "removeOutliers": None
    },
    "featureClass": {
        "firstorder": [], "glcm": [], "gldm": [],
        "glrlm": [], "glszm": [], "ngtdm": []
    }
}

ADDITIONAL_COLUMNS = {
    # "minMax": 1,
    # "fbn": 32768
    "n4itk": 1,
    # "n4itk_mask": 1,
    # "zscore": 1,
    # "normalizeScale": 10,
    "binWidth": 1
}


def get_dicom_folders(base_directory: str) -> list[Path]:
    return [folder for folder in Path(base_directory).rglob('*') if folder.is_dir() and any(f.suffix.lower() == '.dcm' for f in folder.iterdir())]


def load_dicom_series(dicom_folder: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_folder))
    if not series_ids:
        raise RuntimeError(f"Aucune série DICOM détectée dans {dicom_folder}")
    dicom_files = reader.GetGDCMSeriesFileNames(str(dicom_folder), series_ids[0])
    reader.SetFileNames(dicom_files)
    reader.MetaDataDictionaryArrayUpdateOn()
    return reader.Execute()


def load_nifti_mask(mask_path: Path) -> sitk.Image:
    return sitk.ReadImage(str(mask_path))


def extract_radiomics_to_csv(dicom_folder: Path, mask_path: Path, csv_path: Path):
    print(f"Traitement du dossier : {dicom_folder} avec masque {mask_path.stem}")
    
    sitk_image = load_dicom_series(dicom_folder)
    sitk_mask = load_nifti_mask(mask_path)
    
    if sitk_mask.GetSize() != sitk_image.GetSize():
        raise ValueError(f"Le masque et l'image DICOM ont des dimensions incompatibles :\n - Image DICOM : {sitk_image.GetSize()}\n - Masque : {sitk_mask.GetSize()}")

    sitk_mask.CopyInformation(sitk_image)
    
    # ==============================
    # sitk_image = irmtech_lib.min_max_normalization(sitk_image)
    # sitk_image = irmtech_lib.fbn(sitk_image)
    sitk_image = irmtech_lib.n4itk(sitk_image, dicom_folder)
    # ==============================
    extractor = featureextractor.RadiomicsFeatureExtractor(RADIOMICS_CONFIG)
    result = extractor.execute(sitk_image, sitk_mask)
    
    filtered_results = {k: v for k, v in result.items() if k.startswith("original_")}
    
    date_folder = dicom_folder.parent.name
    acquisition_folder = dicom_folder.name
    
    new_columns = ["date", "acquisition", "mask"] + list(ADDITIONAL_COLUMNS.keys()) + list(filtered_results.keys())
    row_data = [date_folder, acquisition_folder, mask_path.stem] + list(ADDITIONAL_COLUMNS.values()) + list(filtered_results.values())

    file_exists = csv_path.exists()
    
    if file_exists:
        with open(csv_path, mode='r', newline='') as csv_file:
            reader = csv.reader(csv_file)
            current_header = next(reader, [])
            rows = list(reader)
    else:
        current_header = []
        rows = []

    if current_header:
        base_columns = ["date", "acquisition", "mask"]
        existing_radiomics = [col for col in current_header if col.startswith("original_")]
        existing_additional = [col for col in current_header if col not in base_columns + existing_radiomics]
        
        updated_header = base_columns + sorted(set(existing_additional + list(ADDITIONAL_COLUMNS.keys()))) + existing_radiomics
    else:
        updated_header = new_columns
    
    updated_rows = []
    for row in rows:
        row_dict = dict(zip(current_header, row))
        updated_row = [row_dict.get(col, "0") for col in updated_header]
        updated_rows.append(updated_row)

    row_dict = dict(zip(new_columns, row_data))
    new_row = [row_dict.get(col, "0") for col in updated_header]
    updated_rows.append(new_row)

    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(updated_header)
        writer.writerows(updated_rows)

    print(f"Features ajoutées au CSV : {csv_path}")


def process_masks_to_csv(base_directory: str, masks_folder: Path, csv_path: Path):
    masks = list(masks_folder.glob("*.nii"))
    if not masks:
        print("Aucun masque NIfTI trouvé.")
        return
    
    dicom_folders = get_dicom_folders(base_directory)
    if not dicom_folders:
        print("Aucun dossier DICOM trouvé.")
        return
    
    for mask_path in masks:
        print(f"Traitement du masque : {mask_path.name}")
        for dicom_folder in dicom_folders:
            extract_radiomics_to_csv(dicom_folder, mask_path, csv_path)


if __name__ == "__main__":
    base_directory = Path("src/data")
    masks_folder = Path("src/masks")
    output_csv_path = Path("radiomics_features.csv")
    
    process_masks_to_csv(base_directory, masks_folder, output_csv_path)