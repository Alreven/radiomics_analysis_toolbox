import numpy as np
import SimpleITK as sitk
from pathlib import Path

# ==============================
# Min Max Normalization
# ==============================
def min_max_normalization(image: sitk.Image) -> sitk.Image:
    array = sitk.GetArrayFromImage(image)
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val - min_val == 0:
        return image
    normalized_array = ((array - min_val) / (max_val - min_val)) * 65535
    normalized_image = sitk.GetImageFromArray(normalized_array.astype(np.uint16))
    normalized_image.CopyInformation(image)
    return normalized_image

# ==============================
# Percentiles Clipped Min Max Normalization
# ==============================
def min_max_normalization_clipped(image: sitk.Image, percentile: int = 5) -> sitk.Image:
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    lower_bound = np.percentile(array, percentile)
    upper_bound = np.percentile(array, 100 - percentile)
    array = np.clip(array, lower_bound, upper_bound)
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val - min_val == 0:
        return image
    normalized_array = ((array - min_val) / (max_val - min_val)) * 65535
    normalized_image = sitk.GetImageFromArray(normalized_array.astype(np.uint16))
    normalized_image.CopyInformation(image)
    return normalized_image

# ==============================
# FBN (Fixed bin number)
# ==============================
def fbn(image: sitk.Image) -> sitk.Image:
    NB_BIN = 256
    array = sitk.GetArrayFromImage(image)
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val - min_val == 0:
        return image
    normalized_array = np.floor( NB_BIN * (array - min_val) / (max_val - min_val)  )
    normalized_array = np.clip( normalized_array , a_min=None , a_max = NB_BIN - 1  )
    normalized_image = sitk.GetImageFromArray(normalized_array.astype(np.uint16))
    normalized_image.CopyInformation(image)
    return normalized_image

# ==============================
# N4ITK avec mise en cache persistante (NIfTI)
# ==============================

cache_dir = Path("cache_n4itk")
cache_dir.mkdir(exist_ok=True)

def save_to_nifti(image: sitk.Image, unique_id: str):
    file_path = cache_dir / f"{unique_id}.nii.gz"
    sitk.WriteImage(image, str(file_path))

def load_from_nifti(unique_id: str) -> sitk.Image:
    file_path = cache_dir / f"{unique_id}.nii.gz"
    if file_path.exists():
        return sitk.ReadImage(str(file_path))
    return None

def n4itk(image: sitk.Image, dicom_folder: Path, iterations=(30, 20, 10)) -> sitk.Image:
    unique_id = f"{dicom_folder.parent.name}{dicom_folder.name}"
    cached_image = load_from_nifti(unique_id)
    if cached_image:
        print(f"Chargement du cache depuis cache_n4itk/{unique_id}.nii.gz")
        cached_image.CopyInformation(image)
        return cached_image
    print(f"Correction N4ITK appliquée pour la première fois sur acquisition {unique_id}")
    image = sitk.Cast(image, sitk.sitkFloat32)
    n4_filter = sitk.N4BiasFieldCorrectionImageFilter()
    n4_filter.SetMaximumNumberOfIterations(iterations)
    corrected_image = n4_filter.Execute(image)
    corrected_image.CopyInformation(image)
    save_to_nifti(corrected_image, unique_id)
    return corrected_image