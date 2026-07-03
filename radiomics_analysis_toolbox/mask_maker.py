import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

DICOM_FOLDER = '.'
X = 28
COMMON_FACES_THRESHOLD = 3
NB_VOXELS_EROSION = 1
VOXEL_LIMIT = 6000

CYL_RADIUS = 4
CYL_LENGTH = 30

SEARCH_RANGE_Z = 10
SEARCH_RANGE_Y = 20
SEARCH_RANGE_X = 10

reader = sitk.ImageSeriesReader()
series_ids = reader.GetGDCMSeriesIDs(DICOM_FOLDER)
if not series_ids:
    raise RuntimeError(f"Aucune série DICOM détectée dans : {DICOM_FOLDER}")

first_series_id = series_ids[0]
dicom_names = reader.GetGDCMSeriesFileNames(DICOM_FOLDER, first_series_id)
if not dicom_names:
    raise RuntimeError(f"Aucun fichier DICOM pour la série {first_series_id} dans {DICOM_FOLDER}")

print(f"Chargement de {len(dicom_names)} fichiers DICOM...")
reader.SetFileNames(dicom_names)
dicom_sitk = reader.Execute()

img_3d = sitk.GetArrayFromImage(dicom_sitk)
num_slices, num_rows, num_cols = img_3d.shape
print(f"Image 3D : shape (Z, Y, X) = {img_3d.shape}")

mask_3d = np.zeros_like(img_3d, dtype=np.uint8)

axial_idx    = num_slices // 2
sagittal_idx = num_cols   // 2
coronal_idx  = num_rows   // 2

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
(axial_ax, sagittal_ax, coronal_ax, instruction_ax) = axes.ravel()

def create_overlay(mask_slice):
    overlay = np.zeros((*mask_slice.shape, 4))
    overlay[..., 0] = 1.0
    overlay[..., 1] = 0.5
    overlay[..., 2] = 0.0
    overlay[..., 3] = mask_slice * 0.7
    return overlay

axial_im    = axial_ax.imshow(img_3d[axial_idx],                cmap='gray', origin='lower')
sagittal_im = sagittal_ax.imshow(img_3d[:, :, sagittal_idx],    cmap='gray', origin='lower')
coronal_im  = coronal_ax.imshow(img_3d[:, coronal_idx, :],      cmap='gray', origin='lower')

axial_mask_im    = axial_ax.imshow(create_overlay(mask_3d[axial_idx]),                origin='lower')
sagittal_mask_im = sagittal_ax.imshow(create_overlay(mask_3d[:, :, sagittal_idx]),    origin='lower')
coronal_mask_im  = coronal_ax.imshow(create_overlay(mask_3d[:, coronal_idx, :]),      origin='lower')

axial_ax.set_title(f'Axial (Z={axial_idx})')
sagittal_ax.set_title(f'Sagittal (X={sagittal_idx})')
coronal_ax.set_title(f'Coronal (Y={coronal_idx})')

instructions = """\
Instructions:
- Zoom : w (in), x (out)
- Ajout région (vue Coronal) : c
- Reset masque : v
- Export masque : b
- Navigation : a (préc.), z (suiv.)
"""
instruction_ax.text(0.5, 0.5, instructions, ha='center', va='center', fontsize=12)
instruction_ax.axis('off')

last_mouse_pos = None
last_mouse_ax  = None
selected_voxel_value = None

def count_common_faces(z, y, x):
    neighbors = [
        (z+1, y,   x), (z-1, y,   x),
        (z,   y+1, x), (z,   y-1, x),
        (z,   y,   x+1), (z,   y,   x-1)
    ]
    c = 0
    for nz, ny, nx in neighbors:
        if 0 <= nz < num_slices and 0 <= ny < num_rows and 0 <= nx < num_cols:
            if mask_3d[nz, ny, nx] == 1:
                c += 1
    return c

def region_growing(seed_z, seed_y, seed_x):
    global selected_voxel_value
    if not (0 <= seed_z < num_slices and 0 <= seed_y < num_rows and 0 <= seed_x < num_cols):
        return
    if mask_3d[seed_z, seed_y, seed_x] == 1:
        return

    threshold = selected_voxel_value - X if selected_voxel_value is not None else 0

    vox_count = 0  

    if img_3d[seed_z, seed_y, seed_x] >= threshold:
        mask_3d[seed_z, seed_y, seed_x] = 1
        vox_count += 1
    elif count_common_faces(seed_z, seed_y, seed_x) >= COMMON_FACES_THRESHOLD:
        mask_3d[seed_z, seed_y, seed_x] = 1
        vox_count += 1
    else:
        return

    queue = deque([(seed_z, seed_y, seed_x)])
    visited = set()

    while queue:
        z, y, x = queue.popleft()
        if (z, y, x) in visited:
            continue
        visited.add((z, y, x))

        if vox_count > VOXEL_LIMIT:
            print(f"\n[INFO] Arrêt de la croissance, {VOXEL_LIMIT} voxels atteints.")
            replace_with_cylinder_3Dsearch(seed_z, seed_y, seed_x, selected_voxel_value)
            return

        neigh = [
            (z+1, y,   x), (z-1, y,   x),
            (z,   y+1, x), (z,   y-1, x),
            (z,   y,   x+1), (z,   y,   x-1)
        ]
        for nz, ny, nx in neigh:
            if 0 <= nz < num_slices and 0 <= ny < num_rows and 0 <= nx < num_cols:
                if mask_3d[nz, ny, nx] == 0:
                    val = img_3d[nz, ny, nx]
                    if val >= threshold:
                        mask_3d[nz, ny, nx] = 1
                        queue.append((nz, ny, nx))
                        vox_count += 1
                    else:
                        if count_common_faces(nz, ny, nx) >= COMMON_FACES_THRESHOLD:
                            mask_3d[nz, ny, nx] = 1
                            queue.append((nz, ny, nx))
                            vox_count += 1

    if vox_count > VOXEL_LIMIT:
        print(f"\n[INFO] Mask dépasse {VOXEL_LIMIT} voxels ({vox_count}).")
        replace_with_cylinder_3Dsearch(seed_z, seed_y, seed_x, selected_voxel_value)

def apply_erosion():
    if np.any(mask_3d):
        temp_sitk = sitk.GetImageFromArray(mask_3d.astype(np.uint8))
        eroder = sitk.BinaryErodeImageFilter()
        eroder.SetKernelRadius(NB_VOXELS_EROSION)
        eroder.SetKernelType(sitk.sitkCross)
        out = eroder.Execute(temp_sitk)
        mask_3d[:, :, :] = sitk.GetArrayFromImage(out).astype(np.uint8)

def replace_with_cylinder_3Dsearch(seed_z, seed_y, seed_x, seed_value):
    mask_3d.fill(0)
    best_score = -1
    best_center = (seed_z, seed_y, seed_x)

    min_intensity = seed_value - X
    max_intensity = seed_value + X
    half_len = CYL_LENGTH // 2
    radius2 = CYL_RADIUS * CYL_RADIUS

    for dz_ in range(-SEARCH_RANGE_Z, SEARCH_RANGE_Z + 1):
        for dy_ in range(-SEARCH_RANGE_Y, SEARCH_RANGE_Y + 1):
            for dx_ in range(-SEARCH_RANGE_X, SEARCH_RANGE_X + 1):
                zC = seed_z + dz_
                yC = seed_y + dy_
                xC = seed_x + dx_

                y_start = yC - half_len
                y_end   = y_start + CYL_LENGTH

                z_min = zC - CYL_RADIUS
                z_max = zC + CYL_RADIUS
                x_min = xC - CYL_RADIUS
                x_max = xC + CYL_RADIUS

                score = 0
                for zz in range(z_min, z_max+1):
                    if not (0 <= zz < num_slices):
                        continue
                    dz2 = (zz - zC)*(zz - zC)
                    for yy in range(y_start, y_end):
                        if not (0 <= yy < num_rows):
                            continue
                        for xx in range(x_min, x_max+1):
                            if not (0 <= xx < num_cols):
                                continue
                            dx2 = (xx - xC)*(xx - xC)
                            if dz2 + dx2 <= radius2:
                                val = img_3d[zz, yy, xx]
                                if min_intensity <= val <= max_intensity:
                                    score += 1

                if score > best_score:
                    best_score = score
                    best_center = (zC, yC, xC)

    print(f"[INFO] Meilleur centre = {best_center} (score={best_score})")
    place_cylinder(*best_center)

def place_cylinder(zC, yC, xC):
    mask_3d.fill(0)
    half_len = CYL_LENGTH // 2
    radius2  = CYL_RADIUS * CYL_RADIUS

    y_start = yC - half_len
    y_end   = y_start + CYL_LENGTH

    z_min = zC - CYL_RADIUS
    z_max = zC + CYL_RADIUS
    x_min = xC - CYL_RADIUS
    x_max = xC + CYL_RADIUS

    for zz in range(z_min, z_max+1):
        if not (0 <= zz < num_slices):
            continue
        dz2 = (zz - zC)*(zz - zC)
        for yy in range(y_start, y_end):
            if not (0 <= yy < num_rows):
                continue
            for xx in range(x_min, x_max+1):
                if not (0 <= xx < num_cols):
                    continue
                dx2 = (xx - xC)*(xx - xC)
                if dz2 + dx2 <= radius2:
                    mask_3d[zz, yy, xx] = 1

def update_views():
    axial_im.set_data(img_3d[axial_idx])
    sagittal_im.set_data(img_3d[:, :, sagittal_idx])
    coronal_im.set_data(img_3d[:, coronal_idx, :])

    axial_mask_im.set_data(create_overlay(mask_3d[axial_idx]))
    sagittal_mask_im.set_data(create_overlay(mask_3d[:, :, sagittal_idx]))
    coronal_mask_im.set_data(create_overlay(mask_3d[:, coronal_idx, :]))

    axial_ax.set_title(f'Axial (Z={axial_idx})')
    sagittal_ax.set_title(f'Sagittal (X={sagittal_idx})')
    coronal_ax.set_title(f'Coronal (Y={coronal_idx})')

    fig.canvas.draw_idle()

def on_mouse_move(event):
    global last_mouse_pos, last_mouse_ax
    if event.inaxes in [axial_ax, sagittal_ax, coronal_ax]:
        last_mouse_pos = (event.xdata, event.ydata)
        last_mouse_ax  = event.inaxes

def zoom(event, zoom_in=True):
    if last_mouse_ax is None or last_mouse_pos is None:
        return
    ax = last_mouse_ax
    x, y = last_mouse_pos
    factor = (1/1.2) if zoom_in else 1.2

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim([x - (x - xlim[0]) * factor,
                 x + (xlim[1] - x) * factor])
    ax.set_ylim([y - (y - ylim[0]) * factor,
                 y + (ylim[1] - y) * factor])
    fig.canvas.draw_idle()

def on_key_press(event):
    global axial_idx, sagittal_idx, coronal_idx, selected_voxel_value

    if event.key == 'c' and last_mouse_ax == coronal_ax and last_mouse_pos:
        x = int(np.clip(last_mouse_pos[0], 0, num_cols   - 1))
        z = int(np.clip(last_mouse_pos[1], 0, num_slices - 1))
        selected_voxel_value = img_3d[z, coronal_idx, x]

        region_growing(z, coronal_idx, x)
        apply_erosion()
        update_views()

    elif event.key == 'v':
        mask_3d.fill(0)
        update_views()

    elif event.key == 'b':
        mask_sitk = sitk.GetImageFromArray(mask_3d.astype(np.uint8))
        mask_sitk.CopyInformation(dicom_sitk)

        sitk.WriteImage(mask_sitk, 'mask_volume.nii')
        print("Masque exporté : mask_volume.nii")

    elif event.key == 'a':
        if last_mouse_ax == axial_ax:
            axial_idx = max(axial_idx - 1, 0)
        elif last_mouse_ax == sagittal_ax:
            sagittal_idx = max(sagittal_idx - 1, 0)
        elif last_mouse_ax == coronal_ax:
            coronal_idx = max(coronal_idx - 1, 0)
        update_views()

    elif event.key == 'z':
        if last_mouse_ax == axial_ax:
            axial_idx = min(axial_idx + 1, num_slices - 1)
        elif last_mouse_ax == sagittal_ax:
            sagittal_idx = min(sagittal_idx + 1, num_cols - 1)
        elif last_mouse_ax == coronal_ax:
            coronal_idx = min(coronal_idx + 1, num_rows - 1)
        update_views()

    elif event.key == 'w':
        zoom(event, True)
    elif event.key == 'x':
        zoom(event, False)

fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
fig.canvas.mpl_connect('key_press_event', on_key_press)
plt.show()