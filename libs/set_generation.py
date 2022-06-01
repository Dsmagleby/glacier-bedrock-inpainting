import numpy as np
import random
import cv2

from time import sleep
from tqdm import tqdm



def create_train_images_small(image, mask, patch_size, max_iter, threshold=None, mode=None, random_state=None, invalid_value=-32767.):

    i_h, i_w = image.shape[:2]
    s_h, s_w = mask.shape[:2]
    p_h, p_w = patch_size

    if p_h > i_h:
        raise ValueError(
            "Height of the patch should be less than the height of the image."
        )

    if p_w > i_w:
        raise ValueError(
            "Width of the patch should be less than the width of the image."
        )

    if i_h != s_h:
        raise ValueError(
            "Height of the mask should equal to the height of the image."
        )

    if i_w != s_w:
        raise ValueError(
            "Width of the mask should equal to the width of the image."
            )

    size = p_h * p_w
    patches = []
    while True:

        for _ in range(max_iter):
            append = True
            rng = random.seed(random_state)
            i_s = random.randint(0, i_h - p_h + 1)
            j_s = random.randint(0, i_w - p_w + 1)
            patch = image[i_s:i_s+p_h, j_s:j_s+p_w]
            mask_patch = mask[i_s:i_s+p_h, j_s:j_s+p_w]

            if invalid_value in patch:
                append = False
            elif np.sum(mask_patch) != 0:
                append = False

            # mode = max:     max of image must be above threshold
            # mode = average: average of image must be above threshold
            elif mode == 'max':
                max = np.max(patch)
                if max < threshold:
                    append = False
            elif mode == 'average':
                avg = np.mean(patch)
                if avg < threshold:
                    append = False

            if append and patch.size == size:
                patches.append(patch)


        if len(patches) > 20:
            break

    return np.array(patches)


def flow_train_images(image, mask, patch_size, max_iter, threshold=None, mode=None, random_state=None, invalid_value=-32767.):
    flow = create_train_images_small(image, mask, patch_size, max_iter, threshold, mode, random_state, invalid_value)
    return flow[..., np.newaxis]


def create_test_images_from_glacier_center(image, mask, patch_size, coords_frame, create_blank=False, random_state=None, invalid_value=-32767.):

    i_h, i_w = image.shape[:2]
    s_h, s_w = mask.shape[:2]
    p_h, p_w = patch_size

    if p_h > i_h:
        raise ValueError(
            "Height of the patch should be less than the height of the image."
        )

    if p_w > i_w:
        raise ValueError(
            "Width of the patch should be less than the width of the image."
        )

    if i_h != s_h:
        raise ValueError(
            "Height of the mask should equal to the height of the image."
        )

    if i_w != s_w:
        raise ValueError(
            "Width of the mask should equal to the width of the image."
            )

    size = p_h * p_w
    patches = []
    RGI = []
    masks = []



    rows = np.array(coords_frame['rows'].tolist())
    cols = np.array(coords_frame['cols'].tolist())
    RGIId = coords_frame['RGIId'].tolist()

    for r, c, id in tqdm(zip(rows, cols, RGIId)):
        r = r - int(p_h/2) if r >= int(p_h/2) else r
        c = c - int(p_w/2) if c >= int(p_w/2) else c

        append = True
        patch = image[r:r+p_h, c:c+p_w]
        mask_patch = mask[r:r+p_h, c:c+p_w]

        # extract glacier at center from mask_patch
        # only return the singular glacier mask
        center_value = mask_patch[int(p_h/2), int(p_w/2)]
        # if glacier is very small, look for it in surrounding pixels

        single_mask = np.zeros_like(mask_patch)
        if center_value == 0:
            mid_right = mask_patch[int(p_w/2),int(p_w/2):]
            mid_left  = mask_patch[int(p_w/2),:int(p_w/2)+1][::-1]
            mid_bot   = mask_patch[int(p_h/2):,int(p_h/2)]
            mid_top   = mask_patch[:int(p_h/2)+1,int(p_h/2)][::-1]

            if np.sum(mid_right) != 0 and np.sum(mid_left) != 0 and np.sum(mid_bot) != 0 and np.sum(mid_top) != 0:
                mask_value = np.array([
                    mid_right[np.nonzero(mid_right)][0],
                    mid_left[np.nonzero(mid_left)][0],
                    mid_bot[np.nonzero(mid_bot)][0],
                    mid_top[np.nonzero(mid_top)][0]
                ])
                if np.all(mask_value == mask_value[0]):
                    mask_coords = np.nonzero(mask_patch == mask_value[0])
                    single_mask[mask_coords] = 1
                else:
                    if not create_blank:
                        append = False

            else:
                if not create_blank:
                    append = False

        else:
            mask_coords = np.nonzero(mask_patch == center_value)
            single_mask[mask_coords] = 1

        #    # 3x3
        #    center_matrix = mask_patch[int(p_h/2)-1:int(p_h/2)+2,
        #                               int(p_w/2)-1:int(p_w/2)+2].flatten()
        #    values, counts = np.unique(center_matrix, return_counts=True)
        #    if np.sum(values) != 0:
        #        center_value = center_matrix[np.nonzero(center_matrix)[0][0]]
        #    else:
        #        append = False

        if invalid_value in patch:
            append = False

        # at low resolution, some glaciers are not visible
        # after translation from coords to xy
        elif np.sum(single_mask) == 0:
            if not create_blank:
                append = False

        if append and patch.size == size:
            patches.append(patch)
            masks.append(single_mask)
            RGI.append(id)

    return np.array(patches), np.array(masks), np.array(RGI)


def create_dataset_train(image, mask, patch_size, max_iter, seen, threshold=None, mode=None, random_state=None, invalid_value=-32767.):

    i_h, i_w = image.shape[:2]
    s_h, s_w = mask.shape[:2]
    p_h, p_w = patch_size

    if p_h > i_h:
        raise ValueError(
            "Height of the patch should be less than the height of the image."
        )

    if p_w > i_w:
        raise ValueError(
            "Width of the patch should be less than the width of the image."
        )

    if i_h != s_h:
        raise ValueError(
            "Height of the mask should equal to the height of the image."
        )

    if i_w != s_w:
        raise ValueError(
            "Width of the mask should equal to the width of the image."
            )

    size = p_h * p_w
    seen = seen
    patches = []

    for _ in range(max_iter):
        append = True
        rng = random.seed(random_state)
        i_s = random.randint(0, i_h - p_h + 1)
        j_s = random.randint(0, i_w - p_w + 1)
        patch = image[i_s:i_s+p_h, j_s:j_s+p_w]

        center_i = int(i_s + (p_h/2))
        center_j = int(j_s + (p_w/2))

        mask_patch = mask[i_s:i_s+p_h, j_s:j_s+p_w]

        if invalid_value in patch:
            append = False
        if np.sum(mask_patch[int(p_h/2)-47:int(p_h/2)+48, int(p_w/2)-47:int(p_w/2)+48]) != 0:
            append = False
        if np.sum(seen[center_i-15:center_i+16, center_j-15:center_j+16]) > 200:
            append = False

        # modified for other mountain ranges
        if np.amax(patch) > 4800:
            append = False

        # mode = max:     max of image must be above threshold
        # mode = average: average of image must be above threshold
        if mode == 'max':
            max = np.max(patch[int(p_h/2)-47:int(p_h/2)+48, int(p_w/2)-47:int(p_w/2)+48])
            if max < threshold:
                append = False
        if mode == 'average':
            avg = np.mean(patch[int(p_h/2)-47:int(p_h/2)+48, int(p_w/2)-47:int(p_w/2)+48])
            if avg < threshold:
                append = False

        if append and patch.size == size:
            seen[center_i-15:center_i+16, center_j-15:center_j+16] += 1
            patches.append(patch)

        if len(patches) == 10:
            break


    return np.array(patches), seen

def flow_train_dataset(image, mask, patch_size, train_path, val_path, threshold=None, mode=None, total=15000, postfix='', random_state=None, invalid_value=-32767.):
    seen = np.zeros_like(image)

    num, limit = 0, total
    pbar = tqdm(total = limit)
    while True:
        flow, seen = create_dataset_train(image, mask, patch_size, 50000, seen, threshold, mode, random_state, invalid_value)
        flow = flow[..., np.newaxis]

        if len(flow) < 10:
            print("Unable to find new patches")
            break

        for img in flow[:-1]:
            cv2.imwrite(train_path + str(num) + postfix + '.tif', img.astype(np.uint16))
            num += 1

        cv2.imwrite(val_path + str(num) + postfix + '.tif', flow[-1].astype(np.uint16))
        num += 1

        pbar.update(10)
        if num >= limit:
            break

    pbar.close()

    return seen


def create_test_images_full_noedge(image, mask, patch_size, coords_frame, create_blank=False, random_state=None, invalid_value=-32767.):

    i_h, i_w = image.shape[:2]
    s_h, s_w = mask.shape[:2]
    p_h, p_w = patch_size

    if p_h > i_h:
        raise ValueError(
            "Height of the patch should be less than the height of the image."
        )

    if p_w > i_w:
        raise ValueError(
            "Width of the patch should be less than the width of the image."
        )

    if i_h != s_h:
        raise ValueError(
            "Height of the mask should equal to the height of the image."
        )

    if i_w != s_w:
        raise ValueError(
            "Width of the mask should equal to the width of the image."
            )

    size = p_h * p_w
    patches = []
    RGI = []
    masks = []

    rows = np.array(coords_frame['rows'].tolist())
    cols = np.array(coords_frame['cols'].tolist())
    RGIId = coords_frame['RGIId'].tolist()

    for r, c, id in tqdm(zip(rows, cols, RGIId)):
        r = r - int(p_h/2) if r >= int(p_h/2) else r
        c = c - int(p_w/2) if c >= int(p_w/2) else c

        append = True
        patch = image[r:r+p_h, c:c+p_w]
        mask_patch = mask[r:r+p_h, c:c+p_w]

        # extract mask_patch edge
        #edge = [mask_patch[0,:-1], mask_patch[:-1,-1], mask_patch[-1,::-1], mask_patch[-2:0:-1,0]]
        #edge = np.concatenate(edge)

        # remove glaciers bordering the edge of the mask
        #for i in np.unique(edge):
        #    if i != 0:
        #        mask_coords = np.nonzero(mask_patch == i)
        #        mask_patch[mask_coords] = 0

        # convert all mask values from labeled to 1
        mask_patch[mask_patch > 0] = 1

        if invalid_value in patch:
            append = False

        # at low resolution, some glaciers are not visible
        # after translation from coords to xy
        elif np.sum(mask_patch) == 0:
            if not create_blank:
                append = False

        if append and patch.size == size:
            patches.append(patch)
            masks.append(mask_patch)
            RGI.append(id)

    return np.array(patches), np.array(masks), np.array(RGI)
