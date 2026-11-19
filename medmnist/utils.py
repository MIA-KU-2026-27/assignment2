import os
from PIL import Image
from tqdm import trange


SPLIT_DICT = {
    "train": "TRAIN",
    "val": "VALIDATION",
    "test": "TEST"
}  # compatible for Google AutoML Vision


def save2d(imgs, labels, img_folder,
           split, postfix, csv_path):
    print(f"Saving {split} set to {img_folder}, csv_path={csv_path}...")
    return save_fn(imgs, labels, img_folder,
                   split, postfix, csv_path,
                   load_fn=lambda arr: Image.fromarray(arr),
                   save_fn=lambda img, path: img.save(path))


def save_fn(imgs, labels, img_folder,
            split, postfix, csv_path,
            load_fn, save_fn):

    assert imgs.shape[0] == labels.shape[0]

    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    if csv_path is not None:
        csv_file = open(csv_path, "a")

    for idx in trange(imgs.shape[0]):

        img = load_fn(imgs[idx])

        label = labels[idx]

        file_name = f"{split}{idx}_{'_'.join(map(str,label))}.{postfix}"

        save_fn(img, os.path.join(img_folder, file_name))

        if csv_path is not None:
            line = f"{SPLIT_DICT[split]},{file_name},{','.join(map(str,label))}\n"
            csv_file.write(line)

    if csv_path is not None:
        csv_file.close()


def load_frames(arr):
    frames = []
    for frame in arr:
        frames.append(Image.fromarray(frame))
    return frames


def save_frames_as_gif(frames, path, duration=200):
    assert path.endswith(".gif")
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0)