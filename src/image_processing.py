import pathlib
import nibabel as nib
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

train_dir = pathlib.Path("../data/train/")
test_dir = pathlib.Path("../data/test/")
patient_filename = pathlib.Path("../data/patient_classifications.csv")


class ImageProcessing:
    def __init__(self, desired_image_size=(150, 150)):
        """Extracts and pre-processes images from MRI scans. Scans should be run through FSL first, and separated by train/test.

        Args:
            desired_image_size (tuple, optional): Standard output image size. Defaults to (150, 150).
        """
        self.desired_image_size = desired_image_size
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.patient_data = pd.read_csv(patient_filename)

        (self.X_test, self.y_test, self.X_train, self.y_train) = self.process_images()

    def process_images(self):
        test_images, test_filenames = self.processing_pipeline(self.test_dir)
        train_images, train_filenames = self.processing_pipeline(self.train_dir)

        test_class = self.get_classifications(test_filenames)
        train_class = self.get_classifications(train_filenames)

        X_train, y_train = train_images, train_class
        X_test, y_test = test_images, test_class

        return X_test, to_categorical(y_test), X_train, to_categorical(y_train)

    def get_classifications(self, filenames, cdr_threshold=1):
        sample_patient_id = [filename.name.split("_")[0] for filename in filenames]
        sample_condition = np.isin(self.patient_data.Subject, sample_patient_id)
        classifications = self.patient_data.loc[sample_condition, "cdr"].values
        return classifications >= cdr_threshold

    def processing_pipeline(self, scan_dir):
        images = []
        scan_filenames = list(scan_dir.glob("*.nii.gz"))
        for scan_filename in scan_filenames:
            scan = nib.load(scan_filename).get_fdata()
            im0, im1, im2 = self.slice_temporal_lobe(scan)

            im = self.normalize_image(im1)  # keep im1 for now
            im = self.crop_image(im)
            images.append(im[np.newaxis])
        images = np.vstack(images)
        return images[:, :, :, np.newaxis], scan_filenames

    def slice_temporal_lobe(self, data):
        n_i, n_j, n_k = data.shape
        center_i = (n_i - 1) // 2
        center_j = (n_j - 1) // 2
        center_k = (n_k - 1) // 2
        slice_0 = data[center_i, :, :]
        slice_1 = data[:, center_j, :]
        slice_2 = data[:, :, center_k]
        return slice_0, slice_1, slice_2

    def normalize_image(self, im):
        return im / np.percentile(im, 99)

    def split_train_test(self, X, y, test_size=0.2, random_state=440):
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=test_size,
            shuffle=True,
            stratify=y,
            random_state=random_state,
        )
        return X_train, X_val, y_train, y_val

    def crop_image(self, im):
        diff_x = im.shape[1] - self.desired_image_size[1]
        diff_y = im.shape[0] - self.desired_image_size[0]
        assert diff_x >= 0, "Attempting to crop a smaller image"
        assert diff_y >= 0, "Attempting to crop a smaller image"

        if not (diff_x % 2):  # even size
            left, right = diff_x // 2, im.shape[1] - diff_x // 2
        else:
            left, right = diff_x // 2, (im.shape[1] - diff_x // 2) - 1
        if not (diff_y % 2):  # even size
            top, bottom = diff_y // 2, im.shape[0] - diff_y // 2
        else:
            top, bottom = diff_y // 2, (im.shape[0] - diff_y // 2) - 1
        return im[top:bottom, left:right]
