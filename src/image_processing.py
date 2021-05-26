from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

SCAN_DIR = Path("../data/scans/")
IMAGE_DIR = Path("../data/images/")
PATIENT_FILE = Path("../data/patient_classifications.csv")
IMAGE_SIZE = (150, 150)

class ImageProcessing:
    def __init__(self):
        """Extracts and pre-processes images from MRI scans. Scans should be run through FSL first, and separated by train/test.

        Args:
            desired_image_size (tuple, optional): Standard output image size. Defaults to (150, 150).
        """

        self.desired_image_size = IMAGE_SIZE
        self.scan_dir = SCAN_DIR
        self.image_dir = IMAGE_DIR

        self.train_dir = self.scan_dir.joinpath('train/')
        self.test_dir = self.scan_dir.joinpath('test/')
        self.patient_data = pd.read_csv(PATIENT_FILE)

        (self.X_test, self.y_test, self.X_train_parent, self.y_train_parent) = self.process_images()
        (self.X_train, self.X_val, self.y_train, self.y_val) = self.split_train_test()

        self.save_images()

    def process_images(self):
        """Run extraction and pre-processing pipeline for images and gather classifications.

        Returns:
            X_test (ndarray): Testing split
            y_test (ndarray): Testing classifications
            test_filenames (list): List of testing scan filenames
            X_train (ndarray): Training split
            y_train (ndarray): Training classifications
            train_filenames (list): List of training scan filenames
        """
        test_images, test_filenames = self.processing_pipeline(self.test_dir)
        train_images, train_filenames = self.processing_pipeline(self.train_dir)

        test_class = self.get_classifications(test_filenames)
        train_class = self.get_classifications(train_filenames)

        X_train, y_train = train_images, train_class
        X_test, y_test = test_images, test_class

        return X_test, to_categorical(y_test), X_train, to_categorical(y_train)

    def get_classifications(self, filenames, cdr_threshold=1):
        """Gather labels for each image.

        Args:
            filenames (list, str): List of filenames + paths per image
            cdr_threshold (int, optional): CDR threshold for considering patients to have dementia. Defaults to 1.

        Returns:
            ndarray: Labels for input filenames
        """
        sample_patient_id = [filename.name.split("_")[0] for filename in filenames]
        sample_condition = np.isin(self.patient_data.Subject, sample_patient_id)
        classifications = self.patient_data.loc[sample_condition, "cdr"].values
        return classifications >= cdr_threshold

    def processing_pipeline(self, scan_dir):
        """Extract slices through the temporal lobe from each scan

        Args:
            scan_dir (str): Path to scans

        Returns:
            ndarray: Output tensor of size (number scans, y size, x size, 1)
            list: List of scan filenames
        """
        images = []
        scan_filenames = np.sort(list(scan_dir.glob("*.nii.gz")))
        for scan_filename in scan_filenames:
            scan = nib.load(scan_filename).get_fdata()
            im0, im1, im2 = self.slice_temporal_lobe(scan)

            im = self.normalize_image(im1)  # keep im1 for now
            im = self.crop_image(im)
            images.append(im[np.newaxis])
        images = np.vstack(images)
        return images[:, :, :, np.newaxis], scan_filenames

    def slice_temporal_lobe(self, data):
        """Slice the input scan through the temporal lobe

        Args:
            data (ndarray): Scan of size (size y, size x, size z)

        Returns:
            ndarray: Slice along y axis
            ndarray: Slice along x axis
            ndarray: Slice along z axis
        """
        n_i, n_j, n_k = data.shape
        center_i = (n_i - 1) // 2
        center_j = (n_j - 1) // 2
        center_k = (n_k - 1) // 2
        slice_0 = data[center_i, :, :]
        slice_1 = data[:, center_j, :]
        slice_2 = data[:, :, center_k]
        return slice_0, slice_1, slice_2

    def normalize_image(self, im):
        """Normalize image to 99th percentile to account for bright pixels

        Args:
            im (ndarray): Input slice of size (y size, x size)

        Returns:
            ndarray: Normalized image slice
        """
        return im / np.percentile(im, 99)

    def split_train_test(self, test_size=0.2, random_state=440):
        """Split training sample into training and validation

        Args:
            X (ndarray, (L, M, N, 1)): Image tensor
            y (ndarray, (L, 1)): Image classifications
            test_size (float, optional): Fraction of training for validation. Defaults to 0.2.
            random_state (int, optional): Random seed. Defaults to 440.

        Returns:
            ndarray: Training and validation splits
        """
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_train_parent,
            self.y_train_parent,
            test_size=test_size,
            shuffle=True,
            stratify=self.y_train_parent,
            random_state=random_state,
        )
        return X_train, X_val, y_train, y_val

    def crop_image(self, im):
        """Crop 2D image to standard shape

        Args:
            im (ndarray): Input image

        Returns:
            ndarray: Cropped output image
        """
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

    def save_images(self):
        """Save training, validation, and testing images + labels
        """
        train_dir = self.image_dir.joinpath("train")
        val_dir = self.image_dir.joinpath("val")
        test_dir = self.image_dir.joinpath("test")

        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        for i, im in enumerate(self.X_train):
            out_file = train_dir.joinpath(str(i)+'.npy')
            with open(out_file, 'wb') as f:
                np.save(f, im)

        with open(train_dir.joinpath('train_labels.npy'), 'wb') as f:
            np.save(f, self.y_train)

        for i, im in enumerate(self.X_val):
            out_file = val_dir.joinpath(str(i)+'.npy')
            with open(out_file, 'wb') as f:
                np.save(f, im)

        with open(val_dir.joinpath('val_labels.npy'), 'wb') as f:
            np.save(f, self.y_val)

        for i, im in enumerate(self.X_test):
            out_file = test_dir.joinpath(str(i)+'.npy')
            with open(out_file, 'wb') as f:
                np.save(f, im)

        with open(test_dir.joinpath('test_labels.npy'), 'wb') as f:
            np.save(f, self.y_test)

