from torch.utils.data import Dataset
import glob
from skimage import io

class CustomImageNetTest(Dataset):
    """Test ImageNet Dataset"""

    def __init__(self, folder_path, transform=None):
        """
        Args:
            folder_path (string): Path to the png file with annotations
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.folder_path = folder_path
        self.transform = transform

        self.image_files = glob.glob(self.folder_path  + "/*.png")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_files[idx]
        image = io.imread(img_name)
        class_id = int(img_name[16:24])

        if self.transform:
            image = self.transform(image)

        return (image,class_id)
