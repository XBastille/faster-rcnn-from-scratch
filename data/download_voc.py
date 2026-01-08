import os
import shutil
import zipfile
import urllib.request
from tqdm import tqdm

def download_file(url, destination):
    """Download file with progress bar."""
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    print(f"Downloading from {url}...")
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=destination.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=destination, reporthook=t.update_to)


def merge_directories(src_dir, dst_dir):
    """Merge source directory into destination."""
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)
        
        if os.path.isdir(src_path):
            merge_directories(src_path, dst_path)
        else:
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)


def download_voc2007(root='./data'):
    """Download and setup VOC 2007."""
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    
    url = "https://www.kaggle.com/api/v1/datasets/download/zaraks/pascal-voc-2007"
    zip_path = os.path.join(root, "pascal_voc_2007.zip")
    
    if not os.path.exists(zip_path):
        print("Downloading dataset zip...")
        try:
            download_file(url, zip_path)
        except Exception as e:
            print(f"Error downloading: {e}")
            print("Please manually download the dataset to:", zip_path)
            return
    else:
        print(f"Found existing zip: {zip_path}")

    extract_path = os.path.join(root, "temp_extract")
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
    os.makedirs(extract_path)
    
    print("Extracting zip (this may take a while)...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    except zipfile.BadZipFile:
        print("Error: Zip file is corrupted. Please delete it and try again.")
        return

    target_devkit = os.path.join(root, 'VOCdevkit')
    target_voc2007 = os.path.join(target_devkit, 'VOC2007')
    os.makedirs(target_voc2007, exist_ok=True)
    print("Organizing files...")
    found_any = False
    for p_root, dirs, files in os.walk(extract_path):
        if 'VOC2007' in dirs:
            src_voc2007 = os.path.join(p_root, 'VOC2007')
            print(f"Found source data at: {src_voc2007}")
            merge_directories(src_voc2007, target_voc2007)
            found_any = True
            
    if not found_any:
        print("Warning: Could not find 'VOC2007' folder structure in zip.")
        print(f"Please check {extract_path} and manually organize.")
        return
    
    print("Cleaning up temporary files...")
    shutil.rmtree(extract_path)
    print(f"Done! Dataset setup at: {target_devkit}")
    print(f"Images: {len(os.listdir(os.path.join(target_voc2007, 'JPEGImages')))}")
    print(f"Annotations: {len(os.listdir(os.path.join(target_voc2007, 'Annotations')))}")


if __name__ == '__main__':
    download_voc2007()
