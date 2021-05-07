import os
import requests
from tqdm import tqdm
from urllib.request import urlretrieve


def download_file(url, path, filesize=None):
    """Download a file with a tqdm progress bar. This code is adapted from an
    example in the tqdm documentation.
    """
    # Check if the file already exists, and don't download it if it does.
    if os.path.exists(path):
        # Make sure that we have a full download
        if filesize is None or os.path.getsize(path) == filesize:
            print("Skipping %s, already exists." % os.path.basename(path))
            return
        else:
            print("Found incomplete download of %s, retrying." %
                  os.path.basename(path))
            os.remove(path)

    class TqdmUpTo(tqdm):
        """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
        def update_to(self, b=1, bsize=1, tsize=None):
            """
            b  : int, optional
                Number of blocks transferred so far [default: 1].
            bsize  : int, optional
                Size of each block (in tqdm units) [default: 1].
            tsize  : int, optional
                Total size (in tqdm units). If [default: None] remains unchanged.
            """
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)  # will also set self.n = b * bsize

    with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                  desc=url.split('/')[-1]) as t:  # all optional kwargs
        urlretrieve(url, filename=path, reporthook=t.update_to, data=None)


def download_zenodo(record, basedir):
    # Make the download directory if it doesn't exist.
    os.makedirs(basedir, exist_ok=True)

    # Download a dataset from zenodo.
    zenodo_url = f"https://zenodo.org/api/records/{record}"
    zenodo_metadata = requests.get(zenodo_url).json()

    for file_metadata in zenodo_metadata['files']:
        path = os.path.join(basedir, file_metadata['key'])
        url = file_metadata['links']['self']
        filesize = file_metadata['size']

        download_file(url, path, filesize)
