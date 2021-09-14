import numpy as np
import os
import requests
import sys
from tqdm import tqdm


_warnings = set()


def warn_first_time(key, message):
    if key not in _warnings:
        print(f"WARNING: {message}", file=sys.stderr)
        _warnings.add(key)


def download_file(url, path, filesize=None):
    """Download a file with a tqdm progress bar.

    This will check if the file already exists, and skip it if it does. If the filesize
    is known, this function verifies that the function on disk has the right size, and
    redownloads it if it doesn't.

    Parameters
    ----------
    url : str
        URL to download from
    path : str
        Path on disk to download the file to.
    filesize : int, optional
        Size of the file (if known), by default None
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

    # Make sure that the directory exists.
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Download with a progress bar.
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get('content-length'))
        chunk_size = 128 * 1024
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True,
                            desc=url.split('/')[-1])
        with open(path, 'wb') as out_file:
            for chunk in response.iter_content(chunk_size):
                progress_bar.update(len(chunk))
                out_file.write(chunk)
        progress_bar.close()


def download_zenodo(record, basedir):
    """Download a record from Zenodo.

    Parameters
    ----------
    record : str
        Zenodo record number.
    basedir : str
        Directory to download the record to.
    """
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


def get_str_dtype_length(dtype):
    """Return the length of a string dtype handling unicode properly.

    Parameters
    ----------
    dtype : dtype
        dtype to check

    Returns
    -------
    int
        length of the string
    """
    if dtype.type == np.unicode_:
        return dtype.itemsize // 4
    else:
        return dtype.itemsize


def verify_unique(list_1, list_2, ignore_failure=False):
    """Verify that two lists have no elements in common.

    Parameters
    ----------
    list_1 : list
        First list
    list_2 : list
        Second list to compare
    ignore_failure : bool
        If True, raise a ValueError on failure. If False, return False

    Returns
    -------
    unique : bool
        Returns True if there are no overlapping elements between the two lists. Returns
        False if there are overlapping elements and ignore_failure is set to False.

    Raises
    ------
    ValueError
        If ignore_failure is False, raises a ValueError if there are overlapping
        elements.
    """
    ids_1 = set(list_1)
    ids_2 = set(list_2)
    common_ids = ids_1.intersection(ids_2)

    if common_ids:
        # Found an overlap.
        if ignore_failure:
            return False
        else:
            raise ValueError(
                f"Found overlap of {len(common_ids)} entries including "
                f"'{common_ids.pop()}'. Can't handle."
            )
    else:
        return True
