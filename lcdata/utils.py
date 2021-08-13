import numpy as np
import os
import sys
from urllib.request import urlretrieve


_warnings = set()


try:
    # Load tqdm if available. If not, print a message every time that it is called
    # instead.
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        print("Install tqdm to see a progress bar.")
        return iterable


def warn_first_time(key, message):
    if key not in _warnings:
        print(f"WARNING: {message}", file=sys.stderr)
        _warnings.add(key)


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

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    if tqdm is not None:
        # Download with a progress bar.
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
    else:
        # Download without a progress bar.
        print(f"Downloading {url}...")
        print("Install tqdm to see a progress bar.")
        urlretrieve(url, filename=path, data=None)


def download_zenodo(record, basedir):
    import requests

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


_session_id = None
_object_id_count = 0


def generate_object_id():
    """Generate a unique object ID for a light curve.

    We want to make this unique but readable. It is also important that if different
    datasets are generated with different runs of they program they have different IDs.
    To accomplish this, we use the format lcdata_[random session string]_[count]. The
    random_session_string will be consistent for all of the light curves generated in
    the same session. The count will start from zero and increase.
    """
    global _session_id
    global _object_id_count

    if _session_id is None:
        import random
        import string
        _session_id = ''.join(random.choice(string.ascii_lowercase) for i in range(6))

    object_id = f'lcdata_{_session_id}_{_object_id_count:07d}'
    _object_id_count += 1

    return object_id
