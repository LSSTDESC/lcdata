import numpy as np
import os
import sys
from urllib.request import urlretrieve


_warnings = set()


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


def find_alias(keyword, names, aliases, ignore_failure=False):
    """Find an alias for a given keyword

    Inspired by and very similar to `sncosmo.alias_map`.

    Parameters
    ----------
    keyword : str
        Keyword to search for
    names : list[str]
        List of names that are available
    aliases : list[str]
        List of aliases to search through. The first one that is available will be
        returned.
    ignore_failure : bool
        If True, raise a ValueError on failure. If False, return None

    Returns
    -------
    alias : str
        The matching alias.
    """
    lowered_names = [i.lower() for i in names]
    for alias in aliases:
        if alias in lowered_names:
            return alias

    if ignore_failure:
        return None
    else:
        raise ValueError(f"Couldn't find key {keyword}. Possible aliases {aliases}.")


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
