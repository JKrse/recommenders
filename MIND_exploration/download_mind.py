# ========================================================================

import os
import urllib.request
import zipfile

# ========================================================================

def mind_url(MIND_type: str="small"):
    """
    Get url for mind dataset
    """
    assert MIND_type in ["small", "large"]

    base_url = 'https://mind201910small.blob.core.windows.net/release'
    if MIND_type == "small":
        return(
        f'{base_url}/MINDsmall_train.zip',
        f'{base_url}/MINDsmall_dev.zip'
        )
    if MIND_type == "large":
        return(
        f'{base_url}/MINDlarge_train.zip',
        f'{base_url}/MINDlarge_dev.zip'    
        )


def download_url(url, temp_dir,
                 destination_filename=None,
                 progress_updater=None,
                 force_download=False,
                 verbose=True):
    """
    Download a URL to a temporary file
    """
    if not verbose:
        progress_updater = None
    # This is not intended to guarantee uniqueness, we just know it happens to guarantee
    # uniqueness for this application.
    if destination_filename is None:
        url_as_filename = url.replace('://', '_').replace('/', '_')
        destination_filename = \
            os.path.join(temp_dir,url_as_filename)
    if (not force_download) and (os.path.isfile(destination_filename)):
        if verbose:
            print('Bypassing download of already-downloaded file {}'.format(
                os.path.basename(url)))
        return destination_filename
    if verbose:
        print('Downloading file {} to {}'.format(os.path.basename(url),
                                                 destination_filename),
              end='')
    urllib.request.urlretrieve(url, destination_filename, progress_updater)
    assert (os.path.isfile(destination_filename))
    nBytes = os.path.getsize(destination_filename)
    if verbose:
        print('...done, {} bytes.'.format(nBytes))
    return destination_filename


def download_wrapper(url: str, temp_dir: str):
    """ 
    Wrapper function for downloading the MIND datasets
    """

    if not os.path.exists(temp_dir): 
        os.makedirs(temp_dir, exist_ok=True)

    file_name = url.split("/")[-1]

    print(f"Downloading {file_name} ...")
    zip_path = download_url(url, temp_dir, verbose=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    print(f"Downloading of {file_name} complete, saved at {temp_dir}")

