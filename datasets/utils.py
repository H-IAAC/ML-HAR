import errno
import hashlib
import os
import os.path
import csv
from torch.utils.model_zoo import tqdm
import torch
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline      # for warping
#from transforms3d.axangles import axangle2mat  # for rotation

def rearrange(a,y, window, overlap):
    l, f = a.shape
    shape = (int( (l-overlap)/(window-overlap) ), window, f)
    stride = (a.itemsize*f*(window-overlap), a.itemsize*f, a.itemsize)
    X = np.lib.stride_tricks.as_strided(a, shape=shape, strides=stride)

    l,f = y.shape
    shape = (int( (l-overlap)/(window-overlap) ), window, f)
    stride = (y.itemsize*f*(window-overlap), y.itemsize*f, y.itemsize)
    Y = np.lib.stride_tricks.as_strided(y, shape=shape, strides=stride)
    Y = Y.max(axis=1)

    return X, Y.flatten()

def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def check_integrity(fpath, md5=None):
    if md5 is None:
        return True
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater()
            )
        except OSError:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater()
                )


def list_dir(root, prefix=False):
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def download_file_from_google_drive(file_id, root, filename=None, md5=None):
    """Download a Google Drive file from  and place it in root.
    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    import requests
    url = "https://docs.google.com/uc?export=download"

    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        session = requests.Session()

        response = session.get(url, params={'id': file_id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(url, params=params, stream=True)

        _save_response_content(response, fpath)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None



def _save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        pbar = tqdm(total=None)
        progress = 0
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                progress += len(chunk)
                pbar.update(progress - pbar.n)
        pbar.close()
        
def concat_samples(datasetDst, datasetSample):
     
     datasetDst.X = torch.cat((datasetDst.X, datasetSample.X),0)
     datasetDst.Y = torch.cat((datasetDst.Y, datasetSample.Y),0)

     return datasetDst
 
    
 
def find_column_index(csv_file, column_name):
    with open(csv_file, 'r', newline='') as csv_file:
        # Set the space as the delimiter for the reader
        reader = csv.reader(csv_file, delimiter=' ')
        header_row = next(reader)
        try:
            column_index = header_row.index(column_name)
            return column_index
        except ValueError:
            raise ValueError(f"Column '{column_name}' not found in the CSV file.")

def extract_column_to_txt(csv_file, column_name, txt_file, column_type):
    column_index = find_column_index(csv_file, column_name)

    with open(csv_file, 'r', newline='') as csv_file:
        # Set the space as the delimiter for the reader
        reader = csv.reader(csv_file, delimiter=' ')
        next(reader)
        column_data = [row[column_index] for row in reader]
        
        
    with open(txt_file, 'w') as txt_file:
         lines = []
         for value in column_data:
             if column_type == 'int':
                lines.append(str(int(float(value))))
             else:
                lines.append(value)
         txt_file.write('\n'.join(lines))
'''
    with open(txt_file, 'w') as txt_file:
        for value in column_data:
            if column_type == 'int':
               txt_file.write(str(int(float(value))) + '\n')
            else:
               txt_file.write(value + '\n') 
'''


def create_directory (path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            pass


def remove_columns(csv_file, columns, txt_file):
    df = pd.read_csv(csv_file, delimiter=' ')
    df = df.drop(columns=columns)
    df.to_csv(txt_file, header=False,index=False, sep=' ')
            
            
def delete_file(file_path):
    path = Path(file_path)
    path.unlink()            
                
def generate_filtered_csv(csv_file, column_name, filter_values, csv_output_file):
    with open(csv_file, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        header_row = next(reader)

        # Find the index of the specified column
        try:
            column_index = header_row.index(column_name)
        except ValueError:
            raise ValueError(f"Column '{column_name}' not found in the CSV file.")

        # Filter rows based on the values of the specified column
        filtered_rows = [row for row in reader if row[column_index] in filter_values]

    with open(csv_output_file, 'w', newline='') as csv_output_file:
        writer = csv.writer(csv_output_file, delimiter=' ')

        # Write the header row
        writer.writerow(header_row)

        # Write the filtered rows
        writer.writerows(filtered_rows)

# Jittering
# "Jittering" can be considered as "applying different noise to each sample".
# sigma = standard devitation (STD) of the noise
# source https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
def DA_Jitter(X, sigma=0.05):
    noise = np.random.normal(loc=0, scale=sigma, size=X.shape)

    return X+noise

# scaling
#"Scaling" can be considered as "applying constant noise to the entire samples"
# sigma = STD of the zoom-in/out factor
# adapted from https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
def DA_Scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(X.shape[1],1))
    X[:,:,:] = X[:,:,:] * scalingFactor[:]

    return X

#source : https://github.com/Human-Signals-Lab/LAPNet-HAR
def DA_MagWarp(x, sigma=0.2, knot=4):

    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper

    return ret

#source : https://github.com/Human-Signals-Lab/LAPNet-HAR

def DA_TimeWarp(x, sigma=0.2, knot=4):
  
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
            scale = (x.shape[1]-1)/time_warp[-1]
            ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T
    return ret

'''
# Rotation
def DA_Rotation(X):
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(X , axangle2mat(axis,angle))
'''
# Permutation
# adapted from https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
#### Hyperparameters :  nPerm = # of segments to permute
#### minSegLength = allowable minimum length for each segment

def DA_Permutation(X, Y, nPerm=4, minSegLength=10):
    X_new = np.zeros(X.shape)
    Y_new = np.zeros(X.shape[0], dtype=int)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(nPerm+1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0]-minSegLength, nPerm-1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:]-segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii]+1],:]
        y_temp = Y[segs[idx[ii]]:segs[idx[ii]+1]]
        X_new[pp:pp+len(x_temp),:] = x_temp
        Y_new[pp:pp+len(x_temp)] = y_temp
        pp += len(x_temp)
    return(X_new, Y_new)



