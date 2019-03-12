import warnings
warnings.filterwarnings("ignore")

import os
import gdal
import osr
import sys

import numpy as np
import tensorflow as tf

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import skimage.exposure as exposure

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from matplotlib.ticker import FuncFormatter

import urllib.request
import zipfile

SEED=1989

def import_repository(dirname, replace_last = False):
  if (replace_last):
    sys.path = sys.path[:-1]
  
  sys.path.append(dirname)

def download_extract_zipfile(file_url, filename, destination='.'):
  print("Downloading from " + file_url)
  urllib.request.urlretrieve(file_url, filename)

  if destination != '.' and not os.path.exists(destination):
    os.makedirs(destination)

  print("Extracting to " + destination)
  zip_ref = zipfile.ZipFile(filename, 'r')
  zip_ref.extractall(destination)
  zip_ref.close()

def rescale_01(data):
    return (data - np.min(data))/(np.max(data) - np.min(data))

def image_info(imagefile):
  print(gdal.Info(imagefile, deserialize=True)) 

def vis_refimage(filepath, color_array=['white','green'], zoom=1):
  
  ds = gdal.Open(filepath)
  data = ds.ReadAsArray()
  
  scale_factor = (1/zoom)

  x_start, pixel_width, _, y_start, _, pixel_height = ds.GetGeoTransform()
  x_end = x_start + ds.RasterXSize * pixel_width * scale_factor
  y_end = y_start + ds.RasterYSize * pixel_height * scale_factor
  
  if zoom > 1:
    x_size, y_size = data.shape
    
    x1 = y1 = 0 # TODO implement offset
    x2 = int(x_size * scale_factor)
    y2 = int(y_size * scale_factor)
    
    data = data[x1:x2, y1:y2]

  _, ax = plt.subplots(figsize=(15, 10))
  extent = (x_start, x_end, y_start, y_end)

  ax.set_title(filepath, fontsize=20)
  img = ax.imshow(data, extent=extent, origin='upper', cmap=colors.ListedColormap(color_array))

def vis_image(filepath, bands=[1,2,3], scale_factor = 1.0, zoom=1):
  ds = gdal.Open(filepath)
  
  data = ds.ReadAsArray()
  data = data*scale_factor
  if (data.ndim == 2):
    data = np.stack([data])
  data_equalized = []
  for band in bands:
      if (len(bands) == 1): # Only one band
        data_equalized.append( exposure.equalize_hist(rescale_01(data[band, :, :])) )
      else:
        data_equalized.append( exposure.equalize_hist(rescale_01(data[band-1, :, :])) )
  
  if (len(bands) == 1):
    cmap = 'binary'
    title = '   B'
    data_equalized = data_equalized[0]
  else:
    cmap = None
    title = '   RGB'
    data_equalized = np.stack(data_equalized)
    data_equalized = data_equalized.transpose((1, 2, 0))
  
  scale_factor = (1/zoom)

  x_start, pixel_width, _, y_start, _, pixel_height = ds.GetGeoTransform()
  x_end = x_start + ds.RasterXSize * pixel_width * scale_factor
  y_end = y_start + ds.RasterYSize * pixel_height * scale_factor
  
  _, ax = plt.subplots(figsize=(15, 10))
  extent = (x_start, x_end, y_start, y_end)

  title = filepath + '   RGB' + str(bands)

  if zoom > 1:
    x_size, y_size, _ = data_equalized.shape
    
    x1 = y1 = 0 # TODO implement offset
    x2 = int(x_size * scale_factor)
    y2 = int(y_size * scale_factor)
    
    data_equalized = data_equalized[x1:x2, y1:y2, :]

  ax.set_title(title, fontsize=20)
  img = ax.imshow(data_equalized, extent=extent, origin='upper', cmap=cmap)

def read_image_data(filepaths):
  
  data_struct = { 'filenames': [], 'bandnames': [], 'data': [], 'nbands':0 }

  for i in range(len(filepaths)):
    
    filepath = filepaths[i]
    ds = gdal.Open(filepath)
    data = ds.ReadAsArray()
      
    if (len(data.shape) == 2): # Only one band
      data = np.stack([data])

    print('Reading ' + filepath + ' ' + str(data.shape) + ' [' + str(data.dtype) + ']')
    
    data_struct['filenames'].append(os.path.basename(filepath))
    
    data_struct['data'].append(data)

  nbands, xsize, ysize = data_struct['data'][0].shape

  for band in range( nbands ):
      bandname = 'B'+str((band+1)).zfill(2)
      data_struct['bandnames'].append(bandname)

  data_struct['nfiles'] = len(data_struct['filenames'])
  data_struct['nbands'] = nbands
  data_struct['xsize'] = xsize
  data_struct['ysize'] = ysize

  data_struct['data'] = np.stack(data_struct['data'])

  print('Creating Data Struct nfiles={:d} nbands={:d} xsize={:d} ysize={:d}'
          .format(data_struct['nfiles'], data_struct['nbands'], data_struct['xsize'], data_struct['ysize']))

  return data_struct

def stats(data_struct, nodata=None, axis=None):

  data = data_struct['data']
  nfiles = data_struct['nfiles']
  nbands = data_struct['nbands']

  if nodata is not None:
  	data = np.ma.masked_array(data, data == nodata)

  minPerBand = np.min(data, axis=(2,3))
  maxPerBand = np.max(data, axis=(2,3))
  meanPerBand = np.mean(data, axis=(2,3))
  medianPerBand = np.median(data, axis=(2,3))
  stdPerBand = np.std(data, axis=(2,3))
  varPerBand = np.var(data, axis=(2,3))

  for i in range(nfiles):
    filename = data_struct['filenames'][i]

    print('\n{}:'.format(filename))

    for j in range(nbands):
      bandname = data_struct['bandnames'][j]
      
      print('  {} min={:.2f} max={:.2f} mean={:.2f} median={:.2f} stdDev={:.2f} variance={:.2f}'
              .format(bandname, minPerBand[i,j], maxPerBand[i,j], meanPerBand[i,j],
                          medianPerBand[i,j], stdPerBand[i,j], varPerBand[i,j]
              ))

  print('\nAll files:')
  for j in range(nbands):
    bandname = data_struct['bandnames'][j]
    print('  {} min={:.2f} max={:.2f} mean={:.2f} median={:.2f} stdDev={:.2f} variance={:.2f}'
                .format(bandname, np.min(data[:,j,:,:]), np.max(data[:,j,:,:]), np.mean(data[:,j,:,:]),
                            np.median(data[:,j,:,:]), np.std(data[:,j,:,:]), np.var(data[:,j,:,:])
                ))
def stats_from_files(files_array, nodata=None):
	data_struct = read_image_data(files_array)
	stats(data_struct, nodata)

def vis_histograms(data_struct, bands=[0,1,2,3], colors=['red', 'green', 'blue', 'purple'], nbins = 256):

  _, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
  positions = [(0,0),(0,1),(1,0),(1,1)]

  data = data_struct['data']

  for file in range(data_struct['nfiles']):
    
    filename = data_struct['filenames'][file]
    
    for band in bands:

      ax_data = np.reshape(data[file,band,:,:],-1)
      ax_position = positions[file]
      ax_color = colors[band]
      
      ax[ax_position].set_ylim([0,100000])        
      ax[ax_position].set_title(filename, fontsize=15)
      ax[ax_position].hist(ax_data, nbins, color=ax_color, alpha=0.4)
          
  plt.show()

def standardize(data_struct):
    
  data = data_struct['data']
  nbands = data_struct['nbands']

  print("Standardizing data " + str(data.shape))

  min = np.min(data, axis=(0,2,3))
  max = np.max(data, axis=(0,2,3))

  data_norm_array = []
  for i in range(nbands):
    bandname = data_struct['bandnames'][i]
  
    #data_norm = ((data[:,i,:,:] - min[i])/(max[i] - min[i])) * 2 - 1
    data_norm = data[:,i,:,:]
    
    median = np.median(data_norm)
    std = np.std(data_norm)
    data_norm = (data_norm - median) / std

    data_norm_array.append(data_norm)
  
  data_norm_struct = data_struct.copy()
  data_norm_struct['data'] = np.stack(data_norm_array, axis=1)

  return data_norm_struct

def generate_chips(data_struct, file_idx=0, chip_size = 128):

  xsize = data_struct['xsize']
  ysize = data_struct['ysize']

  nx_chips = data_struct['xsize'] / chip_size
  ny_chips = data_struct['ysize'] / chip_size

  chip_struct = { 'chip_size':chip_size, 'indexes':[], 'data': [] }

  for xstart in range(0, xsize, chip_size):
    
    xend = xstart + chip_size

    if xend > xsize:
      xend = xsize
      xstart =  xsize - chip_size

    for ystart in range(0, ysize, chip_size):
      yend = ystart + chip_size
      
      if yend > ysize:
        yend = ysize
        ystart =  ysize - chip_size
      
      chip_data = data_struct['data'][file_idx, :, xstart:xend, ystart:yend ]

      chip_struct['data'].append(chip_data)
      chip_struct['indexes'].append({ 'x': xstart, 'y': ystart })

  chip_struct['chips_total'] = len(chip_struct['data'])

  return chip_struct

def vis_chip_from_numpy(input_chip, expect_chip=None, idx=0, bands=[1,2,3], color_array=['white','green'], title_prefix = 'expected'):

  chip_title = 'chip-{}   RGB{}'.format(idx, bands)
  chip_data = input_chip[idx,:,:,:]

  data_equalized = []
  for band in bands:
    data_equalized.append( exposure.equalize_hist(chip_data[:, :, band-1]) )

  data_equalized = np.stack(data_equalized)
  data_equalized = data_equalized.transpose((1, 2, 0))

  if (expect_chip is not None):
    _, ax = plt.subplots(ncols=2, figsize=(8,4))
    ax_data = ax[0]
    ax_ref = ax[1]

    ref_title = title_prefix+'-{}'.format(idx)
    expect_data = expect_chip[idx,:,:,:]

    ax_ref.set_title(ref_title, fontsize=15)
    img = ax_ref.imshow(expect_data[:,:,0], origin='upper', cmap=colors.ListedColormap(color_array))

  else:
    _, ax_data = plt.subplots(figsize=(4,4))

  ax_data.set_title(chip_title, fontsize=15)
  img = ax_data.imshow(data_equalized, origin='upper')

def vis_chip(chip_struct, chip_ref_struct=None, idx=0, bands=[1,2,3], color_array=['white','green']):

  chip_title = 'chip-{}   RGB{}'.format(idx, bands)
  chip_data = chip_struct['data'][idx]

  min = np.min(chip_data)
  max = np.max(chip_data)
  chip_data = np.int8( ((chip_data - min)/(max - min)) * 100 )

  data_equalized = []
  for band in bands:
    data_equalized.append( exposure.equalize_hist(chip_data[band-1, :, :]) )

  data_equalized = np.stack(data_equalized)
  data_equalized = data_equalized.transpose((1, 2, 0))

  if (chip_ref_struct is not None):
    _, ax = plt.subplots(ncols=2, figsize=(8,4))
    ax_data = ax[0]
    ax_ref = ax[1]

    ref_title = 'reference-{}'.format(idx)
    chip_ref = chip_ref_struct['data'][idx]

    ax_ref.set_title(ref_title, fontsize=15)
    img = ax_ref.imshow(chip_ref[0,:,:], origin='upper', cmap=colors.ListedColormap(color_array))

  else:
    _, ax_data = plt.subplots(figsize=(4,4))

  ax_data.set_title(chip_title, fontsize=15)
  img = ax_data.imshow(data_equalized, origin='upper')

def vis_augcases(data, data_aug, dataref_aug=None, idx=0):
  nchips, _, _, _ = data.shape
  nchips_aug, _, _, _ = data_aug.shape

  aug_cases = int(nchips_aug/nchips)

  for idx in range(idx, aug_cases*nchips, nchips):
      vis_chipdata(data_aug, dataref_aug, idx=idx)

def vis_chipdata(chip_data, chip_dataref=None, idx=0, bands=[1,2,3], color_array=['white','green']):

  chip_title = 'chip-{}   RGB{}'.format(idx, bands)
  chip_data = chip_data[idx,:,:,:]

  min = np.min(chip_data)
  max = np.max(chip_data)
  chip_data = np.int8( ((chip_data - min)/(max - min)) * 100 )

  data_equalized = []
  for band in bands:
    data_equalized.append( exposure.equalize_hist(chip_data[:, :, band-1]) )

  data_equalized = np.stack(data_equalized)
  data_equalized = data_equalized.transpose((1, 2, 0))

  if (chip_dataref is not None):
    _, ax = plt.subplots(ncols=2, figsize=(8,4))
    ax_data = ax[0]
    ax_ref = ax[1]

    ref_title = 'reference-{}'.format(idx)
    chip_ref = chip_dataref[idx,:,:,:]

    ax_ref.set_title(ref_title, fontsize=15)
    img = ax_ref.imshow(chip_ref[:,:,0], origin='upper', cmap=colors.ListedColormap(color_array))

  else:
    _, ax_data = plt.subplots(figsize=(4,4))

  ax_data.set_title(chip_title, fontsize=15)
  img = ax_data.imshow(data_equalized, origin='upper')

def split_data(chip_struct, chip_ref_struct, test_val_size=0.3):

  x_train = np.stack(chip_struct['data']).astype(np.float32)
  y_train = np.stack(chip_ref_struct['data']).astype(np.float32)
  
  x_train = np.transpose(x_train, [0,2,3,1])
  y_train = np.transpose(y_train, [0,2,3,1])

  x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_val_size, random_state=SEED)
  x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=SEED)

  print("Splited samples:")
  print(' Train ({:.0f}%): {:d}'.format( (1-test_val_size)*100, len(x_train)))
  print(' Test: ({:.0f}%): {:d}'.format( (test_val_size/2)*100, len(x_test)))
  print(' Validation ({:.0f}%): {:d} '.format( (test_val_size/2)*100, len(x_val)))

  return x_train, x_test, x_val, y_train, y_test, y_val

def train(x_train, y_train, x_test, y_test, hyper_params, model_dir):

  tf.set_random_seed(SEED)
  tf.logging.set_verbosity(tf.logging.INFO)

  data_size, _, _, _ =  x_train.shape

  estimator = tf.estimator.Estimator(model_fn=md.description, model_dir=model_dir, params=hyper_params)
  logging_hook = tf.train.LoggingTensorHook(tensors={}, every_n_iter=data_size)

  for i in range(0, hyper_params['number_epochs']):
    train_input = tf.estimator.inputs.numpy_input_fn(x={"data": x_train}, y=y_train, batch_size=hyper_params['batch_size'], num_epochs=1, shuffle=True)
    train_results = estimator.train(input_fn=train_input, steps=None, hooks=[logging_hook])

    test_input = tf.estimator.inputs.numpy_input_fn(x={"data": x_test}, y=y_test, num_epochs=1, shuffle=False)
    test_results = estimator.evaluate(input_fn=test_input)

def data_augmentation(data):
  data_090 = np.rot90(data, k=1, axes=(1,2))
  data_180 = np.rot90(data, k=2, axes=(1,2))
  data_270 = np.rot90(data, k=3, axes=(1,2))

  data_f =np.fliplr(data)
  data_090_f =np.fliplr(data_090)
  data_180_f =np.fliplr(data_180)
  data_270_f =np.fliplr(data_270)

  result = np.concatenate([
          data, data_090, data_180, data_270,
          data_f, data_090_f, data_180_f, data_270_f
      ])

  print('Input data: {}'.format(data.shape))
  print('Data augmentation result: {}'.format(result.shape))

  return result

def evaluate(data, dataref, hyper_params, model_dir, label_names=['Not-forest','Forest']):

  data_size, _, _, _ =  data.shape

  tf.logging.set_verbosity(tf.logging.WARN)

  estimator = tf.estimator.Estimator(model_fn=md.description, model_dir=model_dir, params=hyper_params)
  logging_hook = tf.train.LoggingTensorHook(tensors={}, every_n_iter=data_size)

  predict_input = tf.estimator.inputs.numpy_input_fn(x={"data": data}, batch_size=hyper_params['batch_size'], shuffle=False)
  predict_result = estimator.predict(input_fn=predict_input)

  pred_flat = []
  ref_flat = []

  for pred, ref in zip(predict_result, dataref):
      pred[ pred > 0.5 ] = 1
      pred[ pred <= 0.5 ] = 0

      pred_flat = np.append(pred_flat, pred.reshape(-1))
      ref_flat = np.append(ref_flat, ref.reshape(-1))
  
  print('\n--------------------------------------------------')
  print('------------- CLASSIFICATION METRICS -------------')
  print('--------------------------------------------------')
  print(classification_report(ref_flat, pred_flat, target_names=label_names))
  
  conf_matrix = confusion_matrix(ref_flat, pred_flat)
  fmt = lambda x,pos: '{0:,}'.format(x)

  ax = plt.subplot()
  sns.heatmap(conf_matrix, annot=True, ax = ax, fmt=",", cmap='RdYlGn', cbar_kws={'format': FuncFormatter(fmt)})

  ax.set_xlabel('Predicted labels');
  ax.set_ylabel('Reference labels'); 
  
  ax.set_title('Confusion Matrix'); 
  ax.xaxis.set_ticklabels( label_names );
  ax.yaxis.set_ticklabels( label_names );

def predict(chip_struct, hyper_params, model_dir):

  tf.set_random_seed(SEED)
  tf.logging.set_verbosity(tf.logging.WARN)
  estimator = tf.estimator.Estimator(model_fn=md.description, model_dir=model_dir, params=hyper_params)

  x_predict = np.stack(chip_struct['data']).astype(np.float32)
  x_predict = np.transpose(x_predict, [0,2,3,1])

  tensors_to_log = {}
  data_size, _, _, _ =  x_predict.shape

  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=data_size)

  predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"data": x_predict}, batch_size=hyper_params['batch_size'], shuffle=False)
  predict_result = estimator.predict(input_fn=predict_input_fn)

  print("Predicting chips " + str(x_predict.shape) + "...")

  result = []
  for predict, dummy in zip(predict_result, x_predict):
    predict[ predict > 0.5 ] = 1
    predict[ predict <= 0.5 ] = 0
    result.append( np.transpose(predict, [2,0,1]) )
  
  predict_struct = chip_struct.copy()
  predict_struct['data'] = result

  return predict_struct

def write_chip(base_filepath, out_filepath, chip_struct, dataType = gdal.GDT_Int16, imageFormat = 'GTiff'):
    
  driver = gdal.GetDriverByName(imageFormat)
  base_ds = gdal.Open(base_filepath)

  x_start, pixel_width, _, y_start, _, pixel_height = base_ds.GetGeoTransform()
  x_size = base_ds.RasterXSize 
  y_size = base_ds.RasterYSize
  
  out_srs = osr.SpatialReference()
  out_srs.ImportFromWkt(base_ds.GetProjectionRef())

  out_ds = driver.Create(out_filepath, x_size, y_size, 1, dataType)
  out_ds.SetGeoTransform((x_start, pixel_width, 0, y_start, 0, pixel_height))
  out_ds.SetProjection(out_srs.ExportToWkt())
  
  out_band = out_ds.GetRasterBand(1)

  for i in range(chip_struct['chips_total']):
    chip_data = chip_struct['data'][i]
    chip_index = chip_struct['indexes'][i]
    out_band.WriteArray(chip_data[0,:,:], chip_index['y'], chip_index['x'])
  
  out_band.FlushCache()