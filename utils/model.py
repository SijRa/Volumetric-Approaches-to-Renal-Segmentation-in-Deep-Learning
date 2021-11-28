from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv3DTranspose, Conv3D, MaxPooling3D, BatchNormalization, ELU, concatenate, add
from tensorflow.keras.losses import mean_squared_error, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def _3dUNet(input_shape=(240, 240, 11, 1), n_filters=16, learning_rate=0.001):

  def Conv_Layer(n_filters, kernel_size=(3,3,3), data_format=None):
    def f(_input):
      conv = Conv3D(n_filters, kernel_size=kernel_size, padding='same', data_format=data_format)(_input)
      norm = BatchNormalization()(conv)
      activation = ELU()(norm)
      return activation
    return f

  def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

  def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
  
  # Input layers
  input_mri = Input(shape=input_shape, name='input_mri')

  x = Conv_Layer(1*n_filters, kernel_size=(3,3,3), data_format="channels_last")(input_mri)
  x = Conv_Layer(1*n_filters, kernel_size=(3,3,3))(x)

  res_1 = x
  
  x = MaxPooling3D(pool_size=(2,2,2), strides=2)(res_1)
  x = Conv_Layer(2*n_filters, kernel_size=(3,3,3))(x)
  x = Conv_Layer(2*n_filters, kernel_size=(3,3,3))(x)
  
  res_2 = x
  
  x = MaxPooling3D(pool_size=(2,2,2), strides=2)(res_2)
  x = Conv_Layer(4*n_filters, kernel_size=(3,3,3))(x)
  x = Conv_Layer(4*n_filters, kernel_size=(3,3,3))(x)
  
  res_3 = x

  x = MaxPooling3D(pool_size=(2,2,2), strides=2)(res_3)
  x = Conv_Layer(8*n_filters, kernel_size=(3,3,3))(x)
  x = Conv_Layer(8*n_filters, kernel_size=(3,3,3))(x)
  
  x = Conv3DTranspose(n_filters*4, kernel_size=(2,2,2), strides=2)(x)
  x = concatenate([x, res_3], axis=3)
  x = Conv_Layer(4*n_filters, kernel_size=(3,3,3))(x)
  x = Conv_Layer(4*n_filters, kernel_size=(3,3,3))(x)  

  x = Conv3DTranspose(n_filters*2, kernel_size=(2,2,2), strides=2)(x)
  x = concatenate([x, res_2], axis=3)
  x = Conv_Layer(2*n_filters, kernel_size=(3,3,3))(x)
  x = Conv_Layer(2*n_filters, kernel_size=(3,3,3))(x)

  x = Conv3DTranspose(n_filters*1, kernel_size=(2,2,2), strides=2)(x)
  x = concatenate([x, res_1], axis=3)
  x = Conv_Layer(1*n_filters, kernel_size=(3,3,3))(x)
  x = Conv_Layer(1*n_filters, kernel_size=(3,3,3))(x)

  output_mask = Conv3D(1, kernel_size=(1,1,27), activation='sigmoid')(x)
  
  # Model compilation
  model = Model(inputs=input_mri, outputs=output_mask, name="UNet_3D")
  optimizer = Adam(learning_rate)
  
  model.compile(
    loss=dice_coef_loss,
    optimizer=optimizer,
    metrics=[dice_coef])

  return model


def _3dUNetPlusPlusL2(input_shape=(240, 240, 11, 1), n_filters=16, learning_rate=0.005):

  def Conv_Layer(n_filters, kernel_size=(3,3,3), data_format=None):
    def f(_input):
      conv = Conv3D(n_filters, kernel_size=kernel_size, padding='same', data_format=data_format)(_input)
      #norm = BatchNormalization()(conv)
      activation = ELU()(conv)
      return activation
    return f

  def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

  def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
  
  # Input layers
  input_mri = Input(shape=input_shape, name='input_mri')
  x00 = Conv_Layer(1*n_filters, kernel_size=(3,3,3), data_format="channels_last")(input_mri)
  x00 = Conv_Layer(1*n_filters, kernel_size=(3,3,3))(x00)
  out0 = MaxPooling3D(pool_size=(1,1,1), strides=2)(x00)

  x10 = Conv_Layer(2*n_filters, kernel_size=(3,3,3))(out0)
  x10 = Conv_Layer(2*n_filters, kernel_size=(3,3,3))(x10)
  out1 = MaxPooling3D(pool_size=(1,1,1), strides=2)(x10)

  x01 = Conv3DTranspose(1*n_filters, kernel_size=(2,2,2), strides=2, padding='same')(x10)
  x01 = x01[:,:,:,:11,:]
  x01 = concatenate([x00, x01], axis=-1)
  x01 = Conv_Layer(1*n_filters, kernel_size=(3,3,3))(x01)
  x01 = Conv_Layer(1*n_filters, kernel_size=(3,3,3))(x01)

  x20 = Conv_Layer(4*n_filters, kernel_size=(3,3,3))(out1)
  x20 = Conv_Layer(4*n_filters, kernel_size=(3,3,3))(x20)
  out2 = MaxPooling3D(pool_size=(1,1,1), strides=2)(x20)

  x11 = Conv3DTranspose(2*n_filters, kernel_size=(2,2,2), strides=2, padding='same')(x20)
  x11 = concatenate([x10, x11], axis=-1)
  x11 = Conv_Layer(2*n_filters, kernel_size=(3,3,3))(x11)
  x11 = Conv_Layer(2*n_filters, kernel_size=(3,3,3))(x11)

  x02 = Conv3DTranspose(1*n_filters, kernel_size=(2,2,2), strides=2, padding='same')(x11)
  x02 = x02[:,:,:,:11,:]
  x02 = concatenate([x00, x01, x02], axis=-1)
  x02 = Conv_Layer(1*n_filters, kernel_size=(3,3,3))(x02)
  x02 = Conv_Layer(1*n_filters, kernel_size=(3,3,3))(x02)


  output_mask = Conv3D(1, kernel_size=(1,1,1), activation='sigmoid')(x02)
  
  # Model compilation
  model = Model(inputs=input_mri, outputs=output_mask, name="L2_UNet_3D")
  optimizer = Adam(learning_rate)
  
  model.compile(
    loss=dice_coef_loss,
    optimizer=optimizer,
    metrics=[dice_coef])

  return model


def _3dUNetPlusPlusL3(input_shape=(240, 240, 11, 1), n_filters=16, learning_rate=0.005):

  def Conv_Layer(n_filters, kernel_size=(3,3,3), data_format=None):
    def f(_input):
      conv = Conv3D(n_filters, kernel_size=kernel_size, padding='same', data_format=data_format)(_input)
      #norm = BatchNormalization()(conv)
      activation = ELU()(conv)
      return activation
    return f

  def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

  def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
  
  # Input layers
  input_mri = Input(shape=input_shape, name='input_mri')
  x00 = Conv_Layer(1*n_filters, kernel_size=(3,3,3), data_format="channels_last")(input_mri)
  x00 = Conv_Layer(1*n_filters, kernel_size=(3,3,3))(x00)
  out0 = MaxPooling3D(pool_size=(1,1,1), strides=2)(x00)

  x10 = Conv_Layer(2*n_filters, kernel_size=(3,3,3))(out0)
  x10 = Conv_Layer(2*n_filters, kernel_size=(3,3,3))(x10)
  out1 = MaxPooling3D(pool_size=(1,1,1), strides=2)(x10)

  x01 = Conv3DTranspose(1*n_filters, kernel_size=(2,2,2), strides=2, padding='same')(x10)
  x01 = x01[:,:,:,:11,:]
  x01 = concatenate([x00, x01], axis=-1)
  x01 = Conv_Layer(1*n_filters, kernel_size=(3,3,3))(x01)
  x01 = Conv_Layer(1*n_filters, kernel_size=(3,3,3))(x01)

  x20 = Conv_Layer(4*n_filters, kernel_size=(3,3,3))(out1)
  x20 = Conv_Layer(4*n_filters, kernel_size=(3,3,3))(x20)
  out2 = MaxPooling3D(pool_size=(1,1,1), strides=2)(x20)

  x11 = Conv3DTranspose(1*n_filters, kernel_size=(2,2,2), strides=2, padding='same')(x20)
  x11 = concatenate([x10, x11], axis=-1)
  x11 = Conv_Layer(1*n_filters, kernel_size=(3,3,3))(x11)
  x11 = Conv_Layer(1*n_filters, kernel_size=(3,3,3))(x11)

  x02 = Conv3DTranspose(1*n_filters, kernel_size=(2,2,2), strides=2, padding='same')(x11)
  x02 = x02[:,:,:,:11,:]
  x02 = concatenate([x00, x01, x02], axis=-1)
  x02 = Conv_Layer(1*n_filters, kernel_size=(3,3,3))(x02)
  x02 = Conv_Layer(1*n_filters, kernel_size=(3,3,3))(x02)

  x30 = Conv_Layer(8*n_filters, kernel_size=(3,3,3))(out2)
  x30 = Conv_Layer(8*n_filters, kernel_size=(3,3,3))(x30)
  out3 = MaxPooling3D(pool_size=(1,1,1), strides=2)(x30)

  x21 = Conv3DTranspose(4*n_filters, kernel_size=(2,2,2), strides=2, padding='same')(x30)
  x21 = x21[:,:,:,:3,:]  
  x21 = concatenate([x20, x21], axis=-1)
  x21 = Conv_Layer(4*n_filters, kernel_size=(3,3,3))(x21)
  x21 = Conv_Layer(4*n_filters, kernel_size=(3,3,3))(x21)

  x12 = Conv3DTranspose(2*n_filters, kernel_size=(2,2,2), strides=2, padding='same')(x21)
  x12 = concatenate([x10, x11, x12], axis=-1)
  x12 = Conv_Layer(2*n_filters, kernel_size=(3,3,3))(x12)
  x12 = Conv_Layer(2*n_filters, kernel_size=(3,3,3))(x12)

  x03 = Conv3DTranspose(1*n_filters, kernel_size=(2,2,2), strides=2, padding='same')(x12)
  x03 = x03[:,:,:,:11,:]  
  x03 = concatenate([x00, x01, x02, x03], axis=-1)
  x03 = Conv_Layer(1*n_filters, kernel_size=(3,3,3))(x03)
  x03 = Conv_Layer(1*n_filters, kernel_size=(3,3,3))(x03)

  output_mask = Conv3D(1, kernel_size=(1,1,1), activation='sigmoid')(x03)
  
  # Model compilation
  model = Model(inputs=input_mri, outputs=output_mask, name="L3_UNet_3D")
  optimizer = Adam(learning_rate)
  
  model.compile(
    loss=dice_coef_loss,
    optimizer=optimizer,
    metrics=[dice_coef])

  return model