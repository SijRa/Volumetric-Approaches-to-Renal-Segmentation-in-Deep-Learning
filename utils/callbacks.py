from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

def Plateau_Decay(metric, factor=0.2, patience=10):
  return ReduceLROnPlateau(monitor=metric, factor=factor, patience=patience, verbose=1, mode='min', min_delta=0.001, cooldown=1, min_lr=0)

def Early_Stopping(metric, patience=10):
  return EarlyStopping(monitor=metric, patience=patience, verbose=1)