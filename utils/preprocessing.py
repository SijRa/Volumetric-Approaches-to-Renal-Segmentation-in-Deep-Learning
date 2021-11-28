from intensity_normalization.normalize import fcm

def fuzzy_normalisation(mri, mask):
  wm_mask = fcm.find_tissue_mask(mri, mask)
  return fcm.fcm_normalize(mri, wm_mask)