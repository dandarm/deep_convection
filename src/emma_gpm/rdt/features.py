"""Raw-Kelvin spectral fields; no radiometric normalization."""
import numpy as np

FEATURE_NAMES = ("BTD_062_108", "BTD_073_108", "BTD_087_108", "BTD_097_108", "BTD_108_120", "BTD_134_108")


def spectral_features(frame):
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[0] != 7:
        raise ValueError("Expected seven-channel C,H,W frame")
    ir = frame[4]
    return np.stack([frame[0]-ir, frame[1]-ir, frame[2]-ir, frame[3]-ir, ir-frame[5], frame[6]-ir])


def robust_spectral_model(features, mask, floor):
    values = features[:, mask]
    centre = np.median(values, axis=1)
    scale = np.maximum(1.4826*np.median(np.abs(values-centre[:, None]), axis=1), floor)
    return centre, scale


def spectral_distance(features, centre, scale):
    return np.max(np.abs(features-centre[:, None, None])/scale[:, None, None], axis=0)
