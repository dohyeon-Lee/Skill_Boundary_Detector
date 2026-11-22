"""Shared odd-window Savitzky–Golay contract for SBD builds and reports."""
from __future__ import annotations


def validate_savgol(window, polyorder, *, window_name='smooth_window', order_name='polyorder'):
    try:
        win, order = int(window), int(polyorder)
        valid_window = not isinstance(window, bool) and win == float(window) and win >= 1 and win % 2 == 1
        valid_order = not isinstance(polyorder, bool) and order == float(polyorder) and order >= 0
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{window_name} must be a positive odd integer; {order_name} must be a non-negative integer') from None
    if not valid_window:
        raise ValueError(f'{window_name} must be a positive odd integer (1 disables smoothing); got {window!r}')
    if not valid_order or (win > 1 and order >= win):
        raise ValueError(f'{order_name} must be non-negative and smaller than {window_name}; got {polyorder!r}')
    return win, order


def smooth_savgol(values, window, polyorder):
    import numpy as np
    from scipy.signal import savgol_filter

    window, polyorder = validate_savgol(window, polyorder)
    values = np.asarray(values, dtype=np.float64)
    if window == 1:
        return values.copy()
    # Short curves may not fill the requested odd window. Keep that fallback
    # odd too; this never changes a user's even window into an accepted input.
    actual = min(window, len(values))
    if actual % 2 == 0:
        actual -= 1
    if actual <= polyorder or actual <= 1:
        return values.copy()
    return savgol_filter(values, window_length=actual, polyorder=polyorder)
