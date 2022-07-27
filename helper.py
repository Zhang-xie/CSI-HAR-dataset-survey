import io
import struct
import zipfile
from collections import namedtuple
from time import time

import numba
import numpy as np


BfResult = namedtuple(
    "BfResult",
    [
        "timestamp_low",
        "bfee_count",
        "Nrx",
        "Ntx",
        "rssi_a",
        "rssi_b",
        "rssi_c",
        "noise",
        "agc",
        "perm",
        "rate",
        "csi",
    ],
)


@numba.njit()
def uchar2char(x):
    return -1 * (x & 0x80) + (x & 0x7F)


@numba.njit()
def read_bfee(in_bytes):
    timestamp_low = (
        in_bytes[0] + (in_bytes[1] << 8) + (in_bytes[2] << 16) + (in_bytes[3] << 24)
    )
    bfee_count = in_bytes[4] + (in_bytes[5] << 8)
    Nrx = in_bytes[8]
    Ntx = in_bytes[9]
    rssi_a = in_bytes[10]
    rssi_b = in_bytes[11]
    rssi_c = in_bytes[12]

    # noise = struct.unpack("b", in_bytes[13:14])[0]
    noise = uchar2char(in_bytes[13])
    agc = in_bytes[14]
    antenna_sel = in_bytes[15]
    lens = in_bytes[16] + (in_bytes[17] << 8)
    fake_rate_n_flags = in_bytes[18] + (in_bytes[19] << 8)
    calc_len = (30 * (Nrx * Ntx * 8 * 2 + 3) + 7) // 8

    index = 0
    payload = in_bytes[20:]
    shape = (Ntx, Nrx, 30)

    out_array = np.zeros(shape, dtype=np.complex128)

    assert lens == calc_len

    for i in range(30):
        index += 3
        remainder = index % 8
        for j in range(Nrx):
            for k in range(Ntx):
                r_part = (payload[index // 8] >> remainder) | (
                    payload[index // 8 + 1] << (8 - remainder)
                ) & 0x00FF

                # r_part = r_part.to_bytes(1, "big")
                r_part = uchar2char(r_part)
                i_part = (payload[index // 8 + 1] >> remainder) | (
                    payload[index // 8 + 2] << (8 - remainder)
                ) & 0x00FF
                i_part = uchar2char(i_part)
                # i_part = i_part.to_bytes(1, "big")
                # r_part, i_part = struct.unpack(">bb", r_part + i_part)

                out_array[k, j, i] = r_part + 1j * i_part

                index += 16
    perm = np.zeros(3, dtype=np.int8)

    perm[0] = ((antenna_sel) & 0x3) + 1
    perm[1] = ((antenna_sel >> 2) & 0x3) + 1
    perm[2] = ((antenna_sel >> 4) & 0x3) + 1

    return BfResult(
        timestamp_low,
        bfee_count,
        Nrx,
        Ntx,
        rssi_a,
        rssi_b,
        rssi_c,
        noise,
        agc,
        perm,
        fake_rate_n_flags,
        out_array,
    )


def read_bf_file(b_all):
    b_f = io.BytesIO(b_all)

    assert b_f.seek(0, io.SEEK_END) == len(b_all) and b_f.seek(0, io.SEEK_SET) == 0
    lens = len(b_all)

    ret = []
    cur = 0
    broken_perm = 0
    triangle = [1, 3, 6]

    while cur < (lens - 3):
        field_len = struct.unpack(">H", b_f.read(2))[0]
        code = b_f.read(1)
        cur = cur + 3

        if code == b"\xbb":
            tmp_bytes_data = b_f.read(field_len - 1)

            cur = cur + field_len - 1
            if len(tmp_bytes_data) != (field_len - 1):
                b_f.close()
                return ret

            bytes_data = struct.pack(
                f"{field_len - 1}B",
                *struct.unpack(f">{field_len - 1}B", tmp_bytes_data),
            )
        else:
            b_f.seek(field_len - 1, io.SEEK_CUR)
            cur = cur + field_len - 1
            continue

        if code == b"\xbb":
            bfresult = read_bfee(bytes_data)

            perm = bfresult.perm
            Nrx = bfresult.Nrx
            if Nrx == 1:
                ret.append(bfresult)
                continue
            if sum(perm) != triangle[Nrx - 1]:
                if broken_perm == 0:
                    broken_perm = 1
                    print("WARNING!")
            else:
                bfresult.csi[:, :Nrx, :] = bfresult.csi[:, perm[:Nrx] - 1, :]
            ret.append(bfresult)
    b_f.close()
    return ret


def dbinv(x):
    return 10 ** (x / 10)


def db(x):
    return (10 * np.log10(x) + 300) - 300


def get_total_rss(bf_result: BfResult):
    rssi_mag = 0
    if bf_result.rssi_a != 0:
        rssi_mag = rssi_mag + dbinv(bf_result.rssi_a)
    if bf_result.rssi_b != 0:
        rssi_mag = rssi_mag + dbinv(bf_result.rssi_b)
    if bf_result.rssi_c != 0:
        rssi_mag = rssi_mag + dbinv(bf_result.rssi_c)

    return db(rssi_mag) - 44 - bf_result.agc


def get_scaled_csi(bf_result: BfResult):
    # Pull out CSI
    csi = bf_result.csi

    # Calculate the scale factor between normalized CSI and RSSI (mW)
    csi_sq = csi * csi.conj()
    csi_pwr = csi_sq.sum()
    rssi_pwr = dbinv(get_total_rss(bf_result))
    # Scale CSI -> Signal power : rssi_pwr / (mean of csi_pwr)
    scale = rssi_pwr / (csi_pwr / 30)

    # Thermal noise might be undefined if the trace was
    # captured in monitor mode.
    # ... If so, set it to -92
    if bf_result.noise == -127:
        noise_db = -92
    else:
        noise_db = bf_result.noise
    thermal_noise_pwr = dbinv(noise_db)

    # % Quantization error: the coefficients in the matrices are
    # % 8-bit signed numbers, max 127/-128 to min 0/1. Given that Intel
    # % only uses a 6-bit ADC, I expect every entry to be off by about
    # % +/- 1 (total across real & complex parts) per entry.
    # %
    # % The total power is then 1^2 = 1 per entry, and there are
    # % Nrx*Ntx entries per carrier. We only want one carrier's worth of
    # % error, since we only computed one carrier's worth of signal above.
    quant_error_pwr = scale * (bf_result.Nrx * bf_result.Ntx)

    # % Total noise and error power
    total_noise_pwr = thermal_noise_pwr + quant_error_pwr

    # % Ret now has units of sqrt(SNR) just like H in textbooks
    ret = csi * np.sqrt(scale / total_noise_pwr)

    if bf_result.Ntx == 2:
        ret = ret * np.sqrt(2)
    elif bf_result.Ntx == 3:
        # % Note: this should be sqrt(3)~ 4.77 dB. But, 4.5 dB is how
        # % Intel (and some other chip makers) approximate a factor of 3
        # %
        # % You may need to change this if your card does the right thing.
        ret = ret * np.sqrt(dbinv(4.5))
    return ret


def get_csi_from_bytes(byte_data):
    csi_trace = read_bf_file(byte_data)
    timestamp = np.zeros(len(csi_trace))
    cfr_array = np.zeros((len(csi_trace), *csi_trace[0].csi.shape), dtype=np.complex128)
    for k in range(len(csi_trace)):
        csi_entry = csi_trace[k]

        csi_all = np.squeeze(get_scaled_csi(csi_entry))
        timestamp[k] = csi_entry.timestamp_low
        cfr_array[k, :, :] = csi_all
    return cfr_array.squeeze()


def get_csi_from_bytes_unscaled(byte_data):
    csi_trace = read_bf_file(byte_data)
    # timestamp = np.zeros(len(csi_trace))
    cfr_array = np.zeros((len(csi_trace), *csi_trace[0].csi.shape), dtype=np.complex64)
    for k in range(len(csi_trace)):
        csi_entry = csi_trace[k]

        csi_all = np.squeeze(csi_entry.csi)
        # timestamp[k] = csi_entry.timestamp_low
        cfr_array[k, :, :] = csi_all
    return cfr_array


def write_array2zip(name, arr, zip_f):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    with io.BytesIO() as iobf:
        np.save(iobf, arr, allow_pickle=False)
        zip_f.writestr(name, iobf.getvalue())


def numpyDict2BfResult(x):
    # dict_keys(['timestamp_low', 'Nrx', 'Ntx', 'noise', 'agc', 'RSSI_a', 'RSSI_b', 'RSSI_c', 'CSI', 'label'])
    return BfResult(
        0,
        0,
        x["Nrx"],
        x["Ntx"],
        x["RSSI_a"],
        x["RSSI_b"],
        x["RSSI_c"],
        x["noise"],
        x["agc"],
        0,
        0,
        x["CSI"],
    )


def numpyrec2BfResult(x, csi):
    # dict_keys(['timestamp_low', 'Nrx', 'Ntx', 'noise', 'agc', 'RSSI_a', 'RSSI_b', 'RSSI_c', 'CSI', 'label'])
    return BfResult(
        0,
        0,
        x["Nrx"],
        x["Ntx"],
        x["rssi_a"],
        x["rssi_b"],
        x["rssi_c"],
        x["noise"],
        x["agc"],
        0,
        0,
        csi,
    )
