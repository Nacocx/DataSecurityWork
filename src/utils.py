"""
数据安全课程大作业 - 辅助工具与评价指标模块
包含：
1. 文本与比特流（0/1数组）的相互转换、打包与解包
2. 图像质量评价指标：PSNR (峰值信噪比)、MSE (均方误差)
3. 容量评价指标：BPP (Bits Per Pixel)
4. 完整性与可逆性校验工具
"""

from __future__ import annotations
import zlib
import numpy as np
from PIL import Image
from pathlib import Path


def text_to_bits(text: str) -> np.ndarray:
    """将 UTF-8 字符串转换为 uint8 类型的 0/1 比特数组（包含32位字节长度头）"""
    data = text.encode("utf-8")
    length = len(data)
    header = length.to_bytes(4, byteorder="big")
    full_bytes = header + data
    bits = np.unpackbits(np.frombuffer(full_bytes, dtype=np.uint8))
    return bits.astype(np.uint8)


def bits_to_text(bits: np.ndarray) -> str:
    """从 0/1 比特数组中解析带32位字节长度头的 UTF-8 字符串"""
    if len(bits) < 32:
        raise ValueError("比特流长度不足 32 位，无法解析长度头")
    header_bytes = np.packbits(bits[:32]).tobytes()
    byte_length = int.from_bytes(header_bytes, byteorder="big")
    total_bits_needed = 32 + byte_length * 8
    if len(bits) < total_bits_needed:
        raise ValueError(f"比特流长度不足：需要 {total_bits_needed} bits，实际只有 {len(bits)} bits")
    payload_bits = bits[32:total_bits_needed]
    payload_bytes = np.packbits(payload_bits).tobytes()
    return payload_bytes.decode("utf-8")


def bytes_to_bits(data: bytes) -> np.ndarray:
    """将原始字节流转换为 0/1 比特数组"""
    if len(data) == 0:
        return np.array([], dtype=np.uint8)
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """将 0/1 比特数组打包为字节流"""
    if len(bits) == 0:
        return b""
    return np.packbits(bits.astype(np.uint8)).tobytes()


def compress_location_map(loc_map: np.ndarray) -> np.ndarray:
    """使用 zlib 无损压缩二值位置图，并转为比特流（含32位压缩后字节长度头 + 32位原始比特数）"""
    raw_bits = loc_map.flatten().astype(np.uint8)
    num_raw_bits = len(raw_bits)
    packed = np.packbits(raw_bits).tobytes()
    compressed = zlib.compress(packed, level=9)
    header = len(compressed).to_bytes(4, "big") + num_raw_bits.to_bytes(4, "big")
    return bytes_to_bits(header + compressed)


def decompress_location_map(bits: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, int]:
    """
    从比特流起始位置解压位置图
    返回：(恢复的二维 loc_map, 消耗的比特数)
    """
    if len(bits) < 64:
        raise ValueError("比特流过短，无法读取位置图头部")
    header_bytes = bits_to_bytes(bits[:64])
    comp_len = int.from_bytes(header_bytes[:4], "big")
    num_raw_bits = int.from_bytes(header_bytes[4:8], "big")
    total_bits_consumed = 64 + comp_len * 8
    if len(bits) < total_bits_consumed:
        raise ValueError("比特流过短，无法读取完整压缩位置图")
    comp_bytes = bits_to_bytes(bits[64:total_bits_consumed])
    decompressed = zlib.decompress(comp_bytes)
    unpacked = np.unpackbits(np.frombuffer(decompressed, dtype=np.uint8))[:num_raw_bits]
    loc_map = unpacked.reshape(shape).astype(np.uint8)
    return loc_map, total_bits_consumed


def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算两幅图像的均方误差 (MSE)"""
    diff = img1.astype(np.float64) - img2.astype(np.float64)
    return float(np.mean(diff ** 2))


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算两幅 8-bit 灰度图像的峰值信噪比 (PSNR, 单位 dB)"""
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def calculate_bpp(num_secret_bits: int, img_shape: tuple[int, int]) -> float:
    """计算每像素嵌入比特数 (BPP - Bits Per Pixel)"""
    total_pixels = img_shape[0] * img_shape[1]
    return float(num_secret_bits / total_pixels)


def verify_reversibility(original_img: np.ndarray, recovered_img: np.ndarray,
                         original_bits: np.ndarray, extracted_bits: np.ndarray) -> dict:
    """验证提取的信息与恢复的图像是否100%无损一致"""
    img_identical = bool(np.array_equal(original_img, recovered_img))
    max_pixel_diff = int(np.max(np.abs(original_img.astype(np.int32) - recovered_img.astype(np.int32))))
    diff_pixels = int(np.sum(original_img != recovered_img))

    min_len = min(len(original_bits), len(extracted_bits))
    bits_identical = bool(
        len(original_bits) == len(extracted_bits) and np.array_equal(original_bits, extracted_bits)
    )
    bit_errors = int(np.sum(original_bits[:min_len] != extracted_bits[:min_len])) + abs(
        len(original_bits) - len(extracted_bits)
    )
    ber = bit_errors / max(1, len(original_bits))

    return {
        "image_identical": img_identical,
        "max_pixel_diff": max_pixel_diff,
        "diff_pixel_count": diff_pixels,
        "bits_identical": bits_identical,
        "bit_error_count": bit_errors,
        "ber": ber,
        "is_fully_reversible": img_identical and bits_identical,
    }


def load_grayscale_image(path: str | Path) -> np.ndarray:
    """读取本地图像并转换为 uint8 灰度二维数组"""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


def save_grayscale_image(img: np.ndarray, path: str | Path) -> None:
    """将 uint8 灰度二维数组保存为无损 PNG 图像"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img.astype(np.uint8), mode="L").save(path, format="PNG")
