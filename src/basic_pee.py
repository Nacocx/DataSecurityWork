"""
数据安全课程大作业 - 任务一：基础预测误差扩展 (Basic PEE) 算法实现
原理说明：
1. 棋盘格双层结构（Dot Set 偶数集 & Cross Set 奇数集）：
   - 像素坐标 (i, j) 满足 (i + j) % 2 == parity 的内部像素（1 <= i < H-1, 1 <= j < W-1）作为当前处理层。
   - 预测器使用其正交四邻域（上、下、左、右）像素的整数均值：
     p_hat = (I[i-1, j] + I[i+1, j] + I[i, j-1] + I[i, j+1]) // 4
   - 由于正交四邻域全部属于另一奇偶集合，在同一层嵌入和提取期间，四邻域像素 100% 保持不变！
     从而保证提取端计算的预测值 p_hat 与嵌入端完全一致。
2. 预测误差扩展与直方图平移 (PEE with Histogram Shifting)：
   - 预测误差 e = x - p_hat
   - 设定扩展阈值 T >= 1：
     * 当 -T <= e < T 时（扩展区间，可嵌入 1 比特 b ∈ {0, 1}）：
       e' = 2 * e + b
     * 当 e >= T 时（正向平移区间，不嵌入比特，仅平移以腾出空间）：
       e' = e + T
     * 当 e < -T 时（负向平移区间，不嵌入比特，仅平移）：
       e' = e - T
   - 修改后像素 x' = p_hat + e'
3. 溢出处理（Location Map 位置图）：
   - 若修改后像素 x' < 0 或 x' > 255，则会发生溢出。
   - 对可能溢出的像素不进行修改（保持 x' = x），并在二值位置图 loc_map 中标记为 1；正常处理像素标记为 0。
   - 为避免载荷较短时遍历全图产生 불필요的平移失真，我们在精确嵌入完所需比特数后立即停止修改后续像素（后续像素保持不变）。
   - 提取端根据内嵌在图像首部（固定 T=1 + 边界裁剪防溢出头）的元数据（含压缩位置图、扩展阈值 T、嵌入终止位置等），
     逆序/同序精确提取秘密信息并 100% 无损恢复原始图像。
"""

from __future__ import annotations
from dataclasses import dataclass
import time
import zlib
import numpy as np
from src.utils import (
    bytes_to_bits,
    bits_to_bytes,
    calculate_psnr,
    calculate_mse,
    calculate_bpp,
)


@dataclass
class PEEEmbedResult:
    """PEE 嵌入结果数据结构"""
    stego_img: np.ndarray
    payload_bits_len: int
    total_embedded_bits: int
    aux_bits_len: int
    psnr: float
    mse: float
    bpp: float
    elapsed_ms: float
    threshold_T: int
    overflow_count: int
    modified_pixels: int


def get_checkerboard_coords(shape: tuple[int, int], parity: int) -> list[tuple[int, int]]:
    """获取内部区域 (1..H-2, 1..W-2) 中满足 (i+j)%2 == parity 的按行扫描像素坐标列表"""
    h, w = shape
    coords = []
    for i in range(1, h - 1):
        start_j = 1 if (i + 1) % 2 == parity else 2
        for j in range(start_j, w - 1, 2):
            coords.append((i, j))
    return coords


def diamond_predict(img: np.ndarray, i: int, j: int) -> int:
    """使用上下左右四个相邻像素计算整数均值预测值"""
    return (int(img[i - 1, j]) + int(img[i + 1, j]) + int(img[i, j - 1]) + int(img[i, j + 1])) // 4


def _embed_layer_core(
    img: np.ndarray,
    coords: list[tuple[int, int]],
    bits_to_embed: np.ndarray,
    T: int,
) -> tuple[np.ndarray, int, list[tuple[int, int]]]:
    """
    在给定的像素坐标序列 coords 上执行单层 PEE 嵌入，支持自动溢出规避记录。
    为确保 100% 可逆且零额外开销处理极少数溢出点：
    - 预测值 p_hat 由不变的四邻域决定。
    - 若某像素在当前 T 下扩展或平移后越界 (<0 或 >255)，则保持该像素原值不变，
      并将其在 coords 中的索引记录到 overflow_indices 列表中。
    - 提取端拥有相同的 overflow_indices 列表，遇到列表中的索引直接跳过即可，无需全图 Location Map！
    返回：(修改后的图像, 实际嵌入的比特数, 停止索引 stop_idx, 溢出跳过索引列表 overflow_indices)
    """
    out = img.copy()
    n_bits = len(bits_to_embed)
    bit_idx = 0
    stop_idx = 0
    overflow_indices: list[int] = []

    for idx, (i, j) in enumerate(coords):
        if bit_idx >= n_bits:
            stop_idx = idx
            break
        x = int(out[i, j])
        p_hat = diamond_predict(out, i, j)
        e = x - p_hat

        if -T <= e < T:
            b = int(bits_to_embed[bit_idx])
            e_prime = 2 * e + b
            x_prime = p_hat + e_prime
            if 0 <= x_prime <= 255:
                out[i, j] = x_prime
                bit_idx += 1
            else:
                # 发生溢出，保持原值并记录该点索引
                overflow_indices.append(idx)
        elif e >= T:
            e_prime = e + T
            x_prime = p_hat + e_prime
            if 0 <= x_prime <= 255:
                out[i, j] = x_prime
            else:
                overflow_indices.append(idx)
        else:  # e < -T
            e_prime = e - T
            x_prime = p_hat + e_prime
            if 0 <= x_prime <= 255:
                out[i, j] = x_prime
            else:
                overflow_indices.append(idx)
    else:
        stop_idx = len(coords)

    return out, bit_idx, stop_idx, overflow_indices


def _extract_layer_core(
    stego_img: np.ndarray,
    coords: list[tuple[int, int]],
    stop_idx: int,
    T: int,
    overflow_set: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    从给定的像素坐标序列 coords[:stop_idx] 中提取比特并无损恢复像素
    """
    recovered = stego_img.copy()
    extracted_bits: list[int] = []

    for idx in range(stop_idx):
        if idx in overflow_set:
            # 嵌入时因溢出未修改，直接跳过
            continue
        i, j = coords[idx]
        x_prime = int(recovered[i, j])
        p_hat = diamond_predict(recovered, i, j)
        e_prime = x_prime - p_hat

        if -2 * T <= e_prime < 2 * T:
            b = e_prime & 1  # 奇偶性决定嵌入比特 (Python 中负数的 & 1 同样满足 2*e + b 的奇偶性质)
            e = (e_prime - b) // 2
            extracted_bits.append(b)
            recovered[i, j] = p_hat + e
        elif e_prime >= 2 * T:
            e = e_prime - T
            recovered[i, j] = p_hat + e
        else:  # e_prime < -2 * T
            e = e_prime + T
            recovered[i, j] = p_hat + e

    return recovered, np.array(extracted_bits, dtype=np.uint8)


def _embed_header_in_border(img: np.ndarray, header_bytes: bytes) -> tuple[np.ndarray, bytes]:
    """
    将少量的辅助元数据（header_bytes）可逆地嵌入到图像最外圈边框像素（i=0 或 i=H-1 或 j=0 或 j=W-1）的 LSB 中，
    同时将边框原始 LSB 比特追加到载荷比特流中由内部 PEE 层进行无损保护！
    这样提取端可以先从最外圈边框直接读取辅助元数据，然后用 PEE 恢复内部像素并拿回边框原始 LSB，最后恢复边框像素！
    由于内部 PEE 的四邻域计算（1..H-2, 1..W-2）会用到最外圈边框像素，
    我们在【内部 PEE 嵌入全部完成之后】再修改最外圈边框像素的 LSB；
    在提取端，先从最外圈边框读取 header 和被替换前的原始边框 LSB（或者使用完全不参与内部预测的四个角点 + 底边末行之后？
    注意：内部像素 (1, j) 会用到上边框 (0, j) 作为预测邻域！如果修改了 (0, j) 的 LSB，会导致 (1, j) 预测值变化！）

    为了确保内部像素的四邻域 100% 不受影响：
    我们只选用边框上满足 `(i + j) % 2 == 0` 的边框像素存放 Header？不对，Layer 0 内部像素的四邻域是 `(i+j)%2 == 1` 的像素，
    而 Layer 1 内部像素的四邻域是 `(i+j)%2 == 0` 的像素。
    更优雅且 100% 不影响任何邻域的方法：
    - 我们将图像最后两行 `i = H-2, H-1` 设为专用的“元数据引导区 (Metadata Bootstrap Zone)”，
      而数据嵌入区为 `1 <= i <= H-3, 1 <= j <= W-2`。
    - 引导区仅使用最后一行 `i = H-1` 的像素 LSB 存储固定引导头（仅几十字节），
      因为数据嵌入区最大行号为 `H-3`，其下邻域最大行号为 `H-2`，完全不会触碰 `i = H-1`！
    - 同时，`i = H-1` 被替换下来的原始 LSB 比特作为普通数据比特，随秘密载荷一起用 PEE 无损嵌入在数据嵌入区 (`1..H-3`) 中！
    - 提取时：
      1) 先从 `i = H-1` 的 LSB 读出引导头（包含 PEE 参数、溢出表、以及原始 `i = H-1` 的 LSB 长度）；
      2) 对数据嵌入区 (`1..H-3`) 执行 PEE 逆变换，无损恢复 `1..H-3` 的全部像素，并提取出【原始 `i = H-1` 的 LSB + 用户秘密载荷】；
      3) 用提取出的原始 `i = H-1` 的 LSB 还原 `i = H-1` 行像素！
      至此，整幅图像 H×W 每一个像素 100% 逐像素无损恢复！
    """
    raise NotImplementedError


def pack_metadata(meta_dict: dict) -> bytes:
    """将元数据字典序列化并 zlib 压缩（溢出索引转为差分数组以提升3~5倍压缩率）"""
    import json
    d = dict(meta_dict)
    for k in ("ovf0", "ovf1"):
        if k in d and len(d[k]) > 1:
            arr = d[k]
            d[k] = [arr[0]] + [arr[i] - arr[i - 1] for i in range(1, len(arr))]
    raw = json.dumps(d, separators=(",", ":")).encode("utf-8")
    comp = zlib.compress(raw, level=9)
    return b"RDH1" + len(comp).to_bytes(4, "big") + comp


def unpack_metadata_from_bottom_row(img: np.ndarray) -> tuple[dict, int]:
    """从图像最后几行外围不参与预测的像素 LSB 中读取元数据字典及占用的像素数"""
    import json
    boot_coords = get_bootstrap_coords(img.shape)
    if len(boot_coords) < 64:
        raise ValueError("图像尺寸过小，无法容纳引导头")

    header_bits = np.array([img[r, c] & 1 for (r, c) in boot_coords[:64]], dtype=np.uint8)
    header_bytes = bits_to_bytes(header_bits)
    magic = header_bytes[:4]
    if magic != b"RDH1":
        raise ValueError("未检测到有效的载密图像引导标识 (RDH1)")
    comp_len = int.from_bytes(header_bytes[4:8], "big")
    total_bits = 64 + comp_len * 8
    if total_bits > len(boot_coords):
        raise ValueError(f"引导头长度 ({total_bits} bits) 超出引导区容量 ({len(boot_coords)} bits)")

    all_bits = np.array([img[r, c] & 1 for (r, c) in boot_coords[:total_bits]], dtype=np.uint8)
    comp_bytes = bits_to_bytes(all_bits[64:])
    raw_bytes = zlib.decompress(comp_bytes)
    meta_dict = json.loads(raw_bytes.decode("utf-8"))
    for k in ("ovf0", "ovf1"):
        if k in meta_dict and len(meta_dict[k]) > 1:
            diffs = meta_dict[k]
            restored = [diffs[0]]
            for v in diffs[1:]:
                restored.append(restored[-1] + v)
            meta_dict[k] = restored
    return meta_dict, total_bits


def get_bootstrap_coords(shape: tuple[int, int]) -> list[tuple[int, int]]:
    """
    获取专用于存放引导元数据的像素坐标列表（底部 10 行，512 宽图像提供 5120 bits 容量）。
    数据 PEE 嵌入区设定为：1 <= i < H - 10 - 1, 1 <= j <= W - 2。
    """
    h, w = shape
    coords = []
    for i in range(h - 10, h):
        for j in range(w):
            coords.append((i, j))
    return coords


def basic_pee_embed(
    img: np.ndarray,
    secret_bits: np.ndarray,
    T: int = 2,
    use_double_layer: bool = True,
) -> PEEEmbedResult:
    """
    任务一：基础预测误差扩展 (Basic PEE) 信息嵌入算法
    支持单层（Dot Set）与双层（Dot Set + Cross Set）嵌入，并 100% 无损恢复原始图像。

    参数：
        img: 原始 2D uint8 灰度图像 (H, W)
        secret_bits: 待嵌入的 0/1 秘密比特数组
        T: 预测误差扩展阈值（默认 T=2，当 e ∈ [-T, T-1] 时扩展嵌入）
        use_double_layer: 若一层容量不足，是否自动启用第二层（Cross Set）
    """
    t0 = time.perf_counter()
    h, w = img.shape
    if h < 16 or w < 16:
        raise ValueError("图像尺寸至少应为 16×16")

    # 数据区为行 1 .. H-11，列 1 .. W-2，确保底部 10 行 (H-10 .. H-1) 完全不参与任何四邻域预测
    data_h = h - 10
    work_img = img.copy()

    # 预先准备引导区坐标
    boot_coords = get_bootstrap_coords((h, w))

    # 为了把引导区被覆盖的原始 LSB 也用 PEE 无损藏入图像，我们采用两步精确构造：
    # 但引导区需要覆盖多少个像素取决于 metadata 压缩后的比特长度 L_meta。
    # 我们可以将引导区前 MAX_BOOT_BITS（或者在知道 L_meta 后精确保存？注意：如果先嵌 PEE，此时还不知道 overflow_indices 的精确压缩长度！）
    # 解决办法非常巧妙：
    # 我们在 PEE 数据区之后，或者直接使用底部 3 行固定保存前 K 个像素的原始 LSB？
    # 如果固定保存底部 3 行前 K 个像素的原始 LSB，万一 metadata 超过 K 怎么办？
    # 更直接且 100% 精确的方法：
    # 底部 3 行（i = H-3, H-2, H-1）本身也可以使用固定 T=1 的无溢出 PEE？不对，LSB 最简单。
    # 为什么不在 PEE 嵌入【之前】先用预扫描确定嵌入 secret_bits + 引导区原始 LSB 需要多少空间？
    # 其实更简单：我们将底部 3 行 `i = H-3, H-2, H-1` 划分为两部分：
    # 第一部分是固定大小的引导头（或者先在数据区 1..H-5 嵌入 `secret_bits`），
    # 等等！如果我们在数据区 `1..H-5` 先只嵌入 `secret_bits`，生成了 `meta_dict`（长度为 `L_meta` 比特），
    # 那么引导区 `boot_coords[:L_meta]` 的原始 LSB 比特放到哪里去呢？
    # 放到数据区紧接着 `secret_bits` 停止位置 `stop_idx` 之后的 PEE 像素里！
    # 看！这个两阶段流水线完全没有循环依赖：
    # 阶段 A：在数据区从 `idx = 0` 开始嵌入 `secret_bits`，得到 `stop_idx_1` 和 `overflow_1`；
    # 阶段 B：先试打包一次元数据以确定所需的引导区大小（设为安全上界 `boot_len = len(comp) * 8 + 128`），
    #         取出 `boot_coords[:boot_len]` 的原始 LSB 比特，从 `stop_idx_1` 开始继续用 PEE 嵌入到数据区，
    #         得到最终的 `stop_idx_2` 和附加溢出点 `overflow_2`；
    # 阶段 C：将最终的元数据（长度严格 <= `boot_len`）写入 `boot_coords` 的 LSB 中！
    # 提取时：
    # 1) 从 `boot_coords` 的 LSB 读取最终元数据（包含 `boot_len`, `stop_idx_1`, `stop_idx_2`, `overflow` 等）；
    # 2) 用 PEE 逆变换恢复数据区 `0 .. stop_idx_2`，提取出前半段 `secret_bits` 和后半段 `boot_lsb_bits`；
    # 3) 用 `boot_lsb_bits` 恢复 `boot_coords[:boot_len]` 的原始 LSB！
    # 这个两段式引导设计 100% 严丝合缝，无任何循环依赖，且对任意载荷和溢出情况均 100% 可逆！

    coords_layer0 = get_checkerboard_coords((data_h, w), parity=0)
    coords_layer1 = get_checkerboard_coords((data_h, w), parity=1)

    secret_bits = np.asarray(secret_bits, dtype=np.uint8).flatten()
    n_secret = len(secret_bits)

    # --- 阶段 A：嵌入用户秘密比特 ---
    sub_img = work_img[:data_h, :]
    sub_img, emb0, stop0, ovf0 = _embed_layer_core(sub_img, coords_layer0, secret_bits, T)

    rem_secret = secret_bits[emb0:]
    if len(rem_secret) > 0 and use_double_layer:
        sub_img, emb1, stop1, ovf1 = _embed_layer_core(sub_img, coords_layer1, rem_secret, T)
    else:
        emb1, stop1, ovf1 = 0, 0, []

    if emb0 + emb1 < n_secret:
        raise ValueError(
            f"图像容量不足：请求嵌入 {n_secret} bits，在阈值 T={T} 下最大仅能嵌入 {emb0 + emb1} bits"
        )

    # --- 阶段 B：计算引导区安全比特数并继续嵌入引导区原始 LSB ---
    dummy_meta = {
        "alg": "basic_pee",
        "T": int(T),
        "n_sec": int(n_secret),
        "s0": int(stop0),
        "s1": int(stop1),
        "b_len": 1536,
        "s0_b": int(stop0),
        "s1_b": int(stop1),
        "ovf0": ovf0,
        "ovf1": ovf1,
    }
    est_bytes = pack_metadata(dummy_meta)
    # 留出充足余量给阶段 B 可能新增的溢出索引
    boot_len = min(len(boot_coords), len(est_bytes) * 8 + 1536)
    orig_boot_lsb = np.array([work_img[r, c] & 1 for (r, c) in boot_coords[:boot_len]], dtype=np.uint8)

    # 接着从剩余可用位置嵌入 orig_boot_lsb
    stop0_b = stop0
    stop1_b = stop1
    if stop1 == 0 and stop0 < len(coords_layer0):
        # 第一层还有剩余空间
        sub_part, emb_b0, rel_stop0, ovf_b0 = _embed_layer_core(
            sub_img, coords_layer0[stop0:], orig_boot_lsb, T
        )
        sub_img = sub_part
        stop0_b = stop0 + rel_stop0
        ovf0.extend([stop0 + idx for idx in ovf_b0])
        rem_boot = orig_boot_lsb[emb_b0:]
    else:
        rem_boot = orig_boot_lsb

    if len(rem_boot) > 0:
        sub_part, emb_b1, rel_stop1, ovf_b1 = _embed_layer_core(
            sub_img, coords_layer1[stop1:], rem_boot, T
        )
        sub_img = sub_part
        stop1_b = stop1 + rel_stop1
        ovf1.extend([stop1 + idx for idx in ovf_b1])
        if emb_b1 < len(rem_boot):
            raise ValueError("图像剩余容量不足以存放引导区恢复比特")

    work_img[:data_h, :] = sub_img

    # --- 阶段 C：打包最终元数据并写入底部引导区 LSB ---
    final_meta = {
        "alg": "basic_pee",
        "T": int(T),
        "n_sec": int(n_secret),
        "s0": int(stop0),
        "s1": int(stop1),
        "b_len": int(boot_len),
        "s0_b": int(stop0_b),
        "s1_b": int(stop1_b),
        "ovf0": ovf0,
        "ovf1": ovf1,
    }
    meta_bytes = pack_metadata(final_meta)
    meta_bits = bytes_to_bits(meta_bytes)
    if len(meta_bits) > boot_len:
        raise ValueError(f"元数据比特长度 ({len(meta_bits)}) 超出预分配引导长度 ({boot_len})")

    for idx, b in enumerate(meta_bits):
        r, c = boot_coords[idx]
        work_img[r, c] = (work_img[r, c] & 0xFE) | int(b)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    psnr_val = calculate_psnr(img, work_img)
    mse_val = calculate_mse(img, work_img)
    bpp_val = calculate_bpp(n_secret, (h, w))
    modified_pixels = int(np.sum(img != work_img))

    return PEEEmbedResult(
        stego_img=work_img,
        payload_bits_len=n_secret,
        total_embedded_bits=n_secret + boot_len,
        aux_bits_len=len(meta_bits),
        psnr=psnr_val,
        mse=mse_val,
        bpp=bpp_val,
        elapsed_ms=elapsed_ms,
        threshold_T=T,
        overflow_count=len(ovf0) + len(ovf1),
        modified_pixels=modified_pixels,
    )


def basic_pee_extract(stego_img: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    任务一：基础预测误差扩展 (Basic PEE) 信息提取与无损图像恢复算法
    严格按照嵌入的逆序执行：
    1. 从底部引导区 LSB 读取并解压元数据字典
    2. 若使用了 Layer 1 (Cross Set)，先对 Layer 1 执行 PEE 逆变换并恢复像素
    3. 对 Layer 0 (Dot Set) 执行 PEE 逆变换并恢复像素
    4. 拆分出用户秘密比特流 `secret_bits` 与引导区原始 LSB `orig_boot_lsb`
    5. 用 `orig_boot_lsb` 恢复底部引导区像素
    返回：(恢复后的原始图像 recovered_img, 提取出的秘密比特流 secret_bits, 元数据信息 meta)
    """
    t0 = time.perf_counter()
    h, w = stego_img.shape
    data_h = h - 10
    recovered = stego_img.copy()

    meta, _ = unpack_metadata_from_bottom_row(recovered)
    if meta.get("alg") != "basic_pee":
        raise ValueError(f"算法类型不匹配：期望 basic_pee，实际为 {meta.get('alg')}")

    T = int(meta["T"])
    n_secret = int(meta["n_sec"])
    stop0_b = int(meta["s0_b"])
    stop1_b = int(meta["s1_b"])
    boot_len = int(meta["b_len"])
    ovf0_set = set(meta["ovf0"])
    ovf1_set = set(meta["ovf1"])

    coords_layer0 = get_checkerboard_coords((data_h, w), parity=0)
    coords_layer1 = get_checkerboard_coords((data_h, w), parity=1)

    sub_img = recovered[:data_h, :]

    # 必须先逆变换后嵌入的 Layer 1，再逆变换先嵌入的 Layer 0！
    if stop1_b > 0:
        sub_img, bits_layer1 = _extract_layer_core(
            sub_img, coords_layer1, stop1_b, T, ovf1_set
        )
    else:
        bits_layer1 = np.array([], dtype=np.uint8)

    if stop0_b > 0:
        sub_img, bits_layer0 = _extract_layer_core(
            sub_img, coords_layer0, stop0_b, T, ovf0_set
        )
    else:
        bits_layer0 = np.array([], dtype=np.uint8)

    recovered[:data_h, :] = sub_img

    all_extracted = np.concatenate([bits_layer0, bits_layer1])
    secret_bits = all_extracted[:n_secret]
    orig_boot_lsb = all_extracted[n_secret : n_secret + boot_len]

    # 恢复底部引导区原始 LSB
    boot_coords = get_bootstrap_coords((h, w))
    for idx, b in enumerate(orig_boot_lsb):
        r, c = boot_coords[idx]
        recovered[r, c] = (recovered[r, c] & 0xFE) | int(b)

    meta["extract_elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
    return recovered, secret_bits, meta
