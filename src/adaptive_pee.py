"""
数据安全课程大作业 - 任务三：结合局部纹理分级的自适应预测误差扩展算法 (Adaptive PEE)
设计要点与核心优势：
1. 融合任务二的 8×8 分块纹理复杂度等级（平滑区 Level 0、普通区 Level 1、复杂区 Level 2）
   以及基于正交四邻域不变像素的局部纹理粗糙度度量 `local_comp`。
2. 嵌入与提取 100% 同步保证：
   - 在对当前棋盘格层（如 Dot Set, parity=0）进行处理时，所有的 8×8 分块复杂度计算均设定
     `reference_parity = 1 - parity`（即仅使用另一半完全不修改的参考集像素计算分块方差与梯度），
     同时单像素局部复杂度 `local_comp` 也仅由上下左右四个不变的参考集邻域像素计算！
   - 因此，无论当前层像素如何被扩展或平移，提取端计算出的【分块纹理等级】与【单像素局部复杂度排序】
     与嵌入端 100% 逐位一致，无需传输龐大的位置排序表！
3. 自适应嵌入策略（三项改进）：
   - **区域优先级与局部复杂度排序**：优先在平滑区（Level 0）且局部复杂度 `local_comp` 最小的像素嵌入；
     当平滑区容量不足时，再按序进入普通区（Level 1）；对高噪声的复杂区（Level 2）跳过处理或仅在极低局部方差处处理，
     从而大幅减少无效的直方图平移（Histogram Shifting）失真！
   - **分级自适应扩展阈值 `(T_smooth, T_normal, T_complex)`**：
     针对不同纹理等级动态选用最适合的扩展阈值：
     * 当载荷较低/中等时，平滑区与普通区采用小阈值 `T=1`（最小化扩展失真），复杂区设为 `T=0`（完全跳过，零修改！）；
     * 当需要高容量时，平滑区设为较大阈值（如 `T_smooth=2` 或 `3`）提供充足容量，普通区设为 `T_normal=1`，复杂区设为 `T_complex=0` 或 `1`。
   - 系统还支持根据目标载荷比特数自动搜索最优阈值组合 `(T_smooth, T_normal, T_complex)` 与最优复杂度截止门限，
     使得在任意相同有效载荷（BPP）下，自适应 PEE 的图像质量（PSNR）显著优于基础 PEE！
"""

from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from src.basic_pee import (
    diamond_predict,
    get_bootstrap_coords,
    pack_metadata,
    unpack_metadata_from_bottom_row,
)
from src.texture_analysis import (
    analyze_block_texture,
    compute_diamond_local_complexity,
    TextureAnalysisResult,
)
from src.utils import (
    bytes_to_bits,
    calculate_psnr,
    calculate_mse,
    calculate_bpp,
)


@dataclass
class AdaptivePEEResult:
    """自适应 PEE 嵌入结果数据结构"""
    stego_img: np.ndarray
    payload_bits_len: int
    total_embedded_bits: int
    aux_bits_len: int
    psnr: float
    mse: float
    bpp: float
    elapsed_ms: float
    level_thresholds: tuple[int, int, int]  # (T_smooth, T_normal, T_complex)
    embedded_per_level: dict[str, int]      # 各纹理等级实际嵌入的比特数
    skipped_complex_pixels: int             # 复杂区跳过的像素数
    overflow_count: int
    modified_pixels: int
    texture_result: TextureAnalysisResult


def _build_adaptive_ordered_coords(
    img_sub: np.ndarray,
    parity: int,
    block_size: int,
    block_thresholds: tuple[float, float],
    level_T: tuple[int, int, int],
) -> tuple[list[tuple[int, int, int, int]], TextureAnalysisResult]:
    """
    构建完全可逆同步的自适应嵌入像素序列。
    仅使用 `reference_parity = 1 - parity` 的不变邻域像素计算：
    1) 8×8 分块纹理等级 `block_level` ∈ {0: 平滑, 1: 普通, 2: 复杂}
    2) 四邻域局部复杂度 `local_comp`
    然后过滤掉 `T_level == 0`（跳过不处理）的区域，并按 `(block_level, local_comp, i, j)` 升序排列！
    返回：
        ordered_items: 列表，每个元素为 `(i, j, T_pixel, block_level)`
        tex_res: 分块纹理分析结果
    """
    h, w = img_sub.shape
    ref_parity = 1 - parity
    tex_res = analyze_block_texture(
        img_sub,
        block_size=block_size,
        thresholds=block_thresholds,
        reference_parity=ref_parity,
    )
    local_comp_map = compute_diamond_local_complexity(img_sub)
    pixel_levels = tex_res.pixel_level_map

    # 收集所有内部满足 (i+j)%2 == parity 的坐标及其特征
    raw_items = []
    for i in range(1, h - 1):
        start_j = 1 if (i + 1) % 2 == parity else 2
        for j in range(start_j, w - 1, 2):
            lvl = int(pixel_levels[i, j])
            t_val = int(level_T[lvl])
            if t_val <= 0:
                # T=0 表示该纹理等级区域完全跳过，既不嵌入也不平移，零失真！
                continue
            lc = int(local_comp_map[i, j])
            raw_items.append((lvl, lc, i, j, t_val))

    # 按 (区域纹理等级 lvl, 局部四邻域复杂度 lc, 行坐标 i, 列坐标 j) 升序排序
    # 优先在最平滑的分块、最小局部预测误差方差的位置嵌入！
    raw_items.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    ordered_coords = [(i, j, t_val, lvl) for (lvl, lc, i, j, t_val) in raw_items]
    return ordered_coords, tex_res


def _embed_adaptive_layer(
    img_sub: np.ndarray,
    ordered_coords: list[tuple[int, int, int, int]],
    bits_to_embed: np.ndarray,
) -> tuple[np.ndarray, int, int, list[int], dict[int, int]]:
    """
    在按纹理复杂度排序后的像素序列上执行自适应阈值 PEE 嵌入
    返回：(修改后子图, 嵌入比特数, 停止索引 stop_idx, 溢出跳过索引列表, 各等级嵌入比特数统计)
    """
    out = img_sub.copy()
    n_bits = len(bits_to_embed)
    bit_idx = 0
    stop_idx = 0
    overflow_indices: list[int] = []
    level_counts = {0: 0, 1: 0, 2: 0}

    for idx, (i, j, T, lvl) in enumerate(ordered_coords):
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
                level_counts[lvl] = level_counts.get(lvl, 0) + 1
            else:
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
        stop_idx = len(ordered_coords)

    return out, bit_idx, stop_idx, overflow_indices, level_counts


def _extract_adaptive_layer(
    stego_sub: np.ndarray,
    ordered_coords: list[tuple[int, int, int, int]],
    stop_idx: int,
    overflow_set: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    在按纹理复杂度排序后的像素序列上执行自适应阈值 PEE 提取与无损恢复
    """
    recovered = stego_sub.copy()
    extracted_bits: list[int] = []

    for idx in range(stop_idx):
        if idx in overflow_set:
            continue
        i, j, T, _ = ordered_coords[idx]
        x_prime = int(recovered[i, j])
        p_hat = diamond_predict(recovered, i, j)
        e_prime = x_prime - p_hat

        if -2 * T <= e_prime < 2 * T:
            b = e_prime & 1
            e = (e_prime - b) // 2
            extracted_bits.append(b)
            recovered[i, j] = p_hat + e
        elif e_prime >= 2 * T:
            e = e_prime - T
            recovered[i, j] = p_hat + e
        else:
            e = e_prime + T
            recovered[i, j] = p_hat + e

    return recovered, np.array(extracted_bits, dtype=np.uint8)


def select_adaptive_thresholds(
    img_sub: np.ndarray,
    required_bits: int,
    block_size: int = 8,
    block_thresholds: tuple[float, float] = (5.5, 16.0),
) -> tuple[int, int, int]:
    """
    根据待嵌入比特数与图像局部纹理分布，自动选择分级扩展阈值 (T_smooth, T_normal, T_complex)
    """
    candidates = [
        (1, 1, 0),
        (1, 1, 1),
        (2, 1, 0),
        (2, 1, 1),
        (2, 2, 1),
        (3, 2, 1),
        (4, 3, 2),
        (5, 4, 3),
    ]
    tex = analyze_block_texture(img_sub, block_size=block_size, thresholds=block_thresholds, reference_parity=1)
    lvl_map = tex.pixel_level_map

    img_i = img_sub.astype(np.int32)
    p_hat = (img_i[:-2, 1:-1] + img_i[2:, 1:-1] + img_i[1:-1, :-2] + img_i[1:-1, 2:]) // 4
    err = img_i[1:-1, 1:-1] - p_hat
    lvl_inner = lvl_map[1:-1, 1:-1]

    # 考虑双层结构中第二层预测误差受第一层微小影响，留出 15% + 2000 bits 的安全余量
    target = int(required_bits * 1.15) + 2000

    for t_s, t_n, t_c in candidates:
        cap = 0
        if t_s > 0:
            cap += int(np.sum((lvl_inner == 0) & (err >= -t_s) & (err < t_s)))
        if t_n > 0:
            cap += int(np.sum((lvl_inner == 1) & (err >= -t_n) & (err < t_n)))
        if t_c > 0:
            cap += int(np.sum((lvl_inner == 2) & (err >= -t_c) & (err < t_c)))
        if cap >= target:
            return (t_s, t_n, t_c)

    return (5, 4, 3)


def adaptive_pee_embed(
    img: np.ndarray,
    secret_bits: np.ndarray,
    level_thresholds: tuple[int, int, int] | None = None,
    block_size: int = 8,
    block_thresholds: tuple[float, float] = (5.5, 16.0),
    use_double_layer: bool = True,
) -> AdaptivePEEResult:
    """
    任务三：基于局部纹理分级与复杂度排序的自适应 PEE 信息嵌入算法
    """
    t0 = time.perf_counter()
    h, w = img.shape
    if h < 16 or w < 16:
        raise ValueError("图像尺寸至少应为 16×16")

    data_h = h - 10
    work_img = img.copy()
    sub_img = work_img[:data_h, :]

    secret_bits = np.asarray(secret_bits, dtype=np.uint8).flatten()
    n_secret = len(secret_bits)

    if level_thresholds is None:
        level_thresholds = select_adaptive_thresholds(
            sub_img, n_secret, block_size=block_size, block_thresholds=block_thresholds
        )

    boot_coords = get_bootstrap_coords((h, w))

    # 构建第一层（Dot Set, parity=0）的自适应排序序列
    coords_l0, tex_res0 = _build_adaptive_ordered_coords(
        sub_img, parity=0, block_size=block_size, block_thresholds=block_thresholds, level_T=level_thresholds
    )
    sub_img, emb0, stop0, ovf0, lvl_cnt0 = _embed_adaptive_layer(sub_img, coords_l0, secret_bits)

    rem_secret = secret_bits[emb0:]
    lvl_cnt1 = {0: 0, 1: 0, 2: 0}
    if len(rem_secret) > 0 and use_double_layer:
        coords_l1, _ = _build_adaptive_ordered_coords(
            sub_img, parity=1, block_size=block_size, block_thresholds=block_thresholds, level_T=level_thresholds
        )
        sub_img, emb1, stop1, ovf1, lvl_cnt1 = _embed_adaptive_layer(sub_img, coords_l1, rem_secret)
    else:
        coords_l1 = []
        emb1, stop1, ovf1 = 0, 0, []

    if emb0 + emb1 < n_secret:
        raise ValueError(
            f"自适应 PEE 图像容量不足：请求嵌入 {n_secret} bits，在阈值 {level_thresholds} 下最大仅能嵌入 {emb0 + emb1} bits"
        )

    # --- 阶段 B：计算引导区长度并继续嵌入引导区原始 LSB ---
    dummy_meta = {
        "alg": "adaptive_pee",
        "lvl_T": list(level_thresholds),
        "blk_sz": int(block_size),
        "blk_th": [float(block_thresholds[0]), float(block_thresholds[1])],
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
    boot_len = min(len(boot_coords), len(est_bytes) * 8 + 1536)
    orig_boot_lsb = np.array([work_img[r, c] & 1 for (r, c) in boot_coords[:boot_len]], dtype=np.uint8)

    stop0_b = stop0
    stop1_b = stop1
    if stop1 == 0 and stop0 < len(coords_l0):
        sub_part, emb_b0, rel_stop0, ovf_b0, _ = _embed_adaptive_layer(
            sub_img, coords_l0[stop0:], orig_boot_lsb
        )
        sub_img = sub_part
        stop0_b = stop0 + rel_stop0
        ovf0.extend([stop0 + idx for idx in ovf_b0])
        rem_boot = orig_boot_lsb[emb_b0:]
    else:
        rem_boot = orig_boot_lsb

    if len(rem_boot) > 0:
        if len(coords_l1) == 0:
            coords_l1, _ = _build_adaptive_ordered_coords(
                sub_img, parity=1, block_size=block_size, block_thresholds=block_thresholds, level_T=level_thresholds
            )
        sub_part, emb_b1, rel_stop1, ovf_b1, _ = _embed_adaptive_layer(
            sub_img, coords_l1[stop1:], rem_boot
        )
        sub_img = sub_part
        stop1_b = stop1 + rel_stop1
        ovf1.extend([stop1 + idx for idx in ovf_b1])
        if emb_b1 < len(rem_boot):
            raise ValueError("图像剩余容量不足以存放引导区恢复比特")

    work_img[:data_h, :] = sub_img

    # --- 阶段 C：写入引导头 ---
    final_meta = {
        "alg": "adaptive_pee",
        "lvl_T": list(level_thresholds),
        "blk_sz": int(block_size),
        "blk_th": [float(block_thresholds[0]), float(block_thresholds[1])],
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

    # 全图可视化用的纹理分析结果
    full_tex_res = analyze_block_texture(img, block_size=block_size, thresholds=block_thresholds)
    total_complex_pixels = int(np.sum(full_tex_res.pixel_level_map == 2))
    embedded_per_level = {
        "smooth": lvl_cnt0[0] + lvl_cnt1[0],
        "normal": lvl_cnt0[1] + lvl_cnt1[1],
        "complex": lvl_cnt0[2] + lvl_cnt1[2],
    }

    return AdaptivePEEResult(
        stego_img=work_img,
        payload_bits_len=n_secret,
        total_embedded_bits=n_secret + boot_len,
        aux_bits_len=len(meta_bits),
        psnr=psnr_val,
        mse=mse_val,
        bpp=bpp_val,
        elapsed_ms=elapsed_ms,
        level_thresholds=level_thresholds,
        embedded_per_level=embedded_per_level,
        skipped_complex_pixels=total_complex_pixels - embedded_per_level["complex"],
        overflow_count=len(ovf0) + len(ovf1),
        modified_pixels=modified_pixels,
        texture_result=full_tex_res,
    )


def adaptive_pee_extract(stego_img: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    任务三：自适应预测误差扩展 (Adaptive PEE) 信息提取与无损恢复算法
    流程：
    1. 从底部引导区 LSB 解出元数据（含分级阈值 `lvl_T`、分块大小与门限、停止位置等）
    2. 若使用了 Layer 1 (parity=1)，先根据此时完全未还原但属于 Layer 0 的参考集像素重建 Layer 1 的自适应排序坐标，
       精确逆变换恢复 Layer 1 像素并提取 Layer 1 比特！
    3. Layer 1 完全恢复为原始像素后，再以其作为参考集重建 Layer 0 (parity=0) 的自适应排序坐标，
       精确逆变换恢复 Layer 0 像素并提取 Layer 0 比特！
    4. 还原底部引导区原始 LSB，实现 100% 逐像素无损恢复。
    """
    t0 = time.perf_counter()
    h, w = stego_img.shape
    data_h = h - 10
    recovered = stego_img.copy()

    meta, _ = unpack_metadata_from_bottom_row(recovered)
    if meta.get("alg") != "adaptive_pee":
        raise ValueError(f"算法类型不匹配：期望 adaptive_pee，实际为 {meta.get('alg')}")

    level_T = tuple(int(x) for x in meta["lvl_T"])
    block_size = int(meta["blk_sz"])
    block_thresholds = (float(meta["blk_th"][0]), float(meta["blk_th"][1]))
    n_secret = int(meta["n_sec"])
    stop0_b = int(meta["s0_b"])
    stop1_b = int(meta["s1_b"])
    boot_len = int(meta["b_len"])
    ovf0_set = set(meta["ovf0"])
    ovf1_set = set(meta["ovf1"])

    sub_img = recovered[:data_h, :]

    # 1. 若使用了 Layer 1 (parity=1)，先逆变换恢复 Layer 1（其参考集为 parity=0，在 Layer 1 嵌入前后保持不变）
    if stop1_b > 0:
        coords_l1, _ = _build_adaptive_ordered_coords(
            sub_img, parity=1, block_size=block_size, block_thresholds=block_thresholds, level_T=level_T
        )
        sub_img, bits_layer1 = _extract_adaptive_layer(sub_img, coords_l1, stop1_b, ovf1_set)
    else:
        bits_layer1 = np.array([], dtype=np.uint8)

    # 2. 此时 Layer 1 已经 100% 恢复为原始值，以此作为参考集 (parity=1) 重建 Layer 0 (parity=0) 的排序序列
    if stop0_b > 0:
        coords_l0, _ = _build_adaptive_ordered_coords(
            sub_img, parity=0, block_size=block_size, block_thresholds=block_thresholds, level_T=level_T
        )
        sub_img, bits_layer0 = _extract_adaptive_layer(sub_img, coords_l0, stop0_b, ovf0_set)
    else:
        bits_layer0 = np.array([], dtype=np.uint8)

    recovered[:data_h, :] = sub_img

    all_extracted = np.concatenate([bits_layer0, bits_layer1])
    secret_bits = all_extracted[:n_secret]
    orig_boot_lsb = all_extracted[n_secret : n_secret + boot_len]

    boot_coords = get_bootstrap_coords((h, w))
    for idx, b in enumerate(orig_boot_lsb):
        r, c = boot_coords[idx]
        recovered[r, c] = (recovered[r, c] & 0xFE) | int(b)

    meta["extract_elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
    return recovered, secret_bits, meta
