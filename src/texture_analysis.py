"""
数据安全课程大作业 - 任务二：图像局部纹理复杂度分析模块
功能：
1. 将输入图像划分为固定大小的区域（默认 8×8 像素块）
2. 计算每个分块的局部方差与相邻像素绝对差均值（梯度复杂度）
3. 根据复杂度阈值将图像区域划分为三个等级：
   - 等级 0：平滑区 (Smooth)
   - 等级 1：普通区 (Normal)
   - 等级 2：复杂区 (Complex)
4. 统计不同等级区域的数量、占比和分布情况
5. 支持基于棋盘格参考集（Cross/Dot 不变邻域像素）计算完全可逆同步的分块复杂度与单像素局部复杂度
6. 生成热力图与灰度/彩色分级图展示分析结果
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 配置中文字体支持
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


@dataclass
class TextureAnalysisResult:
    """局部纹理复杂度分析结果数据结构"""
    block_size: int
    grid_shape: tuple[int, int]
    block_variance_map: np.ndarray      # (grid_h, grid_w) 每个分块的局部方差
    block_gradient_map: np.ndarray      # (grid_h, grid_w) 每个分块的相邻像素差均值
    block_score_map: np.ndarray         # (grid_h, grid_w) 综合复杂度得分
    block_level_map: np.ndarray         # (grid_h, grid_w) 0:平滑, 1:普通, 2:复杂
    pixel_level_map: np.ndarray         # (H, W) 映射回原图尺寸的分级图
    pixel_score_map: np.ndarray         # (H, W) 映射回原图尺寸的复杂度热力图
    thresholds: tuple[float, float]     # (T_low, T_high) 分级阈值
    counts: dict[str, int]              # 各等级分块数量 {"smooth": .., "normal": .., "complex": ..}
    ratios: dict[str, float]            # 各等级分块占比


def analyze_block_texture(
    img: np.ndarray,
    block_size: int = 8,
    thresholds: tuple[float, float] | None = None,
    reference_parity: int | None = None,
) -> TextureAnalysisResult:
    """
    对输入灰度图像进行固定分块（默认 8x8）局部纹理复杂度分析。

    参数：
        img: 输入 2D uint8 灰度图像 (H, W)
        block_size: 分块大小，默认 8
        thresholds: (T_low, T_high) 复杂度得分阈值。若为 None，则采用经验固定阈值 (6.0, 18.0)
                    （使用相邻像素差分绝对值均值 + 标准差综合度量）
        reference_parity: 若为 None，使用块内全部像素计算（用于直接图像分析展示）；
                          若为 0 或 1，仅使用块内 (i+j)%2 == reference_parity 的像素计算，
                          保证在另一半像素被嵌入修改后，提取端计算出的分块等级 100% 一致！
    """
    h, w = img.shape
    grid_h = (h + block_size - 1) // block_size
    grid_w = (w + block_size - 1) // block_size

    var_map = np.zeros((grid_h, grid_w), dtype=np.float64)
    grad_map = np.zeros((grid_h, grid_w), dtype=np.float64)
    score_map = np.zeros((grid_h, grid_w), dtype=np.float64)

    img_f = img.astype(np.float64)

    for bi in range(grid_h):
        r0 = bi * block_size
        r1 = min(h, r0 + block_size)
        for bj in range(grid_w):
            c0 = bj * block_size
            c1 = min(w, c0 + block_size)
            block = img_f[r0:r1, c0:c1]

            if reference_parity is None:
                # 使用完整 8x8 块计算局部方差与相邻像素差值
                var_val = float(np.var(block))
                dh = np.abs(block[:, 1:] - block[:, :-1]) if block.shape[1] > 1 else np.zeros(1)
                dv = np.abs(block[1:, :] - block[:-1, :]) if block.shape[0] > 1 else np.zeros(1)
                grad_val = float((np.mean(dh) + np.mean(dv)) / 2.0)
            else:
                # 仅使用指定奇偶集合的像素（保证嵌入前后100%不变）
                rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
                mask = ((rr + cc) % 2) == reference_parity
                ref_pixels = block[mask]
                var_val = float(np.var(ref_pixels)) if len(ref_pixels) > 1 else 0.0
                if len(ref_pixels) > 1:
                    grad_val = float(np.mean(np.abs(np.diff(ref_pixels)))) * 0.85
                else:
                    grad_val = float(np.std(ref_pixels))

            # 综合纹理复杂度得分：结合局部标准差 sqrt(var) 与局部平均梯度
            score = 0.5 * np.sqrt(var_val) + 0.5 * grad_val
            var_map[bi, bj] = var_val
            grad_map[bi, bj] = grad_val
            score_map[bi, bj] = score

    if thresholds is None:
        t_low, t_high = 5.5, 16.0
    else:
        t_low, t_high = thresholds

    level_map = np.zeros((grid_h, grid_w), dtype=np.uint8)
    level_map[(score_map >= t_low) & (score_map < t_high)] = 1
    level_map[score_map >= t_high] = 2

    # 映射回像素尺寸
    pixel_level_map = np.zeros((h, w), dtype=np.uint8)
    pixel_score_map = np.zeros((h, w), dtype=np.float64)
    for bi in range(grid_h):
        r0 = bi * block_size
        r1 = min(h, r0 + block_size)
        for bj in range(grid_w):
            c0 = bj * block_size
            c1 = min(w, c0 + block_size)
            pixel_level_map[r0:r1, c0:c1] = level_map[bi, bj]
            pixel_score_map[r0:r1, c0:c1] = score_map[bi, bj]

    total_blocks = grid_h * grid_w
    n_smooth = int(np.sum(level_map == 0))
    n_normal = int(np.sum(level_map == 1))
    n_complex = int(np.sum(level_map == 2))

    counts = {"smooth": n_smooth, "normal": n_normal, "complex": n_complex, "total": total_blocks}
    ratios = {
        "smooth": n_smooth / total_blocks,
        "normal": n_normal / total_blocks,
        "complex": n_complex / total_blocks,
    }

    return TextureAnalysisResult(
        block_size=block_size,
        grid_shape=(grid_h, grid_w),
        block_variance_map=var_map,
        block_gradient_map=grad_map,
        block_score_map=score_map,
        block_level_map=level_map,
        pixel_level_map=pixel_level_map,
        pixel_score_map=pixel_score_map,
        thresholds=(t_low, t_high),
        counts=counts,
        ratios=ratios,
    )


def compute_diamond_local_complexity(img: np.ndarray) -> np.ndarray:
    """
    计算每个像素点周围4个菱形正交邻域像素（上、下、左、右）的局部复杂度。
    特点：由于 (i, j) 的四个正交邻域全部属于相反的棋盘格奇偶集合，
    在单层嵌入过程中，这4个邻域像素完全不变，因此嵌入端和提取端计算出的局部复杂度 100% 相同！
    计算公式：四个邻域像素两两差值绝对值之和 + 局部极差。
    """
    img_i = img.astype(np.int32)
    h, w = img_i.shape
    comp = np.zeros((h, w), dtype=np.int32)

    u = img_i[:-2, 1:-1]   # (i-1, j)
    d = img_i[2:, 1:-1]    # (i+1, j)
    l = img_i[1:-1, :-2]   # (i, j-1)
    r = img_i[1:-1, 2:]    # (i, j+1)

    # 局部方向差分与环形差分之和，准确反映中心像素所在位置的局部纹理粗糙度
    local_var = (
        np.abs(u - d)
        + np.abs(l - r)
        + np.abs(u - l)
        + np.abs(l - d)
        + np.abs(d - r)
        + np.abs(r - u)
    ) // 3
    comp[1:-1, 1:-1] = local_var
    return comp


def plot_texture_analysis(
    img: np.ndarray,
    result: TextureAnalysisResult,
    save_path: str | Path | None = None,
    title_prefix: str = "",
) -> plt.Figure:
    """
    绘制图像局部纹理复杂度分析四联图：
    1. 原始灰度图像
    2. 8x8 分块局部复杂度热力图 (Heatmap)
    3. 三级纹理分级图 (平滑/普通/复杂)
    4. 各等级区域数量与占比统计柱状图
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=150)

    # 1. 原始图像
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"{title_prefix}原始灰度图像 ({img.shape[1]}×{img.shape[0]})", fontsize=11)
    axes[0].axis("off")

    # 2. 复杂度热力图
    im1 = axes[1].imshow(result.pixel_score_map, cmap="inferno")
    axes[1].set_title(f"局部纹理复杂度热力图 ({result.block_size}×{result.block_size}分块)", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="复杂度得分")

    # 3. 纹理三级分级图（蓝绿：平滑，金黄：普通，绯红：复杂）
    cmap_levels = ListedColormap(["#2b83ba", "#fdae61", "#d7191c"])
    im2 = axes[2].imshow(result.pixel_level_map, cmap=cmap_levels, vmin=0, vmax=2)
    axes[2].set_title("区域三级纹理分级图", fontsize=11)
    axes[2].axis("off")
    cbar = plt.colorbar(im2, ax=axes[2], ticks=[0.33, 1.0, 1.67], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(["平滑区 (0)", "普通区 (1)", "复杂区 (2)"])

    # 4. 区域分布统计图
    labels = ["平滑区\n(Smooth)", "普通区\n(Normal)", "复杂区\n(Complex)"]
    counts = [result.counts["smooth"], result.counts["normal"], result.counts["complex"]]
    ratios = [result.ratios["smooth"] * 100, result.ratios["normal"] * 100, result.ratios["complex"] * 100]
    colors = ["#2b83ba", "#fdae61", "#d7191c"]

    bars = axes[3].bar(labels, counts, color=colors, edgecolor="#333333", width=0.55)
    axes[3].set_title(f"分块等级统计 (共 {result.counts['total']} 块)", fontsize=11)
    axes[3].set_ylabel("分块数量 (个)")
    axes[3].grid(axis="y", linestyle="--", alpha=0.4)
    max_c = max(counts) if max(counts) > 0 else 1
    axes[3].set_ylim(0, max_c * 1.25)

    for bar, cnt, rat in zip(bars, counts, ratios):
        axes[3].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_c * 0.03,
            f"{cnt}块\n({rat:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
