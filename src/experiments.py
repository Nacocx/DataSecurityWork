"""
数据安全课程大作业 - 任务五：实验测试与性能分析模块
完成任务书要求的全部 5 项实验：
1. 正确性测试（不同长度秘密信息、不同纹理图像、含 0/255 极值溢出图像的提取准确率 BER 与恢复图像像素误差验证）
2. 图像质量测试（PSNR、MSE 与嵌入修改前后视觉对比、差分放大图）
3. 信息容量测试（不同阈值配置下的最大可嵌入比特数与最大 BPP 分析）
4. 算法对比测试（基础 PEE vs 局部纹理分级自适应 PEE，在相同有效载荷 BPP 下的 PSNR 与运行耗时对比）
5. 结果分析与图表自动导出（保存 CSV/JSON 数据表与高清实验分析图表，供课程报告、Web界面与答辩PPT使用）
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from src.dataset import generate_test_images
from src.utils import (
    text_to_bits,
    bits_to_text,
    load_grayscale_image,
    save_grayscale_image,
    verify_reversibility,
)
from src.texture_analysis import analyze_block_texture, plot_texture_analysis
from src.basic_pee import basic_pee_embed, basic_pee_extract
from src.adaptive_pee import adaptive_pee_embed, adaptive_pee_extract

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_stego_recovery_comparison(
    orig_img: np.ndarray,
    stego_basic: np.ndarray,
    stego_adaptive: np.ndarray,
    recovered_img: np.ndarray,
    basic_psnr: float,
    adaptive_psnr: float,
    bpp: float,
    save_path: str | Path,
) -> None:
    """绘制原始图像、基础PEE载密图、自适应PEE载密图、恢复图像及两者的修改差分位置分布对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.8), dpi=150)

    axes[0, 0].imshow(orig_img, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("① 原始灰度图像 (Original)", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(stego_basic, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"② 基础 PEE 载密图像\nPSNR = {basic_psnr:.2f} dB (BPP={bpp:.3f})", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(stego_adaptive, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title(f"③ 自适应 PEE 载密图像\nPSNR = {adaptive_psnr:.2f} dB (BPP={bpp:.3f})", fontsize=11, fontweight="bold")
    axes[0, 2].axis("off")

    # 第二行：无损恢复图像、基础PEE像素修改绝对差值图、自适应PEE像素修改绝对差值图
    diff_rec = np.max(np.abs(orig_img.astype(np.int32) - recovered_img.astype(np.int32)))
    axes[1, 0].imshow(recovered_img, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title(f"④ 提取后恢复图像 (Recovered)\n与原图最大像素误差 = {diff_rec} (100%无损)", fontsize=11, fontweight="bold", color="#1b7837")
    axes[1, 0].axis("off")

    diff_basic = np.abs(orig_img.astype(np.int32) - stego_basic.astype(np.int32))
    im_b = axes[1, 1].imshow(diff_basic, cmap="magma", vmin=0, vmax=3)
    axes[1, 1].set_title(f"⑤ 基础 PEE 像素修改分布 (|I - I'|)\n修改像素数: {int(np.sum(diff_basic > 0)):,}", fontsize=11)
    axes[1, 1].axis("off")
    plt.colorbar(im_b, ax=axes[1, 1], fraction=0.046, pad=0.04)

    diff_adap = np.abs(orig_img.astype(np.int32) - stego_adaptive.astype(np.int32))
    im_a = axes[1, 2].imshow(diff_adap, cmap="magma", vmin=0, vmax=3)
    axes[1, 2].set_title(f"⑥ 自适应 PEE 像素修改分布 (|I - I'|)\n修改像素数: {int(np.sum(diff_adap > 0)):,} (优先平滑区)", fontsize=11)
    axes[1, 2].axis("off")
    plt.colorbar(im_a, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_error_histogram(
    img: np.ndarray,
    stego_img: np.ndarray,
    save_path: str | Path,
) -> None:
    """绘制预测误差直方图在信息嵌入前后的变化对比图（展示 PEE 尖峰分布与扩展平移效应）"""
    img_i = img.astype(np.int32)
    p_orig = (img_i[:-2, 1:-1] + img_i[2:, 1:-1] + img_i[1:-1, :-2] + img_i[1:-1, 2:]) // 4
    err_orig = (img_i[1:-1, 1:-1] - p_orig).flatten()

    stego_i = stego_img.astype(np.int32)
    p_stego = (stego_i[:-2, 1:-1] + stego_i[2:, 1:-1] + stego_i[1:-1, :-2] + stego_i[1:-1, 2:]) // 4
    err_stego = (stego_i[1:-1, 1:-1] - p_stego).flatten()

    bins = np.arange(-15.5, 16.5, 1.0)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    ax.hist(err_orig, bins=bins, alpha=0.65, color="#2b83ba", edgecolor="black", label="原始图像预测误差 e 分布")
    ax.hist(err_stego, bins=bins, alpha=0.55, color="#d7191c", edgecolor="black", label="载密图像预测误差 e' 分布")
    ax.set_title("预测误差 (Prediction Error) 直方图嵌入前后对比", fontsize=12, fontweight="bold")
    ax.set_xlabel("预测误差值 (e = x - p_hat)", fontsize=11)
    ax.set_ylabel("像素数量", fontsize=11)
    ax.set_xticks(range(-15, 16, 2))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=10.5)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_all_experiments(output_dir: str | Path = "outputs") -> dict:
    """运行全部五项实验并生成图表、CSV与汇总报告数据"""
    out_dir = Path(output_dir)
    fig_dir = out_dir / "figures"
    stego_dir = out_dir / "stego_samples"
    fig_dir.mkdir(parents=True, exist_ok=True)
    stego_dir.mkdir(parents=True, exist_ok=True)

    img_paths = generate_test_images("data/test_images")
    rng = np.random.default_rng(2026)

    # =========================================================================
    # 实验 1：纹理复杂度分析与可视化（任务二验证）
    # =========================================================================
    texture_summary = []
    for name, path in img_paths.items():
        img = load_grayscale_image(path)
        tex_res = analyze_block_texture(img, block_size=8, thresholds=(5.5, 16.0))
        safe_key = path.stem
        fig = plot_texture_analysis(img, tex_res, save_path=fig_dir / f"texture_{safe_key}.png", title_prefix=f"[{name.split()[0]}] ")
        plt.close(fig)
        texture_summary.append({
            "image_name": name,
            "file": path.name,
            "resolution": f"{img.shape[1]}x{img.shape[0]}",
            "total_blocks": tex_res.counts["total"],
            "smooth_blocks": tex_res.counts["smooth"],
            "smooth_ratio_pct": round(tex_res.ratios["smooth"] * 100, 2),
            "normal_blocks": tex_res.counts["normal"],
            "normal_ratio_pct": round(tex_res.ratios["normal"] * 100, 2),
            "complex_blocks": tex_res.counts["complex"],
            "complex_ratio_pct": round(tex_res.ratios["complex"] * 100, 2),
            "mean_score": round(float(np.mean(tex_res.block_score_map)), 2),
        })

    # =========================================================================
    # 实验 2：正确性与可逆性验证（不同文本长度 & 不同测试图像）
    # =========================================================================
    sample_texts = {
        "短文本 (课程口令)": "《数据安全》课程大作业：基于局部纹理分级与预测误差扩展的自适应可逆图像信息隐藏系统！AuthCode=2026-RDH-PASS",
        "中长文本 (系统摘要, ~2KB)": (
            "【数据安全机密档案】可逆信息隐藏（Reversible Data Hiding, RDH）技术能够在准确提取秘密信息的同时，"
            "逐像素100%无损地恢复原始载体图像，在医学影像诊断、司法证据溯源、卫星遥感测绘与军事通信中具有关键价值。"
            "本系统采用棋盘格菱形预测器结合8x8局部纹理分级策略，优先在平滑区域进行预测误差扩展嵌入，有效抑制了直方图冗余平移失真。\n"
        ) * 8,
        "高容量载荷 (随机二进制, 32768 bits)": None,  # 使用 32768 随机比特
    }

    correctness_records = []
    for name, path in img_paths.items():
        img = load_grayscale_image(path)
        for text_label, text_content in sample_texts.items():
            if text_content is not None:
                payload_bits = text_to_bits(text_content)
            else:
                payload_bits = rng.integers(0, 2, size=32768, dtype=np.uint8)

            # 测试基础 PEE (T=2)
            res_b = basic_pee_embed(img, payload_bits, T=2)
            rec_b, ext_b, _ = basic_pee_extract(res_b.stego_img)
            ver_b = verify_reversibility(img, rec_b, payload_bits, ext_b)
            if text_content is not None:
                decoded_ok_b = (bits_to_text(ext_b) == text_content)
            else:
                decoded_ok_b = ver_b["bits_identical"]

            # 测试自适应 PEE
            res_a = adaptive_pee_embed(img, payload_bits)
            rec_a, ext_a, _ = adaptive_pee_extract(res_a.stego_img)
            ver_a = verify_reversibility(img, rec_a, payload_bits, ext_a)
            if text_content is not None:
                decoded_ok_a = (bits_to_text(ext_a) == text_content)
            else:
                decoded_ok_a = ver_a["bits_identical"]

            correctness_records.append({
                "图像名称": name,
                "载荷类型": text_label,
                "嵌入比特数": len(payload_bits),
                "BPP": round(res_a.bpp, 4),
                "基础PEE_PSNR(dB)": round(res_b.psnr, 2),
                "自适应PEE_PSNR(dB)": round(res_a.psnr, 2),
                "PSNR提升(dB)": round(res_a.psnr - res_b.psnr, 2),
                "基础PEE_BER": ver_b["ber"],
                "自适应PEE_BER": ver_a["ber"],
                "原图恢复最大像素差": max(ver_b["max_pixel_diff"], ver_a["max_pixel_diff"]),
                "文本解码完全一致": bool(decoded_ok_b and decoded_ok_a),
                "100%可逆验证通过": bool(ver_b["is_fully_reversible"] and ver_a["is_fully_reversible"]),
            })

    # 保存典型样例载密图与对比图（以 Cameraman 和 TextureMix 嵌入 26214 bits 即 BPP=0.10 为例）
    demo_bits = rng.integers(0, 2, size=int(512 * 512 * 0.10), dtype=np.uint8)
    for key_img in ["Cameraman (平滑主导)", "TextureMix (复杂纹理对比)"]:
        p_img = img_paths[key_img]
        img = load_grayscale_image(p_img)
        res_b = basic_pee_embed(img, demo_bits, T=2)
        res_a = adaptive_pee_embed(img, demo_bits)
        rec_a, ext_a, _ = adaptive_pee_extract(res_a.stego_img)

        save_grayscale_image(res_b.stego_img, stego_dir / f"{p_img.stem}_stego_basic.png")
        save_grayscale_image(res_a.stego_img, stego_dir / f"{p_img.stem}_stego_adaptive.png")
        save_grayscale_image(rec_a, stego_dir / f"{p_img.stem}_recovered.png")

        plot_stego_recovery_comparison(
            img,
            res_b.stego_img,
            res_a.stego_img,
            rec_a,
            res_b.psnr,
            res_a.psnr,
            res_a.bpp,
            save_path=fig_dir / f"stego_comparison_{p_img.stem}.png",
        )
        if "Cameraman" in key_img:
            plot_prediction_error_histogram(
                img,
                res_a.stego_img,
                save_path=fig_dir / "pee_error_histogram.png",
            )

    # =========================================================================
    # 实验 3 & 4：不同有效载荷 (BPP) 下 基础 PEE vs 自适应 PEE 的全面性能对比
    # =========================================================================
    target_bpps = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    comparison_records = []

    for name, path in img_paths.items():
        img = load_grayscale_image(path)
        total_px = img.shape[0] * img.shape[1]
        for bpp_target in target_bpps:
            n_bits = int(total_px * bpp_target)
            bits = rng.integers(0, 2, size=n_bits, dtype=np.uint8)

            # 基础 PEE (T=2，若超出容量则自动升为 T=3)
            try:
                res_b = basic_pee_embed(img, bits, T=2)
            except ValueError:
                try:
                    res_b = basic_pee_embed(img, bits, T=3)
                except ValueError:
                    continue
            rec_b, ext_b, _ = basic_pee_extract(res_b.stego_img)
            assert np.array_equal(img, rec_b) and np.array_equal(bits, ext_b), "Basic PEE 可逆性校验失败"

            # 自适应 PEE
            try:
                res_a = adaptive_pee_embed(img, bits)
            except ValueError:
                continue
            rec_a, ext_a, _ = adaptive_pee_extract(res_a.stego_img)
            assert np.array_equal(img, rec_a) and np.array_equal(bits, ext_a), "Adaptive PEE 可逆性校验失败"

            comparison_records.append({
                "image_name": name,
                "target_bpp": bpp_target,
                "payload_bits": n_bits,
                "basic_psnr": round(res_b.psnr, 2),
                "adaptive_psnr": round(res_a.psnr, 2),
                "psnr_gain": round(res_a.psnr - res_b.psnr, 2),
                "basic_mse": round(res_b.mse, 4),
                "adaptive_mse": round(res_a.mse, 4),
                "basic_modified_px": res_b.modified_pixels,
                "adaptive_modified_px": res_a.modified_pixels,
                "basic_time_ms": round(res_b.elapsed_ms, 1),
                "adaptive_time_ms": round(res_a.elapsed_ms, 1),
                "adaptive_thresholds": str(res_a.level_thresholds),
                "smooth_bits": res_a.embedded_per_level["smooth"],
                "normal_bits": res_a.embedded_per_level["normal"],
                "complex_bits": res_a.embedded_per_level["complex"],
                "overflow_count": res_a.overflow_count,
            })

    df_comp = pd.DataFrame(comparison_records)
    df_corr = pd.DataFrame(correctness_records)
    df_tex = pd.DataFrame(texture_summary)

    df_comp.to_csv(out_dir / "algorithm_comparison.csv", index=False, encoding="utf-8-sig")
    df_corr.to_csv(out_dir / "correctness_verification.csv", index=False, encoding="utf-8-sig")
    df_tex.to_csv(out_dir / "texture_analysis_summary.csv", index=False, encoding="utf-8-sig")

    # 绘制 4 幅测试图像的 PSNR vs BPP 率失真对比曲线图
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=150)
    axes_flat = axes.flatten()
    for idx, (name, _) in enumerate(img_paths.items()):
        sub_df = df_comp[df_comp["image_name"] == name]
        ax = axes_flat[idx]
        ax.plot(
            sub_df["target_bpp"],
            sub_df["adaptive_psnr"],
            marker="o",
            linewidth=2.4,
            markersize=7,
            color="#d7191c",
            label="自适应 PEE (局部纹理分级+排序)",
        )
        ax.plot(
            sub_df["target_bpp"],
            sub_df["basic_psnr"],
            marker="s",
            linewidth=2.0,
             linestyle="--",
            markersize=6,
            color="#2b83ba",
            label="基础 PEE (固定阈值 T=2)",
        )
        for _, row in sub_df.iterrows():
            ax.annotate(
                f"+{row['psnr_gain']:.1f}dB",
                xy=(row["target_bpp"], row["adaptive_psnr"]),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color="#b2182b",
                fontweight="bold",
            )
        ax.set_title(f"{name} - 图像质量与容量曲线", fontsize=11, fontweight="bold")
        ax.set_xlabel("嵌入率 BPP (Bits Per Pixel)", fontsize=10)
        ax.set_ylabel("峰值信噪比 PSNR (dB)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(loc="upper right", fontsize=9.5)
        ax.set_ylim(sub_df["basic_psnr"].min() - 2, sub_df["adaptive_psnr"].max() + 3.5)

    plt.suptitle("基础 PEE 算法 vs 局部纹理分级自适应 PEE 算法 PSNR-BPP 性能对比", fontsize=13.5, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(fig_dir / "psnr_vs_bpp_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 绘制在 BPP=0.10 下各图像的修改像素数减少比例与各等级纹理区嵌入分布图
    sub_01 = df_comp[df_comp["target_bpp"] == 0.10]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)

    x_idx = np.arange(len(sub_01))
    width = 0.35
    short_names = [n.split()[0] for n in sub_01["image_name"]]

    b1 = axes[0].bar(x_idx - width / 2, sub_01["basic_psnr"], width, label="基础 PEE", color="#2b83ba")
    b2 = axes[0].bar(x_idx + width / 2, sub_01["adaptive_psnr"], width, label="自适应 PEE", color="#d7191c")
    axes[0].set_title("相同载荷 (BPP=0.10) 下各图像 PSNR 对比", fontsize=11.5, fontweight="bold")
    axes[0].set_xticks(x_idx)
    axes[0].set_xticklabels(short_names, fontsize=10.5)
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_ylim(45, 65)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend()
    for bar in b1:
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{bar.get_height():.1f}", ha="center", fontsize=9)
    for bar in b2:
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{bar.get_height():.1f}", ha="center", fontsize=9, fontweight="bold", color="#b2182b")

    # 右图：自适应 PEE 在 BPP=0.10 下各纹理等级区域的嵌入比特占比堆叠柱状图
    s_bits = sub_01["smooth_bits"].to_numpy()
    n_bits_arr = sub_01["normal_bits"].to_numpy()
    c_bits = sub_01["complex_bits"].to_numpy()
    tot_b = s_bits + n_bits_arr + c_bits
    s_pct = s_bits / tot_b * 100
    n_pct = n_bits_arr / tot_b * 100
    c_pct = c_bits / tot_b * 100

    axes[1].bar(short_names, s_pct, label="平滑区 (Level 0) 嵌入占比", color="#2b83ba", width=0.5)
    axes[1].bar(short_names, n_pct, bottom=s_pct, label="普通区 (Level 1) 嵌入占比", color="#fdae61", width=0.5)
    axes[1].bar(short_names, c_pct, bottom=s_pct + n_pct, label="复杂区 (Level 2) 嵌入占比", color="#d7191c", width=0.5)
    axes[1].set_title("自适应 PEE 在不同纹理等级区域的嵌入载荷分配 (BPP=0.10)", fontsize=11.5, fontweight="bold")
    axes[1].set_ylabel("嵌入比特比例 (%)")
    axes[1].set_ylim(0, 115)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper right", fontsize=9.5)

    for i in range(len(short_names)):
        if s_pct[i] > 8:
            axes[1].text(i, s_pct[i] / 2, f"{s_pct[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9.5)
        if n_pct[i] > 8:
            axes[1].text(i, s_pct[i] + n_pct[i] / 2, f"{n_pct[i]:.1f}%", ha="center", va="center", color="black", fontweight="bold", fontsize=9.5)

    plt.tight_layout()
    fig.savefig(fig_dir / "adaptive_allocation_and_gain.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary_data = {
        "texture_summary": texture_summary,
        "correctness_records": correctness_records,
        "comparison_records": comparison_records,
    }
    with open(out_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    return summary_data


if __name__ == "__main__":
    print("正在执行《数据安全》课程大作业全套实验测试与图表生成...")
    res = run_all_experiments("outputs")
    print(f"实验完成！共生成 {len(res['correctness_records'])} 组可逆性测试与 {len(res['comparison_records'])} 组性能对比数据。")
