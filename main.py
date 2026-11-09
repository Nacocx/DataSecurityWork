"""
《数据安全》课程大作业 - 主入口程序 (CLI 命令行与系统启动器)
支持：
1. 运行完整演示流水线（纹理分析 -> 秘密信息嵌入 -> 提取与恢复 -> 逐像素验证）
2. 对指定图像嵌入自定义秘密文本并保存载密图像
3. 从载密图像中提取秘密信息并无损恢复原图
4. 一键运行全套对比实验并生成报告图表
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

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
from src.experiments import run_all_experiments


def run_demo() -> None:
    """运行交互式命令行完整演示，展示验收要求的全部 6 项核心能力"""
    print("=" * 78)
    print("  《数据安全》课程综合实践大作业 — 自适应可逆图像信息隐藏系统演示")
    print("=" * 78)

    img_paths = generate_test_images("data/test_images")
    demo_path = img_paths["Cameraman (平滑主导)"]
    img = load_grayscale_image(demo_path)
    print(f"\n[1] 导入载体图像: {demo_path} (分辨率: {img.shape[1]}×{img.shape[0]}, 灰度无损 PNG)")

    # 任务二：局部纹理分析
    tex_res = analyze_block_texture(img, block_size=8, thresholds=(5.5, 16.0))
    print(f"[2] 8×8 局部纹理复杂度三级划分结果 (共 {tex_res.counts['total']} 块):")
    print(f"    - 平滑区 (Level 0): {tex_res.counts['smooth']:4d} 块 ({tex_res.ratios['smooth']*100:5.1f}%)")
    print(f"    - 普通区 (Level 1): {tex_res.counts['normal']:4d} 块 ({tex_res.ratios['normal']*100:5.1f}%)")
    print(f"    - 复杂区 (Level 2): {tex_res.counts['complex']:4d} 块 ({tex_res.ratios['complex']*100:5.1f}%)")

    # 准备秘密文本
    secret_msg = (
        "【数据安全课程验收测试】基于局部纹理分级与预测误差扩展(PEE)的自适应可逆图像信息隐藏系统。"
        "VerifyCode=2026-RDH-100%-LOSSLESS!"
    )
    payload_bits = text_to_bits(secret_msg * 20)  # 约 2.6 KB 秘密文本
    print(f"\n[3] 待嵌入秘密文本长度: {len(secret_msg * 20)} 字符 ({len(payload_bits)} bits, BPP={len(payload_bits)/img.size:.4f})")

    # 任务一：基础 PEE
    res_b = basic_pee_embed(img, payload_bits, T=2)
    rec_b, ext_b, meta_b = basic_pee_extract(res_b.stego_img)
    ver_b = verify_reversibility(img, rec_b, payload_bits, ext_b)
    print(f"\n[4] 基础 PEE 算法 (固定阈值 T=2):")
    print(f"    - 载密图像 PSNR      : {res_b.psnr:.2f} dB (MSE={res_b.mse:.5f}, 修改像素数={res_b.modified_pixels:,})")
    print(f"    - 提取文本完全一致   : {bits_to_text(ext_b) == secret_msg * 20} (BER={ver_b['ber']:.6f})")
    print(f"    - 原图100%无损恢复   : {ver_b['image_identical']} (最大像素误差={ver_b['max_pixel_diff']}, 差异像素数={ver_b['diff_pixel_count']})")

    # 任务三：自适应 PEE
    res_a = adaptive_pee_embed(img, payload_bits)
    rec_a, ext_a, meta_a = adaptive_pee_extract(res_a.stego_img)
    ver_a = verify_reversibility(img, rec_a, payload_bits, ext_a)
    print(f"\n[5] 自适应 PEE 算法 (局部纹理分级阈值 {res_a.level_thresholds} + 局部方差排序):")
    print(f"    - 载密图像 PSNR      : {res_a.psnr:.2f} dB (相比基础 PEE 提升 +{res_a.psnr - res_b.psnr:.2f} dB!)")
    print(f"    - 各级区域嵌入比特数 : 平滑区={res_a.embedded_per_level['smooth']}, 普通区={res_a.embedded_per_level['normal']}, 复杂区={res_a.embedded_per_level['complex']}")
    print(f"    - 提取文本完全一致   : {bits_to_text(ext_a) == secret_msg * 20} (BER={ver_a['ber']:.6f})")
    print(f"    - 原图100%无损恢复   : {ver_a['image_identical']} (最大像素误差={ver_a['max_pixel_diff']}, 差异像素数={ver_a['diff_pixel_count']})")
    print(f"    - 提取出的秘密信息前80字: {bits_to_text(ext_a)[:80]}...")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="基于局部纹理分级与预测误差扩展的自适应可逆图像信息隐藏系统")
    parser.add_argument("--mode", choices=["demo", "exp", "embed", "extract"], default="demo", help="运行模式")
    parser.add_argument("--image", type=str, help="输入图像路径")
    parser.add_argument("--output", type=str, default="outputs/custom_stego.png", help="输出图像路径")
    parser.add_argument("--secret", type=str, default="Hello Reversible Data Hiding!", help="待隐藏的秘密文本")
    parser.add_argument("--alg", choices=["adaptive", "basic"], default="adaptive", help="算法选择")
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo()
    elif args.mode == "exp":
        run_all_experiments("outputs")
    elif args.mode == "embed":
        if not args.image:
            raise ValueError("请通过 --image 指定输入图像路径")
        img = load_grayscale_image(args.image)
        bits = text_to_bits(args.secret)
        if args.alg == "adaptive":
            res = adaptive_pee_embed(img, bits)
        else:
            res = basic_pee_embed(img, bits, T=2)
        save_grayscale_image(res.stego_img, args.output)
        print(f"嵌入成功！载密图像已保存至: {args.output} | PSNR={res.psnr:.2f} dB | BPP={res.bpp:.4f}")
    elif args.mode == "extract":
        if not args.image:
            raise ValueError("请通过 --image 指定载密图像路径")
        stego = load_grayscale_image(args.image)
        if args.alg == "adaptive":
            rec, bits, _ = adaptive_pee_extract(stego)
        else:
            rec, bits, _ = basic_pee_extract(stego)
        text = bits_to_text(bits)
        save_grayscale_image(rec, args.output)
        print(f"提取出的秘密文本: {text}")
        print(f"无损恢复的原始图像已保存至: {args.output}")


if __name__ == "__main__":
    main()
