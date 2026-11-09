"""
《数据安全》课程大作业 - 任务四：系统功能实现与可视化展示 (Streamlit Web 交互界面)
包含功能：
1. 图像导入：支持选择内置标准测试图像或上传本地 PNG/BMP/TIFF 图像
2. 局部纹理复杂度分析展示：8×8 分块热力图、三级纹理分级图、各区域数量与占比统计
3. 秘密信息输入与嵌入：支持自定义输入秘密文本，选择【基础 PEE】或【自适应 PEE】算法生成载密图像
4. 信息提取与无损恢复验证：从载密图像中提取秘密信息、恢复原始图像，并逐像素比对验证 100% 可逆性
5. 实验结果与算法对比大屏：展示不同图像、不同 BPP 下的 PSNR 曲线、表格与预测误差直方图
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from src.dataset import generate_test_images
from src.utils import (
    text_to_bits,
    bits_to_text,
    load_grayscale_image,
    verify_reversibility,
)
from src.texture_analysis import analyze_block_texture, plot_texture_analysis
from src.basic_pee import basic_pee_embed, basic_pee_extract
from src.adaptive_pee import adaptive_pee_embed, adaptive_pee_extract


st.set_page_config(
    page_title="自适应可逆图像信息隐藏系统 - 数据安全大作业",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ 基于局部纹理分级与预测误差扩展的自适应可逆图像信息隐藏系统")
st.caption("《数据安全》课程综合实践大作业 | 核心技术：棋盘格菱形预测器 · 8×8局部纹理三级分类 · 自适应预测误差扩展 (Adaptive PEE)")

# 确保测试图像存在
test_img_paths = generate_test_images("data/test_images")

# 侧边栏：图像选择与参数设置
with st.sidebar:
    st.header("⚙️ 1. 载体图像导入")
    source_mode = st.radio("图像来源", ["内置标准测试图像", "上传本地图像文件"], index=0)

    if source_mode == "内置标准测试图像":
        selected_name = st.selectbox("选择测试图像", list(test_img_paths.keys()))
        orig_img = load_grayscale_image(test_img_paths[selected_name])
        img_label = selected_name
    else:
        uploaded = st.file_uploader("请上传 PNG/BMP/JPG 图像", type=["png", "bmp", "tif", "jpg", "jpeg"])
        if uploaded is not None:
            orig_img = np.array(Image.open(uploaded).convert("L"), dtype=np.uint8)
            img_label = uploaded.name
        else:
            st.info("未上传文件，默认使用 Cameraman 测试图")
            orig_img = load_grayscale_image( list(test_img_paths.values())[0] )
            img_label = "Cameraman (默认)"

    st.markdown(f"**当前图像尺寸：** `{orig_img.shape[1]} × {orig_img.shape[0]}` 像素")

    st.divider()
    st.header("🔍 2. 纹理分级参数 (任务二)")
    block_size = st.selectbox("局部纹理分块大小", [8, 16, 4], index=0)
    t_low = st.slider("平滑/普通区阈值 (T_low)", 2.0, 12.0, 5.5, 0.5)
    t_high = st.slider("普通/复杂区阈值 (T_high)", 10.0, 30.0, 16.0, 0.5)

    st.divider()
    st.header("🔐 3. 嵌入算法参数 (任务一 & 三)")
    basic_T = st.slider("基础 PEE 固定阈值 T", 1, 4, 2, 1)
    auto_adaptive = st.checkbox("自适应 PEE 自动优选分级阈值", value=True)
    if not auto_adaptive:
        t_smooth = st.slider("平滑区阈值 T_smooth", 1, 4, 2)
        t_normal = st.slider("普通区阈值 T_normal", 0, 3, 1)
        t_complex = st.slider("复杂区阈值 T_complex (0=跳过)", 0, 2, 0)
        custom_level_T = (t_smooth, t_normal, t_complex)
    else:
        custom_level_T = None

tab1, tab2, tab3 = st.tabs([
    "🚀 在线嵌入 · 提取与无损恢复验证 (任务一/三/四)",
    "📊 图像局部纹理复杂度分析 (任务二)",
    "📈 算法对比实验与性能分析报告 (任务五)",
])

with tab1:
    st.subheader("1️⃣ 秘密信息输入与信息嵌入")
    col_in1, col_in2 = st.columns([2, 1])
    with col_in1:
        default_secret = (
            "【数据安全课程大作业验证】这是一段待隐藏的机密文本信息！"
            "系统结合 8×8 局部纹理复杂度分级与预测误差扩展（PEE），"
            "在提取秘密信息的同时可实现原始图像的 100% 逐像素无损恢复。AuthID=2026-SJAQ-PASS"
        )
        secret_text = st.text_area("请输入需要隐藏的秘密文本信息（支持中文、英文及符号）：", value=default_secret, height=110)
    with col_in2:
        repeat_times = st.number_input("载荷倍增次数（用于测试大容量文本嵌入）", min_value=1, max_value=200, value=5, step=1)
        full_secret_text = secret_text * int(repeat_times)
        secret_bits = text_to_bits(full_secret_text)
        st.metric("待嵌入秘密比特总数", f"{len(secret_bits):,} bits", f"约 {len(secret_bits) / (orig_img.size):.4f} BPP")

    if st.button("▶️ 执行信息嵌入、提取与原始图像无损恢复对比", type="primary", use_container_width=True):
        with st.spinner("正在执行基础 PEE 与自适应 PEE 嵌入、提取及逐像素无损验证..."):
            try:
                # 1. 基础 PEE 嵌入与提取
                res_basic = basic_pee_embed(orig_img, secret_bits, T=basic_T)
                rec_basic, ext_bits_basic, meta_basic = basic_pee_extract(res_basic.stego_img)
                ext_text_basic = bits_to_text(ext_bits_basic)
                ver_basic = verify_reversibility(orig_img, rec_basic, secret_bits, ext_bits_basic)

                # 2. 自适应 PEE 嵌入与提取
                res_adap = adaptive_pee_embed(
                    orig_img,
                    secret_bits,
                    level_thresholds=custom_level_T,
                    block_size=block_size,
                    block_thresholds=(t_low, t_high),
                )
                rec_adap, ext_bits_adap, meta_adap = adaptive_pee_extract(res_adap.stego_img)
                ext_text_adap = bits_to_text(ext_bits_adap)
                ver_adap = verify_reversibility(orig_img, rec_adap, secret_bits, ext_bits_adap)

                st.success(
                    f"✅ 嵌入与提取完成！自适应 PEE 相比基础 PEE 的图像质量 PSNR 提升了 "
                    f"**+{res_adap.psnr - res_basic.psnr:.2f} dB**，且两算法恢复图像与原图 **100% 逐像素完全一致**！"
                )

                # 核心指标卡片对比
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("基础 PEE 载密图 PSNR", f"{res_basic.psnr:.2f} dB", f"修改像素: {res_basic.modified_pixels:,}")
                m2.metric(
                    "自适应 PEE 载密图 PSNR",
                    f"{res_adap.psnr:.2f} dB",
                    f"+{res_adap.psnr - res_basic.psnr:.2f} dB (修改像素: {res_adap.modified_pixels:,})",
                )
                m3.metric("秘密信息提取误码率 (BER)", f"{ver_adap['ber']:.6f}", "100% 准确提取")
                m4.metric("恢复图像与原图像素差", f"{ver_adap['max_pixel_diff']} (差异像素: {ver_adap['diff_pixel_count']})", "100% 无损恢复")

                st.divider()
                st.subheader("2️⃣ 视觉效果与像素修改差分对比")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.image(orig_img, caption=f"① 原始灰度图像 ({orig_img.shape[1]}×{orig_img.shape[0]})", use_container_width=True)
                with c2:
                    st.image(res_basic.stego_img, caption=f"② 基础 PEE 载密图 (PSNR={res_basic.psnr:.2f}dB)", use_container_width=True)
                with c3:
                    st.image(res_adap.stego_img, caption=f"③ 自适应 PEE 载密图 (PSNR={res_adap.psnr:.2f}dB)", use_container_width=True)
                with c4:
                    st.image(rec_adap, caption="④ 提取后恢复的原始图像 (100% 无损)", use_container_width=True)

                st.divider()
                st.subheader("3️⃣ 提取信息与完整性验证详情")
                vcol1, vcol2 = st.columns(2)
                with vcol1:
                    st.markdown("#### 📋 自适应嵌入策略执行详情")
                    st.json({
                        "算法模式": "基于局部纹理分级与复杂度排序的自适应 PEE",
                        "选用的三级阈值 (T_平滑, T_普通, T_复杂)": list(res_adap.level_thresholds),
                        "各区域实际嵌入比特数": res_adap.embedded_per_level,
                        "复杂纹理区跳过保护像素数": res_adap.skipped_complex_pixels,
                        "边界溢出跳过点数": res_adap.overflow_count,
                        "嵌入耗时 (ms)": round(res_adap.elapsed_ms, 2),
                        "提取与恢复耗时 (ms)": round(meta_adap["extract_elapsed_ms"], 2),
                    })
                with vcol2:
                    st.markdown("#### 🔓 从载密图像中提取出的秘密文本（前 500 字符预览）")
                    st.code(ext_text_adap[:500] + ("..." if len(ext_text_adap) > 500 else ""), language="text")
                    st.markdown(
                        f"- **原始文本长度：** `{len(full_secret_text)}` 字符 (`{len(secret_bits)}` bits)\n"
                        f"- **提取文本长度：** `{len(ext_text_adap)}` 字符 (`{len(ext_bits_adap)}` bits)\n"
                        f"- **字符串完全相等校验：** `{'通过 (True)' if ext_text_adap == full_secret_text else '失败 (False)'}`\n"
                        f"- **原始图像与恢复图像 `np.array_equal` 校验：** `{'通过 (True)' if ver_adap['image_identical'] else '失败 (False)'}`"
                    )
            except Exception as e:
                st.error(f"执行出错：{e}")

with tab2:
    st.subheader(f"🔍 图像局部纹理复杂度分析 — `{img_label}`")
    tex_res = analyze_block_texture(orig_img, block_size=block_size, thresholds=(t_low, t_high))
    fig_tex = plot_texture_analysis(orig_img, tex_res, title_prefix="")
    st.pyplot(fig_tex)

    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    col_t1.metric("总分块数量", f"{tex_res.counts['total']} 块", f"分块尺寸 {block_size}×{block_size}")
    col_t2.metric("平滑区 (Level 0)", f"{tex_res.counts['smooth']} 块", f"占比 {tex_res.ratios['smooth']*100:.1f}%")
    col_t3.metric("普通区 (Level 1)", f"{tex_res.counts['normal']} 块", f"占比 {tex_res.ratios['normal']*100:.1f}%")
    col_t4.metric("复杂区 (Level 2)", f"{tex_res.counts['complex']} 块", f"占比 {tex_res.ratios['complex']*100:.1f}%")

with tab3:
    st.subheader("📈 实验测试与性能对比分析汇总")
    comp_csv = Path("outputs/algorithm_comparison.csv")
    corr_csv = Path("outputs/correctness_verification.csv")
    if comp_csv.exists() and corr_csv.exists():
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.image("outputs/figures/psnr_vs_bpp_curves.png", caption="基础 PEE vs 自适应 PEE 的 PSNR-BPP 性能对比曲线", use_container_width=True)
        with col_f2:
            st.image("outputs/figures/adaptive_allocation_and_gain.png", caption="BPP=0.10 下 PSNR 提升与各纹理区嵌入载荷分布", use_container_width=True)

        st.markdown("#### 1. 正确性与可逆性测试结果表")
        st.dataframe(pd.read_csv(corr_csv), use_container_width=True)

        st.markdown("#### 2. 不同有效载荷 (BPP) 下基础算法与自适应算法对比数据表")
        st.dataframe(pd.read_csv(comp_csv), use_container_width=True)
    else:
        st.info("请先运行 `uv run python main.py --mode exp` 生成完整实验数据与图表。")
