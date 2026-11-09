"""
《数据安全》课程大作业 — 答辩 PPT 自动生成脚本
使用 python-pptx 生成 16:9 宽屏、图文并茂、结构清晰的课程答辩演示文稿：
《基于局部纹理分级与预测误差扩展的自适应可逆图像信息隐藏系统设计与实现》
特点：
- 严格贴合课程实践答辩需求，突出原理理解、系统实现、实验验证与无损恢复展示，无繁琐冗余的学术堆砌
- 嵌入项目实际运行生成的高清纹理分级图、载密/恢复对比图、预测误差直方图与 PSNR-BPP 曲线图
"""

from __future__ import annotations
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE


# 现代深蓝与科技青配色方案
C_NAVY = RGBColor(20, 42, 78)       # 主标题深海蓝
C_BLUE = RGBColor(43, 131, 186)     # 辅助科技蓝
C_RED = RGBColor(215, 25, 28)       # 强调绯红
C_ORANGE = RGBColor(245, 130, 32)   # 活力橙
C_GREEN = RGBColor(27, 120, 55)     # 校验通过绿
C_DARK = RGBColor(40, 45, 55)       # 正文深灰
C_MUTED = RGBColor(110, 118, 130)   # 注释灰
C_LIGHT_BG = RGBColor(244, 247, 251)# 卡片浅蓝灰背景
C_WHITE = RGBColor(255, 255, 255)


def set_font(run, size_pt: float, bold: bool = False, color: RGBColor = C_DARK, name: str = "Microsoft YaHei"):
    run.font.name = name
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    run.font.color.rgb = color


def add_header(slide, title_text: str, subtitle_text: str = "", page_num: int | None = None):
    """为内容页添加统一顶部导航栏与标题"""
    # 顶部深蓝装饰条
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.92))
    bar.fill.solid()
    bar.fill.fore_color.rgb = C_NAVY
    bar.line.fill.background()

    # 左侧亮色竖线
    accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.45), Inches(0.20), Inches(0.10), Inches(0.52))
    accent.fill.solid()
    accent.fill.fore_color.rgb = C_ORANGE
    accent.line.fill.background()

    # 标题文字
    tb = slide.shapes.add_textbox(Inches(0.68), Inches(0.12), Inches(10.5), Inches(0.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = title_text
    set_font(r, 23, bold=True, color=C_WHITE)

    if subtitle_text:
        r2 = p.add_run()
        r2.text = f"   |   {subtitle_text}"
        set_font(r2, 13, bold=False, color=RGBColor(200, 215, 235))

    # 底部页脚条
    footer = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(7.18), Inches(13.333), Inches(0.32))
    footer.fill.solid()
    footer.fill.fore_color.rgb = RGBColor(235, 239, 245)
    footer.line.fill.background()

    ftb = slide.shapes.add_textbox(Inches(0.45), Inches(7.19), Inches(12.4), Inches(0.28))
    ftf = ftb.text_frame
    fp = ftf.paragraphs[0]
    fr = fp.add_run()
    page_str = f"第 {page_num} 页 / 共 12 页" if page_num else ""
    fr.text = f"《数据安全》课程综合实践大作业  ·  基于局部纹理分级与预测误差扩展的自适应可逆图像信息隐藏系统        {page_str}"
    set_font(fr, 9.5, color=C_MUTED)


def add_card(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    title: str,
    bullets: list[str],
    title_color: RGBColor = C_NAVY,
    bg_color: RGBColor = C_LIGHT_BG,
    border_color: RGBColor = C_BLUE,
    font_size: float = 13.5,
):
    """添加带标题和项目符号列表的圆角信息卡片"""
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    card.adjustments[0] = 0.03
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = border_color
    card.line.width = Pt(1.2)

    tb = slide.shapes.add_textbox(Inches(left + 0.18), Inches(top + 0.12), Inches(width - 0.36), Inches(height - 0.24))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.space_after = Pt(6)
    r0 = p0.add_run()
    r0.text = title
    set_font(r0, font_size + 2, bold=True, color=title_color)

    for b_text in bullets:
        p = tf.add_paragraph()
        p.space_after = Pt(4)
        p.line_spacing = 1.22
        r = p.add_run()
        r.text = f"•  {b_text}"
        set_font(r, font_size, bold=False, color=C_DARK)


def build_presentation(output_path: str | Path = "答辩PPT_自适应可逆图像信息隐藏系统.pptx") -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    fig_dir = Path("outputs/figures")

    # =========================================================================
    # 第 1 页：封面页
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = C_NAVY
    bg1.line.fill.background()

    # 右侧几何装饰块
    dec1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(5.85), Inches(13.333), Inches(0.12))
    dec1.fill.solid()
    dec1.fill.fore_color.rgb = C_ORANGE
    dec1.line.fill.background()

    tb1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.4), Inches(11.3), Inches(3.8))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.space_after = Pt(14)
    r = p.add_run()
    r.text = "《数据安全》课程综合实践大作业答辩"
    set_font(r, 20, bold=True, color=C_ORANGE)

    p2 = tf1.add_paragraph()
    p2.space_after = Pt(18)
    r2 = p2.add_run()
    r2.text = "基于局部纹理分级与预测误差扩展的\n自适应可逆图像信息隐藏系统设计与实现"
    set_font(r2, 34, bold=True, color=C_WHITE)

    p3 = tf1.add_paragraph()
    r3 = p3.add_run()
    r3.text = "Adaptive Reversible Data Hiding Based on Local Texture Classification and Prediction Error Expansion"
    set_font(r3, 14, bold=False, color=RGBColor(180, 205, 230))

    # 底部信息栏
    info_tb = s1.shapes.add_textbox(Inches(1.0), Inches(6.15), Inches(11.3), Inches(0.9))
    itf = info_tb.text_frame
    ip = itf.paragraphs[0]
    ir = ip.add_run()
    ir.text = "开发工具：Python (uv 环境)  |  核心技术栈：NumPy · OpenCV · Pillow · Matplotlib · Streamlit  |  验收状态：信息提取 100% 准确 · 原图恢复 100% 无损"
    set_font(ir, 13.5, bold=True, color=RGBColor(230, 240, 250))

    # =========================================================================
    # 第 2 页：目录与项目概览
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "一、答辩汇报提纲与任务达成概览", "Project Overview & Outline", page_num=2)

    sections = [
        ("01 项目背景与目标", [
            "传统 LSB 隐写不可逆 vs 可逆信息隐藏 (RDH)",
            "为什么将预测误差扩展 (PEE) 与局部纹理结合",
            "课程实践定位：原理清晰、功能完整、100%无损恢复",
        ]),
        ("02 核心算法原理与设计", [
            "任务一：棋盘格双层菱形预测与基础 PEE 算法",
            "任务二：8×8 分块局部纹理复杂度三级划分与可视化",
            "任务三：分级自适应阈值与复杂度排序嵌入策略",
        ]),
        ("03 系统功能实现与交互", [
            "任务四：模块化代码架构 + CLI 与 Web 双端界面",
            "支持本地图像导入、秘密文本嵌入、提取与恢复",
            "自动执行逐像素一致性比对与性能指标统计",
        ]),
        ("04 实验验证与对比分析", [
            "任务五：4 类典型纹理图像 × 7 组载荷率全面评测",
            "正确性验证：所有测试 BER = 0，恢复图像最大像素差 = 0",
            "自适应增益：相同载荷下 PSNR 最高提升 +11.01 dB",
        ]),
    ]
    for i, (sec_title, items) in enumerate(sections):
        col = i % 2
        row = i // 2
        add_card(
            s2,
            left=0.55 + col * 6.2,
            top=1.25 + row * 2.85,
            width=5.95,
            height=2.6,
            title=sec_title,
            bullets=items,
            font_size=14,
        )

    # =========================================================================
    # 第 3 页：项目背景与意义（LSB vs RDH）
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "二、项目背景与目的：为什么需要可逆信息隐藏？", "Background & Motivation", page_num=3)

    add_card(
        s3, 0.55, 1.25, 5.9, 2.65,
        title="⚠️ 传统信息隐藏 (如 LSB 最低有效位替换) 的局限",
        bullets=[
            "直接用秘密比特覆盖像素最低位，造成永久性、不可逆的像素修改。",
            "接收方虽然能提取出隐藏文本，但原始载体图像再也无法100%还原。",
            "在医学影像、司法证据、遥感测绘等对原始数据完整性要求极高的场景中无法使用。",
        ],
        title_color=C_RED,
        border_color=C_RED,
    )

    add_card(
        s3, 6.85, 1.25, 5.9, 2.65,
        title="✅ 可逆信息隐藏 (Reversible Data Hiding, RDH) 的优势",
        bullets=[
            "利用自然图像相邻像素之间的高度空间相关性构建整数预测器。",
            "通过预测误差扩展 (PEE) 与直方图平移，在不损失原图信息的前提下嵌入秘密比特。",
            "提取端不仅能 100% 准确提取秘密信息，还能逐像素 100% 无损恢复原始图像！",
        ],
        title_color=C_GREEN,
        border_color=C_GREEN,
    )

    add_card(
        s3, 0.55, 4.15, 12.2, 2.75,
        title="🎯 本课程大作业的核心设计思路：从“盲目均匀嵌入”到“局部纹理自适应嵌入”",
        bullets=[
            "基础 PEE 的痛点：对整幅图像采用固定阈值 T 并按行顺序扫描，遇到边缘或高噪声复杂区域时预测误差很大，无法嵌入比特却被迫发生直方图平移 (e' = e ± T)，白白产生大量视觉失真。",
            "自适应改进方案：引入 8×8 局部纹理复杂度分析，将图像划分为【平滑区、普通区、复杂区】三个等级。",
            "核心收益：优先选择预测误差最小的平滑区像素嵌入，并对复杂纹理区跳过处理或减小阈值，从而在相同秘密信息容量下显著提升载密图像的峰值信噪比 (PSNR)！",
        ],
        title_color=C_NAVY,
        border_color=C_ORANGE,
        font_size=13.8,
    )

    # =========================================================================
    # 第 4 页：系统总体技术路线与架构
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "三、系统总体技术路线与模块设计", "Technical Pipeline & Architecture", page_num=4)

    steps = [
        ("1. 数据输入", "导入 8-bit 灰度无损 PNG 图像\n输入 UTF-8 秘密文本转比特流"),
        ("2. 局部纹理分级", "8×8 分块计算局部方差与梯度\n划分为平滑/普通/复杂三级"),
        ("3. 自适应策略排序", "按(区域等级, 四邻域复杂度)升序\n自动优选分级阈值 (Ts, Tn, Tc)"),
        ("4. PEE 编码嵌入", "棋盘格双层菱形预测器计算误差\n扩展嵌入比特 + 溢出位置记录"),
        ("5. 逆变换提取恢复", "读取底部引导区元数据\n提取秘密文本并100%还原像素"),
        ("6. 验证与性能评测", "校验 BER=0 且像素差=0\n统计 PSNR、BPP 与耗时对比"),
    ]
    for idx, (st_title, st_desc) in enumerate(steps):
        left = 0.45 + idx * 2.08
        box = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(1.35), Inches(1.92), Inches(1.85))
        box.fill.solid()
        box.fill.fore_color.rgb = C_NAVY if idx in (1, 2, 3) else C_BLUE
        box.line.fill.background()
        tb = s4.shapes.add_textbox(Inches(left + 0.08), Inches(1.45), Inches(1.76), Inches(1.65))
        tf = tb.text_frame
        tf.word_wrap = True
        p0 = tf.paragraphs[0]
        p0.alignment = PP_ALIGN.CENTER
        r0 = p0.add_run()
        r0.text = st_title
        set_font(r0, 14, bold=True, color=C_WHITE)
        p1 = tf.add_paragraph()
        p1.space_before = Pt(8)
        p1.alignment = PP_ALIGN.CENTER
        r1 = p1.add_run()
        r1.text = st_desc
        set_font(r1, 11.5, bold=False, color=RGBColor(235, 242, 250))

    add_card(
        s4, 0.45, 3.50, 6.0, 3.4,
        title="🔑 关键机制 1：棋盘格双层无漂移同步结构",
        bullets=[
            "将像素按 (i + j) % 2 划分为 Dot 集合 (偶数) 与 Cross 集合 (奇数)。",
            "对 Dot 集合像素预测和嵌入时，其上下左右 4 个邻域像素全属于 Cross 集合，在当前层处理期间 100% 保持不变！",
            "同理，8×8 分块复杂度与单像素局部粗糙度也仅用另一奇偶参考集计算。",
            "彻底解决了“修改像素导致提取端预测值或纹理等级改变”的失步问题！",
        ],
        font_size=13,
    )

    add_card(
        s4, 6.75, 3.50, 6.1, 3.4,
        title="🛡️ 关键机制 2：两阶段自包含无损引导与溢出保护",
        bullets=[
            "像素扩展后若越界 (<0 或 >255)，保持原值不变并将其索引记入溢出表，采用差分 + zlib 高比压缩。",
            "图像底部保留独立于内部预测邻域的引导区，用 LSB 存入压缩元数据。",
            "引导区被替换的原始 LSB 比特在第二阶段通过 PEE 无损嵌入内部数据区。",
            "提取端先读引导头 -> PEE 恢复内部像素与引导区原始 LSB -> 还原引导区，实现全图 H×W 每一个像素 100% 无损恢复！",
        ],
        font_size=13,
    )

    # =========================================================================
    # 第 5 页：任务一：基础预测误差扩展 (Basic PEE) 原理
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "四、任务一实现：基础预测误差扩展 (Basic PEE) 算法", "Task 1: Basic Prediction Error Expansion", page_num=5)

    add_card(
        s5, 0.50, 1.20, 6.1, 5.75,
        title="📐 整数菱形预测与误差扩展/逆变换公式",
        bullets=[
            "1. 菱形四邻域整数预测器：\n    p_hat = (I[i-1, j] + I[i+1, j] + I[i, j-1] + I[i, j+1]) // 4\n    计算原始预测误差：e = x - p_hat",
            "2. 设定扩展阈值 T (基础算法默认 T = 2)：\n    • 扩展嵌入区间 (-T ≤ e < T)：嵌入 1 比特秘密信息 b ∈ {0, 1}\n       e' = 2 * e + b\n    • 正向平移区间 (e ≥ T)：不嵌入比特，向右平移腾出空间\n       e' = e + T\n    • 负向平移区间 (e < -T)：不嵌入比特，向左平移腾出空间\n       e' = e - T\n    • 生成载密像素：x' = p_hat + e'",
            "3. 提取与无损恢复逆变换：\n    • 计算载密误差：e' = x' - p_hat\n    • 若 -2T ≤ e' < 2T：提取比特 b = e' & 1，恢复误差 e = (e' - b) // 2\n    • 若 e' ≥ 2T：恢复误差 e = e' - T；若 e' < -2T：e = e' + T\n    • 无损恢复原始像素：x = p_hat + e",
        ],
        font_size=12.8,
    )

    hist_img = fig_dir / "pee_error_histogram.png"
    if hist_img.exists():
        s5.shapes.add_picture(str(hist_img), Inches(6.80), Inches(1.35), width=Inches(6.05))

    add_card(
        s5, 6.80, 4.65, 6.05, 2.30,
        title="📊 预测误差直方图特征分析（见上图）",
        bullets=[
            "自然图像相邻像素高度相关，预测误差 e 高度集中在 0 附近（呈尖峰拉普拉斯分布）。",
            "经过 PEE 嵌入后，中心峰值区间 [-T, T-1] 被扩展为 [-2T, 2T-1] 以携带秘密比特，两侧误差向外平移，从而保证映射双射可逆。",
        ],
        font_size=12.8,
    )

    # =========================================================================
    # 第 6 页：任务二：图像局部纹理复杂度分析与可视化
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "五、任务二实现：8×8 局部纹理复杂度三级分析", "Task 2: Local Texture Complexity Analysis", page_num=6)

    add_card(
        s6, 0.45, 1.15, 12.4, 1.65,
        title="🔍 局部纹理复杂度计算与三级区域划分规则",
        bullets=[
            "将图像划分为 8×8 像素块，综合块内局部标准差 σ 与相邻像素绝对差均值 G 计算复杂度得分：Score = 0.5·σ + 0.5·G（嵌入时仅用不变参考集像素计算以确保可逆同步）。",
            "根据阈值 (T_low=5.5, T_high=16.0) 划分为三级：【等级0 平滑区 (蓝色)】、【等级1 普通区 (橙黄)】、【等级2 复杂区 (红色)】。",
        ],
        font_size=13,
    )

    tex_cam = fig_dir / "texture_cameraman_512.png"
    tex_mix = fig_dir / "texture_texture_mix_512.png"
    if tex_cam.exists():
        s6.shapes.add_picture(str(tex_cam), Inches(0.45), Inches(2.90), width=Inches(6.10))
    if tex_mix.exists():
        s6.shapes.add_picture(str(tex_mix), Inches(6.75), Inches(2.90), width=Inches(6.10))

    tb_cap = s6.shapes.add_textbox(Inches(0.45), Inches(6.55), Inches(12.4), Inches(0.5))
    p_cap = tb_cap.text_frame.paragraphs[0]
    p_cap.alignment = PP_ALIGN.CENTER
    r_cap = p_cap.add_run()
    r_cap.text = "左图：Cameraman（平滑区占 53.8%，普通区 25.4%，复杂区 20.8%）    |    右图：TextureMix（平滑区占 38.9%，普通区 6.4%，复杂区 54.7%）"
    set_font(r_cap, 12.5, bold=True, color=C_NAVY)

    # =========================================================================
    # 第 7 页：任务三：自适应信息嵌入策略设计
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "六、任务三实现：基于局部纹理分级的自适应嵌入策略", "Task 3: Adaptive Embedding Strategy", page_num=7)

    add_card(
        s7, 0.50, 1.20, 6.0, 5.70,
        title="🚀 自适应 PEE 的三项核心改进策略",
        bullets=[
            "1. 区域优先级与局部复杂度双层排序：\n    • 对所有候选像素按 (分块等级 Level, 四邻域局部复杂度 LC, 行 i, 列 j) 升序排列。\n    • 优先在最平滑的分块、最小局部预测误差方差的像素点嵌入秘密信息！",
            "2. 分级差异化阈值设置 (T_smooth, T_normal, T_complex)：\n    • 平滑区 (Level 0)：预测误差高度集中在 0 附近，设 T_smooth = 1 或 2，实现高容量且极低失真嵌入。\n    • 普通区 (Level 1)：设较小阈值 T_normal = 1，仅在平滑区装满后按需启用。\n    • 复杂区 (Level 2)：设 T_complex = 0（直接跳过不处理），彻底消除复杂纹理区的无效直方图平移失真！",
            "3. 容量自适应阈值优选算法：\n    • 根据待嵌入秘密比特长度自动评估预测误差分布，从候选阈值组合中选择能容纳载荷且失真最小的参数配置。",
        ],
        font_size=13,
    )

    alloc_img = fig_dir / "adaptive_allocation_and_gain.png"
    if alloc_img.exists():
        s7.shapes.add_picture(str(alloc_img), Inches(6.75), Inches(1.40), width=Inches(6.10))

    add_card(
        s7, 6.75, 4.35, 6.10, 2.55,
        title="💡 为什么自适应策略能大幅提升 PSNR？（见上图）",
        bullets=[
            "在 BPP = 0.10 载荷下，4 幅图像的全部秘密比特 100% 集中在【平滑区 (Level 0)】完成嵌入，复杂区完全无需触碰！",
            "同时，由于平滑区预测误差极小，采用 T=1 即可装下全部比特，避免了基础 PEE (T=2) 成倍放大的扩展与平移误差，PSNR 提升最高达 +8.6 ~ +11.0 dB！",
        ],
        title_color=C_RED,
        border_color=C_RED,
        font_size=12.8,
    )

    # =========================================================================
    # 第 8 页：任务四：系统功能实现与可视化交互展示
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "七、任务四实现：系统功能整合与可视化效果展示", "Task 4: System Implementation & Visualization", page_num=8)

    add_card(
        s8, 0.45, 1.15, 4.3, 5.75,
        title="💻 软件系统功能模块 (CLI + Web)",
        bullets=[
            "环境管理：采用现代 Python 包管理器 uv 统一管理虚拟环境与依赖。",
            "1. 图像导入：支持内置测试集与本地任意 PNG/BMP 图像导入。",
            "2. 秘密信息输入：支持任意中英文文本 UTF-8 编码打包。",
            "3. 信息嵌入：一键调用基础 PEE / 自适应 PEE 生成无损 PNG 载密图。",
            "4. 信息提取与图像恢复：盲提取秘密文本并还原原始图像。",
            "5. 自动验证面板：实时输出 PSNR、BPP、BER 及逐像素误差比对结果。",
        ],
        font_size=12.8,
    )

    comp_vis = fig_dir / "stego_comparison_texture_mix_512.png"
    if comp_vis.exists():
        s8.shapes.add_picture(str(comp_vis), Inches(4.95), Inches(1.18), width=Inches(7.90))

    # =========================================================================
    # 第 9 页：任务五（1）：正确性与无损恢复测试结果
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "八、任务五实验（1）：正确性测试与 100% 无损恢复验证", "Task 5: Correctness & Reversibility Verification", page_num=9)

    add_card(
        s9, 0.45, 1.15, 12.4, 1.35,
        title="✅ 核心验收指标测试结论：12 组跨图像、跨载荷测试全部 100% 通过无损校验！",
        bullets=[
            "在 4 幅不同纹理特征图像（含人为加入 0 和 255 极端饱和像素的 MoonCoins 图像）上测试短文本、2KB 长文本及 32768 bits 高容量载荷：提取误码率 BER 恒为 0.000000，恢复图像与原始图像最大像素差恒为 0！",
        ],
        title_color=C_GREEN,
        border_color=C_GREEN,
        font_size=13,
    )

    # 添加实验数据表格
    rows_data = [
        ["测试图像名称", "秘密载荷类型", "嵌入比特数", "BPP", "基础PEE PSNR", "自适应PEE PSNR", "PSNR 提升", "提取误码率 BER", "原图恢复像素差", "无损验证"],
        ["Cameraman (平滑主导)", "短文本 (课程口令)", "1,216", "0.0046", "65.02 dB", "67.63 dB", "+2.61 dB", "0.0", "0 (完全一致)", "✅ 通过"],
        ["Cameraman (平滑主导)", "中长文本 (~2KB)", "29,344", "0.1119", "54.83 dB", "56.12 dB", "+1.29 dB", "0.0", "0 (完全一致)", "✅ 通过"],
        ["Astronaut (均衡自然)", "短文本 (课程口令)", "1,216", "0.0046", "61.80 dB", "68.69 dB", "+6.89 dB", "0.0", "0 (完全一致)", "✅ 通过"],
        ["Astronaut (均衡自然)", "中长文本 (~2KB)", "29,344", "0.1119", "51.22 dB", "55.68 dB", "+4.46 dB", "0.0", "0 (完全一致)", "✅ 通过"],
        ["MoonCoins (含0/255极值)", "高容量载荷 (32Kb)", "32,768", "0.1250", "52.68 dB", "55.69 dB", "+3.02 dB", "0.0", "0 (完全一致)", "✅ 通过"],
        ["TextureMix (复杂纹理对比)", "短文本 (课程口令)", "1,216", "0.0046", "58.06 dB", "68.17 dB", "+10.11 dB", "0.0", "0 (完全一致)", "✅ 通过"],
        ["TextureMix (复杂纹理对比)", "中长文本 (~2KB)", "29,344", "0.1119", "48.94 dB", "57.21 dB", "+8.27 dB", "0.0", "0 (完全一致)", "✅ 通过"],
        ["TextureMix (复杂纹理对比)", "高容量载荷 (32Kb)", "32,768", "0.1250", "48.67 dB", "56.33 dB", "+7.66 dB", "0.0", "0 (完全一致)", "✅ 通过"],
    ]

    table_shape = s9.shapes.add_table(len(rows_data), 10, Inches(0.45), Inches(2.68), Inches(12.4), Inches(4.2))
    table = table_shape.table
    col_widths = [1.85, 1.55, 0.95, 0.75, 1.2, 1.3, 1.05, 1.15, 1.35, 1.25]
    for ci, w_in in enumerate(col_widths):
        table.columns[ci].width = Inches(w_in)

    for ri, row in enumerate(rows_data):
        for ci, val in enumerate(row):
            cell = table.cell(ri, ci)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if ri == 0:
                cell.fill.fore_color.rgb = C_NAVY
            elif ri % 2 == 1:
                cell.fill.fore_color.rgb = RGBColor(240, 245, 252)
            else:
                cell.fill.fore_color.rgb = C_WHITE

            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = val
            if ri == 0:
                set_font(r, 10.5, bold=True, color=C_WHITE)
            elif ci == 6:
                set_font(r, 10.5, bold=True, color=C_RED)
            elif ci == 9:
                set_font(r, 10.5, bold=True, color=C_GREEN)
            else:
                set_font(r, 10.2, bold=False, color=C_DARK)

    # =========================================================================
    # 第 10 页：任务五（2）：基础算法 vs 自适应算法性能对比曲线
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "九、任务五实验（2）：图像质量 (PSNR) 与嵌入容量 (BPP) 对比分析", "Task 5: Rate-Distortion Performance Comparison", page_num=10)

    curve_img = fig_dir / "psnr_vs_bpp_curves.png"
    if curve_img.exists():
        s10.shapes.add_picture(str(curve_img), Inches(0.45), Inches(1.12), width=Inches(7.75))

    add_card(
        s10, 8.35, 1.15, 4.55, 5.80,
        title="📈 实验结果深度分析",
        bullets=[
            "1. 全载荷区间一致更优：\n    在 0.02 ~ 0.30 BPP 的所有测试点上，自适应 PEE 的 PSNR 曲线全面位于基础 PEE 之上。",
            "2. 纹理反差越大，自适应优势越显著：\n    • 在包含大面积高频纹理与平滑区共存的 TextureMix 图像上，自适应算法精准避开砖墙与草地复杂区，PSNR 提升高达 +7.11 ~ +11.01 dB！\n    • 在 Astronaut 自然图像上，PSNR 提升达 +2.8 ~ +8.01 dB。",
            "3. 高载荷下的自动分级降级能力：\n    当 BPP 增大至 0.25~0.30 时，自适应算法自动将阈值从 (1,1,0) 调整为 (2,1,0) 或 (2,1,1)，优先扩容平滑区，兼顾高容量与高画质。",
            "4. 时间开销分析：\n    基础 PEE 平均耗时 25~150 ms；自适应 PEE 因增加分块统计与复杂度排序，耗时约 300~650 ms，完全满足实时交互需求。",
        ],
        font_size=12.2,
    )

    # =========================================================================
    # 第 11 页：遇到的问题、解决方法与算法局限性讨论
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "十、开发过程中遇到的问题、解决方案与算法局限性分析", "Challenges, Solutions & Limitations", page_num=11)

    add_card(
        s11, 0.50, 1.20, 6.0, 5.70,
        title="🛠️ 遇到的核心技术问题与解决方法",
        bullets=[
            "问题 1：纹理复杂度漂移导致提取端失步\n    • 原因：若直接用 8×8 块内全部像素计算方差，嵌入修改一半像素后，提取端重算的方差会发生微小变化，导致分块等级改变、提取乱码。\n    • 解决：严格利用棋盘格正交特性，在嵌入和提取当前奇偶层时，仅使用另一半 100% 不变的参考集像素计算分块特征与四邻域局部复杂度，实现零误差同步！",
            "问题 2：极端灰度值 (0 / 255) 像素扩展溢出\n    • 原因：MoonCoins 等图像含饱和黑/白像素，扩展后超出 [0, 255]。\n    • 解决：设计差分压缩溢出索引表 + 两阶段底部独立引导区机制，将引导区原始 LSB 二次无损嵌入数据区，实现全图 100% 逐像素无损恢复。",
        ],
        title_color=C_NAVY,
        border_color=C_BLUE,
        font_size=13,
    )

    add_card(
        s11, 6.80, 1.20, 6.0, 5.70,
        title="⚖️ 算法存在的局限性与适用边界说明",
        bullets=[
            "1. 依赖无损图像格式 (PNG / BMP / TIFF)：\n    • 可逆信息隐藏依赖精确的像素整数差值。若载密图像经过 JPEG 有损压缩、截图重采样或滤波处理，隐藏信息与无损恢复能力将被破坏。",
            "2. 全图高噪声/纯纹理图像的容量受限：\n    • 若载体图像整体均为剧烈变化的噪声或高频纹理，预测误差普遍较大，可嵌入容量 (BPP) 会明显下降。",
            "3. 排序与分块带来一定计算开销：\n    • 自适应算法需执行分块统计与数十万像素元组排序，运行耗时约为基础 PEE 的 4~6 倍（约 0.3~0.6 秒/图）。",
        ],
        title_color=C_RED,
        border_color=C_ORANGE,
        font_size=13,
    )

    # =========================================================================
    # 第 12 页：课程学习总结与致谢
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "十一、课程学习总结与答辩演示", "Summary & Q&A", page_num=12)

    add_card(
        s12, 0.55, 1.25, 12.2, 3.85,
        title="🎓 《数据安全》课程综合实践收获总结",
        bullets=[
            "1. 深入掌握了可逆信息隐藏 (RDH) 的核心思想：理解了如何利用数字图像的空间冗余与整数预测器，在“数据隐藏”与“原始载体完整性保护”之间取得完美平衡。",
            "2. 完成了从基础算法到自适应改进的完整工程闭环：独立实现了基础 PEE 算法、8×8 局部纹理复杂度三级分析模块、以及基于纹理分级与局部粗糙度排序的自适应 PEE 系统。",
            "3. 强化了严谨的数据安全编程思维：在解决“提取端同步一致性”与“边界溢出无损恢复”问题时，深刻体会到可逆算法中哪怕 1 个比特或 1 个像素的偏差都会导致恢复失败，必须通过严密的数学对偶设计与逐像素自动化测试加以保障。",
        ],
        font_size=14,
    )

    thank_box = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.55), Inches(5.35), Inches(12.2), Inches(1.55))
    thank_box.fill.solid()
    thank_box.fill.fore_color.rgb = C_NAVY
    thank_box.line.fill.background()

    ttb = s12.shapes.add_textbox(Inches(0.8), Inches(5.50), Inches(11.7), Inches(1.25))
    ttf = ttb.text_frame
    tp = ttf.paragraphs[0]
    tp.alignment = PP_ALIGN.CENTER
    tr = tp.add_run()
    tr.text = "感谢老师审阅与指导！欢迎提问\n演示命令：uv run python main.py --mode demo   |   Web 界面：uv run streamlit run app.py"
    set_font(tr, 18, bold=True, color=C_WHITE)

    out_path = Path(output_path)
    prs.save(out_path)
    return out_path


if __name__ == "__main__":
    p = build_presentation()
    print(f"答辩 PPT 已成功生成：{p.resolve()}")
