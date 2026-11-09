# 基于局部纹理分级与预测误差扩展的自适应可逆图像信息隐藏系统

《数据安全》课程综合实践大作业项目。本项目采用 `uv` 进行 Python 环境与依赖管理，完整实现了**基础预测误差扩展算法（Basic PEE）**、**8×8 图像局部纹理复杂度三级分析**、**自适应预测误差扩展算法（Adaptive PEE）**、**Streamlit 交互式可视化界面**、**全套性能对比实验**以及**课程答辩 PPT 自动生成**。

---

## 一、运行环境与安装说明

本项目使用 `uv` 管理虚拟环境（Python >= 3.11）。在项目根目录下执行以下命令即可一键同步并安装全部依赖：

```powershell
# 同步虚拟环境与依赖 (NumPy, OpenCV, Pillow, Matplotlib, Streamlit, python-pptx 等)
uv sync
```

---

## 二、基本操作步骤与运行指令

### 1. 运行命令行快速演示（核心验收功能一键展示）
执行以下命令将自动导入测试图像、输出 8×8 局部纹理分级统计、分别运行基础 PEE 与自适应 PEE 的嵌入与提取，并执行逐像素 100% 无损恢复验证：

```powershell
uv run python main.py --mode demo
```

### 2. 启动可视化 Web 交互系统（任务四可视化展示）
提供图形化界面，支持选择内置图像或上传本地图像、调整纹理分级参数、输入自定义秘密文本、对比基础/自适应算法效果及查看完整实验大屏：

```powershell
uv run streamlit run app.py
```

### 3. 自定义图像嵌入与提取（CLI 工具模式）
- **将秘密文本嵌入指定图像：**
  ```powershell
  uv run python main.py --mode embed --alg adaptive --image data/test_images/cameraman_512.png --secret "这是机密测试信息2026" --output outputs/my_stego.png
  ```
- **从载密图像中提取秘密信息并无损恢复原图：**
  ```powershell
  uv run python main.py --mode extract --alg adaptive --image outputs/my_stego.png --output outputs/my_recovered.png
  ```

### 4. 重新运行全套实验测试与图表生成（任务五）
```powershell
uv run python main.py --mode exp
```

### 5. 重新生成课程答辩 PPT
```powershell
uv run python -m src.generate_ppt
```

---

## 三、输入输出格式与项目交付物清单

| 交付类别 | 文件路径 | 说明 |
| :--- | :--- | :--- |
| **1. 项目源代码** | `src/`、`main.py`、`app.py` | 含基础 PEE、局部纹理分析、自适应 PEE、实验评测与 Web 界面 |
| **2. 运行说明** | `README.md` | 环境配置、依赖安装与操作指令说明 |
| **3. 测试样例** | `data/test_images/*.png`、`outputs/stego_samples/*.png` | 4 幅不同纹理特征的 512×512 无损 PNG 测试图及对应载密图、恢复图 |
| **4. 实验结果** | `outputs/figures/*.png`、`outputs/*.csv` | 纹理热力图与分级图、载密恢复对比图、误差直方图、PSNR-BPP 曲线及统计表 |
| **5. 课程作业报告** | `课程作业报告.md` | 包含背景目的、算法原理、系统实现、实验对比、问题解决与总结 |
| **6. 课程答辩 PPT** | `答辩PPT_自适应可逆图像信息隐藏系统.pptx` | 12 页 16:9 宽屏图文并茂答辩演示文稿 |
