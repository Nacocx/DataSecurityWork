"""
数据安全课程大作业 - 标准测试图像集与示例数据生成模块
生成 4 幅具有典型不同局部纹理特征的 512×512 灰度无损 PNG 测试图像：
1. cameraman_512.png  (平滑与边缘混合型：大片天空平滑背景 + 人物/三脚架边缘，来自 skimage.data.camera)
2. astronaut_512.png  (均衡自然图像型：人像、宇航服与器具细节均衡，来自 skimage.data.astronaut)
3. coins_brick_512.png (中等混合纹理型：光滑背景 + 浮雕硬币与砖墙纹理合成)
4. grass_text_512.png (高复杂度丰富纹理型：草地/织物高频纹理与几何平滑区组合)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
from skimage import data, color, transform
from src.utils import save_grayscale_image


def generate_test_images(output_dir: str | Path = "data/test_images") -> dict[str, Path]:
    """生成并保存 4 幅不同纹理特征的 512×512 标准测试灰度图像"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # 1. Cameraman (512x512)：经典标准测试图，包含大面积平滑天空与清晰建筑/人物边缘
    cam = data.camera()
    if cam.shape != (512, 512):
        cam = (transform.resize(cam, (512, 512), anti_aliasing=True) * 255).astype(np.uint8)
    p1 = out_dir / "cameraman_512.png"
    save_grayscale_image(cam, p1)
    paths["Cameraman (平滑主导)"] = p1

    # 2. Astronaut (512x512)：标准宇航员灰度图像，平滑、普通、复杂区域分布均衡
    astro = data.astronaut()
    astro_gray = (color.rgb2gray(astro) * 255.0).round().astype(np.uint8)
    if astro_gray.shape != (512, 512):
        astro_gray = (transform.resize(astro_gray, (512, 512), anti_aliasing=True) * 255).astype(np.uint8)
    p2 = out_dir / "astronaut_512.png"
    save_grayscale_image(astro_gray, p2)
    paths["Astronaut (均衡自然)"] = p2

    # 3. Moon & Coins 混合纹理图 (512x512)：模拟含极端灰度值(0/255附近)与中等纹理的图像
    moon = data.moon()
    if moon.shape != (512, 512):
        moon = (transform.resize(moon, (512, 512), anti_aliasing=True) * 255).astype(np.uint8)
    coins = data.coins()
    coins_resized = (transform.resize(coins, (256, 256), anti_aliasing=True) * 255).astype(np.uint8)
    hybrid = moon.copy()
    hybrid[128:384, 128:384] = coins_resized
    # 故意加入少量 0 和 255 边界饱和像素以验证溢出处理机制 (Location Map / Overflow Table)
    hybrid[20:36, 20:36] = 255
    hybrid[20:36, 40:56] = 0
    p3 = out_dir / "moon_coins_512.png"
    save_grayscale_image(hybrid, p3)
    paths["MoonCoins (含边界极值)"] = p3

    # 4. Grass & Brick 强纹理对比图 (512x512)：上半部分与右侧含高频复杂草地/砖墙纹理，左下半含渐变平滑块
    grass = data.grass()
    if grass.shape != (512, 512):
        grass = (transform.resize(grass, (512, 512), anti_aliasing=True) * 255).astype(np.uint8)
    brick = data.brick()
    if brick.shape != (512, 512):
        brick = (transform.resize(brick, (512, 512), anti_aliasing=True) * 255).astype(np.uint8)
    tex_mix = grass.copy()
    tex_mix[:256, :256] = brick[:256, :256]
    # 左下角构造平滑渐变区域，形成鲜明的“平滑+普通+高复杂”三区共存图像
    yy, xx = np.meshgrid(np.linspace(60, 200, 256), np.linspace(60, 200, 256), indexing="ij")
    tex_mix[256:, :256] = ((yy + xx) / 2).astype(np.uint8)
    p4 = out_dir / "texture_mix_512.png"
    save_grayscale_image(tex_mix, p4)
    paths["TextureMix (复杂纹理对比)"] = p4

    return paths
