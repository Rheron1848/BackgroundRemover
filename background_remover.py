import os
from pathlib import Path

# 设置模型存储路径（必须在导入rembg之前设置）
SCRIPT_DIR = Path(__file__).parent.absolute()
MODELS_DIR = SCRIPT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
os.environ['U2NET_HOME'] = str(MODELS_DIR)  # 使用U2NET_HOME环境变量

import gradio as gr
from rembg import remove, new_session
from PIL import Image
import io
import numpy as np
import logging
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"模型将存储在: {MODELS_DIR}")
logger.info("首次使用时会自动下载所需模型，请耐心等待...")

def remove_background(input_image, alpha_matting=False, alpha_matting_foreground_threshold=240,
                     alpha_matting_background_threshold=10, alpha_matting_erode_size=10, model_name="u2net"):
    """
    移除图片背景

    参数:
    - input_image: 输入的图像，类型应为 NumPy 数组或 PIL 图像。
    - alpha_matting: 是否启用 Alpha 遮罩。
    - alpha_matting_foreground_threshold: 前景阈值。
    - alpha_matting_background_threshold: 背景阈值。
    - alpha_matting_erode_size: 边缘侵蚀大小。
    - model_name: 使用的模型名称，可选值：
        - u2net (默认): 通用模型
        - u2netp: 快速模型
        - u2net_human_seg: 人物分割模型
        - silueta: 剪影模型
        - isnet-general-use: 通用模型
        - isnet-anime: 动漫模型

    返回:
    - 处理后的图像，类型为 NumPy 数组。
    """
    logger.info("开始处理图片...")
    logger.info(f"使用模型: {model_name}")
    logger.info(f"Alpha matting 启用状态: {alpha_matting}")

    # 输入验证
    if not isinstance(input_image, (np.ndarray, Image.Image)):
        logger.error("输入图像格式错误")
        raise ValueError("输入图像必须是 NumPy 数组或 PIL 图像。")

    # 验证模型名称
    valid_models = ["u2net", "u2netp", "u2net_human_seg", "silueta", "isnet-general-use", "isnet-anime"]
    if model_name not in valid_models:
        logger.error(f"无效的模型名称: {model_name}")
        raise ValueError(f"无效的模型名称。可用模型: {', '.join(valid_models)}")

    # 转换参数类型
    alpha_matting_foreground_threshold = int(alpha_matting_foreground_threshold)
    alpha_matting_background_threshold = int(alpha_matting_background_threshold)
    alpha_matting_erode_size = int(alpha_matting_erode_size)
    
    try:
        logger.info("正在移除背景...")
        # 创建新的会话并指定模型
        session = new_session(model_name)
        # 处理图片
        output = remove(
            input_image,
            session=session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size
        )
        logger.info("背景移除完成！")
    except Exception as e:
        logger.error(f"处理图像时发生错误: {e}")
        raise RuntimeError(f"处理图像时发生错误: {e}")
    
    return output

# 创建Gradio界面
logger.info("正在初始化Gradio界面...")

with gr.Blocks(title="图片背景去除工具", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🖼️ 图片背景去除工具")
    gr.Markdown("上传图片，自动识别主体并去除背景。可以调整参数以获得更好的效果。")
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            input_image = gr.Image(label="原始图片", type="numpy")
            with gr.Row():
                alpha_matting = gr.Checkbox(
                    label="启用Alpha遮罩",
                    value=False,
                    info="启用后可以更好地处理半透明边缘和细节（如头发），但会降低处理速度。对于边缘复杂的图片建议启用。"
                )
            
            with gr.Row():
                model_name = gr.Dropdown(
                    choices=["u2net", "u2netp", "u2net_human_seg", "silueta", "isnet-general-use", "isnet-anime"],
                    value="u2net",
                    label="选择模型",
                    info="u2net: 通用模型\nu2netp: 快速模型\nu2net_human_seg: 人物分割\nsilueta: 剪影\nisnet-general-use: 通用模型\nisnet-anime: 动漫模型"
                )
            
            with gr.Row():
                alpha_matting_foreground_threshold = gr.Slider(
                    minimum=0, maximum=255, value=240, step=1,
                    label="前景阈值",
                    info="控制前景识别的严格程度。值越高，越多的像素会被认为是背景。对于主体较暗或需要保留更多细节时，可以适当降低此值。"
                )
                alpha_matting_background_threshold = gr.Slider(
                    minimum=0, maximum=255, value=10, step=1,
                    label="背景阈值",
                    info="控制背景识别的严格程度。值越低，越多的像素会被认为是前景。当背景较复杂或需要更干净的背景去除效果时，可以适当提高此值。"
                )
                alpha_matting_erode_size = gr.Slider(
                    minimum=0, maximum=50, value=10, step=1,
                    label="边缘侵蚀大小",
                    info="控制边缘过渡区域的大小。值越大，边缘过渡越平滑，但可能会损失一些细节。对于需要更自然的边缘过渡效果时，可以适当增加此值。"
                )
            
            process_btn = gr.Button("开始处理", variant="primary")
            
        with gr.Column(scale=1):
            output_image = gr.Image(
                label="处理结果 ",
                type="numpy",
                interactive=False,
                elem_id="output_image",
                show_download_button=True,
                show_label=True,
                image_mode='RGBA'
            )
            with gr.Row():
                clear_btn = gr.Button("清除", variant="secondary")
                download_btn = gr.Button("下载结果", variant="secondary")
            with gr.Row():
                download_file = gr.File(label="下载")

    # 处理函数
    output = process_btn.click(
        fn=remove_background,
        inputs=[
            input_image,
            alpha_matting,
            alpha_matting_foreground_threshold,
            alpha_matting_background_threshold,
            alpha_matting_erode_size,
            model_name
        ],
        outputs=output_image
    )

    # 清除功能
    clear_btn.click(
        lambda: None,
        inputs=None,
        outputs=output_image,
        show_progress="hidden",
    )

    # 下载功能
    def save_image(img):
        if img is None:
            return None
        
        try:
            # 确保图像是 numpy 数组
            if isinstance(img, np.ndarray):
                # 检查是否是 RGB 格式
                if len(img.shape) == 3 and img.shape[2] == 3:
                    # 创建透明通道
                    alpha = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)
                    # 将原始图像中的黑色背景设置为透明
                    mask = np.all(img == [0, 0, 0], axis=2)
                    alpha[mask] = 0
                    # 合并 RGB 和 alpha 通道
                    img_rgba = np.dstack((img, alpha))
                    output_img = Image.fromarray(img_rgba)
                # 如果已经是 RGBA 格式
                elif len(img.shape) == 3 and img.shape[2] == 4:
                    output_img = Image.fromarray(img)
                else:
                    logger.error(f"不支持的图像格式: shape={img.shape}")
                    return None
            elif isinstance(img, Image.Image):
                output_img = img
            else:
                logger.error(f"不支持的图像类型: {type(img)}")
                return None

            # 确保是 RGBA 模式
            if output_img.mode != 'RGBA':
                output_img = output_img.convert('RGBA')

            # 保存为 PNG 格式
            temp_file_path = "output_image.png"
            output_img.save(temp_file_path, format='PNG')
            logger.info("图像已成功保存为PNG格式，包含透明通道")
            return temp_file_path

        except Exception as e:
            logger.error(f"保存图像时发生错误: {e}")
            return None

    download_btn.click(
        fn=save_image,
        inputs=[output_image],
        outputs=[download_file],
        show_progress="hidden",
    )

if __name__ == "__main__":
    logger.info("启动Web服务...")
    demo.launch(share=False, server_name="0.0.0.0") 