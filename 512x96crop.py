import os
from PIL import Image

input_folder = "Test_images/Slight under focus"
output_folder = "Test_images/512x96crop"

crop_box = (220, 45, 220 + 512, 45 + 96)

for filename in os.listdir(input_folder):
    if filename.endswith(".tiff"):

        file_path = os.path.join(input_folder, filename)
        
        img = Image.open(file_path)
        
        # 裁剪图片
        cropped_img = img.crop(crop_box)
        
        # 构建输出文件路径
        output_path = os.path.join(output_folder, filename)
        
        # 保存裁剪后的图片
        cropped_img.save(output_path)
        
        print(f"裁剪后的圖片已保存到: {output_path}")

print("所有圖片裁剪完成。")

