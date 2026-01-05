import cv2
import numpy as np
from tkinter import Tk, Frame, Button, Canvas, Scrollbar, Label, Scale, HORIZONTAL, filedialog
from PIL import Image, ImageTk

# 使用固定的 HSV 範圍提取膚色區域
def detect_skin_hsv(image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 40, 70], dtype="uint8")
        upper_skin = np.array([25, 255, 255], dtype="uint8")
        skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

        return skin_mask

class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DIP 期末專題")

        # 初始化變數
        self.image = None  # 原始圖片
        self.processed_image = None  # 處理後圖片
        self.history = []  # Undo 堆疊
        self.redo_stack = []  # Redo 堆疊
        self.skin_mask = None  # 膚色掩膜

        self.character_filters = {
            "goblin": {"brightness": 21, "contrast": 0.94, "gamma": 0.46, "hue": 32.5, "texture": "gaussian_v"},
            "na_vi": {"brightness": 9, "contrast": 1.05, "gamma": 0.57, "hue": 96.0, "texture": "gaussian"},
            "minion": {"brightness": 15, "contrast": 1.16, "gamma": 0.52, "hue": 16.0, "texture": "smooth_blur"},
            "patrick": {"brightness": 10, "contrast": 1.12, "gamma": 0.91, "hue": -10.0, "texture": "smooth_gaussianBlurh"}
        }
    
        # 建立界面
        self.create_ui()

    def create_ui(self):
        # 主框架
        main_frame = Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        # 工具列 (左側)
        self.tool_frame = Frame(main_frame, width=200, bg="lightgray")
        self.tool_frame.pack(side="left", fill="y")

        # 畫布區域 (中央)
        self.canvas = Canvas(main_frame, bg="white")
        self.canvas.pack(side="left", fill="both", expand=True)

        # 操作按鈕
        self.control_frame = Frame(main_frame, width=200)
        self.control_frame.pack(side="right", fill='y')
        self.original_canvas = Canvas(self.control_frame, width=150, height=150, bg="gray")
        self.original_canvas.pack(pady=10)
        Button(self.control_frame, text="另存新檔", command=self.save_image).pack(fill="x", pady=2)
        Button(self.control_frame, text="重置", command=self.reset_image).pack(fill="x", pady=2)
        Button(self.control_frame, text="復原", command=self.undo).pack(fill="x", pady=2)
        Button(self.control_frame, text="取消復原", command=self.redo).pack(fill="x", pady=2)

        # 顯示主工具列
        self.show_main_tools()
            
    # 工具列
    def show_main_tools(self):
        self.clear_tool_frame()
        Label(self.tool_frame, text="工具列", bg="lightgray", font=("Arial", 16)).pack(pady=10)
        Button(self.tool_frame, text="開啟圖片", command=self.open_image).pack(pady=5)
        Button(self.tool_frame, text="膚色校正", command=self.show_skin_correction_tools).pack(pady=5)
        Button(self.tool_frame, text="角色濾鏡", command=self.show_skin_tone_tools).pack(pady=5)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if file_path:
            self.image = cv2.imread(file_path)
        self.processed_image = self.image.copy()
        self.history = [self.image.copy()]
        self.display_image(self.image)  # 顯示處理畫布中的圖片
        self.display_original_image()  # 顯示右上角的原始影像

    def display_original_image(self):
        if self.image is not None:
            # 縮放圖片至 150x150 的範圍內
            img_height, img_width = self.image.shape[:2]
            scale = min(150 / img_width, 150 / img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)

            # 轉換成 Tkinter 可用的圖片格式
            resized_image = cv2.resize(self.image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            self.original_tk_image = ImageTk.PhotoImage(pil_image)

            # 清空並顯示圖片於原始影像 Canvas
            self.original_canvas.delete("all")
            self.original_canvas.create_image(0, 0, anchor="nw", image=self.original_tk_image)


    def display_image(self, img):
        # 取得處理後圖片的寬高
        img_height, img_width = img.shape[:2]

        # 取得畫布大小
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # 計算等比例縮放比例
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        # 縮放圖片
        resized_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # 轉換成 Tkinter 可用的格式
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        self.tk_image = ImageTk.PhotoImage(pil_image)

        # 清空畫布，將圖片置於中央
        self.canvas.delete("all")
        x_offset = (canvas_width - new_width) // 2  # 水平置中
        y_offset = (canvas_height - new_height) // 2  # 垂直置中
        self.canvas.create_image(x_offset, y_offset, anchor="nw", image=self.tk_image)

        # 更新畫布範圍
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def show_skin_correction_tools(self):
        self.clear_tool_frame()
        Label(self.tool_frame, text="膚色校正", bg="lightgray", font=("Arial", 14)).pack(pady=5)
        self.add_adjustment_sliders()
        Button(self.tool_frame, text="套用", command=self.apply_changes).pack(pady=5)
        Button(self.tool_frame, text="上一頁", command=self.show_main_tools).pack(pady=5)

    # 更新滑動條，綁定即時更新的函數
    def add_adjustment_sliders(self):
        Label(self.tool_frame, text="亮度").pack()
        self.brightness_slider = Scale(self.tool_frame, from_=-50, to=50, resolution=1, orient=HORIZONTAL, command=self.update_skin_correction)
        self.brightness_slider.pack()

        Label(self.tool_frame, text="對比度").pack()
        self.contrast_slider = Scale(self.tool_frame, from_=0.5, to=2.0, resolution=0.01, orient=HORIZONTAL, command=self.update_skin_correction)
        self.contrast_slider.pack()

        Label(self.tool_frame, text="Gamma").pack()
        self.gamma_slider = Scale(self.tool_frame, from_=0.1, to=2.0, resolution=0.01, orient=HORIZONTAL, command=self.update_skin_correction)
        self.gamma_slider.pack()

        Label(self.tool_frame, text="色調調整").pack()
        self.hue_shift_slider = Scale(self.tool_frame, from_=-10, to=150, resolution=0.5, orient=HORIZONTAL, command=self.update_skin_correction)
        self.hue_shift_slider.pack()

    def clear_tool_frame(self):
        for widget in self.tool_frame.winfo_children():
            widget.destroy()

    # 即時更新膚色校正效果，但不存入歷史堆疊
    def update_skin_correction(self, event=None):
        if self.processed_image is not None:
            # 使用暫時變數 self.preview_image
            self.preview_image = self.processed_image.copy()

            # 檢測膚色區域
            self.skin_mask = detect_skin_hsv(self.preview_image)
            skin_region = cv2.bitwise_and(self.preview_image, self.preview_image, mask=self.skin_mask)

            # 取得滑動條參數
            alpha = self.contrast_slider.get()  # 對比度
            beta = self.brightness_slider.get()  # 亮度
            gamma = self.gamma_slider.get()  # Gamma
            hue_shift = self.hue_shift_slider.get()  # 色調調整

            # 進行亮度、對比度、Gamma 與色調調整
            corrected = cv2.convertScaleAbs(skin_region, alpha=alpha, beta=beta)
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
            gamma_corrected = cv2.LUT(corrected, table)

            hsv_image = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_image)
            h = cv2.add(h, hue_shift)
            hsv_corrected = cv2.merge([h, s, v])
            self.preview_image[self.skin_mask > 0] = cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)[self.skin_mask > 0]

            # 即時顯示效果
            self.display_image(self.preview_image)

    # 將即時效果套用到 processed_image 並儲存變更
    def apply_changes(self):
        if hasattr(self, 'preview_image') and self.preview_image is not None:
            self.processed_image = self.preview_image.copy()
            self.history.append(self.processed_image.copy())  # 保存步驟
            self.redo_stack.clear()  # 清空 redo 堆疊
            self.display_image(self.processed_image)
            print("套用變更")

    def show_skin_tone_tools(self):
        self.clear_tool_frame()
        Label(self.tool_frame, text="角色濾鏡", bg="lightgray", font=("Arial", 14)).pack(pady=5)
        Button(self.tool_frame, text="哥布林", command=lambda: self.apply_character_filter("goblin")).pack(pady=2)
        Button(self.tool_frame, text="納美人", command=lambda: self.apply_character_filter("na_vi")).pack(pady=2)
        Button(self.tool_frame, text="Minion", command=lambda: self.apply_character_filter("minion")).pack(pady=2)
        Button(self.tool_frame, text="派大星", command=lambda: self.apply_character_filter("patrick")).pack(pady=2)
        Button(self.tool_frame, text="上一頁", command=self.show_main_tools).pack(pady=5)

    # 套用角色濾鏡，包括亮度、對比度、Gamma 和噪點/紋理效果
    def apply_character_filter(self, character):
        if self.processed_image is not None:
            params = self.character_filters[character]
            self.skin_mask = detect_skin_hsv(self.processed_image)

            # 提取膚色區域
            skin_region = cv2.bitwise_and(self.processed_image, self.processed_image, mask=self.skin_mask)

            # 1. 亮度、對比度、Gamma 調整
            adjusted = cv2.convertScaleAbs(skin_region, alpha=params["contrast"], beta=params["brightness"])
            inv_gamma = 1.0 / params["gamma"]
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
            gamma_corrected = cv2.LUT(adjusted, table)

            # 2. 色調調整
            hsv_image = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_image)
            h = cv2.add(h, params["hue"])
            color_corrected = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

            # 3. 紋理/噪點效果
            if params["texture"] == "gaussian_v":
                # 在 V 通道加高斯噪點
                hsv_image = cv2.cvtColor(color_corrected, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv_image)
                noise = np.random.normal(0, 10, v.shape).astype(np.int16)  # 標準差減小到10
                v = np.clip(v + noise, 0, 255).astype(np.uint8)  # 限制像素範圍
                hsv_noisy = cv2.merge([h, s, v])
                color_corrected = cv2.cvtColor(hsv_noisy, cv2.COLOR_HSV2BGR)
            elif params["texture"] == "gaussian":
                # 增加高斯雜訊
                noise = np.random.normal(0, 15, color_corrected.shape).astype(np.int16)
                color_corrected = np.clip(color_corrected + noise, 0, 255).astype(np.uint8)
            elif params["texture"] == "smooth_blur":
                # 均值濾波
                color_corrected = cv2.blur(color_corrected, ksize=(3, 3))
            elif params["texture"] == "smooth_gaussianBlur":
                color_corrected = cv2.GaussianBlur(color_corrected, (3, 3), 0)

            # 合併回原圖
            result = self.processed_image.copy()
            result[self.skin_mask > 0] = color_corrected[self.skin_mask > 0]
            self.processed_image = result.copy()

            # 保存步驟並顯示
            self.history.append(self.processed_image.copy())
            self.redo_stack.clear()
            self.display_image(self.processed_image)

    # 右側按鈕
    def save_image(self):
        if self.processed_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                     filetypes=[("Image files", "*.jpg *.png *.jpeg")])
            if file_path:
                cv2.imwrite(file_path, self.processed_image)

    def reset_image(self):
        if self.image is not None:
            self.processed_image = self.image.copy()
            self.display_image(self.processed_image)

    def undo(self):
        if len(self.history) > 1:
            self.redo_stack.append(self.history.pop())  # 將當前步驟加入 redo 堆疊
            self.processed_image = self.history[-1].copy()  # 復原至上一個步驟
            self.display_image(self.processed_image)
            print("undo")

    def redo(self):
        if self.redo_stack:
            self.history.append(self.redo_stack.pop())  # 將 redo 堆疊中的步驟恢復
            self.processed_image = self.history[-1].copy()
            self.display_image(self.processed_image)
            print("redo")

if __name__ == "__main__":
    root = Tk()
    app = ImageEditorApp(root)
    root.mainloop()
