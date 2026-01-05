import tkinter as tk
from tkinter import filedialog, Scale, HORIZONTAL
from PIL import Image, ImageTk
import cv2
import numpy as np

class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DIP 期末專題 - 影像處理工具")
        self.root.geometry("1000x700")

        # --- 資料狀態管理 ---
        self.original_img = None  
        self.current_img = None
        
        # 紀錄步驟用的堆疊
        self.history = []     
        self.redo_stack = []  
        
        # 角色濾鏡參數表
        self.character_filters = {
            "goblin": {"brightness": 21, "contrast": 0.94, "gamma": 0.46, "hue": 32.5, "texture": "gaussian_v"},
            "na_vi": {"brightness": 9, "contrast": 1.05, "gamma": 0.57, "hue": 96.0, "texture": "gaussian"},
            "minion": {"brightness": 15, "contrast": 1.16, "gamma": 0.52, "hue": 16.0, "texture": "smooth_blur"},
            "patrick": {"brightness": 10, "contrast": 1.12, "gamma": 0.91, "hue": -10.0, "texture": "smooth_gaussianBlur"}
        }

        # 介面顯示用的變數
        self.tk_main_img = None
        self.tk_side_img = None
        self.image_loaded = False

        # 綁定視窗縮放事件
        self.root.bind("<Configure>", self.onResize)
        
        # 建置畫面
        self.buildGUI()

    # =================================== GUI 介面佈局 ===================================
    def buildGUI(self):
        # 左側工具列容器
        self.left_toolbar = tk.Frame(self.root, width=150, bg="#d3d3d3", padx=5, pady=5)
        self.left_toolbar.pack(side="left", fill="y")
        
        # 初始顯示主工具選單
        self.showMainTools() 

        # 右側功能欄
        self.right_sidebar = tk.Frame(self.root, width=200, bg="#f0f0f0", padx=10, pady=10)
        self.right_sidebar.pack(side="right", fill="y")
        
        # 右上角原始影像預覽框 (Canvas)
        self.side_canvas = tk.Canvas(self.right_sidebar, width=180, height=150, bg="#808080", highlightthickness=0)
        self.side_canvas.pack(pady=(10, 20))
        
        # 右側按鈕
        tk.Button(self.right_sidebar, text="另存新檔", width=20, command=self.saveImg).pack(pady=2)
        tk.Button(self.right_sidebar, text="重置", width=20, command=self.resetImg).pack(pady=2)
        tk.Button(self.right_sidebar, text="復原", width=20, command=self.undo).pack(pady=2)
        tk.Button(self.right_sidebar, text="取消復原", width=20, command=self.redo).pack(pady=2)

        # 中央畫布
        self.canvas_area = tk.Frame(self.root, bg="white", highlightbackground="orange", highlightthickness=1)
        self.canvas_area.pack(side="left", fill="both", expand=True)

        self.main_display = tk.Label(self.canvas_area, text="畫布區", font=("System", 30), fg="orange", bg="white")
        self.main_display.pack(fill="both", expand=True)

    def clearToolFrame(self):
        for widget in self.left_toolbar.winfo_children():
            widget.destroy()

    def showMainTools(self):
        self.clearToolFrame()
        tk.Label(self.left_toolbar, text="工具列", bg="#d3d3d3", font=("Arial", 12, "bold")).pack(pady=10)
        
        tk.Button(self.left_toolbar, text="開啟圖片", width=12, command=self.openImg).pack(pady=5)
        tk.Button(self.left_toolbar, text="膚色校正", width=12, command=self.showSkinCorrectionTools).pack(pady=5)
        tk.Button(self.left_toolbar, text="角色濾鏡", width=12, command=self.showCharacterFilterTools).pack(pady=5)

    # =================================== 影像顯示與縮放邏輯 ===================================
    def onResize(self, event):
        if self.image_loaded and event.widget == self.root:
            self.displayAll()

    def displayAll(self):
        self.displayMiddleImg()
        self.displayOriginalImg()

    def getResized(self, img, max_w, max_h):
        h, w = img.shape[:2]
        # 計算縮放比例，取較小值以確保圖片完整顯示
        ratio = min(max_w / w, max_h / h)
        new_w, new_h = max(int(w * ratio), 1), max(int(h * ratio), 1)
        
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def displayMiddleImg(self, img_to_show=None):
        img = img_to_show if img_to_show is not None else self.current_img
        
        if img is None: 
            return
        
        self.root.update_idletasks()
        canvas_w = max(self.canvas_area.winfo_width(), 100)
        canvas_h = max(self.canvas_area.winfo_height(), 100)

        # 邊距
        resized_img = self.getResized(img, canvas_w - 20, canvas_h - 20)
        
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        self.tk_main_img = ImageTk.PhotoImage(Image.fromarray(rgb_img))
        self.main_display.config(image=self.tk_main_img, text="")

    def displayOriginalImg(self):
        if self.original_img is None: 
            return
            
        cw, ch = 180, 150
        resized_img = self.getResized(self.original_img, cw, ch)
        
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        self.tk_side_img = ImageTk.PhotoImage(Image.fromarray(rgb_img))
        
        # 清除舊圖並置中顯示
        self.side_canvas.delete("all") 
        self.side_canvas.create_image(cw//2, ch//2, image=self.tk_side_img, anchor="center")

    def openImg(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            img = cv2.imread(file_path)
            
            if img is not None:
                self.original_img = img.copy()
                self.current_img = img.copy()
                self.history = [img.copy()]
                self.redo_stack = []
                self.image_loaded = True 
                self.displayAll() 

    # =================================== 膚色偵測與校正功能 ===================================
    def detectSkinHSV(self, img):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 膚色範圍
        lower_skin = np.array([0, 40, 70], dtype="uint8")
        upper_skin = np.array([25, 255, 255], dtype="uint8")
        
        return cv2.inRange(hsv_img, lower_skin, upper_skin)

    def showSkinCorrectionTools(self):
        if not self.image_loaded: 
            return
            
        self.clearToolFrame()

        # 備份
        self.pre_correction_img = self.current_img.copy()

        tk.Label(self.left_toolbar, text="膚色校正", bg="#d3d3d3", font=("Arial", 12, "bold")).pack(pady=5)
        
        # 滑桿
        tk.Label(self.left_toolbar, text="亮度", bg="#d3d3d3").pack()
        self.br_scale = Scale(self.left_toolbar, from_=-50, to=50, orient=HORIZONTAL, command=self.updateSkinPreview)
        self.br_scale.set(0); self.br_scale.pack()

        tk.Label(self.left_toolbar, text="對比度", bg="#d3d3d3").pack()
        self.ct_scale = Scale(self.left_toolbar, from_=0.5, to=2.0, resolution=0.1, orient=HORIZONTAL, command=self.updateSkinPreview)
        self.ct_scale.set(1.0); self.ct_scale.pack()

        tk.Label(self.left_toolbar, text="Gamma", bg="#d3d3d3").pack()
        self.gm_scale = Scale(self.left_toolbar, from_=0.1, to=2.0, resolution=0.1, orient=HORIZONTAL, command=self.updateSkinPreview)
        self.gm_scale.set(1.0); self.gm_scale.pack()

        tk.Label(self.left_toolbar, text="色調調整", bg="#d3d3d3").pack()
        self.hue_scale = Scale(self.left_toolbar, from_=-10, to=150, resolution=1, orient=HORIZONTAL, command=self.updateSkinPreview)
        self.hue_scale.set(0); self.hue_scale.pack()

        # 功能按鈕
        tk.Button(self.left_toolbar, text="套用變更", command=self.applyProcessedImg).pack(pady=10)
        tk.Button(self.left_toolbar, text="取消/返回", command=self.cancelCorrection).pack()

    def updateSkinPreview(self, event=None):
        if self.current_img is None: 
            return
        
        img = self.pre_correction_img.copy()

        mask = self.detectSkinHSV(img)
        skin_region = cv2.bitwise_and(img, img, mask=mask)

        # 抓滑桿參數
        alpha = self.ct_scale.get() # 對比度
        beta = self.br_scale.get() # 亮度
        gamma = self.gm_scale.get() # Gamma
        hue_shift = self.hue_scale.get()   # 色調調整

        # 1.亮度與對比
        corrected = cv2.convertScaleAbs(skin_region, alpha=alpha, beta=beta)
        
        # 2.Gamma 校正
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
        gamma_corrected = cv2.LUT(corrected, table)

        # 3.色調調整
        hsv_res = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_res)
        h = cv2.add(h, hue_shift)
        hsv_res = cv2.merge([h, s, v])
        
        # 合併回原圖
        img[mask > 0] = cv2.cvtColor(hsv_res, cv2.COLOR_HSV2BGR)[mask > 0]
        
        # 暫存結果給套用按鈕用
        self.temp_preview = img 
        self.displayMiddleImg(img) 

    def applyProcessedImg(self):
        # 寫入歷史紀錄
        if hasattr(self, 'temp_preview'):
            self.current_img = self.temp_preview.copy()
            self.history.append(self.current_img.copy()) 
            self.redo_stack.clear()
            self.displayAll()
            self.showMainTools()

    def cancelCorrection(self):
        # 不要了
        if hasattr(self, 'pre_correction_img'):
            self.current_img = self.pre_correction_img.copy()
            self.displayAll()
        self.showMainTools()

    # =================================== 角色濾鏡功能 ===================================
    def showCharacterFilterTools(self):
        if not self.image_loaded: 
            return
            
        self.clearToolFrame()

        self.pre_filter_img = self.current_img.copy() 
        self.temp_preview = self.current_img.copy() 

        tk.Label(self.left_toolbar, text="角色濾鏡", bg="#d3d3d3", font=("Arial", 12, "bold")).pack(pady=10)
        
        for name in self.character_filters.keys():
            tk.Button(self.left_toolbar, text=name.capitalize(), width=12, 
                      command=lambda n=name: self.previewCharacterFilter(n)).pack(pady=2)
            
        tk.Frame(self.left_toolbar, height=20, bg="#d3d3d3").pack() 

        tk.Button(self.left_toolbar, text="套用變更", width=12, bg="#d3d3d3", command=self.applyFilterChanges).pack(pady=5)
        tk.Button(self.left_toolbar, text="取消/返回", width=12, bg="#d3d3d3", command=self.cancelFilter).pack(pady=5)

    def previewCharacterFilter(self, character):
        if self.pre_filter_img is None: 
            return
        
        params = self.character_filters[character]
        img = self.pre_filter_img.copy() 
        mask = self.detectSkinHSV(img)
        
        skin_region = cv2.bitwise_and(img, img, mask=mask)
        
        # 進行色彩與紋理調整
        adjusted = cv2.convertScaleAbs(skin_region, alpha=params["contrast"],\
                                        beta=params["brightness"])
        
        inv_gamma = 1.0 / params["gamma"]
        table = np.array([((i / 255.0) ** inv_gamma) \
                          * 255 for i in np.arange(256)]).astype("uint8")
        gamma_res = cv2.LUT(adjusted, table)

        hsv_res = cv2.cvtColor(gamma_res, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_res)
        h = cv2.add(h, params["hue"])
        color_corrected = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

        # 處理特殊紋理效果
        if params["texture"] == "gaussian_v":
            hsv_noise = cv2.cvtColor(color_corrected, cv2.COLOR_BGR2HSV)
            h_n, s_n, v_n = cv2.split(hsv_noise)
            noise = np.random.normal(0, 10, v_n.shape).astype(np.int16)
            v_n = np.clip(v_n + noise, 0, 255).astype(np.uint8)
            color_corrected = cv2.cvtColor(cv2.merge([h_n, s_n, v_n]), cv2.COLOR_HSV2BGR)
            
        elif params["texture"] == "gaussian":
            noise = np.random.normal(0, 15, color_corrected.shape).astype(np.int16)
            color_corrected = np.clip(color_corrected + noise, 0, 255).astype(np.uint8)
            
        elif params["texture"] == "smooth_blur":
            color_corrected = cv2.blur(color_corrected, (3, 3))
            
        elif params["texture"] == "smooth_gaussianBlur":
            color_corrected = cv2.GaussianBlur(color_corrected, (3, 3), 0)

        # 合併顯示
        img[mask > 0] = color_corrected[mask > 0]
        self.temp_preview = img.copy() 
        self.displayMiddleImg(img)

    def applyFilterChanges(self):
        if hasattr(self, 'temp_preview'):
            self.current_img = self.temp_preview.copy()
            self.history.append(self.current_img.copy())
            self.redo_stack.clear()
            self.displayAll()
            self.showMainTools()

    def cancelFilter(self):
        if hasattr(self, 'pre_filter_img'):
            self.current_img = self.pre_filter_img.copy()
            self.displayAll()
        self.showMainTools()
    
    # =================================== 輔助功能 (Undo/Redo/Save) ===================================    
    def undo(self):
        if len(self.history) > 1:
            self.redo_stack.append(self.history.pop())
            self.current_img = self.history[-1].copy()
            self.displayAll()

    def redo(self):
        if self.redo_stack:
            img = self.redo_stack.pop()
            self.history.append(img.copy())
            self.current_img = img.copy()
            self.displayAll()

    def saveImg(self):
        if self.current_img is None: 
            return
            
        path = filedialog.asksaveasfilename(defaultextension=".png", 
                                            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")])
        if path: 
            cv2.imwrite(path, self.current_img)

    def resetImg(self):
        if self.original_img is not None:
            self.current_img = self.original_img.copy()
            self.history = [self.original_img.copy()]
            self.redo_stack = []
            self.displayAll()
            self.showMainTools()
        
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditorApp(root)
    root.mainloop()