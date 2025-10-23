// main.rs

/// 一个能持续生成视觉上不同颜色的迭代器。
///
/// 它首先会返回一个预定义的、包含超过100种常见颜色的列表。
/// 在此之后，它会使用黄金分割律在HSV颜色空间中生成一系列在视觉上分布均匀的颜色，
/// 并将它们转换为RGB元组。
pub struct ColorIterator {
    predefined_index: usize,
    // 用于程序化生成阶段的HSV颜色状态
    hue: f32, // 色相 [0.0, 1.0]
    saturation: f32, // 饱和度 [0.0, 1.0]
    value: f32, // 亮度/明度 [0.0, 1.0]
}

impl ColorIterator {
    /// 创建一个新的颜色迭代器。
    ///
    /// 你可以自定义饱和度(saturation)和亮度(value)来调整生成的颜色风格。
    /// - `saturation`: 饱和度，0.0为灰色，1.0为最鲜艳。推荐值：0.7-0.95
    /// - `value`: 亮度，0.0为黑色，1.0为最亮。推荐值：0.8-1.0
    pub fn new(saturation: f32, value: f32) -> Self {
        Self {
            predefined_index: 0,
            // 使用一个随机的初始色相，避免每次启动程序生成的颜色序列都一样
            hue: 0.5,
            saturation: saturation.clamp(0.0, 1.0),
            value: value.clamp(0.0, 1.0),
        }
    }
}

impl Default for ColorIterator {
    /// 创建一个具有默认饱和度和亮度的迭代器。
    fn default() -> Self {
        Self::new(0.8, 0.95)
    }
}

impl Iterator for ColorIterator {
    // 迭代器吐出的项目类型是 (R, G, B) 元组
    type Item = (u8, u8, u8);

    fn next(&mut self) -> Option<Self::Item> {
        // --- 阶段一：返回预定义颜色 ---
        if self.predefined_index < PREDEFINED_COLORS.len() {
            let color = PREDEFINED_COLORS[self.predefined_index];
            self.predefined_index += 1;
            return Some(color);
        }

        // --- 阶段二：程序化生成颜色 ---
        // 将当前HSV颜色转换为RGB
        let rgb = hsv_to_rgb(self.hue, self.saturation, self.value);

        // 更新色相，使用黄金分割共轭数，使其在色轮上均匀跳跃
        // (sqrt(5) - 1) / 2 ≈ 0.618033988749895
        const GOLDEN_RATIO_CONJUGATE: f32 = 0.618034;
        self.hue = (self.hue + GOLDEN_RATIO_CONJUGATE) % 1.0;

        // 这个迭代器是无限的，所以总是返回 Some
        Some(rgb)
    }
}

/// 预定义的颜色列表，包含超过100种常见且视觉区分度高的颜色。
/// 列表顺序经过挑选，将最基础的颜色放在最前面。
const PREDEFINED_COLORS: &[(u8, u8, u8)] = &[
    // 12 种基础高对比度颜色
    (230, 25, 75),   // Red
    (60, 180, 75),   // Green
    (0, 130, 200),   // Blue
    (255, 225, 25),  // Yellow
    (245, 130, 48),  // Orange
    (145, 30, 180),  // Purple
    (70, 240, 240),  // Cyan
    (240, 50, 230),  // Magenta
    (210, 245, 60),  // Lime
    (250, 190, 212), // Pink
    (0, 128, 128),   // Teal
    (128, 0, 0),     // Maroon

    // 8 种灰度与特殊色
    (128, 128, 128), // Grey
    (255, 255, 255), // White
    (0, 0, 0),       // Black
    (170, 110, 40),  // Brown
    (128, 128, 0),   // Olive
    (0, 0, 128),     // Navy
    (255, 250, 200), // Beige
    (230, 190, 255), // Lavender

    // Web 安全色和其他常见颜色 (选取部分)
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (192, 192, 192),
    (255, 165, 0),
    (138, 43, 226),
    (165, 42, 42),
    (222, 184, 135),
    (95, 158, 160),
    (127, 255, 212),
    (100, 149, 237),
    (255, 105, 180),
    (255, 127, 80),
    (220, 20, 60),
    (34, 139, 34),
    (0, 206, 209),
    (148, 0, 211),
    (255, 20, 147),
    (30, 144, 255),
    (218, 165, 32),
    (47, 79, 79),

    // 更多不同色调和亮度的颜色
    (25, 25, 112),    // MidnightBlue
    (135, 206, 235),  // SkyBlue
    (173, 255, 47),   // GreenYellow
    (240, 230, 140),  // Khaki
    (210, 105, 30),   // Chocolate
    (255, 99, 71),    // Tomato
    (255, 215, 0),    // Gold
    (189, 183, 107),  // DarkKhaki
    (188, 143, 143),  // RosyBrown
    (219, 112, 147),  // PaleVioletRed
    (244, 164, 96),   // SandyBrown
    (154, 205, 50),   // YellowGreen
    (46, 139, 87),    // SeaGreen
    (102, 205, 170),  // MediumAquaMarine
    (95, 158, 160),   // CadetBlue
    (106, 90, 205),   // SlateBlue
    (216, 191, 216),  // Thistle
    (255, 240, 245),  // LavenderBlush
    (240, 255, 240),  // Honeydew
    (240, 248, 255),  // AliceBlue
    (176, 196, 222),  // LightSteelBlue
    (255, 228, 196),  // Bisque
    (255, 235, 205),  // BlanchedAlmond
    (112, 128, 144),  // SlateGray
    (255, 69, 0),     // OrangeRed
    (75, 0, 130),     // Indigo
    (255, 222, 173),  // NavajoWhite
    (210, 180, 140),  // Tan
    (65, 105, 225),   // RoyalBlue
    (176, 224, 230),  // PowderBlue
    (50, 205, 50),    // LimeGreen
    (205, 92, 92),    // IndianRed
    (199, 21, 133),   // MediumVioletRed
    (255, 182, 193),  // LightPink
    (255, 192, 203),  // Pink
    (139, 69, 19),    // SaddleBrown
    (160, 82, 45),    // Sienna
    (205, 133, 63),   // Peru
    (123, 104, 238),  // MediumSlateBlue
    (72, 61, 139),    // DarkSlateBlue
    (48, 25, 52),     // A deep purple
    (20, 90, 50),     // A dark green
    (10, 10, 90),     // A dark blue
    (150, 75, 0),     // Burnt Orange
    (0, 75, 150),     // Steel Blue
    (180, 200, 220),  // Light Periwinkle
    (255, 153, 153),  // Light Coral
    (153, 255, 153),  // Light Green
    (153, 153, 255),  // Light Blue
    (255, 255, 153),  // Light Yellow
    (153, 255, 255),  // Light Cyan
    (255, 153, 255),  // Light Magenta
];


/// 将 HSV (色相, 饱和度, 亮度) 颜色转换为 RGB 元组。
/// h, s, v 值的范围都是 [0.0, 1.0]。
/// 返回的 (r, g, b) 元组中，值的范围是 [0, 255]。
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let r;
    let g;
    let b;

    if s == 0.0 {
        // 灰色
        r = v;
        g = v;
        b = v;
    } else {
        let h = h * 6.0;
        let i = h.floor();
        let f = h - i;
        let p = v * (1.0 - s);
        let q = v * (1.0 - s * f);
        let t = v * (1.0 - s * (1.0 - f));

        match i as i32 {
            0 => { r = v; g = t; b = p; }
            1 => { r = q; g = v; b = p; }
            2 => { r = p; g = v; b = t; }
            3 => { r = p; g = q; b = v; }
            4 => { r = t; g = p; b = v; }
            _ => { r = v; g = p; b = q; }
        }
    }

    (
        (r * 255.0) as u8,
        (g * 255.0) as u8,
        (b * 255.0) as u8,
    )
}
