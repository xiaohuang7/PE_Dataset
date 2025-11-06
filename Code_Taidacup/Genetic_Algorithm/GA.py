import streamlit as st
import random 
import numpy as np
import time
import sympy
from sympy import symbols, Eq, Matrix, solve, simplify
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import os
from fitness import fitness_function

# 遗传算法参数
POPULATION_SIZE = 500
GENES = [-1, 0, 1]
NUM_VARIABLES = 24
MAX_GENERATIONS = 2000
def set_global_style():
    # 自定义CSS：强制所有元素（包括st.write）字体放大到18px
    st.markdown("""
    <style>
        /* 基础样式 - 强制所有元素字体大小 */
        html, body, p, span, div, [class*="st-"], [data-testid*="st-"] {
            font-family: "Microsoft YaHei", "Inter", sans-serif !important;
            font-size: 18px !important;
            line-height: 1.6 !important; /* 增加行高，提高可读性 */
        }
        
        /* 标题字体层级 */
        h1 { font-size: 30px !important; }
        h2 { font-size: 28px !important; }
        h3 { font-size: 24px !important; }
        h4 { font-size: 22px !important; }
        h5 { font-size: 20px !important; }
        
        /* 输入框样式 */
        input[type="number"] {
            font-size: 18px !important;
            padding: 8px !important;
            height: 40px !important; /* 增加输入框高度 */
        }
        
        /* 按钮样式 */
        button {
            font-size: 18px !important;
            padding: 10px 20px !important;
            height: auto !important;
        }
        
        /* 浅蓝色标记样式 */
        .light-blue-text {
            color: #165DFF !important;
            font-weight: 500 !important;
            font-size: 18px !important;
            display: inline-block;
            margin-bottom: 4px;
        }
        
        /* 下标样式 */
        .subscript {
            position: relative;
            display: inline-block;
            margin-right: 8px; /* 增加间距 */
        }
        .subscript span {
            position: absolute;
            bottom: -8px;
            right: -8px;
            font-size: 14px !important;
        }
    </style>
    """, unsafe_allow_html=True)

# 设置全局字体样式
set_global_style()
st.title("宽增益DCDC变换器拓扑生成")

# 输入区
with st.form("input_form"):
    col1, col2 = st.columns([0.4, 1], vertical_alignment="center")

    # 第一列放文字（默认居左）
    with col1:
        st.write("任意三阶宽增益DCDC变换器的表达式如下：")

    # 第二列放图片（关闭容器自适应宽度，使用自定义width，默认居左）
    with col2:
        st.image(
            "/root/multiport-RL/imgs/表达式.png",
            width=400,
            use_container_width=False  # 替代原use_column_width，确保自定义宽度生效
        )
    # 电压增益系数（浅蓝色标记，用自定义布局实现）
    st.write("请输入电压增益系数：")
    cols = st.columns(8, gap="small")
    
    # m0-m3和n0-n3用"文本+输入框"组合实现浅蓝色标记
    # 生成带下标的标签（m₀、m₁等）
    def create_subscript_label(letter, index):
        """生成带下标的HTML标签（如m₀）"""
        return f"""
        <div class="light-blue-text subscript">
            {letter}<span>{index}</span>
        </div>
        """
    
    # 创建带下标和浅蓝色的输入框
    def create_labeled_input(container, letter, index, value, key):
        with container:
            st.markdown(create_subscript_label(letter, index), unsafe_allow_html=True)
            return st.number_input(
                "", 
                value=value, 
                format="%d", 
                key=key, 
                label_visibility="collapsed"
            )
    
    # 电压增益系数输入（m0-m3和n0-n3）
    m0 = create_labeled_input(cols[0], "m", "0", 1, "m0")
    m1 = create_labeled_input(cols[1], "m", "1", 0, "m1")
    m2 = create_labeled_input(cols[2], "m", "2", 0, "m2")
    m3 = create_labeled_input(cols[3], "m", "3", 0, "m3")
    n0 = create_labeled_input(cols[4], "n", "0", 0, "n0")
    n1 = create_labeled_input(cols[5], "n", "1", 0, "n1")
    n2 = create_labeled_input(cols[6], "n", "2", 1, "n2")
    n3 = create_labeled_input(cols[7], "n", "3", 0, "n3")
    st.write("请输入优化权重α为电流应力系数，β是电压应力系数（α+β=10）：")
    cols_x = st.columns(2, gap="medium")
    
    # α和β的浅蓝色标记
    with cols_x[0]:
        st.markdown(f"<span class='light-blue-text'>α</span>", unsafe_allow_html=True)
        Alpha = st.number_input("", value=5, format="%d", key="alpha", label_visibility="collapsed")
    
    with cols_x[1]:
        st.markdown(f"<span class='light-blue-text'>β</span>", unsafe_allow_html=True)
        Beta = st.number_input("", value=5, format="%d", key="beta", label_visibility="collapsed")
    st.write("m₀-m₃及n₀-n₃的取值范围规定如下：")
    st.image("/root/multiport-RL/imgs/范围.png",width=400)
    button_cols = st.columns([1, 1, 1])  # 三列布局，左右列留白，中间列放按钮
    with button_cols[1]:
        # 用两列分割中间区域，放置两个按钮
        btn1, btn2 = st.columns(2, gap="small")
        
        # 先注入立体按钮的CSS样式
        st.markdown("""
        <style>
            /* 立体按钮基础样式 */
            .stButton > button, .stFormSubmitButton > button {
                /* 主体样式 */
                background-color: #165DFF !important;
                color: white !important;
                border: none !important;
                border-radius: 8px !important;
                padding: 12px 24px !important;
                font-size: 18px !important;
                font-weight: 500 !important;
                
                /* 立体效果核心：多层阴影 */
                box-shadow: 
                    0 4px 0 #0E42CC,  /* 底部深色阴影（模拟厚度） */
                    0 6px 8px rgba(0, 0, 0, 0.15) !important;  /* 环境阴影 */
                
                /* 过渡动画（确保hover效果平滑） */
                transition: all 0.2s ease !important;
                transform: translateY(0) !important;  /* 初始位置 */
            }
            
            /* 鼠标悬停时：轻微上浮+阴影变化 */
            .stButton > button:hover, .stFormSubmitButton > button:hover {
                box-shadow: 
                    0 6px 0 #0E42CC,  /* 底部阴影上移 */
                    0 8px 12px rgba(0, 0, 0, 0.2) !important;
                transform: translateY(-2px) !important;  /* 上移2px，增强立体感 */
            }
            
            /* 点击时：下沉效果 */
            .stButton > button:active, .stFormSubmitButton > button:active {
                box-shadow: 
                    0 2px 0 #0E42CC,  /* 底部阴影缩小 */
                    0 4px 6px rgba(0, 0, 0, 0.1) !important;
                transform: translateY(2px) !important;  /* 下沉2px，模拟按压 */
            }
        </style>
        """, unsafe_allow_html=True)
        
        # 渲染按钮（保持原有功能）
        online_submit = btn1.form_submit_button("在线运行", use_container_width=True)
        offline_submit = btn2.form_submit_button("离线运行", use_container_width=True)

D1, D2 = symbols('D1 D2', real=True, positive=True)

def draw_polyline(ax, p1, p2, used_points=None,all_mid_points=None, used_lines=None):
    # 记录已用折点，避免重叠
    if used_points is None:
        used_points = set()
    if all_mid_points is None:
        all_mid_points = []
    if used_lines is None:
        used_lines = set()

    # 检查是否已连过这条线（无向边，顺序无关）
    line_key = tuple(sorted([p1, p2]))
    if line_key in used_lines:
        return used_points, all_mid_points, used_lines  # 不再连线
    
    # 获取所有元器件的x坐标
    component_xs = {1.2, 1.6, 2.0, 2.2, -1, 0, 0.8}

    while True:
        mid_x = round(random.uniform(min(p1[0], p2[0]) - 1.2, max(p1[0], p2[0]) + 1.2), 2)
        # 与所有元件x坐标的差值绝对值都大于等于0.1
        # 与所有已用折点x坐标的差值绝对值都大于等于0.05
        if (
        all(abs(mid_x - cx) >= 0.2 for cx in component_xs)
        and all(abs(mid_x - pt[0]) >= 0.1 for pt in used_points)
        ):
            break

    # 两个折点
    mid1 = (mid_x, p1[1])
    mid2 = (mid_x, p2[1])

    # 记录已用折点
    used_points.add(mid1)
    used_points.add(mid2)
    all_mid_points.append(mid1)
    all_mid_points.append(mid2)
    # 记录已连线
    used_lines.add(line_key)

    ax.plot([p1[0], mid1[0]], [p1[1], mid1[1]], 'k-', lw=1, zorder=1)
    ax.plot([mid1[0], mid2[0]], [mid1[1], mid2[1]], 'k-', lw=1, zorder=1)
    ax.plot([mid2[0], p2[0]], [mid2[1], p2[1]], 'k-', lw=1, zorder=1)
    return used_points,all_mid_points,used_lines

def draw_circuit_with_images(matrix):
    import matplotlib.image as mpimg
    fig, ax = plt.subplots(figsize=(6, 4))
    # 坐标定义（手动调整，仿真电路图布局）
    # 元件布局（可微调）
    L_pos = [(-1, 2.5), (0, 1.5), (0.8, 0.5)]
    C_pos = [(1.2, 3), (1.6, 2), (2.0, 1)]
    Vin_pos = (2.4, 0)

    # 画元件图片
    L_img = mpimg.imread('/root/multiport-RL/imgs/L.png')
    C_img = mpimg.imread('/root/multiport-RL/imgs/C.png')
    Vin_img = mpimg.imread('/root/multiport-RL/imgs/Vin.png')
    # 电感
    for i, (x, y) in enumerate(L_pos):
        ax.imshow(L_img, extent=[x-0.2, x+0.2, y-0.2, y+0.2], zorder=2)
        ax.text(x-0.1, y, f"L{i+1}", fontsize=6, va='center', ha='right', color='k')
    # 电容
    for i, (x, y) in enumerate(C_pos):
        ax.imshow(C_img, extent=[x-0.2, x+0.2, y-0.2, y+0.2], zorder=2)
        ax.text(x-0.1, y+0.05, f"C{i+1}", fontsize=6, va='bottom', ha='center', color='k')
    # Vin
    ax.imshow(Vin_img, extent=[Vin_pos[0]-0.2, Vin_pos[0]+0.2, Vin_pos[1]-0.2, Vin_pos[1]+0.2], zorder=2)
    ax.text(Vin_pos[0]+0.1, Vin_pos[1]+0.15, "Vin", fontsize=6, va='top', ha='center', color='k')
        # 折线连线，自动避让
    used_points = set()
    all_mid_points = []
    used_lines = set()
    # 画每一行的环路（每个电感为起点，按矩阵连接到对应的C或Vin）
    for i in range(3):
        # 电感正极坐标
        last_point = (L_pos[i][0]-0.02, L_pos[i][1]+0.17)
        # 依次扫描四列
        for j in range(4):
            val = matrix[i, j] if j < matrix.shape[1] else 0
            if val == 0:
                continue
            # 连接到对应电容或Vin
            if j < 3:
                if val == 1:
                    # 连到电容正极（上端）
                    target = (C_pos[j][0], C_pos[j][1]+0.19)
                    next_polarity = -1  # 下一条线从电容负极出发
                elif val == -1:
                    # 连到电容负极（下端）
                    target = (C_pos[j][0], C_pos[j][1]-0.17)
                    next_polarity = 1   # 下一条线从电容正极出发
            else:
                if val == 1:
                    # 连到Vin正极（上端）
                    target = (Vin_pos[0], Vin_pos[1]+0.19)
                    next_polarity = -1  # 下一条线从Vin负极出发
                elif val == -1:
                    # 连到Vin负极（下端）
                    target = (Vin_pos[0], Vin_pos[1]-0.16)
                    next_polarity = 1 # 下一条线从Vin正极出发
            used_points,all_mid_points,used_lines = draw_polyline(ax, last_point, target,used_points,all_mid_points,used_lines)
            if j < 3:
                # 电容出口
                if next_polarity == 1:
                    last_point = (C_pos[j][0], C_pos[j][1]+0.19)
                else:
                    last_point = (C_pos[j][0], C_pos[j][1]-0.17)
            else:
                # Vin出口
                if next_polarity == 1:
                    last_point = (Vin_pos[0], Vin_pos[1]+0.19)
                else:
                    last_point = (Vin_pos[0], Vin_pos[1]-0.16)
        end = (L_pos[i][0]-0.02, L_pos[i][1]-0.17)
        used_points,all_mid_points,used_lines = draw_polyline(ax, last_point, end,used_points,all_mid_points,used_lines)

    from collections import defaultdict
    y_dict = defaultdict(list)
    for pt in all_mid_points:
        y_dict[pt[1]].append(pt)
    for pts in y_dict.values():
        if len(pts) >= 2:
            for pt in pts:
                ax.plot(pt[0], pt[1], 'ko', markersize=4, zorder=3)
    ax.set_xlim(-2.3, 3.7)
    ax.set_ylim(-0.5, 3.5)
    ax.axis('off')
    fig.tight_layout()
    return fig

def initialize_population():
    return [list(np.random.choice(GENES, NUM_VARIABLES)) for _ in range(POPULATION_SIZE)]

def calculate_fitness(population, m0, m1, m2, m3, n0, n1, n2, n3):
    return np.array([fitness_function(s, m0, m1, m2, m3, n0, n1, n2, n3) for s in population])

def selection(population, fitness_values):
    parents = []
    tournament_size = 5
    best_index = np.argmax(fitness_values)
    parents.append(population[best_index])
    for _ in range(len(population)-1):
        indices = np.random.choice(len(population), tournament_size)
        winner = indices[np.argmax(fitness_values[indices])]
        parents.append(population[winner])
    return parents

def crossover(parents, population_size):
    offspring = parents[0:1]
    for _ in range(population_size-1):
        p1, p2 = random.sample(parents, 2)
        point = random.randint(1, NUM_VARIABLES-1)
        child = p1[:point] + p2[point:]
        offspring.append(child)
    return offspring

def mutate(offspring):
    mutation_rate = 0.5
    for i in range(1, len(offspring)):
        for j in range(NUM_VARIABLES):
            if random.random() < mutation_rate:
                offspring[i][j] = random.choice(GENES)

def genetic_algorithm(m0, m1, m2, m3, n0, n1, n2, n3):
    population = initialize_population()
    best_cost_history = []
    for generation in range(MAX_GENERATIONS):
        fitness_values = calculate_fitness(population, m0, m1, m2, m3, n0, n1, n2, n3)
        parents = selection(population, fitness_values)
        offspring = crossover(parents, POPULATION_SIZE)
        mutate(offspring)
        population = offspring
        best_index = np.argmax(fitness_values)
        best_chromosome = population[best_index]
        best_fitness = fitness_values[best_index]
        best_cost_history.append(best_fitness)
    return best_chromosome, best_fitness, best_cost_history

if online_submit:
    with st.spinner("正在运行..."):
        start_time = time.time()    
        best_chromosome, best_fitness, best_cost_history = genetic_algorithm(
            m0, m1, m2, m3, n0, n1, n2, n3
        )
        end_time = time.time()
        if best_fitness < 460:
            st.error("本次运行未找到可行解，请尝试多次运行或更改输入参数。")
        else:
            st.success(f"运行完成！最佳适应度: {best_fitness}")
            st.write(f"程序运行时间：{end_time - start_time:.2f} 秒")

            # 输出矩阵
            augmented_matrix = Matrix([
                [best_chromosome[1] * D1 + best_chromosome[5] * D2, best_chromosome[2] * D1 + best_chromosome[6] * D2, best_chromosome[3] * D1 + best_chromosome[7] * D2, best_chromosome[0] * D1 + best_chromosome[4] * D2],
                [best_chromosome[9] * D1 + best_chromosome[13] * D2, best_chromosome[10] * D1 + best_chromosome[14] * D2, best_chromosome[11] * D1 + best_chromosome[15] * D2, best_chromosome[8] * D1 + best_chromosome[12] * D2],
                [best_chromosome[17] * D1 + best_chromosome[21] * D2, best_chromosome[18] * D1 + best_chromosome[22] * D2, best_chromosome[19] * D1 + best_chromosome[23] * D2, best_chromosome[16] * D1 + best_chromosome[20] * D2]
            ])
            st.write("增广矩阵：")
            st.latex("A\\_matrix=" + sympy.latex(simplify(augmented_matrix)))

            # 拆分为 D1 矩阵和 D2 矩阵
            D1_matrix = augmented_matrix.applyfunc(lambda expr: expr.as_coefficients_dict().get(D1, 0))
            D2_matrix = augmented_matrix.applyfunc(lambda expr: expr.as_coefficients_dict().get(D2, 0))

            st.latex("D1\\_matrix = " + sympy.latex(D1_matrix))
            st.latex("D2\\_matrix = " + sympy.latex(D2_matrix))
            
            # 绘制优化曲线
            st.write("迭代曲线：")
            fig, ax = plt.subplots(figsize=(3, 2))
            ax.plot(range(1, len(best_cost_history)+1), best_cost_history)
            ax.set_xlabel('Generation', fontsize=6)
            ax.set_ylabel('Best Fitness', fontsize=6)
            ax.set_title('GA Optimization', fontsize=8)
            ax.tick_params(axis='both', labelsize=4)
            fig.tight_layout()

            # 居中显示
            center = st.columns([2, 3, 2])
            with center[1]:
                st.pyplot(fig)
    # 离线运行逻辑
if offline_submit:

    # 定义两个条件
    condition1 = (n3 == 1 and m0 == 1 and m1 == 0 and m2 == 0 and m3 == 0 and n0 == 0 and n1 == 0 and n2 == 0)
    condition2 = (n3 == 0 and m0 == 1 and m1 == -3 and m2 == 3 and m3 == -1 and n0 == 1 and n1 == 0 and n2 == 0)

    if condition1:
        img_folder = "/root/multiport-RL/imgs/D^3"
        
        # 读取并处理图片（展示所有图片，取消数量限制）
        img_paths = [
            os.path.join(img_folder, img) 
            for img in os.listdir(img_folder) 
            if img.endswith(('.png', '.jpg', '.jpeg'))
        ]
        img_paths.sort()  # 按文件名排序（保持原排序逻辑）

        if len(img_paths) > 0:
            # 计算总行数（向上取整，确保最后一行不足4张也能显示）
            total_rows = (len(img_paths) + 3) // 4
            
            # 逐行展示图片
            for row_idx in range(total_rows):
                # 为每行创建4列容器
                cols = st.columns(4, gap="small")
                
                # 计算当前行的图片索引范围
                start_idx = row_idx * 4
                end_idx = start_idx + 4
                row_imgs = img_paths[start_idx:end_idx]  # 可能不足4张（最后一行）
                
                # 在当前行的列中展示图片
                for col_idx, (col, img_path) in enumerate(zip(cols, row_imgs)):
                    time.sleep(0.5)  # 每张延迟0.5秒（可调整）
                    with col:
                        img_num = start_idx + col_idx + 1  # 图片编号（从1开始连续编号）
                        st.image(
                            img_path,
                            width=250,
                            use_container_width=True
                        )

    # ---------------------- condition2：2张图（1行排列） ----------------------
    elif condition2:  # 用elif避免两个条件同时执行
            img_folder = "/root/multiport-RL/imgs/(1-D)^-3"
            
            # 读取并处理图片（展示所有图片，不限制数量）
            img_paths = [
                os.path.join(img_folder, img) 
                for img in os.listdir(img_folder) 
                if img.endswith(('.png', '.jpg', '.jpeg'))
            ]
            img_paths.sort()  # 按文件名排序

            if len(img_paths) > 0:
                total_rows = (len(img_paths) + 3) // 4  # 总行数（向上取整，确保最后一行不足4张也能显示）
                
                for row_idx in range(total_rows):
                    cols = st.columns(4, gap="small")  # 每行4列
                    start_idx = row_idx * 4
                    end_idx = start_idx + 4
                    row_imgs = img_paths[start_idx:end_idx]  # 当前行的图片
                    
                    # 逐个显示当前行的图片
                    for col_idx, (col, img_path) in enumerate(zip(cols, row_imgs)):
                        time.sleep(0.5)  # 延迟可调整
                        with col:
                            img_num = start_idx + col_idx + 1  # 图片编号（从1开始）
                            st.image(
                                img_path,
                                width=250,
                                use_container_width=True
                            )


