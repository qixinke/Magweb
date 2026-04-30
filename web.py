import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import openai
import matplotlib.pyplot as plt
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

# ================== 页面配置 ==================
st.set_page_config(page_title="材料数据智能平台", layout="wide", initial_sidebar_state="expanded")

# ================== 全局自定义 CSS 美化 ==================
st.markdown("""
<style>
    /* 全局字体放大 */
    html, body, [class*="css"] {
        font-size: 16px;
    }
    .main > div {
        padding: 1rem 1.5rem;
    }
    /* 侧边栏样式 */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fc;
        border-right: 1px solid #e9ecef;
        padding-top: 2rem;
    }
    .stRadio label {
        font-size: 1rem;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        transition: 0.2s;
    }
    .stRadio label:hover {
        background-color: #e9ecef;
    }
    /* 按钮样式 */
    .stButton button {
        border-radius: 40px !important;
        font-weight: 500 !important;
        background-color: #4361ee !important;
        color: white !important;
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(67,97,238,0.3);
        background-color: #3a56d4 !important;
    }
    /* 输入框圆角 */
    .stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox select {
        border-radius: 28px !important;
        border: 1px solid #cbd5e1 !important;
        padding: 0.6rem 1rem !important;
    }
    /* 数据表格 */
    .dataframe {
        font-size: 0.85rem !important;
    }
    /* 标题 */
    h1 {
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: #1e293b;
    }
    h2 {
        font-size: 1.5rem !important;
    }
    /* 登录卡片容器 */
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 80vh;
    }
    .login-card {
        background: #ffffff;
        border-radius: 32px;
        padding: 2rem 2rem 2.5rem 2rem;
        box-shadow: 0 20px 35px -10px rgba(0,0,0,0.1);
        text-align: center;
        width: 100%;
        max-width: 450px;
        border: 1px solid #eef2f6;
        transition: transform 0.2s;
    }
    .login-card:hover {
        transform: translateY(-5px);
    }
    /* 返回按钮行 */
    .back-button-row {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ================== 模型持久化路径 ==================
MODEL_SAVE_PATH = "saved_model.pkl"
MODEL_META_PATH = "model_meta.pkl"

# ================== 初始化session ==================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'dataset_name' not in st.session_state:
    st.session_state.dataset_name = None
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'uploaded_df_name' not in st.session_state:
    st.session_state.uploaded_df_name = "📤 用户上传数据集"
if 'page_override' not in st.session_state:
    st.session_state.page_override = None
if 'dataset_tab' not in st.session_state:
    st.session_state.dataset_tab = None


# 加载已保存的模型
def load_saved_model():
    if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(MODEL_META_PATH):
        try:
            model = joblib.load(MODEL_SAVE_PATH)
            meta = joblib.load(MODEL_META_PATH)
            st.session_state.model = model
            st.session_state.scaler = meta['scaler']
            st.session_state.feature_cols = meta['feature_cols']
            st.session_state.target_col = meta['target_col']
            st.session_state.model_type = meta['model_type']
            st.session_state.dataset_name = meta['dataset_name']
            st.session_state.trained = True
            return True
        except Exception as e:
            st.warning(f"加载历史模型失败: {e}")
    return False


# 保存模型到本地
def save_model_to_disk(model, scaler, feature_cols, target_col, model_type, dataset_name):
    joblib.dump(model, MODEL_SAVE_PATH)
    joblib.dump({
        'scaler': scaler,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'model_type': model_type,
        'dataset_name': dataset_name
    }, MODEL_META_PATH)


# 读取所有CSV文件
@st.cache_resource
def load_data():
    files = {
        "离子液体": "离子液体.csv",
        "镁二次电池": "镁二次电池数据库2024.11.12.csv",
        "镁合金储氢": "镁合金储氢数据库.csv",
        "镁空气电池": "镁空气电池数据收集.csv",
        '镁合金力学性能': 'Mag力学性能.csv',
        '镁合金腐蚀数据': 'Mag腐蚀数据.csv',
        '储氢催化剂': '储氢催化剂实验数据.csv',
        '电催化数据': '电催化.csv',
        '自修复聚氨酯': '自修复聚氨酯1.csv',
    }
    data_dict = {}
    for name, filename in files.items():
        try:
            for enc in ['gbk', 'utf-8', 'gb2312', 'latin1']:
                try:
                    df = pd.read_csv(filename, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                st.error(f"无法解码文件 {filename}")
                continue
            df.columns = df.columns.str.strip()
            data_dict[name] = df
        except FileNotFoundError:
            st.error(f"文件不存在: {filename}，请确保文件在正确目录下。")
            continue
    return data_dict if data_dict else None


# ================== 登录 ==================
def show_login():
    import os
    st.markdown("""
    <style>
    .login-left-image {
        border-radius: 32px;
        overflow: hidden;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        aspect-ratio: 1 / 1.2;
    }
    .login-left-image img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    .login-right-card {
        background-color: #ffffff;
        border-radius: 32px;
        padding: 2rem 2rem 2.5rem 4rem;   /* 增加上下内边距，使卡片更高 */
        box-shadow: 0 20px 35px -10px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #eef2f6;
        transition: transform 0.2s;
        width: 100%;
    }
    .login-right-card:hover {
        transform: translateY(-5px);
    }
    .stButton button {
        width: 100%;
    }
    .login-right-card h3 {
        font-size: 5rem !important;
        margin-bottom: 0.2rem !important;
        color: #1e293b;
        text-align: left !important;
    }
    .login-right-card h1 {
        font-size: 4rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.2rem !important;
        color: #0f172a;
        text-align: left !important;
    }
    .login-right-card .stCaption {
        font-size: 2rem !important;
        color: #475569;
        margin-bottom: 1rem;
        text-align: left !important;
    }
    .login-right-card .stTextInput,
    .login-right-card .stTextInput > div,
    .login-right-card .stForm {
        text-align: center !important;
    }
    .login-right-card .stTextInput input {
        text-align: left;
    }
    /* 左侧logo图片限制大小 */
    .login-right-card img {
        max-width: 90px !important;
        margin-top: 0.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # 调整列宽比例，使整体宽度变小
    col1, col2 = st.columns([1, 2])

    with col1:
        if os.path.exists("login_side.png"):
            st.image("login_side.png")
        else:
            st.markdown(
                '<div class="login-left-image" style="flex-direction: column; color: white; text-align: center;">'
                '<div style="font-size: 4rem;">🔬⚡</div>'
                '<div style="font-size: 5rem; font-weight: bold; margin-top: 0.8rem;">轻金属材料实验室</div>'
                '</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('----------------------------------------------------------------------')

        # Logo 和标题文字并排放置
        title_col1, title_col2 = st.columns([1, 6])
        with title_col1:
            if os.path.exists("logo.png"):
                st.image("logo.png", width=150)   # 稍微增大Logo
            else:
                st.markdown('<div style="font-size: 2rem;">⚙️🔬</div>', unsafe_allow_html=True)
        with title_col2:
            st.markdown('<h3 style="font-size: 50px;">轻金属材料与安全储能重点实验室</h3>', unsafe_allow_html=True)
            st.markdown('<h1 style="font-size: 40px;">材料数据智能平台</h1>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 30px; color: #475569;">AI for Materials Science</p>',
                        unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("用户名", placeholder="admin")
            password = st.text_input("密码", type="password", placeholder="••••••")
            submitted = st.form_submit_button("登录")
            if submitted:
                if username == "admin" and password == "123456":
                    st.session_state.logged_in = True
                    st.success("登录成功！")
                    st.rerun()
                else:
                    st.error("用户名或密码错误")

        st.caption("© 2025 河南省轻金属材料与安全储能重点实验室")
        # 增加一点底部留白（通过一个空行实现）
        st.markdown('----------------------------------------------------------------------')

# ================== 返回首页按钮组件 ==================
def back_to_home_button():
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("← 返回首页", key="back_home"):
            st.session_state.page_override = None
            st.session_state.dataset_tab = None
            st.rerun()
    st.markdown("---")


# ================== 首页（数据概览） ==================
def show_home(df_dict):
    st.markdown("---")
    st.title("🏠 数据平台首页")
    st.markdown("欢迎使用**轻金属材料与安全储能重点实验室**材料数据智能平台。")

    if df_dict is None or len(df_dict) == 0:
        st.error("没有加载到任何数据集，请检查数据文件。")
        return

    total_rows = sum(df.shape[0] for df in df_dict.values())
    total_cols = sum(df.shape[1] for df in df_dict.values())
    total_numeric = sum(df.select_dtypes(include=[np.number]).shape[1] for df in df_dict.values())

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 1rem; text-align: center; color: white;">
            <div style="font-size: 2rem;">📚</div>
            <div style="font-size: 1.8rem; font-weight: bold;">{len(df_dict)}</div>
            <div>数据集总数</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 20px; padding: 1rem; text-align: center; color: white;">
            <div style="font-size: 2rem;">📊</div>
            <div style="font-size: 1.8rem; font-weight: bold;">{total_rows}</div>
            <div>总行数</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 20px; padding: 1rem; text-align: center; color: white;">
            <div style="font-size: 2rem;">📋</div>
            <div style="font-size: 1.8rem; font-weight: bold;">{total_cols}</div>
            <div>总列数</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); border-radius: 20px; padding: 1rem; text-align: center; color: white;">
            <div style="font-size: 2rem;">🔢</div>
            <div style="font-size: 1.8rem; font-weight: bold;">{total_numeric}</div>
            <div>数值列总数</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("📊 数据集规模对比")
    df_sizes = pd.DataFrame({
        "数据集": list(df_dict.keys()),
        "行数": [df.shape[0] for df in df_dict.values()],
        "列数": [df.shape[1] for df in df_dict.values()]
    })
    fig = px.bar(df_sizes, x="数据集", y="行数", text="行数", title="各数据集行数分布",
                 color="行数", color_continuous_scale="Blues", height=450)
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_tickangle=-30, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("🗂️ 数据集预览卡片")
    cols_per_row = 3
    dataset_list = list(df_dict.items())
    for i in range(0, len(dataset_list), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i + j
            if idx < len(dataset_list):
                name, df = dataset_list[idx]
                with cols[j]:
                    numeric_count = df.select_dtypes(include=[np.number]).shape[1]
                    missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
                    st.markdown(f"""
                    <div style="background: white; border-radius: 20px; padding: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid #eef2f6;">
                        <h4 style="margin: 0 0 0.5rem 0;">📄 {name}</h4>
                        <div style="font-size: 0.9rem; color: #4a5568;">行数: <b>{df.shape[0]}</b> &nbsp;|&nbsp; 列数: <b>{df.shape[1]}</b></div>
                        <div style="font-size: 0.9rem; color: #4a5568;">数值列: <b>{numeric_count}</b> &nbsp;|&nbsp; 缺失占比: <b>{missing_ratio:.1f}%</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"🔍 查看 {name}", key=f"card_btn_{name}"):
                        st.session_state.page_override = "数据库管理"
                        st.session_state.dataset_tab = name
                        st.rerun()

    st.markdown("---")
    st.subheader("⚡ 快速操作")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("🚀 训练新模型", use_container_width=True):
            st.session_state.page_override = "模型训练"
            st.rerun()
    with col_b:
        if st.button("🔮 批量预测", use_container_width=True):
            st.session_state.page_override = "模型预测"
            st.rerun()
    with col_c:
        if st.button("💬 DeepSeek 助手", use_container_width=True):
            st.session_state.page_override = "DeepSeek对话"
            st.rerun()
    st.info("💡 提示：点击数据集卡片下方的「查看」按钮可进入数据库管理页面进行编辑和下载。")


# ================== 数据库管理 ==================
def show_database(df_dict):
    st.markdown("---")
    back_to_home_button()
    st.title("📁 数据库管理")
    if df_dict is None or len(df_dict) == 0:
        st.error("没有成功加载任何数据集，请检查文件")
        return
    tab_names = list(df_dict.keys())
    tabs = st.tabs(tab_names)
    for i, (name, df) in enumerate(df_dict.items()):
        with tabs[i]:
            st.subheader(f"数据集: {name}")
            st.info(f"数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
            col1, col2 = st.columns(2)
            with col1:
                search_col = st.selectbox("选择搜索列", ["所有列"] + list(df.columns), key=f"search_col_{i}")
            with col2:
                search_term = st.text_input("搜索关键词", key=f"search_term_{i}")
            if search_term:
                if search_col == "所有列":
                    mask = df.astype(str).apply(lambda row: row.str.contains(search_term, case=False, na=False).any(), axis=1)
                else:
                    mask = df[search_col].astype(str).str.contains(search_term, case=False, na=False)
                filtered_df = df[mask]
                st.write(f"找到 {len(filtered_df)} 条记录")
            else:
                filtered_df = df
            st.data_editor(filtered_df, use_container_width=True, height=400)
            csv = filtered_df.to_csv(index=False).encode()
            st.download_button(f"下载 {name} 数据", csv, f"{name}.csv", "text/csv")
    # 清除临时覆盖变量
    if st.session_state.page_override:
        st.session_state.page_override = None
    if st.session_state.dataset_tab:
        st.session_state.dataset_tab = None


# ================== 模型训练 ==================
def show_model_train(df_dict):
    st.markdown("---")
    back_to_home_button()
    st.title("🧠 模型训练")
    if df_dict is None or len(df_dict) == 0:
        st.error("数据加载失败或无有效数据集")
        return

    if st.session_state.trained:
        st.info("当前已有训练好的模型，重新训练将覆盖原有模型。")

    st.markdown("### 📂 上传自定义数据集（CSV格式）")
    uploaded_file = st.file_uploader("选择CSV文件", type=["csv"], key="train_upload")
    if uploaded_file is not None:
        try:
            for enc in ['gbk', 'utf-8', 'gb2312', 'latin1']:
                try:
                    uploaded_df = pd.read_csv(uploaded_file, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                st.error("无法解码上传的文件，请检查编码")
                uploaded_df = None
            if uploaded_df is not None:
                uploaded_df.columns = uploaded_df.columns.str.strip()
                st.session_state.uploaded_df = uploaded_df
                st.success(f"成功加载数据集，共 {uploaded_df.shape[0]} 行 {uploaded_df.shape[1]} 列")
                st.dataframe(uploaded_df.head(3))
        except Exception as e:
            st.error(f"读取文件失败: {e}")

    dataset_options = list(df_dict.keys())
    if st.session_state.uploaded_df is not None:
        dataset_options.append(st.session_state.uploaded_df_name)

    dataset_choice = st.selectbox("选择数据集", dataset_options, key="dataset_select")

    if dataset_choice == st.session_state.get("uploaded_df_name", ""):
        df_raw = st.session_state.uploaded_df.copy()
        st.info("当前使用自定义上传的数据集")
    else:
        df_raw = df_dict[dataset_choice].copy()

    st.write("数据预览:")
    st.dataframe(df_raw.head(5))

    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("该数据集没有数值型列，无法训练")
        return

    with st.form("train_form"):
        feature_cols = st.multiselect("选择特征列", options=numeric_cols,
                                      default=numeric_cols[:min(3, len(numeric_cols))])
        target_col = st.selectbox("选择目标列 (回归)", options=numeric_cols)
        model_type = st.selectbox("选择算法", ["随机森林回归", "线性回归", "支持向量回归(SVR)"])
        test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, 0.05)
        use_tuning = st.checkbox("启用简易超参数调优（仅随机森林）", value=False) if model_type == "随机森林回归" else False
        submitted = st.form_submit_button("开始训练")

    if submitted:
        if len(feature_cols) == 0:
            st.error("请至少选择一个特征列")
            st.stop()
        df = df_raw[feature_cols + [target_col]].dropna()
        if len(df) == 0:
            st.error("剔除缺失值后无有效数据")
            st.stop()
        X = df[feature_cols]
        y = df[target_col]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

        with st.spinner("模型训练中..."):
            if model_type == "随机森林回归":
                if use_tuning:
                    param_grid = {
                        'n_estimators': [50, 100, 150],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5]
                    }
                    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
                    grid = GridSearchCV(base_model, param_grid, cv=3, scoring='r2', n_jobs=-1)
                    grid.fit(X_train, y_train)
                    model = grid.best_estimator_
                    st.success(f"最优参数: {grid.best_params_}")
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    model.fit(X_train, y_train)
            elif model_type == "支持向量回归(SVR)":
                model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
                model.fit(X_train, y_train)
            else:
                model = LinearRegression()
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.feature_cols = feature_cols
        st.session_state.target_col = target_col
        st.session_state.model_type = model_type
        st.session_state.dataset_name = dataset_choice
        st.session_state.trained = True
        save_model_to_disk(model, scaler, feature_cols, target_col, model_type, dataset_choice)

        st.success("训练完成并已保存！")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R²", f"{r2:.4f}")
        col2.metric("MSE", f"{mse:.4f}")
        col3.metric("RMSE", f"{rmse:.4f}")
        col4.metric("MAE", f"{mae:.4f}")

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual values")
        ax.set_ylabel("Predicted values")
        st.pyplot(fig)

        if isinstance(model, RandomForestRegressor):
            importances = model.feature_importances_
            if len(importances) != len(feature_cols):
                st.error(f"特征重要性长度 {len(importances)} 与特征列数量 {len(feature_cols)} 不匹配，请重新训练模型。")
            else:
                indices = np.argsort(importances)[::-1]
                fig2, ax2 = plt.subplots()
                ax2.barh(range(len(feature_cols)), importances[indices])
                ax2.set_yticks(range(len(feature_cols)))
                ax2.set_yticklabels([feature_cols[i] for i in indices])
                ax2.set_xlabel("Feature importanc")
                ax2.set_title("RF model")
                st.pyplot(fig2)

        model_bytes = io.BytesIO()
        joblib.dump({
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'target_col': target_col,
            'model_type': model_type,
            'dataset': dataset_choice
        }, model_bytes)
        model_bytes.seek(0)
        st.download_button("下载模型 (.pkl)", model_bytes, "trained_model.pkl", "application/octet-stream")


# ================== 模型预测 ==================
def show_model_predict():
    st.markdown("---")
    back_to_home_button()
    st.title("🔮 模型预测")
    if not st.session_state.trained or st.session_state.model is None:
        st.warning("尚未训练模型，请先前往「模型训练」页面训练模型，或确保已存在历史模型。")
        return

    st.info(
        f"当前模型: {st.session_state.model_type} | 数据集: {st.session_state.dataset_name} | 目标: {st.session_state.target_col}")

    st.markdown("### 单点预测")
    input_values = {}
    cols = st.columns(2)
    for i, col in enumerate(st.session_state.feature_cols):
        with cols[i % 2]:
            input_values[col] = st.number_input(f"{col}", value=0.0, step=0.01, format="%.6f", key=f"single_{col}")
    if st.button("预测", type="primary"):
        input_df = pd.DataFrame([input_values])[st.session_state.feature_cols]
        input_scaled = st.session_state.scaler.transform(input_df.values)
        pred = st.session_state.model.predict(input_scaled)[0]
        st.success(f"预测 {st.session_state.target_col} : **{pred:.4f}**")
        if isinstance(st.session_state.model, RandomForestRegressor):
            preds = [tree.predict(input_scaled)[0] for tree in st.session_state.model.estimators_]
            ci_low = np.percentile(preds, 2.5)
            ci_high = np.percentile(preds, 97.5)
            st.write(f"预测区间 (95% 置信度): [{ci_low:.4f}, {ci_high:.4f}]")

    st.markdown("### 批量预测")
    batch_file = st.file_uploader("上传CSV文件进行批量预测（需包含所有特征列）", type=["csv"])
    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            missing_cols = set(st.session_state.feature_cols) - set(batch_df.columns)
            if missing_cols:
                st.error(f"上传文件缺少特征列: {missing_cols}")
            else:
                X_batch = batch_df[st.session_state.feature_cols]
                X_scaled = st.session_state.scaler.transform(X_batch.values)
                preds = st.session_state.model.predict(X_scaled)
                batch_df['预测_' + st.session_state.target_col] = preds
                st.write("预测结果预览:")
                st.dataframe(batch_df.head(10))
                csv_result = batch_df.to_csv(index=False).encode()
                st.download_button("下载预测结果", csv_result, "batch_prediction.csv", "text/csv")
        except Exception as e:
            st.error(f"读取文件失败: {e}")

    st.markdown("---")
    st.markdown("### 预训练模型")
    st.markdown("## 1. 自修复聚氨酯性能预测")
    external_url = "https://spu-prediction.streamlit.app/"
    st.markdown(
        f'<a href="{external_url}" target="_blank" style="display:block; width:100%; background-color:#4361ee; color:white; text-align:center; padding:12px; border-radius:40px; text-decoration:none; font-weight:bold; margin-top:1rem;">🔬 访问 SPU 预测平台 | 点击进入</a>',
        unsafe_allow_html=True
    )


# ================== DeepSeek 对话 ==================
def show_chat():
    st.markdown("---")
    back_to_home_button()
    st.title("💬 DeepSeek 智能助手")
    st.markdown("向 DeepSeek AI 提问材料科学相关问题")

    # 直接使用硬编码 API Key（请确保安全）
    api_key = "sk-5ec9a4d6c43149eaa26c126aa369ff4e"
    st.session_state.api_key = api_key

    client = openai.OpenAI(api_key=st.session_state.api_key, base_url="https://api.deepseek.com")

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("输入问题..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=st.session_state.chat_messages,
                        temperature=0.7,
                        stream=False
                    )
                    answer = response.choices[0].message.content
                    st.markdown(answer)
                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"调用失败: {e}")

    if st.button("清空对话历史"):
        st.session_state.chat_messages = []
        st.rerun()


# ================== 主程序 ==================
def main():
    if not st.session_state.logged_in:
        show_login()
        return

    if 'data_loaded' not in st.session_state:
        st.session_state.df_dict = load_data()
        st.session_state.data_loaded = True
    df_dict = st.session_state.df_dict

    if not st.session_state.trained:
        load_saved_model()

    with st.sidebar:
        st.title("导航菜单")
        page = st.radio("选择页面", ["首页", "数据库管理", "模型训练", "模型预测", "DeepSeek对话"])
        if st.button("退出登录"):
            st.session_state.logged_in = False
            st.session_state.model = None
            st.session_state.trained = False
            st.session_state.chat_messages = []
            st.session_state.page_override = None
            st.session_state.dataset_tab = None
            st.rerun()

    if st.session_state.page_override:
        page = st.session_state.page_override

    if page == "首页":
        show_home(df_dict)
    elif page == "数据库管理":
        show_database(df_dict)
    elif page == "模型训练":
        show_model_train(df_dict)
    elif page == "模型预测":
        show_model_predict()
    elif page == "DeepSeek对话":
        show_chat()


if __name__ == "__main__":
    main()