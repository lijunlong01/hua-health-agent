# -*- coding: utf-8 -*-
"""Hyperuricemia Health Agent v3-cloud (Streamlit Community Cloud edition)

与本地版（新世界/网页智能体/app.py）行为一致，仅4处云端适配：
1. 模型/中位数/背景矩阵改为同目录文件（webapp_models.pkl + medians_bg.pkl，不再读训练xlsx）
2. 上传图片写入临时目录（云容器应用目录只读）
3. OCR：Windows 走 PowerShell WinRT；Linux 云端走 pytesseract（eng+chi_sim）；均不可用给明确提示
4. 其余逻辑逐行同本地版

【不变式清单——每次修改前必须核对，禁止回退以下既有功能】
1. 分级回应策略：缺0项=完整评估；缺1-3项=出结果+info注明估计项；缺4-8项=初步结果+warning；
   缺>8项=拒绝出数并引导补充（handle_free_text 与 guided 模式都要遵守）
2. 指标优先判定：含>=2项指标且非问句的消息先走评估，不被知识库抢答
3. 关键词匹配统一小写（英文缩写 BMI/eGFR/TG 等必须可用）
4. 知识库关键词不得含宽泛单字（如"内脏"）以免误触发
5. 结果区特征名必须英文（NAME_EN 映射）；估计项必须披露
6. 英文缩写关键词匹配、XGBoost explainer 用 tree_path_dependent、LR shap_values 取正类
"""
import re
import os
import subprocess
import tempfile
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import shap
import plotly.graph_objects as go
from kb_data import kb_answer, detect_cn

st.set_page_config(page_title='Hyperuricemia Health Agent', page_icon='🩺', layout='wide')

st.markdown("""
<style>
.main {background: linear-gradient(160deg,#f7fafc 0%,#eef4fb 100%);}
[data-testid="stSidebar"] {background:#ffffff;}
.block-container {padding-top: 1.6rem;}
h1 {color:#1a3a5c;}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'webapp_models.pkl')
SUPP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'medians_bg.pkl')


@st.cache_resource
def load_all():
    obj = pickle.load(open(MODEL_PATH, 'rb'))
    models = obj['models']
    prep = obj['prep']
    cont_cols = list(prep.transformers_[0][2])
    scaler = prep.named_transformers_['num']
    cont_idx = {c: i for i, c in enumerate(cont_cols)}
    names_clean = obj['names_clean']
    EV = obj['EV_VOTE']
    supp = pickle.load(open(SUPP_PATH, 'rb'))
    return models, cont_idx, scaler, names_clean, EV, supp['medians'], supp['X_bg_s']


models, cont_idx, scaler, names_clean, EV, MED, X_BG_S = load_all()
CUTOFF = 0.233

FEATS = [
    ('BMI', 'Body mass index BMI (weight kg ÷ height m²)', (14, 45)),
    ('eGFR(mL/min/1.73m2)', 'Estimated glomerular filtration rate eGFR', (5, 200)),
    ('内脏脂肪等级', 'Visceral fat level (body composition report)', (0.5, 40)),
    ('尿素测定(UREA)mmol/L', 'Urea UREA (mmol/L)', (1, 35)),
    ('红细胞计数(RBC)10^12/L', 'Red blood cell count RBC (10^12/L)', (2.5, 8)),
    ('舒张压一次/mmHg', 'Diastolic blood pressure (mmHg)', (35, 150)),
    ('血清γ-谷氨酰基转移酶测定(γ-GT)μ/L', 'γ-glutamyl transferase γ-GT (μ/L)', (1, 600)),
    ('血清低密度脂蛋白胆固醇测定(LDL-C)mmol/L', 'LDL cholesterol (mmol/L)', (0.3, 10)),
    ('血清天门冬氨酸氨基转移酶测定(AST)μ/L', 'Aspartate aminotransferase AST (μ/L)', (2, 400)),
    ('血清甘油三酯测定(TG)mmol/L', 'Triglycerides TG (mmol/L)', (0.2, 20)),
    ('血清白蛋白测定(ALB)g/L', 'Albumin ALB (g/L)', (20, 65)),
    ('血红蛋白(HGB)g/L', 'Hemoglobin HGB (g/L)', (55, 220)),
    ('血葡萄糖测定(Glu)mmol/L', 'Fasting glucose Glu (mmol/L)', (1.5, 33)),
    ('身体年龄/year', 'Body age (body composition report)', (15, 90)),
    ('骨量/kg', 'Bone mass (kg, body composition report)', (1, 12)),
]
STEP_NAMES = [f[0] for f in FEATS]
NAME_EN = {'BMI': 'BMI', 'eGFR(mL/min/1.73m2)': 'eGFR', '内脏脂肪等级': 'Visceral fat level',
           '尿素测定(UREA)mmol/L': 'Urea', '红细胞计数(RBC)10^12/L': 'RBC count',
           '舒张压一次/mmHg': 'Diastolic BP', '血清γ-谷氨酰基转移酶测定(γ-GT)μ/L': 'γ-GT',
           '血清低密度脂蛋白胆固醇测定(LDL-C)mmol/L': 'LDL-C',
           '血清天门冬氨酸氨基转移酶测定(AST)μ/L': 'AST',
           '血清甘油三酯测定(TG)mmol/L': 'Triglycerides', '血清白蛋白测定(ALB)g/L': 'Albumin',
           '血红蛋白(HGB)g/L': 'Hemoglobin', '血葡萄糖测定(Glu)mmol/L': 'Glucose',
           '身体年龄/year': 'Body age', '骨量/kg': 'Bone mass'}


def parse_number(text):
    nums = re.findall(r'-?\d+(?:\.\d+)?', str(text).replace('．', '.').replace('。', '.'))
    return float(nums[0]) if nums else None


# ================= OCR =================
OCR_PS = r'''
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTaskGeneric = ([System.Runtime.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($WinRtTask, $ResultType) {
    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $netTask = $asTask.Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    $netTask.Result
}
[Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.BitmapDecoder,Windows.Foundation,ContentType=WindowsRuntime] | Out-Null
[Windows.Storage.StorageFile,Windows.Storage,ContentType=WindowsRuntime] | Out-Null
$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($args[0])) ([Windows.Storage.StorageFile])
$stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
$decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
$result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
$result.Text
'''

# 双语关键词 -> 特征
OCR_MAP = [
    ('serum uric acid', 'UA'), ('uric acid', 'UA'), ('尿酸', 'UA'), ('血清尿酸', 'UA'),
    ('triglyceride', 'TG'), ('甘油三酯', 'TG'), ('TG', 'TG'),
    ('LDL', 'LDL'), ('low density', 'LDL'), ('低密度', 'LDL'),
    ('aspartate', 'AST'), ('AST', 'AST'), ('天门冬氨酸', 'AST'), ('谷草', 'AST'),
    ('alanine', 'ALT'), ('ALT', 'ALT'), ('丙氨酸', 'ALT'), ('谷丙', 'ALT'),
    ('γ-GT', 'γ-GT'), ('gamma-GT', 'γ-GT'), ('GGT', 'γ-GT'), ('γ-谷氨酰', 'γ-GT'), ('谷氨酰', 'γ-GT'),
    ('eGFR', 'eGFR'), ('GFR', 'eGFR'), ('glomerular', 'eGFR'), ('肾小球滤过率', 'eGFR'),
    ('creatinine', 'CREA'), ('肌酐', 'CREA'),
    ('urea', 'UREA'), ('尿素', 'UREA'), ('BUN', 'UREA'),
    ('albumin', 'ALB'), ('白蛋白', 'ALB'), ('ALB', 'ALB'),
    ('hemoglobin', 'HGB'), ('血红蛋白', 'HGB'), ('HGB', 'HGB'),
    ('fasting glucose', 'Glu'), ('glucose', 'Glu'), ('blood sugar', 'Glu'), ('血糖', 'Glu'), ('空腹血糖', 'Glu'),
    ('red blood cell', 'RBC'), ('RBC', 'RBC'), ('红细胞', 'RBC'),
    ('diastolic', 'DBP'), ('舒张压', 'DBP'), ('低压', 'DBP'), ('DBP', 'DBP'),
    ('body mass index', 'BMI'), ('BMI', 'BMI'), ('体质指数', 'BMI'),
    ('visceral fat', 'VF'), ('内脏脂肪', 'VF'),
    ('body age', 'BA'), ('身体年龄', 'BA'),
    ('bone mass', 'BM'), ('骨量', 'BM'),
]
FEAT_OF = {'TG': '血清甘油三酯测定(TG)mmol/L', 'LDL': '血清低密度脂蛋白胆固醇测定(LDL-C)mmol/L',
           'AST': '血清天门冬氨酸氨基转移酶测定(AST)μ/L', 'γ-GT': '血清γ-谷氨酰基转移酶测定(γ-GT)μ/L',
           'eGFR': 'eGFR(mL/min/1.73m2)', 'UREA': '尿素测定(UREA)mmol/L', 'ALB': '血清白蛋白测定(ALB)g/L',
           'HGB': '血红蛋白(HGB)g/L', 'Glu': '血葡萄糖测定(Glu)mmol/L', 'RBC': '红细胞计数(RBC)10^12/L',
           'DBP': '舒张压一次/mmHg', 'BMI': 'BMI', 'VF': '内脏脂肪等级', 'BA': '身体年龄/year', 'BM': '骨量/kg'}


def _parse_ocr_text(raw):
    text = raw.replace(' ', '').replace('\n', '').lower()
    found = {}
    for kw, key in OCR_MAP:
        m = re.search(re.escape(kw.lower()) + r'[^0-9-]*(-?\d+(?:\.\d+)?)', text)
        if m and key in FEAT_OF:
            found[FEAT_OF[key]] = float(m.group(1))
    return text, found


def ocr_extract(img_path):
    if os.name == 'nt':
        try:
            out = subprocess.run(['powershell', '-NoProfile', '-Command', OCR_PS, img_path],
                                 capture_output=True, text=True, timeout=60)
            return _parse_ocr_text(out.stdout)
        except Exception:
            return None, {}
    # Linux 云端：pytesseract（packages.txt 已装 tesseract + chi_sim）
    try:
        import pytesseract
        text = None
        for lang in ('eng+chi_sim', 'eng'):
            try:
                text = pytesseract.image_to_string(img_path, lang=lang)
                if text and text.strip():
                    break
            except Exception:
                continue
        if text:
            return _parse_ocr_text(text)
        return None, {}
    except Exception:
        return None, {}


# ================= 模型 =================
def full_vals(vals):
    return {f: vals.get(f, MED[f]) for f in STEP_NAMES}


def predict(vals):
    vals = full_vals(vals)
    Xs = np.array([[ (vals[f] - scaler.mean_[cont_idx[f]]) / scaler.scale_[cont_idx[f]] for f in STEP_NAMES]])
    probs = [models[m].predict_proba(Xs)[:, 1] for m in models]
    return float(np.mean(probs))


def explain(vals):
    vals = full_vals(vals)
    Xs = np.array([[ (vals[f] - scaler.mean_[cont_idx[f]]) / scaler.scale_[cont_idx[f]] for f in STEP_NAMES]])
    X_bg_s = X_BG_S[:300]
    sv_sum = np.zeros(len(STEP_NAMES))
    for m in ['GBM', 'CatBoost', 'XGBoost', 'RF', 'LR']:
        est = models[m]
        if m == 'LR':
            raw = shap.LinearExplainer(est, X_bg_s).shap_values(Xs)
            sv = np.asarray(raw[1] if isinstance(raw, list) else raw)
        elif m == 'XGBoost':
            expl = shap.TreeExplainer(est, feature_perturbation='tree_path_dependent')
            sv = np.asarray(expl.shap_values(Xs, check_additivity=False))
        else:
            expl = shap.TreeExplainer(est, X_bg_s)
            sv = np.asarray(expl.shap_values(Xs, check_additivity=False))
        if sv.ndim == 3:
            sv = sv[..., 1]
        sv_sum += sv.ravel() / 5
    return sorted(zip(STEP_NAMES, sv_sum), key=lambda t: -abs(t[1]))


def risk_text(p):
    if p >= CUTOFF:
        return 'High', 'See a doctor soon for a serum uric acid test, and watch your lipids, glucose and kidney function.'
    if p >= 0.15:
        return 'Moderate', 'Schedule regular uric acid checks; limit seafood, alcohol and sugary drinks.'
    return 'Low', 'Your risk is low. Keep a healthy lifestyle and an annual check-up.'


# ================= 会话状态 =================
if 'msgs' not in st.session_state:
    st.session_state.msgs = [('assistant',
        'Hello! I am the Hyperuricemia Health Agent 🩺\n\n'
        'You can:\n① **Ask any question** about uric acid, gout, diet or medication (Chinese input is also welcome);\n'
        '② **Send your lab values** — e.g. "triglycerides 2.1, glucose 5.6";\n'
        '③ Upload a **lab report photo** in the sidebar;\n④ Or click **Guided assessment**.')]
if 'vals' not in st.session_state:
    st.session_state.vals = {}
if 'step' not in st.session_state:
    st.session_state.step = None
if 'last_p' not in st.session_state:
    st.session_state.last_p = None
if 'imp_n' not in st.session_state:
    st.session_state.imp_n = 0


def add_msg(role, text):
    st.session_state.msgs.append((role, text))


def handle_free_text(t):
    if '重新开始' in t or 'restart' in t.lower():
        st.session_state.vals = {}; st.session_state.step = None
        st.session_state.last_p = None; st.session_state.imp_n = 0
        return 'Cleared. You can start over.'
    got = {}
    tl = t.lower()
    for kw, key in OCR_MAP:
        if key not in FEAT_OF:
            continue
        m = re.search(re.escape(kw.lower()) + r'[^0-9-]*?(-?\d+(?:\.\d+)?)', tl)
        if m:
            got[FEAT_OF[key]] = float(m.group(1))
    st.session_state.vals.update(got)
    missing = [f for f in STEP_NAMES if f not in st.session_state.vals]
    # 若消息明显是数据输入（提取到>=2项指标，或无问号且含指标），优先评估而非知识库
    is_question = ('?' in t or '？' in t or t.strip().endswith(('吗', '呢', 'how', 'what', 'why', 'can', 'should', 'is', 'are')))
    if got and len(got) >= 2 and not is_question:
        if len(missing) <= 8:
            for f in missing:
                st.session_state.vals[f] = MED[f]
            st.session_state.imp_n = len(missing)
            st.session_state.imp_note = '、'.join(NAME_EN.get(f, f) for f in missing)
            return None
        else:
            return ('Recorded: %s. You have provided only %d/15 indicators — the assessment would be unreliable. '
                    'Please add more lab values (e.g., triglycerides, γ-GT, glucose, hemoglobin, urea, albumin, LDL, AST, RBC, diastolic BP, eGFR), '
                    'or use "Guided assessment" in the sidebar.') % (', '.join(got), 15 - len(missing))
    ans = kb_answer(t)
    if ans:
        return ans
    if got and len(missing) <= 8:
        for f in missing:
            st.session_state.vals[f] = MED[f]
        st.session_state.imp_n = len(missing)
        st.session_state.imp_note = '、'.join(missing)
        return None
    if got and len(missing) > 8:
        return ('Recorded: %s. You have provided only %d/15 indicators — the assessment would be unreliable. '
                'Please add more lab values (e.g., triglycerides, γ-GT, glucose, hemoglobin, urea, albumin, LDL, AST, RBC, diastolic BP, eGFR), '
                'or use "Guided assessment" in the sidebar.') % (', '.join(got), 15 - len(missing))
    if detect_cn(t):
        return '这个问题我暂时无法回答。您可以问我高尿酸/痛风/饮食/用药相关问题，或直接发送指标数值（如"甘油三酯 2.1，血糖 5.6"）。'
    return 'I could not answer that. Try asking about hyperuricemia, gout, diet or medication, or send me lab values (e.g., "triglycerides 2.1, glucose 5.6").'


# ================= 侧边栏 =================
with st.sidebar:
    st.markdown('### 🩺 Hyperuricemia Health Agent')
    st.markdown('**Model**: soft-voting ensemble (6 algorithms)\n\n'
                '**Training cohort**: 6,779 community adults in Hainan, China\n\n'
                '**Internal test AUC** 0.810 | **External validation AUC** 0.810')
    st.divider()
    st.markdown('#### 📷 Upload a lab report photo')
    up = st.file_uploader('jpg/png — metrics will be auto-recognized', type=['jpg', 'jpeg', 'png'])
    if up is not None and st.button('Recognize & analyze', use_container_width=True):
        tmp = os.path.join(tempfile.gettempdir(), '_upload.png')
        with open(tmp, 'wb') as fh:
            fh.write(up.getbuffer())
        text, found = ocr_extract(tmp)
        if text is None:
            st.error('Recognition failed. Please use a clearer photo or type the values.')
        else:
            st.session_state.vals.update(found)
            msg = ('Recognized and recorded: ' + ('; '.join('%s=%.2f' % (k, v) for k, v in found.items())) if found
                   else 'No metric values were recognized in this photo.')
            if found:
                missing = [f for f in STEP_NAMES if f not in st.session_state.vals]
                for f in ['内脏脂肪等级', '身体年龄/year', '骨量/kg']:
                    if f in missing:
                        st.session_state.vals[f] = MED[f]
                missing = [f for f in STEP_NAMES if f not in st.session_state.vals]
                if missing:
                    msg += ' Still missing %d: %s' % (len(missing), ', '.join(missing[:4]))
                else:
                    st.session_state.last_p = predict(st.session_state.vals)
                    st.session_state.imp_n = 0
            add_msg('assistant', msg)
            st.rerun()
    st.divider()
    if st.button('🔄 Guided assessment (15 items)', use_container_width=True):
        st.session_state.step = 0
        st.session_state.vals = {}
        st.session_state.last_p = None
        st.session_state.imp_n = 0
        st.session_state.guide_imp = []
        add_msg('assistant', 'Starting guided assessment. ' + FEATS[0][1])
        st.rerun()
    st.caption('⚠️ This agent is for health reference only and is not medical advice.')

# ================= 主区 =================
st.markdown('# 🩺 Hyperuricemia Health Agent')
st.caption('Conversational risk assessment · Medical Q&A · Lab-photo recognition · English / 中文')

for role, text in st.session_state.msgs:
    with st.chat_message(role):
        st.markdown(text)

if st.session_state.step is not None and st.session_state.step < len(FEATS):
    st.info('**Question**: ' + FEATS[st.session_state.step][1] +
            ' (plausible range %.1f – %.1f; answer "unknown" if unsure)' % FEATS[st.session_state.step][2])

user_in = st.chat_input('Ask a question or enter lab values…')
if user_in:
    add_msg('user', user_in)
    if st.session_state.step is not None and st.session_state.step < len(FEATS):
        feat, q, rng = FEATS[st.session_state.step]
        if '不清楚' in user_in or '不知道' in user_in or 'unknown' in user_in.lower():
            st.session_state.vals[feat] = MED[feat]
            st.session_state.guide_imp.append(feat)
            st.session_state.step += 1
            nxt = FEATS[st.session_state.step][1] if st.session_state.step < len(FEATS) else None
            reply = 'Estimated with the population-typical value %.1f.' % st.session_state.vals[feat]
            if nxt:
                reply += '\n\n' + nxt
            else:
                st.session_state.last_p = predict(st.session_state.vals)
                st.session_state.step = None
                st.session_state.imp_n = len(st.session_state.guide_imp)
                st.session_state.imp_note = '、'.join(NAME_EN.get(f, f) for f in st.session_state.guide_imp)
            add_msg('assistant', reply)
        else:
            num = parse_number(user_in)
            if num is None or not (rng[0] <= num <= rng[1]):
                add_msg('assistant', 'Please reply with a number between %.1f and %.1f.' % rng)
            else:
                st.session_state.vals[feat] = num
                st.session_state.step += 1
                if st.session_state.step < len(FEATS):
                    add_msg('assistant', 'Recorded. ' + FEATS[st.session_state.step][1])
                else:
                    st.session_state.last_p = predict(st.session_state.vals)
                    st.session_state.step = None
                    st.session_state.imp_n = 0
                    add_msg('assistant', 'All 15 indicators collected. Computing your assessment…')
    else:
        reply = handle_free_text(user_in)
        if reply is None:
            st.session_state.last_p = predict(st.session_state.vals)
        else:
            add_msg('assistant', reply)
    st.rerun()

# 风险结果
if st.session_state.last_p is not None:
    p = st.session_state.last_p
    level, advice = risk_text(p)
    contrib = explain(st.session_state.vals)
    st.divider()
    col1, col2 = st.columns([1, 2])
    with col1:
        fig = go.Figure(go.Indicator(mode='gauge+number', value=p * 100,
                                     number={'suffix': '%', 'font': {'size': 34}},
                                     gauge={'axis': {'range': [0, 100]},
                                            'bar': {'color': '#1a3a5c'},
                                            'steps': [{'range': [0, 15], 'color': '#d7f0d8'},
                                                      {'range': [15, 23.3], 'color': '#fdf3c2'},
                                                      {'range': [23.3, 100], 'color': '#fad7d7'}],
                                            'threshold': {'line': {'color': 'red', 'width': 3},
                                                          'thickness': 0.8, 'value': CUTOFF * 100}}))
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('**%s risk** (cutoff %.1f%%)\n\n%s' % (level, CUTOFF * 100, advice))
    with col2:
        st.markdown('#### Key factors driving your risk' +
                    (' (preliminary)' if st.session_state.imp_n > 3 else ''))
        for name, v in contrib[:5]:
            direction = '🔺 Increases' if v > 0 else '🔻 Decreases'
            st.markdown('- %s: **%s** (contribution %+.3f)' % (direction, NAME_EN.get(name, name), v))
        if st.session_state.imp_n > 0:
            if st.session_state.imp_n > 3:
                st.warning('🔶 %d indicators were unknown and estimated with population-typical values (%s); '
                           'this preliminary result may deviate substantially.' % (st.session_state.imp_n,
                                                                                    getattr(st.session_state, 'imp_note', '')))
            else:
                st.info('Estimated with population-typical values: %s' % st.session_state.imp_note)
        st.caption('Explanation based on SHAP (5-member approximation), for reference only.')
