# -*- coding: utf-8 -*-
"""
app.py - 語言學習助教（無TTS版本）
"""

import os
import datetime
import gradio as gr
import random
import numpy as np
from PIL import Image
import wave
import struct

from models import get_model_manager
from processors import get_conversation_manager

# 創建必要的目錄
for dir_name in ["scenario_images", "temp_audio", "user_recordings", "generations"]:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# 載入外部CSS文件
def load_css_file(css_file_path):
    try:
        with open(css_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"⚠️  CSS文件未找到: {css_file_path}")
        return ""
    except Exception as e:
        print(f"⚠️  讀取CSS文件時出錯: {e}")
        return ""

# 載入樣式
css_content = load_css_file("styles.css")

# 預設場景範例
scenario_examples = [
    {
        "name": "機場對話",
        "image_path": "scenario_images/airport.jpg",
        "preview_text": "練習機場通關、登機和問詢的對話",
        "scenario": "機場對話 (Airport Conversation)",
        "icon": "✈️"
    },
    {
        "name": "餐廳點餐",
        "image_path": "scenario_images/restaurant.jpg",
        "preview_text": "練習餐廳點餐、詢問菜單和結帳的對話",
        "scenario": "餐廳點餐 (Restaurant Ordering)",
        "icon": "🍽️"
    },
    {
        "name": "求職面試",
        "image_path": "scenario_images/interview.jpg",
        "preview_text": "練習工作面試中的自我介紹和問答",
        "scenario": "求職面試 (Job Interview)",
        "icon": "💼"
    },
    {
        "name": "日常社交",
        "image_path": "scenario_images/socializing.jpg",
        "preview_text": "練習日常問候、閒聊和社交對話",
        "scenario": "日常社交 (Daily Social Conversation)",
        "icon": "🤝"
    },
    {
        "name": "醫療諮詢",
        "image_path": "scenario_images/medical.jpg",
        "preview_text": "練習在診所或醫院的醫療對話",
        "scenario": "醫療諮詢 (Medical Consultation)",
        "icon": "🏥"
    },
    {
        "name": "學術討論",
        "image_path": "scenario_images/academic.jpg",
        "preview_text": "練習課堂或研討會的學術討論",
        "scenario": "學術討論 (Academic Discussion)",
        "icon": "📚"
    }
]

# 預設場景設定
scenario_presets = {
    "機場對話 (Airport Conversation)": {
        "description": "在機場通關、護照檢查和登機的相關對話情境",
        "roles": {
            "assistant": "機場工作人員/海關人員",
            "user": "旅客"
        },
        "sample_dialog": {
            "assistant": "Good morning. Passport please?",
            "user": "Good morning. Here is my passport.",
            "next_prompt": "Thank you. Where are you traveling to today?"
        }
    },
    "餐廳點餐 (Restaurant Ordering)": {
        "description": "在餐廳點餐、詢問菜單和結帳的對話情境",
        "roles": {
            "assistant": "服務生/餐廳工作人員",
            "user": "顧客"
        },
        "sample_dialog": {
            "assistant": "Hello, welcome to our restaurant. Are you ready to order?",
            "user": "Hi, yes. Could I see the menu please?",
            "next_prompt": "Of course, here's our menu. Today's special is grilled salmon with vegetables."
        }
    },
    "求職面試 (Job Interview)": {
        "description": "求職面試中的自我介紹和問答情境",
        "roles": {
            "assistant": "面試官/招聘人員",
            "user": "求職者"
        },
        "sample_dialog": {
            "assistant": "Thank you for coming in today. Could you tell us a bit about yourself?",
            "user": "Thank you for having me. I graduated from...",
            "next_prompt": "That's interesting. What would you say are your greatest strengths?"
        }
    },
    "日常社交 (Daily Social Conversation)": {
        "description": "日常問候、閒聊和社交對話情境",
        "roles": {
            "assistant": "朋友/同事",
            "user": "您自己"
        },
        "sample_dialog": {
            "assistant": "Hey there! How's your day going so far?",
            "user": "Hi! It's going well, thanks for asking. How about yours?",
            "next_prompt": "Pretty good! I just got back from that new coffee shop downtown."
        }
    },
    "醫療諮詢 (Medical Consultation)": {
        "description": "在診所或醫院與醫生進行病情諮詢的對話",
        "roles": {
            "assistant": "醫生/護士",
            "user": "病人"
        },
        "sample_dialog": {
            "assistant": "Good afternoon. What seems to be the problem today?",
            "user": "I've been having a headache for three days.",
            "next_prompt": "I see. Can you describe the pain and when it started?"
        }
    },
    "學術討論 (Academic Discussion)": {
        "description": "課堂或研討會中的學術討論和問答",
        "roles": {
            "assistant": "教授/演講者",
            "user": "學生/聽眾"
        },
        "sample_dialog": {
            "assistant": "The research shows significant results in this area. Does anyone have questions?",
            "user": "Yes, I'm wondering about the methodology used in the study.",
            "next_prompt": "That's a good question. The methodology involved a mixed-methods approach..."
        }
    }
}

# 初始化模型和處理器
print("正在初始化系統...")
GPU_MEMORY_LIMIT = 20
model_manager = get_model_manager(gpu_memory_limit=GPU_MEMORY_LIMIT)
conversation_manager = get_conversation_manager()
device_info = model_manager.get_device_info()

print(f"🔒 GPU記憶體限制: {GPU_MEMORY_LIMIT}GB")
print(f"📊 當前記憶體使用: {device_info.get('current_memory_usage', 0):.2f}GB")

# 全域變數
current_scenario_name = "機場對話 (Airport Conversation)"
current_language = "英文 (English)"
current_difficulty = "初級 (TOEIC 405-600分)"

def ensure_scenario_images():
    for example in scenario_examples:
        image_path = example["image_path"]
        if not os.path.exists(image_path):
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            colors = [
                (73, 109, 137),   # 機場 - 藍灰色
                (139, 69, 19),    # 餐廳 - 棕色
                (25, 25, 112),    # 面試 - 深藍色
                (34, 139, 34),    # 社交 - 森林綠
                (220, 20, 60),    # 醫療 - 深紅色
                (72, 61, 139)     # 學術 - 深紫色
            ]
            color_index = scenario_examples.index(example) % len(colors)
            img = Image.new('RGB', (400, 300), color=colors[color_index])
            img.save(image_path)
            print(f"已創建佔位圖片: {image_path}")

def generate_temp_audio(prefix):
    os.makedirs("temp_audio", exist_ok=True)
    temp_audio_path = os.path.join("temp_audio", f"{prefix}_{random.randint(1000, 9999)}.wav")

    if not os.path.exists(temp_audio_path):
        with wave.open(temp_audio_path, 'w') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(44100)
            duration = 2
            values = []
            for i in range(duration * 44100):
                freq = 440 + 100 * np.sin(i * 0.001)
                amplitude = 0.1 * np.exp(-i / 22050)
                values.append(int(32767.0 * amplitude * np.sin(i * 2 * np.pi * freq / 44100)))
            f.writeframes(struct.pack('h' * len(values), *values))

    return temp_audio_path

def get_memory_status():
    try:
        status = model_manager.get_memory_status()
        if status:
            gpu_info = status.get("gpu_memory", {})
            cpu_info = status.get("cpu_memory", {})
            limits = status.get("limits", {})
            
            status_text = "=== 記憶體使用狀況 ===\n"
            
            if gpu_info:
                for gpu_id, info in gpu_info.items():
                    usage_status = "🔴" if info['reserved'] > limits.get('gpu_limit_gb', 20) * 0.8 else "🟢"
                    status_text += f"{usage_status} {gpu_id}: {info['reserved']:.2f}GB / {info['total']:.2f}GB ({info['usage_percent']:.1f}%)\n"
            
            if cpu_info:
                cpu_status = "🔴" if cpu_info['process_usage'] > limits.get('cpu_limit_gb', 32) * 0.8 else "🟢"
                status_text += f"{cpu_status} CPU進程: {cpu_info['process_usage']:.2f}GB\n"
                status_text += f"🖥️  系統記憶體: {cpu_info['used']:.2f}GB / {cpu_info['total']:.2f}GB ({cpu_info['percent']:.1f}%)\n"
            
            status_text += f"\n設定限制:\n"
            status_text += f"GPU限制: {limits.get('gpu_limit_gb', 'N/A')}GB\n"
            status_text += f"CPU限制: {limits.get('cpu_limit_gb', 'N/A')}GB\n"
            status_text += f"監控狀態: {'✅ 運行中' if status.get('monitoring', False) else '❌ 未運行'}"
            
            return status_text
        else:
            return "記憶體監控未啟動或無法獲取狀態"
    except Exception as e:
        return f"獲取記憶體狀態失敗: {str(e)}"

def get_system_stats():
    stats = {
        "📊 總對話次數": len(conversation_manager.conversation_history),
        "🎤 Whisper狀態": "✅ 已載入" if device_info["whisper_available"] else "❌ 未載入",
        "🧠 Audio-LLM狀態": "✅ 已載入" if device_info["use_audio_llm"] else "❌ 未載入",
        "🚀 GPU加速": "✅ 已啟用" if device_info["use_gpu"] else "❌ 使用CPU",
        "🎭 當前場景": current_scenario_name,
        "🌍 學習語言": current_language,
        "📊 難度級別": current_difficulty,
        "💾 記憶體使用": f"{device_info.get('current_memory_usage', 0):.2f}GB / {device_info.get('memory_limit_gb', 0)}GB"
    }
    return stats

# 界面控制函數
def update_language_difficulty(language, difficulty):
    global current_language, current_difficulty
    current_language = language
    current_difficulty = difficulty
    return f"✅ 已設定語言: {language}, 難度: {difficulty}"

def show_preset_mode():
    return (
        gr.update(visible=False),  # 隱藏初始設定
        gr.update(visible=True),   # 顯示場景選擇
        gr.update(visible=False),  # 隱藏對話區域
        gr.update(visible=False),  # 隱藏自由對話
        gr.update(visible=True),   # 顯示返回按鈕
        "preset_selection"
    )

def show_free_dialog_mode():
    return (
        gr.update(visible=False),  # 隱藏初始設定
        gr.update(visible=False),  # 隱藏場景選擇
        gr.update(visible=False),  # 隱藏對話區域
        gr.update(visible=True),   # 顯示自由對話
        gr.update(visible=True),   # 顯示返回按鈕
        "free_dialog"
    )

def back_to_initial():
    return (
        gr.update(visible=True),   # 顯示初始設定
        gr.update(visible=False),  # 隱藏場景選擇
        gr.update(visible=False),  # 隱藏對話區域
        gr.update(visible=False),  # 隱藏自由對話
        gr.update(visible=False),  # 隱藏返回按鈕
        "initial"
    )

def select_scenario(example_index):
    global current_scenario_name
    
    selected = scenario_examples[example_index]
    preset_name = selected["scenario"]
    preset = scenario_presets[preset_name]
    
    current_scenario_name = preset_name

    return (
        gr.update(visible=False),  # 隱藏場景選擇
        gr.update(visible=True),   # 顯示對話區域
        preset["roles"]["assistant"],  # 助教角色
        preset["roles"]["user"],       # 用戶角色
        preset["sample_dialog"]["assistant"],  # 助教開場白
        f"🎭 當前場景：{selected['name']}"  # 場景標題
    )

def start_free_conversation(scenario_text):
    if not scenario_text or scenario_text.strip() == "":
        response_text = "請輸入您想要的場景或問題，以便我能更好地幫助您。"
        return response_text, gr.update(visible=False)

    response_text = f"您好！我是您的語言學習助教。我了解您提到的「{scenario_text}」。請開始對話吧！"

    return (
        response_text,
        gr.update(visible=True)
    )

def process_user_audio(audio_path, language, difficulty, focus_area, feedback_detail,
                      pronunciation_focus, accent_preference, track_progress):
    if audio_path is None:
        return "", "請先錄製您的回應", 0, 0, "", []

    try:
        print(f"處理音頻文件: {audio_path}")
        print(f"使用語言設定: {language}")
        print(f"使用難度設定: {difficulty}")
        
        conversation_context = conversation_manager.get_conversation_context()
        
        # 傳遞難度參數到處理函數
        result = conversation_manager.process_user_input(
            audio_path, 
            current_scenario_name, 
            conversation_context,
            difficulty  # 新增難度參數
        )
        
        if not result["success"]:
            return "", result["error_message"], 0, 0, "", []
        
        # 根據回饋詳細程度調整顯示內容
        if feedback_detail == "基本回饋":
            feedback = f"您說的是：'{result['recognized_text']}'\n發音得分：{result['pronunciation_score']}/100"
        elif feedback_detail == "詳細回饋":
            feedback = f"您說的是：'{result['recognized_text']}'\n\n發音分析：\n{result['pronunciation_analysis'][:300]}..."
        else:
            feedback = f"您說的是：'{result['recognized_text']}'\n\n詳細發音分析：\n{result['pronunciation_analysis']}"

        if pronunciation_focus:
            additional_tips = []
            if "子音發音" in pronunciation_focus:
                additional_tips.append("💡 注意子音的清晰發音")
            if "母音發音" in pronunciation_focus:
                additional_tips.append("💡 練習母音的準確度")
            if "語調" in pronunciation_focus:
                additional_tips.append("💡 注意語調的起伏變化")
            
            if additional_tips:
                feedback += "\n\n重點提醒：\n" + "\n".join(additional_tips)

        history_entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": current_scenario_name,
            "difficulty": difficulty,  # 記錄難度
            "score": result["pronunciation_score"],
            "feedback": feedback[:50] + "..." if len(feedback) > 50 else feedback
        }

        history = [history_entry]

        return (
            result["recognized_text"], 
            feedback, 
            result["pronunciation_score"], 
            result["fluency_score"],
            result["response_text"], 
            history
        )
        
    except Exception as e:
        print(f"處理用戶音頻時出錯: {e}")
        return "", f"處理過程出現錯誤: {str(e)}", 0, 0, "", []

def process_free_user_audio(audio_path, language, difficulty, scenario_text):
    if audio_path is None:
        return "", "請先錄製您的回應"

    try:
        # 傳遞難度參數到自由對話處理
        result = conversation_manager.process_user_input(
            audio_path, 
            "自由對話",
            f"Context: {scenario_text}",
            difficulty  # 新增難度參數
        )
        
        if not result["success"]:
            return "", result["error_message"]

        return result["recognized_text"], result["response_text"]
        
    except Exception as e:
        print(f"自由對話處理錯誤: {str(e)}")
        return "處理錯誤", "抱歉，處理您的語音時出現問題。請重試。"

def update_history(history):
    if not history:
        return [], []

    gallery_images = []
    for _ in range(min(len(history), 4)):
        gallery_images.append(random.choice([e["image_path"] for e in scenario_examples]))

    history_data = [
        [
            entry["timestamp"],
            entry["scenario"],
            entry.get("difficulty", "未記錄"),  # 顯示難度
            f"{entry['score']}分",
            entry["feedback"]
        ]
        for entry in history
    ]

    return gallery_images, history_data

def clear_conversation_history():
    conversation_manager.clear_history()
    return "✅ 對話歷史已清除", []

def export_conversation_history():
    if not conversation_manager.conversation_history:
        return "❌ 沒有對話記錄可導出"
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_history_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"語言學習助教 - 對話歷史記錄\n")
        f.write(f"導出時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"語言設定: {current_language}\n")
        f.write(f"難度設定: {current_difficulty}\n")
        f.write("=" * 50 + "\n\n")
        
        for entry in conversation_manager.conversation_history:
            f.write(f"時間: {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"場景: {entry['scenario']}\n")
            f.write(f"用戶: {entry['user']}\n")
            f.write(f"助教: {entry['assistant']}\n")
            f.write("-" * 50 + "\n")
    
    return f"✅ 對話歷史已導出至: {filename}"

# 初始化
ensure_scenario_images()

# 主Gradio界面
with gr.Blocks(css=css_content, title="語言學習助教", theme=gr.themes.Soft()) as demo:
    history_state = gr.State([])
    current_mode = gr.State("initial")

    # 主容器
    with gr.Column(elem_classes="main-container fade-in-up"):
        
        # 標題區域
        with gr.Column(elem_classes="header-section"):
            gr.HTML("""
                <div class="header-title">🗣️ 口說語言學習互動助教</div>
                <div class="header-subtitle">清大113061529 電機所碩一 楊傑翔 Final Project</div>
                <div style="margin-top: 15px; font-size: 1rem; opacity: 0.8;">
                    提升您的口語表達能力，接收即時專業發音回饋，輕鬆學習語言！
                </div>
            """)

        # 系統狀態顯示
        with gr.Column(elem_classes="status-card"):
            if device_info["use_gpu"]:
                current_usage = device_info.get('current_memory_usage', 0)
                status_html = f"""
                <div style="display: flex; align-items: center; gap: 20px; flex-wrap: wrap;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.2rem;">🚀</span>
                        <strong>GPU加速已啟用</strong>
                        <span style="color: #666;">- {device_info['gpu_name']} ({device_info['gpu_memory']:.1f}GB)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.2rem;">💾</span>
                        <strong>記憶體使用:</strong>
                        <span style="color: #4CAF50; font-weight: 600;">{current_usage:.2f}GB / {device_info['memory_limit_gb']}GB</span>
                    </div>
                """
                if device_info["use_audio_llm"]:
                    status_html += """
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.2rem;">✅</span>
                        <strong>Audio-LLM已載入</strong>
                    </div>
                    """
                else:
                    status_html += """
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.2rem;">⚠️</span>
                        <strong>Audio-LLM未載入</strong>
                        <span style="color: #666;">(使用簡化分析)</span>
                    </div>
                    """
                status_html += "</div>"
            else:
                status_html = """
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.2rem;">💻</span>
                    <strong>使用CPU模式</strong>
                </div>
                """
            gr.HTML(status_html)

        # 1. 初始設定區域（語言和難度選擇）
        with gr.Column(elem_classes="initial-settings", visible=True) as initial_settings:
            gr.HTML("<h3 style='text-align: center; margin-bottom: 25px; color: #374151;'>⚙️ 系統設定</h3>")
            
            with gr.Row():
                language = gr.Dropdown(
                    ["英文 (English)"], 
                    label="🌍 學習語言", 
                    value="英文 (English)",
                    elem_classes="gradio-dropdown"
                )
                difficulty = gr.Dropdown([
                    "初學者 (TOEIC 250-400分)",
                    "初級 (TOEIC 405-600分)",
                    "中級 (TOEIC 605-780分)",
                    "中高級 (TOEIC 785-900分)",
                    "高級 (TOEIC 905+分)"
                ], label="📊 難度級別", value="初級 (TOEIC 405-600分)",
                elem_classes="gradio-dropdown")
            
            # 設定確認按鈕
            settings_status = gr.Textbox(label="設定狀態", interactive=False, visible=False)
            confirm_settings_btn = gr.Button("✅ 確認設定", elem_classes="primary-btn")

            gr.HTML("<h3 style='text-align: center; margin: 30px 0 20px 0; color: #374151;'>🎯 選擇對話模式</h3>")
            
            with gr.Row():
                preset_scenario_btn = gr.Button(
                    "🎭 預設場景對話", 
                    elem_classes="mode-btn"
                )
                free_dialog_btn = gr.Button(
                    "💭 自由對話", 
                    elem_classes="mode-btn"
                )

        # 2. 預設場景選擇區域
        with gr.Column(visible=False, elem_classes="fade-in-up") as preset_scenario_selection:
            gr.HTML("<h2 style='text-align: center; margin-bottom: 30px; color: #374151;'>🎭 選擇練習場景</h2>")
            
            with gr.Row(equal_height=True):
                for i, example in enumerate(scenario_examples):
                    with gr.Column(elem_classes="scenario-card"):
                        gr.HTML(f"""
                            <div style="text-align: center; margin-bottom: 15px;">
                                <div style="font-size: 2rem; margin-bottom: 8px;">{example['icon']}</div>
                                <h3 style="margin: 0; color: #374151;">{example['name']}</h3>
                            </div>
                        """)
                        
                        gr.Image(
                            example["image_path"], 
                            height=200, 
                            show_label=False,
                            container=False,
                            elem_classes="scenario-image"
                        )
                        
                        gr.HTML(f"<p style='text-align: center; color: #6b7280; margin: 10px 0;'>{example['preview_text']}</p>")
                        
                        if i == 0:
                            scenario_btn_1 = gr.Button("選擇此場景", elem_classes="scenario-btn")
                        elif i == 1:
                            scenario_btn_2 = gr.Button("選擇此場景", elem_classes="scenario-btn")
                        elif i == 2:
                            scenario_btn_3 = gr.Button("選擇此場景", elem_classes="scenario-btn")
                        elif i == 3:
                            scenario_btn_4 = gr.Button("選擇此場景", elem_classes="scenario-btn")
                        elif i == 4:
                            scenario_btn_5 = gr.Button("選擇此場景", elem_classes="scenario-btn")
                        elif i == 5:
                            scenario_btn_6 = gr.Button("選擇此場景", elem_classes="scenario-btn")

        # 3. 預設場景對話區域（選擇場景後才顯示）
        with gr.Column(visible=False, elem_classes="fade-in-up") as preset_conversation_area:
            scenario_title = gr.HTML("<h2 style='text-align: center; margin-bottom: 20px; color: #374151;'>🎭 當前場景</h2>")
            
            # 角色設定
            with gr.Row():
                assistant_role = gr.Textbox(
                    label="🤖 助教角色",
                    interactive=False,
                    elem_classes="gradio-textbox"
                )
                user_role = gr.Textbox(
                    label="👤 您的角色",
                    interactive=False,
                    elem_classes="gradio-textbox"
                )

            # 對話區域
            with gr.Column(elem_classes="conversation-area"):
                gr.HTML("<h3 style='margin-bottom: 20px; color: #374151;'>💬 對話區域</h3>")
                
                with gr.Column(elem_classes="dialog-box"):
                    assistant_text = gr.Textbox(
                        label="🤖 助教",
                        lines=3,
                        interactive=False,
                        elem_classes="gradio-textbox"
                    )

                    with gr.Column(elem_classes="user-input"):
                        user_audio_input = gr.Audio(
                            label="🎤 錄製您的回應", 
                            type="filepath", 
                            sources=["microphone"],
                            elem_classes="gradio-audio"
                        )
                        
                        user_text = gr.Textbox(
                            label="📝 語音識別結果", 
                            interactive=False,
                            elem_classes="gradio-textbox"
                        )

                        with gr.Row():
                            retry_btn = gr.Button("🔄 重新錄製", elem_classes="secondary-btn")
                            submit_audio_btn = gr.Button("🚀 提交回應", elem_classes="primary-btn")

            # 回饋區域 - 擴大對話框
            with gr.Accordion("📝 發音回饋與分析", open=True, elem_classes="advanced-section"):
                with gr.Column(elem_classes="feedback-panel"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            feedback_text = gr.Textbox(
                                label="📋 詳細評估與改進建議", 
                                lines=8,  # 增加行數讓對話框更大
                                interactive=False,
                                elem_classes="gradio-textbox"
                            )

                        with gr.Column(scale=1):
                            with gr.Row():
                                pronunciation_score = gr.Number(
                                    label="🎯 發音得分", 
                                    value=0, 
                                    interactive=False,
                                    elem_classes="score-display"
                                )
                                fluency_score = gr.Number(
                                    label="⚡ 流暢度", 
                                    value=0, 
                                    interactive=False,
                                    elem_classes="score-display"
                                )

            # 進階設定
            with gr.Accordion("⚙️ 進階功能設定", open=False, elem_classes="advanced-section"):
                gr.HTML("<h4 style='margin: 15px 0; color: #374151;'>🔊 發音評估設定</h4>")
                
                with gr.Row():
                    with gr.Column():
                        pronunciation_focus = gr.CheckboxGroup(
                            ["子音發音", "母音發音", "連音", "重音", "語調", "節奏"],
                            value=["子音發音", "母音發音", "語調"],
                            label="🎯 發音重點關注"
                        )

                    with gr.Column():
                        accent_preference = gr.Radio(
                            ["美式英文", "英式英文", "不指定"],
                            value="美式英文",
                            label="🌍 發音口音偏好"
                        )

                gr.HTML("<h4 style='margin: 25px 0 15px 0; color: #374151;'>📈 學習追蹤</h4>")
                
                with gr.Row():
                    track_progress = gr.Checkbox(label="📊 記錄學習進度", value=True)
                    show_comparison = gr.Checkbox(label="📋 顯示與標準發音比較", value=True)
                    
                with gr.Row():
                    feedback_detail = gr.Radio(
                        ["基本回饋", "詳細回饋", "專家級分析"],
                        value="詳細回饋",
                        label="📝 回饋詳細程度"
                    )

                with gr.Accordion("📊 學習歷程", open=False):
                    with gr.Row():
                        clear_history_btn = gr.Button("🗑️ 清除歷史", elem_classes="secondary-btn")
                        export_history_btn = gr.Button("📥 導出歷史", elem_classes="secondary-btn")
                    
                    history_status = gr.Textbox(label="操作狀態", interactive=False)
                    
                    history_gallery = gr.Gallery(
                        label="最近練習的對話", 
                        columns=4, 
                        object_fit="contain", 
                        height="200px"
                    )
                    
                    history_info = gr.Dataframe(
                        headers=["時間", "場景", "難度", "得分", "重點改進項目"],
                        datatype=["str", "str", "str", "str", "str"],
                        label="練習記錄"
                    )

        # 4. 自由對話模式
        with gr.Column(visible=False, elem_classes="fade-in-up") as free_dialog_mode:
            gr.HTML("<h2 style='text-align: center; margin-bottom: 30px; color: #374151;'>💭 自由對話</h2>")

            custom_scenario = gr.Textbox(
                label="📝 描述您想要的場景或問題",
                placeholder="例如：我想練習在餐廳點餐的對話，我是客人，助教扮演服務生；或者直接提問：如何改善我的英文發音？",
                lines=4,
                elem_classes="gradio-textbox"
            )

            start_free_dialog_btn = gr.Button("🚀 開始對話", elem_classes="primary-btn")

            with gr.Column(elem_classes="conversation-area", visible=False) as free_dialog_area:
                free_assistant_text = gr.Textbox(
                    label="🤖 助教回應", 
                    lines=4,
                    interactive=False,
                    elem_classes="gradio-textbox"
                )

                with gr.Column(elem_classes="user-input"):
                    free_user_audio_input = gr.Audio(
                        label="🎤 錄製您的回應", 
                        type="filepath", 
                        sources=["microphone"],
                        elem_classes="gradio-audio"
                    )
                    free_user_text = gr.Textbox(
                        label="📝 語音識別結果", 
                        interactive=False,
                        elem_classes="gradio-textbox"
                    )

                    with gr.Row():
                        free_retry_btn = gr.Button("🔄 重新錄製", elem_classes="secondary-btn")
                        free_submit_audio_btn = gr.Button("🚀 提交回應", elem_classes="primary-btn")

        # 返回按鈕
        with gr.Column(visible=False) as back_btn_group:
            back_btn = gr.Button("← 返回主選單", elem_classes="back-btn")

        # 系統監控和統計
        with gr.Accordion("📊 系統監控與統計", open=False, elem_classes="advanced-section"):
            with gr.Row():
                with gr.Column():
                    gr.HTML("<h4 style='margin-bottom: 15px; color: #374151;'>💾 記憶體使用狀況</h4>")
                    memory_status_display = gr.Textbox(
                        label="記憶體監控",
                        lines=8,
                        interactive=False,
                        elem_classes="memory-info"
                    )
                    memory_refresh_btn = gr.Button("🔄 刷新記憶體狀態", elem_classes="secondary-btn")
                    
                with gr.Column():
                    gr.HTML("<h4 style='margin-bottom: 15px; color: #374151;'>📈 系統統計</h4>")
                    stats_display = gr.JSON(label="系統狀態", elem_classes="stats-panel")
                    stats_refresh_btn = gr.Button("🔄 刷新統計", elem_classes="secondary-btn")

    # 隱藏的默認值組件
    default_focus_area = gr.Textbox(value="綜合練習", visible=False)

    # === 事件綁定 ===
    
    # 設定確認 - 修復難度選擇問題
    def update_settings_and_show_status(lang, diff):
        global current_language, current_difficulty
        current_language = lang
        current_difficulty = diff
        status_msg = f"✅ 已設定語言: {lang}, 難度: {diff}"
        print(f"設定更新: 語言={lang}, 難度={diff}")  # 調試用
        return status_msg, gr.update(visible=True)
    
    confirm_settings_btn.click(
        fn=update_settings_and_show_status,
        inputs=[language, difficulty],
        outputs=[settings_status, settings_status]
    )
    
    # 語言和難度改變時也自動更新（即時更新）
    language.change(
        fn=lambda lang, diff: update_language_difficulty(lang, diff),
        inputs=[language, difficulty],
        outputs=[]
    )
    
    difficulty.change(
        fn=lambda lang, diff: update_language_difficulty(lang, diff),
        inputs=[language, difficulty], 
        outputs=[]
    )
    
    # 模式選擇
    preset_scenario_btn.click(
        fn=show_preset_mode,
        outputs=[initial_settings, preset_scenario_selection, preset_conversation_area, free_dialog_mode, back_btn_group, current_mode]
    )

    free_dialog_btn.click(
        fn=show_free_dialog_mode,
        outputs=[initial_settings, preset_scenario_selection, preset_conversation_area, free_dialog_mode, back_btn_group, current_mode]
    )

    back_btn.click(
        fn=back_to_initial,
        outputs=[initial_settings, preset_scenario_selection, preset_conversation_area, free_dialog_mode, back_btn_group, current_mode]
    )

    # 場景選擇按鈕
    scenario_btn_1.click(
        fn=select_scenario,
        inputs=[gr.Number(value=0, visible=False)],
        outputs=[preset_scenario_selection, preset_conversation_area, assistant_role, user_role, assistant_text, scenario_title]
    )

    scenario_btn_2.click(
        fn=select_scenario,
        inputs=[gr.Number(value=1, visible=False)],
        outputs=[preset_scenario_selection, preset_conversation_area, assistant_role, user_role, assistant_text, scenario_title]
    )

    scenario_btn_3.click(
        fn=select_scenario,
        inputs=[gr.Number(value=2, visible=False)],
        outputs=[preset_scenario_selection, preset_conversation_area, assistant_role, user_role, assistant_text, scenario_title]
    )

    scenario_btn_4.click(
        fn=select_scenario,
        inputs=[gr.Number(value=3, visible=False)],
        outputs=[preset_scenario_selection, preset_conversation_area, assistant_role, user_role, assistant_text, scenario_title]
    )

    scenario_btn_5.click(
        fn=select_scenario,
        inputs=[gr.Number(value=4, visible=False)],
        outputs=[preset_scenario_selection, preset_conversation_area, assistant_role, user_role, assistant_text, scenario_title]
    )

    scenario_btn_6.click(
        fn=select_scenario,
        inputs=[gr.Number(value=5, visible=False)],
        outputs=[preset_scenario_selection, preset_conversation_area, assistant_role, user_role, assistant_text, scenario_title]
    )

    # 自由對話按鈕
    start_free_dialog_btn.click(
        fn=start_free_conversation,
        inputs=[custom_scenario],
        outputs=[free_assistant_text, free_dialog_area]
    )

    # 提交音頻回應 - 預設場景
    submit_audio_btn.click(
        fn=process_user_audio,
        inputs=[
            user_audio_input, language, difficulty,
            default_focus_area, feedback_detail,
            pronunciation_focus, accent_preference, track_progress
        ],
        outputs=[
            user_text, feedback_text, pronunciation_score,
            fluency_score, assistant_text, history_state
        ]
    ).then(
        fn=update_history,
        inputs=[history_state],
        outputs=[history_gallery, history_info]
    )

    # 提交音頻回應 - 自由對話
    free_submit_audio_btn.click(
        fn=process_free_user_audio,
        inputs=[free_user_audio_input, language, difficulty, custom_scenario],
        outputs=[free_user_text, free_assistant_text]
    )

    # 重試按鈕
    retry_btn.click(
        fn=lambda: [None, ""],
        outputs=[user_audio_input, user_text]
    )

    free_retry_btn.click(
        fn=lambda: [None, ""],
        outputs=[free_user_audio_input, free_user_text]
    )
    
    # 歷史記錄管理
    clear_history_btn.click(
        fn=clear_conversation_history,
        outputs=[history_status, history_info]
    )
    
    export_history_btn.click(
        fn=export_conversation_history,
        outputs=[history_status]
    )
    
    # 記憶體狀態查看
    memory_refresh_btn.click(
        fn=get_memory_status,
        outputs=[memory_status_display]
    )
    
    # 系統統計查看
    stats_refresh_btn.click(
        fn=get_system_stats,
        outputs=[stats_display]
    )

    # 初始載入狀態
    demo.load(
        fn=get_memory_status,
        outputs=[memory_status_display]
    )
    
    demo.load(
        fn=get_system_stats,
        outputs=[stats_display]
    )

if __name__ == "__main__":
    print("=== 啟動語言學習助教（無TTS版本）===")
    print(f"使用設備: {device_info['device']}")
    print(f"Whisper可用: {device_info['whisper_available']}")
    print(f"Audio-LLM可用: {device_info['use_audio_llm']}")
    
    if css_content:
        print("✅ CSS樣式文件載入成功")
    else:
        print("⚠️  CSS樣式文件載入失敗，使用默認樣式")
    
    # 啟動參數
    launch_kwargs = {
        "share": True,
        "debug": True,
        "server_name": "0.0.0.0",
        "server_port": 7861,
        "show_error": True,
        "favicon_path": None,
        "ssl_verify": False
    }
    
    if device_info["use_gpu"]:
        print("GPU模式已啟用，建議確保有足夠的VRAM")
        print(f"當前GPU記憶體使用: {device_info.get('current_memory_usage', 0):.2f}GB")
    
    try:
        demo.launch(**launch_kwargs)
    except Exception as e:
        print(f"啟動失敗: {e}")
        print("嘗試使用自動端口...")
        launch_kwargs.pop("server_port")
        demo.launch(**launch_kwargs)