#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_advanced_features.py - 進階功能完整測試腳本
"""

import sys
import os
import tempfile
import wave
import struct
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processors import get_conversation_manager, create_advanced_prompt, DIFFICULTY_CONFIGS

def create_test_audio_file(duration=2, frequency=440, sample_rate=44100):
    """創建測試用音頻文件"""
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_file.close()
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(frequency * 2 * np.pi * t) * 0.3
    
    with wave.open(temp_file.name, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wav_file.writeframes(audio_int16.tobytes())
    
    return temp_file.name

def test_prompt_generation():
    """測試 System Prompt 生成功能"""
    print("🧪 測試 System Prompt 生成...")
    
    test_cases = [
        {
            "name": "初學者 + 基本設定",
            "scenario": "機場對話 (Airport Conversation)",
            "difficulty": "初學者 (TOEIC 250-400分)",
            "pronunciation_focus": ["子音發音", "母音發音"],
            "accent_preference": "美式英文",
            "feedback_detail": "基本回饋",
            "show_comparison": True
        },
        {
            "name": "高級 + 專家分析",
            "scenario": "求職面試 (Job Interview)",
            "difficulty": "高級 (TOEIC 905+分)",
            "pronunciation_focus": ["語調", "節奏", "連音"],
            "accent_preference": "英式英文",
            "feedback_detail": "專家級分析",
            "show_comparison": True
        },
        {
            "name": "中級 + 全功能",
            "scenario": "餐廳點餐 (Restaurant Ordering)",
            "difficulty": "中級 (TOEIC 605-780分)",
            "pronunciation_focus": ["子音發音", "母音發音", "語調", "重音"],
            "accent_preference": "不指定",
            "feedback_detail": "詳細回饋",
            "show_comparison": True
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- 測試案例 {i}: {test_case['name']} ---")
        
        prompt = create_advanced_prompt(
            scenario=test_case["scenario"],
            difficulty=test_case["difficulty"],
            pronunciation_focus=test_case["pronunciation_focus"],
            accent_preference=test_case["accent_preference"],
            feedback_detail=test_case["feedback_detail"],
            show_comparison=test_case["show_comparison"],
            conversation_history="Test context"
        )
        
        print(f"✅ Prompt 長度: {len(prompt)} 字符")
        
        checks = [
            (test_case["difficulty"].split()[0], "難度級別"),
            (test_case["accent_preference"], "口音偏好"),
            (test_case["feedback_detail"], "回饋級別"),
            ("SUGGESTED NEXT RESPONSES", "建議回覆功能")
        ]
        
        for keyword, description in checks:
            if keyword in prompt:
                print(f"  ✅ {description}: 已包含")
            else:
                print(f"  ❌ {description}: 未包含")
        
        focus_found = sum(1 for focus in test_case["pronunciation_focus"] if focus in prompt)
        print(f"  ✅ 發音重點: {focus_found}/{len(test_case['pronunciation_focus'])} 項已包含")
        
        print(f"  📄 Prompt 預覽:")
        print(f"    {prompt[:200]}...")

def test_conversation_manager():
    """測試對話管理器的進階功能整合"""
    print("\n🧪 測試對話管理器...")
    
    test_audio_path = create_test_audio_file()
    
    try:
        manager = get_conversation_manager()
        
        test_params = {
            "audio_path": test_audio_path,
            "scenario": "機場對話 (Airport Conversation)",
            "conversation_context": "Test conversation context",
            "difficulty": "中級 (TOEIC 605-780分)",
            "pronunciation_focus": ["子音發音", "語調"],
            "accent_preference": "美式英文",
            "feedback_detail": "詳細回饋",
            "show_comparison": True
        }
        
        print("📤 發送測試請求...")
        print(f"  音頻文件: {test_audio_path}")
        print(f"  場景: {test_params['scenario']}")
        print(f"  難度: {test_params['difficulty']}")
        print(f"  發音重點: {test_params['pronunciation_focus']}")
        print(f"  口音偏好: {test_params['accent_preference']}")
        print(f"  回饋級別: {test_params['feedback_detail']}")
        
        result = manager.process_user_input(**test_params)
        
        print("\n📥 處理結果:")
        print(f"  ✅ 處理成功: {result['success']}")
        
        if result['success']:
            print(f"  📝 識別文字: {result.get('recognized_text', 'N/A')}")
            print(f"  🎯 發音得分: {result.get('pronunciation_score', 0)}/100")
            print(f"  ⚡ 流暢度: {result.get('fluency_score', 0)}/100")
            print(f"  🤖 助教回應: {result.get('response_text', 'N/A')[:100]}...")
            
            if 'suggested_responses' in result and result['suggested_responses']:
                print(f"  💡 建議回覆數量: {len(result['suggested_responses'])}")
                for i, suggestion in enumerate(result['suggested_responses'][:2], 1):
                    print(f"    {i}. {suggestion}")
            else:
                print(f"  ⚠️  建議回覆: 未生成")
            
            analysis = result.get('pronunciation_analysis', '')
            if analysis:
                print(f"  📊 分析內容長度: {len(analysis)} 字符")
                print(f"  📊 分析預覽: {analysis[:150]}...")
            else:
                print(f"  ⚠️  發音分析: 未生成")
        else:
            print(f"  ❌ 錯誤訊息: {result.get('error_message', 'Unknown error')}")
    
    finally:
        if os.path.exists(test_audio_path):
            os.unlink(test_audio_path)
            print(f"🗑️  已清理測試文件: {test_audio_path}")

def test_difficulty_configs():
    """測試難度配置系統"""
    print("\n🧪 測試難度配置系統...")
    
    for difficulty_name, config in DIFFICULTY_CONFIGS.items():
        print(f"\n--- {difficulty_name} ---")
        print(f"  級別: {config['level']}")
        print(f"  TOEIC範圍: {config['toeic_range']}")
        print(f"  描述: {config['description']}")
        print(f"  評估標準: {config['evaluation_criteria']}")
        print(f"  鼓勵程度: {config['encouragement_level']}")
        print(f"  分數調整: {config['score_adjustment']:+d}")
        print(f"  詞彙水平: {config['vocabulary_level']}")
        print(f"  句型複雜度: {config['sentence_complexity']}")

def test_scenario_responses():
    """測試場景回應生成"""
    print("\n🧪 測試場景回應生成...")
    
    from processors import get_scenario_responses
    
    scenarios = [
        "機場對話 (Airport Conversation)",
        "餐廳點餐 (Restaurant Ordering)",
        "求職面試 (Job Interview)",
        "日常社交 (Daily Social Conversation)"
    ]
    
    difficulties = ["初學者 (TOEIC 250-400分)", "中級 (TOEIC 605-780分)", "高級 (TOEIC 905+分)"]
    
    for scenario in scenarios:
        print(f"\n--- {scenario} ---")
        for difficulty in difficulties:
            responses = get_scenario_responses(scenario, difficulty)
            print(f"  {difficulty}: {len(responses)} 個回應")
            if responses:
                print(f"    範例: {responses[0]}")

def test_memory_integration():
    """測試記憶體監控整合"""
    print("\n🧪 測試記憶體監控整合...")
    
    try:
        from models import get_model_manager
        manager = get_model_manager()
        
        device_info = manager.get_device_info()
        print("💾 設備資訊:")
        for key, value in device_info.items():
            print(f"  {key}: {value}")
        
        memory_status = manager.get_memory_status()
        if memory_status:
            print("\n📊 記憶體狀態:")
            print(f"  監控運行: {memory_status.get('monitoring', False)}")
            
            gpu_memory = memory_status.get('gpu_memory', {})
            if gpu_memory:
                for gpu_id, info in gpu_memory.items():
                    print(f"  {gpu_id}: {info['reserved']:.2f}GB / {info['total']:.2f}GB")
        else:
            print("⚠️  記憶體監控未啟動")
    
    except Exception as e:
        print(f"❌ 記憶體監控測試失敗: {e}")

def main():
    """主測試函數"""
    print("🚀 開始進階功能完整測試")
    print(f"📅 測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # 1. 測試 Prompt 生成
        test_prompt_generation()
        
        # 2. 測試難度配置
        test_difficulty_configs()
        
        # 3. 測試場景回應
        test_scenario_responses()
        
        # 4. 測試記憶體整合
        test_memory_integration()
        
        # 5. 測試對話管理器
        print("\n" + "="*60)
        print("⚠️  以下測試需要載入模型，可能需要較長時間...")
        user_input = input("是否繼續進行對話管理器測試？(y/n): ")
        
        if user_input.lower() == 'y':
            test_conversation_manager()
        else:
            print("⏩ 跳過對話管理器測試")
        
        print("\n" + "="*60)
        print("✅ 所有測試完成！")
        
    except KeyboardInterrupt:
        print("\n❌ 測試被用戶中斷")
    except Exception as e:
        print(f"\n❌ 測試過程中出現錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()