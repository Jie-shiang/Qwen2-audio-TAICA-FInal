# -*- coding: utf-8 -*-
"""
processors.py - 音頻處理和分析核心
負責音頻處理、語音識別、發音分析等功能
"""

import os
import random
import re
import datetime
from models import get_model_manager

DIFFICULTY_CONFIGS = {
    "初學者 (TOEIC 250-400分)": {
        "level": "beginner",
        "toeic_range": "250-400",
        "description": "Basic vocabulary and simple sentence structures",
        "evaluation_criteria": "Focus on basic pronunciation clarity and simple grammar",
        "encouragement_level": "very_encouraging",
        "score_adjustment": +15,
        "vocabulary_level": "basic",
        "sentence_complexity": "simple"
    },
    "初級 (TOEIC 405-600分)": {
        "level": "elementary",
        "toeic_range": "405-600",
        "description": "Elementary vocabulary with basic conversational skills",
        "evaluation_criteria": "Evaluate basic conversation flow and pronunciation accuracy",
        "encouragement_level": "encouraging",
        "score_adjustment": +10,
        "vocabulary_level": "elementary",
        "sentence_complexity": "basic"
    },
    "中級 (TOEIC 605-780分)": {
        "level": "intermediate",
        "toeic_range": "605-780",
        "description": "Intermediate vocabulary and complex sentence structures",
        "evaluation_criteria": "Assess fluency, natural expression, and grammar accuracy",
        "encouragement_level": "balanced",
        "score_adjustment": 0,
        "vocabulary_level": "intermediate",
        "sentence_complexity": "moderate"
    },
    "中高級 (TOEIC 785-900分)": {
        "level": "upper_intermediate",
        "toeic_range": "785-900",
        "description": "Advanced vocabulary with nuanced expressions",
        "evaluation_criteria": "Focus on natural flow, idiomatic expressions, and subtle pronunciation",
        "encouragement_level": "constructive",
        "score_adjustment": -5,
        "vocabulary_level": "advanced",
        "sentence_complexity": "complex"
    },
    "高級 (TOEIC 905+分)": {
        "level": "advanced",
        "toeic_range": "905+",
        "description": "Professional-level vocabulary and sophisticated expressions",
        "evaluation_criteria": "Evaluate native-like fluency, sophisticated vocabulary usage, and professional communication",
        "encouragement_level": "detailed",
        "score_adjustment": -10,
        "vocabulary_level": "professional",
        "sentence_complexity": "sophisticated"
    }
}

def create_advanced_prompt(scenario, difficulty, pronunciation_focus, accent_preference, 
                          feedback_detail, show_comparison, conversation_history=""):
    """創建整合進階功能的完整 prompt"""
    
    difficulty_config = DIFFICULTY_CONFIGS.get(difficulty, DIFFICULTY_CONFIGS["中級 (TOEIC 605-780分)"])
    level = difficulty_config["level"]
    criteria = difficulty_config["evaluation_criteria"]
    encouragement = difficulty_config["encouragement_level"]
    vocab_level = difficulty_config["vocabulary_level"]
    sentence_complexity = difficulty_config["sentence_complexity"]
    
    pronunciation_instructions = ""
    if pronunciation_focus:
        focus_areas = []
        if "子音發音" in pronunciation_focus:
            focus_areas.append("consonant clarity and accuracy")
        if "母音發音" in pronunciation_focus:
            focus_areas.append("vowel precision and positioning")
        if "連音" in pronunciation_focus:
            focus_areas.append("linking sounds and connected speech")
        if "重音" in pronunciation_focus:
            focus_areas.append("word stress and sentence stress patterns")
        if "語調" in pronunciation_focus:
            focus_areas.append("intonation patterns and pitch variation")
        if "節奏" in pronunciation_focus:
            focus_areas.append("rhythm, pacing, and natural flow")
        
        if focus_areas:
            pronunciation_instructions = f"""
PRONUNCIATION FOCUS AREAS (Priority Analysis):
- Pay special attention to: {', '.join(focus_areas)}
- Provide specific feedback on these aspects
- Give targeted improvement suggestions for these areas
"""

    accent_instructions = ""
    if accent_preference == "美式英文":
        accent_instructions = """
ACCENT TARGET: American English (General American)
- Evaluate based on American pronunciation standards
- Focus on rhotic 'r' sounds, flat 'a' in words like 'dance'
- American intonation patterns and stress
"""
    elif accent_preference == "英式英文":
        accent_instructions = """
ACCENT TARGET: British English (Received Pronunciation)
- Evaluate based on British pronunciation standards  
- Focus on non-rhotic features, long 'a' in words like 'dance'
- British intonation patterns and received pronunciation
"""
    else:
        accent_instructions = """
ACCENT APPROACH: Flexible/International English
- Accept both American and British variations
- Focus on clarity and intelligibility over specific accent
"""

    feedback_instructions = ""
    if feedback_detail == "基本回饋":
        feedback_instructions = """
FEEDBACK LEVEL: Basic (Concise)
- Provide simple, easy-to-understand feedback
- Focus on 1-2 main improvement points
- Keep suggestions practical and actionable
"""
    elif feedback_detail == "詳細回饋":
        feedback_instructions = """
FEEDBACK LEVEL: Detailed (Comprehensive)
- Provide thorough analysis of pronunciation aspects
- Include specific examples and comparisons
- Offer multiple improvement strategies
"""
    else:
        feedback_instructions = """
FEEDBACK LEVEL: Expert Analysis (In-depth)
- Provide linguistic analysis of pronunciation features
- Include phonetic explanations and technical details
- Offer advanced practice techniques and exercises
"""

    scenario_base_prompts = {
        "機場對話 (Airport Conversation)": f"""You are an airport staff member helping a traveler at {level} English level (TOEIC {difficulty_config['toeic_range']}). 

ROLE & SCENARIO: Airport staff assisting with check-in, security, customs, or boarding procedures.

LANGUAGE LEVEL: Use {vocab_level} vocabulary and {sentence_complexity} sentence structures appropriate for {level} learners.""",

        "餐廳點餐 (Restaurant Ordering)": f"""You are a restaurant server taking orders from a {level} English learner (TOEIC {difficulty_config['toeic_range']}). 

ROLE & SCENARIO: Friendly restaurant server helping with menu selection, taking orders, and providing dining assistance.

LANGUAGE LEVEL: Use {vocab_level} vocabulary and {sentence_complexity} sentence structures appropriate for {level} learners.""",

        "求職面試 (Job Interview)": f"""You are a professional interviewer speaking with a {level} English candidate (TOEIC {difficulty_config['toeic_range']}). 

ROLE & SCENARIO: Professional interviewer conducting a job interview, asking relevant questions and providing follow-ups.

LANGUAGE LEVEL: Use {vocab_level} vocabulary and {sentence_complexity} sentence structures appropriate for {level} learners.""",

        "日常社交 (Daily Social Conversation)": f"""You are a friendly conversation partner with a {level} English speaker (TOEIC {difficulty_config['toeic_range']}). 

ROLE & SCENARIO: Casual friend or acquaintance engaging in everyday social conversation.

LANGUAGE LEVEL: Use {vocab_level} vocabulary and {sentence_complexity} sentence structures appropriate for {level} learners.""",

        "醫療諮詢 (Medical Consultation)": f"""You are a healthcare professional speaking with a {level} English patient (TOEIC {difficulty_config['toeic_range']}). 

ROLE & SCENARIO: Doctor, nurse, or medical staff conducting consultation and providing medical guidance.

LANGUAGE LEVEL: Use {vocab_level} vocabulary and {sentence_complexity} sentence structures appropriate for {level} learners.""",

        "學術討論 (Academic Discussion)": f"""You are an academic professional (professor/researcher) with a {level} English student (TOEIC {difficulty_config['toeic_range']}). 

ROLE & SCENARIO: Academic setting with professor or researcher engaging in educational discussion.

LANGUAGE LEVEL: Use {vocab_level} vocabulary and {sentence_complexity} sentence structures appropriate for {level} learners.""",

        "自由對話": f"""You are a helpful language learning assistant engaging with a {level} English learner (TOEIC {difficulty_config['toeic_range']}).

ROLE & SCENARIO: Adaptive conversation partner for the user's specified scenario or topic.

LANGUAGE LEVEL: Use {vocab_level} vocabulary and {sentence_complexity} sentence structures appropriate for {level} learners."""
    }

    base_prompt = scenario_base_prompts.get(scenario, scenario_base_prompts["日常社交 (Daily Social Conversation)"])

    system_prompt = f"""{base_prompt}

{pronunciation_instructions}
{accent_instructions}
{feedback_instructions}

ANALYSIS REQUIREMENTS:
1. {criteria}
2. Provide {encouragement} feedback with {difficulty_config['encouragement_level']} tone
3. Use examples appropriate for {level} proficiency level
4. Focus on practical improvement suggestions

RESPONSE FORMAT:
Your response must include EXACTLY these sections:

**PRONUNCIATION ANALYSIS:**
- Overall pronunciation score: [X]/100
- Specific feedback on the student's pronunciation quality
- Highlight both strengths and areas for improvement
{pronunciation_instructions.replace('PRONUNCIATION FOCUS AREAS (Priority Analysis):', '- Special attention to:') if pronunciation_focus else ''}

**CONVERSATION RESPONSE:**
[Your natural response as the role character, continuing the conversation]

**SUGGESTED NEXT RESPONSES:**
Provide 2-3 suggested responses the student could use to continue this conversation:
1. [First suggestion - basic response]
2. [Second suggestion - intermediate response] 
3. [Third suggestion - more advanced response]

Each suggestion should be appropriate for the {level} level and include brief explanations of when to use each option.

CONVERSATION CONTEXT: {conversation_history}"""

    return system_prompt

def get_scenario_responses(scenario, difficulty="中級 (TOEIC 605-780分)"):
    """獲取基於難度的場景回應（備用簡化模式使用）"""
    
    difficulty_config = DIFFICULTY_CONFIGS.get(difficulty, DIFFICULTY_CONFIGS["中級 (TOEIC 605-780分)"])
    level = difficulty_config["level"]
    
    responses_by_level = {
        "beginner": {
            "機場對話 (Airport Conversation)": [
                "Hello! Passport, please?",
                "Where are you going today?", 
                "How long will you stay?",
                "Any bags to check?",
                "Gate 12. Have a nice flight!",
                "Thank you. Next, please!"
            ],
            "餐廳點餐 (Restaurant Ordering)": [
                "Hi! Table for how many?",
                "Here's the menu. Take your time.",
                "Ready to order?",
                "What would you like to drink?",
                "Great choice! Anything else?",
                "Your meal will be ready soon."
            ],
            "求職面試 (Job Interview)": [
                "Nice to meet you. Please sit down.",
                "Tell me about yourself.",
                "Why do you want this job?",
                "What are your strengths?",
                "Do you have questions for us?",
                "Thank you for coming today."
            ],
            "日常社交 (Daily Social Conversation)": [
                "Hi! How are you today?",
                "Nice weather, isn't it?",
                "What do you do for work?",
                "Do you live around here?",
                "Have a great day!",
                "See you later!"
            ]
        },
        "intermediate": {
            "機場對話 (Airport Conversation)": [
                "Good morning! May I see your passport and ticket?",
                "What's the purpose of your visit to our country?",
                "How long are you planning to stay?",
                "Do you have anything to declare?",
                "Please proceed to gate 15. Boarding starts at 3 PM.",
                "Have a pleasant journey!"
            ],
            "餐廳點餐 (Restaurant Ordering)": [
                "Welcome! Do you have a reservation?",
                "Would you prefer a table by the window?",
                "Can I get you started with something to drink?",
                "Our special today is grilled salmon with vegetables.",
                "How would you like your steak cooked?",
                "Would you care for dessert or coffee?"
            ]
        },
        "advanced": {
            "機場對話 (Airport Conversation)": [
                "Good afternoon. I'll need to verify your travel documents.",
                "Could you clarify the nature of your business visit?",
                "I notice your return flight is quite far out. Any particular reason for the extended stay?",
                "For customs purposes, are you carrying any items that exceed the duty-free allowance?",
                "Your gate assignment is B7, and I'd recommend arriving 30 minutes before boarding.",
                "I hope you have a productive and enjoyable trip."
            ]
        }
    }
    
    level_responses = responses_by_level.get(level, responses_by_level["intermediate"])
    scenario_responses = level_responses.get(scenario, level_responses.get("日常社交 (Daily Social Conversation)", ["Hello!", "How can I help you?"]))
    
    return scenario_responses

class AudioProcessor:
    """音頻處理類（改進版）"""
    
    def __init__(self):
        self.model_manager = get_model_manager()
    
    def transcribe_speech(self, audio_path):
        """語音識別"""
        if not audio_path or not os.path.exists(audio_path):
            return None, "音頻文件不存在"
        
        try:
            recognized_text = self.model_manager.transcribe_audio(audio_path)
            
            if not recognized_text or len(recognized_text.strip()) < 2:
                return None, "語音識別失敗，請重新錄製"
            
            return recognized_text, "識別成功"
            
        except Exception as e:
            return None, f"語音識別錯誤: {str(e)}"
    
    def analyze_pronunciation(self, audio_path, transcribed_text, scenario, conversation_history="", 
                            difficulty="中級 (TOEIC 605-780分)", pronunciation_focus=None, 
                            accent_preference="不指定", feedback_detail="詳細回饋", 
                            show_comparison=True, **kwargs):
        """發音分析 - 完整整合進階功能"""
        try:
            analysis_result = self._analyze_with_audio_llm(
                audio_path, transcribed_text, scenario, conversation_history, 
                difficulty, pronunciation_focus, accent_preference, 
                feedback_detail, show_comparison, **kwargs
            )
            
            if analysis_result:
                return analysis_result
            else:
                return self._analyze_with_simple_method(
                    transcribed_text, scenario, difficulty, pronunciation_focus, 
                    accent_preference, feedback_detail
                )
                
        except Exception as e:
            print(f"發音分析錯誤: {e}")
            return self._analyze_with_simple_method(
                transcribed_text, scenario, difficulty, pronunciation_focus, 
                accent_preference, feedback_detail
            )
    
    def _analyze_with_audio_llm(self, audio_path, transcribed_text, scenario, conversation_history, 
                               difficulty, pronunciation_focus, accent_preference, feedback_detail, 
                               show_comparison, **kwargs):
        """使用Audio-LLM進行詳細分析 - 整合所有進階功能"""
        try:
            system_prompt = create_advanced_prompt(
                scenario, difficulty, pronunciation_focus, accent_preference,
                feedback_detail, show_comparison, conversation_history
            )
            
            difficulty_config = DIFFICULTY_CONFIGS.get(difficulty, DIFFICULTY_CONFIGS["中級 (TOEIC 605-780分)"])
            
            full_prompt = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
<|AUDIO|>
The student said: "{transcribed_text}"

Please analyze their pronunciation and provide your response according to the specified format, considering their {difficulty_config['level']} proficiency level.
<|im_end|>
<|im_start|>assistant
"""
            
            response = self.model_manager.generate_audio_response(audio_path, full_prompt)
            
            if response:
                return self._parse_llm_response(response, transcribed_text, scenario, difficulty)
            else:
                return None
                
        except Exception as e:
            print(f"Audio-LLM分析失敗: {e}")
            return None
    
    def _analyze_with_simple_method(self, transcribed_text, scenario, difficulty, 
                                   pronunciation_focus=None, accent_preference="不指定", 
                                   feedback_detail="詳細回饋"):
        """簡化分析模式 - 也整合進階功能設定"""
        
        difficulty_config = DIFFICULTY_CONFIGS.get(difficulty, DIFFICULTY_CONFIGS["中級 (TOEIC 605-780分)"])
        
        pronunciation_score = self._calculate_pronunciation_score(transcribed_text)
        fluency_score = self._calculate_fluency_score(transcribed_text)
        
        pronunciation_score += difficulty_config["score_adjustment"]
        pronunciation_score = max(40, min(100, pronunciation_score))
        
        pronunciation_analysis = self._generate_advanced_analysis(
            transcribed_text, pronunciation_score, difficulty_config, 
            pronunciation_focus, accent_preference, feedback_detail
        )
        
        responses = get_scenario_responses(scenario, difficulty)
        response_text = random.choice(responses)
        
        suggested_responses = self._generate_suggested_responses(scenario, difficulty_config, transcribed_text)
        
        return {
            "pronunciation_analysis": pronunciation_analysis,
            "response_text": response_text,
            "suggested_responses": suggested_responses,
            "pronunciation_score": pronunciation_score,
            "fluency_score": fluency_score
        }
    
    def _generate_advanced_analysis(self, text, score, difficulty_config, pronunciation_focus, 
                                   accent_preference, feedback_detail):
        """根據進階設定生成分析內容"""
        
        level = difficulty_config["level"]
        encouragement = difficulty_config["encouragement_level"]
        
        analysis = f"📊 發音評분: {score}/100 ({level} 級別標準)\n\n"
        
        if feedback_detail == "基本回饋":
            if score >= 80:
                analysis += f"🌟 表現很好！繼續保持這樣的練習。"
            elif score >= 70:
                analysis += f"👍 基礎不錯，建議繼續改進發音準確度。"
            else:
                analysis += f"💪 有進步空間，建議多練習基本發音。"
        
        elif feedback_detail == "詳細回饋":
            analysis += self._get_detailed_feedback(text, score, level, pronunciation_focus)
            
        else:
            analysis += self._get_expert_feedback(text, score, level, pronunciation_focus, accent_preference)
        
        if pronunciation_focus:
            analysis += "\n\n🎯 重點改進建議：\n"
            focus_tips = {
                "子音發音": "練習清晰的子音發音，特別注意 th, r, l 等音素",
                "母音發音": "注意母音的精確度，避免音位偏移",
                "連音": "練習自然的連音技巧，讓語流更順暢",
                "重音": "掌握單字重音和句子重音的規律",
                "語調": "練習適當的語調變化，增加表達的自然度",
                "節奏": "控制說話節奏，適當的停頓和語速"
            }
            
            for focus in pronunciation_focus:
                if focus in focus_tips:
                    analysis += f"• {focus_tips[focus]}\n"
        
        if accent_preference != "不指定":
            analysis += f"\n🌍 口音建議：針對{accent_preference}發音特點進行練習"
        
        return analysis
    
    def _get_detailed_feedback(self, text, score, level, pronunciation_focus):
        """生成詳細回饋"""
        word_count = len(text.split())
        
        feedback = ""
        if score >= 85:
            feedback += f"🎉 優秀的{level}水平表現！語音清晰，表達自然。\n\n"
        elif score >= 75:
            feedback += f"✅ 良好的{level}水平，有明顯的語言能力基礎。\n\n"
        else:
            feedback += f"📚 {level}水平的基礎練習，建議加強基本發音。\n\n"
        
        feedback += f"📝 詳細分析：\n"
        feedback += f"- 發音長度: {word_count} 個單字\n"
        feedback += f"- 語言複雜度: {'適中' if 5 <= word_count <= 15 else '較簡單' if word_count < 5 else '較複雜'}\n"
        feedback += f"- 整體流暢度: {'良好' if score >= 80 else '尚可' if score >= 70 else '需改進'}\n"
        
        return feedback
    
    def _get_expert_feedback(self, text, score, level, pronunciation_focus, accent_preference):
        """生成專家級回饋"""
        feedback = f"🔬 專家級語音分析 ({level} 水平)：\n\n"
        
        feedback += f"📊 語音質量評估：\n"
        feedback += f"- 音素準確度: {score}%\n"
        feedback += f"- 韻律特徵: {'自然' if score >= 85 else '可改進'}\n"
        feedback += f"- 語流連貫性: {'流暢' if score >= 80 else '需加強'}\n\n"
        
        feedback += f"🎯 專業改進建議：\n"
        if pronunciation_focus:
            feedback += f"- 重點練習領域: {', '.join(pronunciation_focus)}\n"
        
        if accent_preference != "不指定":
            feedback += f"- 目標口音: {accent_preference}標準\n"
            feedback += f"- 建議練習材料: 針對{accent_preference}的語音資源\n"
        
        feedback += f"- 練習頻率建議: 每日15-20分鐘專項練習\n"
        feedback += f"- 進階練習: 影子跟讀、語調模仿、錄音對比\n"
        
        return feedback
    
    def _generate_suggested_responses(self, scenario, difficulty_config, user_text):
        """生成建議回覆句子"""
        level = difficulty_config["level"]
        
        suggestions = {
            "機場對話 (Airport Conversation)": {
                "beginner": [
                    "Thank you. Here is my passport.",
                    "I am here for vacation.",
                    "I will stay for one week."
                ],
                "intermediate": [
                    "Thank you. Here are my travel documents.",
                    "I'm visiting for tourism purposes.",
                    "I plan to stay for about ten days."
                ],
                "advanced": [
                    "Certainly. Here are my passport and boarding pass.",
                    "I'm here on a business trip with some leisure time.",
                    "I'll be staying for approximately two weeks for both business and tourism."
                ]
            },
            "餐廳點餐 (Restaurant Ordering)": {
                "beginner": [
                    "I want a burger, please.",
                    "Can I have water?",
                    "How much is it?"
                ],
                "intermediate": [
                    "I'd like to order the grilled chicken, please.",
                    "Could I have a glass of water with that?",
                    "What's the total cost?"
                ],
                "advanced": [
                    "I'd be interested in trying your signature dish.",
                    "Could you recommend a wine pairing with that?",
                    "I'd like to split the bill, if that's possible."
                ]
            }
        }
        
        scenario_suggestions = suggestions.get(scenario, {})
        level_suggestions = scenario_suggestions.get(level, [
            "That sounds good.",
            "I understand. Thank you.",
            "Could you please explain more?"
        ])
        
        return level_suggestions
    
    def _calculate_pronunciation_score(self, text):
        """計算發音分數"""
        base_score = 70
        
        word_count = len(text.split())
        length_bonus = min(15, word_count * 2)
        
        grammar_bonus = 0
        if any(word in text.lower() for word in ['please', 'thank you', 'excuse me']):
            grammar_bonus += 5
        if '?' in text:
            grammar_bonus += 5
        
        random_factor = random.randint(-5, 10)
        
        final_score = base_score + length_bonus + grammar_bonus + random_factor
        return max(60, min(95, final_score))
    
    def _calculate_fluency_score(self, text):
        """計算流暢度分數"""
        base_score = 75
        
        sentence_count = text.count('.') + text.count('?') + text.count('!')
        if sentence_count == 0:
            sentence_count = 1
        
        words_per_sentence = len(text.split()) / sentence_count
        
        if 5 <= words_per_sentence <= 15:
            structure_bonus = 10
        elif words_per_sentence < 5:
            structure_bonus = -5
        else:
            structure_bonus = 0
        
        random_factor = random.randint(-8, 12)
        
        final_score = base_score + structure_bonus + random_factor
        return max(65, min(90, final_score))
    
    def _parse_llm_response(self, response, transcribed_text, scenario, difficulty):
        """解析LLM回應 - 提取建議回覆"""
        lines = response.split('\n')
        
        difficulty_config = DIFFICULTY_CONFIGS.get(difficulty, DIFFICULTY_CONFIGS["中級 (TOEIC 605-780分)"])
        
        base_score = 85 + difficulty_config["score_adjustment"]
        pronunciation_score = max(40, min(100, base_score))
        fluency_score = 80
        pronunciation_analysis = ""
        response_text = ""
        suggested_responses = []
        
        current_section = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if "**PRONUNCIATION ANALYSIS:**" in line or "PRONUNCIATION ANALYSIS:" in line:
                current_section = "analysis"
                continue
            elif "**CONVERSATION RESPONSE:**" in line or "CONVERSATION RESPONSE:" in line:
                current_section = "response"
                continue
            elif "**SUGGESTED NEXT RESPONSES:**" in line or "SUGGESTED NEXT RESPONSES:" in line:
                current_section = "suggestions"
                continue
            
            if current_section == "analysis":
                if "score:" in line.lower() and any(char.isdigit() for char in line):
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        score = int(numbers[0])
                        if 0 <= score <= 100:
                            pronunciation_score = score
                pronunciation_analysis += line + "\n"
            
            elif current_section == "response":
                if not line.startswith("**") and not line.startswith("SUGGESTED"):
                    response_text += line + " "
            
            elif current_section == "suggestions":
                if line.startswith(("1.", "2.", "3.", "-", "•")):
                    suggestion = re.sub(r'^[123\-•]\s*', '', line)
                    suggestion = re.sub(r'\[.*?\]', '', suggestion).strip()
                    if suggestion:
                        suggested_responses.append(suggestion)
        
        if not suggested_responses:
            suggested_responses = self._generate_suggested_responses(scenario, difficulty_config, transcribed_text)
        
        if not response_text.strip():
            responses = get_scenario_responses(scenario, difficulty)
            response_text = random.choice(responses)
        
        if not pronunciation_analysis.strip():
            pronunciation_analysis = f"發音評分: {pronunciation_score}/100 ({difficulty_config['level']} 級別)\n整體表現良好，繼續保持！"
        
        return {
            "pronunciation_analysis": pronunciation_analysis.strip(),
            "response_text": response_text.strip(),
            "suggested_responses": suggested_responses,
            "pronunciation_score": pronunciation_score,
            "fluency_score": fluency_score
        }

class ConversationManager:    
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.conversation_history = []
    
    def process_user_input(self, audio_path, scenario, conversation_context="", difficulty="中級 (TOEIC 605-780分)", 
                          pronunciation_focus=None, accent_preference="不指定", feedback_detail="詳細回饋", 
                          show_comparison=True, **kwargs):
        """處理用戶輸入的完整流程 - 整合所有進階功能"""
        result = {
            "recognized_text": "",
            "pronunciation_analysis": "",
            "response_text": "",
            "suggested_responses": [],
            "pronunciation_score": 0,
            "fluency_score": 0,
            "success": False,
            "error_message": ""
        }
        
        try:
            recognized_text, transcribe_status = self.audio_processor.transcribe_speech(audio_path)
            
            if not recognized_text:
                result["error_message"] = transcribe_status
                return result
            
            result["recognized_text"] = recognized_text
            
            analysis_result = self.audio_processor.analyze_pronunciation(
                audio_path=audio_path,
                transcribed_text=recognized_text, 
                scenario=scenario, 
                conversation_history=conversation_context,
                difficulty=difficulty,
                pronunciation_focus=pronunciation_focus,
                accent_preference=accent_preference,
                feedback_detail=feedback_detail,
                show_comparison=show_comparison,
                **kwargs
            )
            
            result.update(analysis_result)
            result["success"] = True
            
            self._update_conversation_history(scenario, recognized_text, result["response_text"])
            
            return result
            
        except Exception as e:
            result["error_message"] = f"處理過程出錯: {str(e)}"
            return result
    
    def _update_conversation_history(self, scenario, user_text, assistant_text):
        entry = {
            "timestamp": datetime.datetime.now(),
            "scenario": scenario,
            "user": user_text,
            "assistant": assistant_text
        }
        
        self.conversation_history.append(entry)
        
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_conversation_context(self, max_entries=3):
        """獲取對話上下文"""
        if not self.conversation_history:
            return ""
        
        recent_history = self.conversation_history[-max_entries:]
        context_lines = []
        
        for entry in recent_history:
            context_lines.append(f"User: {entry['user']}")
            context_lines.append(f"Assistant: {entry['assistant']}")
        
        return "\n".join(context_lines)
    
    def clear_history(self):
        """清除對話歷史"""
        self.conversation_history = []

conversation_manager = ConversationManager()

def get_conversation_manager():
    """獲取對話管理器實例"""
    return conversation_manager

if __name__ == "__main__":
    print("測試音頻處理器...")
    processor = AudioProcessor()
    print("AudioProcessor初始化完成")
    
    manager = ConversationManager()
    print("ConversationManager初始化完成")
    
    test_focus = ["子音發音", "語調"]
    test_accent = "美式英文"
    test_feedback = "專家級分析"
    
    for difficulty in DIFFICULTY_CONFIGS:
        print(f"難度: {difficulty}")
        prompt = create_advanced_prompt(
            "機場對話 (Airport Conversation)", 
            difficulty, 
            test_focus, 
            test_accent, 
            test_feedback, 
            True
        )
        print(f"Prompt長度: {len(prompt)} 字符")
        print("---")