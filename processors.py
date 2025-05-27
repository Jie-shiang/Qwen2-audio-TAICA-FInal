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
        "score_adjustment": +15
    },
    "初級 (TOEIC 405-600分)": {
        "level": "elementary",
        "toeic_range": "405-600",
        "description": "Elementary vocabulary with basic conversational skills",
        "evaluation_criteria": "Evaluate basic conversation flow and pronunciation accuracy",
        "encouragement_level": "encouraging",
        "score_adjustment": +10
    },
    "中級 (TOEIC 605-780分)": {
        "level": "intermediate",
        "toeic_range": "605-780",
        "description": "Intermediate vocabulary and complex sentence structures",
        "evaluation_criteria": "Assess fluency, natural expression, and grammar accuracy",
        "encouragement_level": "balanced",
        "score_adjustment": 0
    },
    "中高級 (TOEIC 785-900分)": {
        "level": "upper_intermediate",
        "toeic_range": "785-900",
        "description": "Advanced vocabulary with nuanced expressions",
        "evaluation_criteria": "Focus on natural flow, idiomatic expressions, and subtle pronunciation",
        "encouragement_level": "constructive",
        "score_adjustment": -5
    },
    "高級 (TOEIC 905+分)": {
        "level": "advanced",
        "toeic_range": "905+",
        "description": "Professional-level vocabulary and sophisticated expressions",
        "evaluation_criteria": "Evaluate native-like fluency, sophisticated vocabulary usage, and professional communication",
        "encouragement_level": "detailed",
        "score_adjustment": -10
    }
}

def get_scenario_prompt(scenario, difficulty="中級 (TOEIC 605-780分)"):
    """獲取基於難度的場景prompt"""
    
    difficulty_config = DIFFICULTY_CONFIGS.get(difficulty, DIFFICULTY_CONFIGS["中級 (TOEIC 605-780分)"])
    level = difficulty_config["level"]
    criteria = difficulty_config["evaluation_criteria"]
    encouragement = difficulty_config["encouragement_level"]
    
    base_prompts = {
        "機場對話 (Airport Conversation)": f"""You are an airport staff member helping a traveler at {level} English level (TOEIC {difficulty_config['toeic_range']}). 

Analyze the user's spoken English considering their proficiency level:
1. {criteria}
2. Check if they used appropriate phrases for airport scenarios at their level
3. Provide {encouragement} feedback and corrections
4. Continue the conversation naturally as airport staff
5. Adjust your language complexity to match their {level} level

Keep responses helpful and appropriate for {level} learners.""",

        "餐廳點餐 (Restaurant Ordering)": f"""You are a restaurant waiter taking orders from a {level} English learner (TOEIC {difficulty_config['toeic_range']}). 

Analyze the customer's spoken English:
1. {criteria}
2. Check if they used polite ordering phrases appropriate for their level
3. Suggest better expressions with {encouragement} tone
4. Respond naturally as a waiter would, matching their {level} level
5. Focus on food-related vocabulary suitable for their proficiency

Be friendly and patient with {level} learners.""",

        "求職面試 (Job Interview)": f"""You are a job interviewer speaking with a {level} English candidate (TOEIC {difficulty_config['toeic_range']}). 

Analyze the candidate's spoken English:
1. {criteria}
2. Evaluate their professional vocabulary usage at {level} level
3. Assess confidence and communication skills appropriate for their proficiency
4. Ask follow-up questions suitable for {level} speakers
5. Provide {encouragement} feedback on their interview performance

Maintain a professional but supportive tone for {level} learners.""",

        "日常社交 (Daily Social Conversation)": f"""You are a friendly conversation partner with a {level} English speaker (TOEIC {difficulty_config['toeic_range']}). 

Analyze their English conversation skills:
1. {criteria}
2. Check natural expression usage at their proficiency level
3. Suggest more native-like phrases with {encouragement} approach
4. Keep the conversation flowing naturally at {level} complexity
5. Focus on casual expressions appropriate for their level

Be supportive and encouraging with {level} learners.""",

        "醫療諮詢 (Medical Consultation)": f"""You are a doctor/nurse speaking with a {level} English patient (TOEIC {difficulty_config['toeic_range']}). 

Analyze the patient's spoken English:
1. {criteria}
2. Check if they can express health concerns at their proficiency level
3. Provide {encouragement} feedback on medical vocabulary usage
4. Respond professionally but clearly for {level} speakers
5. Use medical terms appropriate for their English level

Be patient and clear with {level} English patients.""",

        "學術討論 (Academic Discussion)": f"""You are a professor/lecturer with a {level} English student (TOEIC {difficulty_config['toeic_range']}). 

Analyze the student's academic English:
1. {criteria}
2. Evaluate their ability to discuss academic topics at {level} proficiency
3. Assess their use of academic vocabulary and expressions
4. Provide {encouragement} feedback on their academic communication
5. Ask questions appropriate for {level} academic discussions

Maintain an academic but supportive tone for {level} learners."""
    }
    
    return base_prompts.get(scenario, base_prompts["日常社交 (Daily Social Conversation)"])

def get_scenario_responses(scenario, difficulty="中級 (TOEIC 605-780分)"):
    """獲取基於難度的場景回應"""
    
    difficulty_config = DIFFICULTY_CONFIGS.get(difficulty, DIFFICULTY_CONFIGS["中級 (TOEIC 605-780分)"])
    level = difficulty_config["level"]
    
    if level in ["beginner", "elementary"]:
        responses = {
            "機場對話 (Airport Conversation)": [
                "Thank you. What is your destination?",
                "How long will you stay?",
                "Do you have anything to declare?",
                "Please go to gate 12.",
                "Your boarding pass, please.",
                "Window or aisle seat?"
            ],
            "餐廳點餐 (Restaurant Ordering)": [
                "What would you like to drink?",
                "Are you ready to order?",
                "Would you like appetizers?",
                "How do you want your steak?",
                "Would you like dessert?",
                "Here is your check."
            ],
            "求職面試 (Job Interview)": [
                "Tell me about yourself.",
                "What are your strengths?",
                "Why do you want this job?",
                "Do you have questions?",
                "When can you start?",
                "Thank you for coming."
            ],
            "日常社交 (Daily Social Conversation)": [
                "How was your weekend?",
                "What do you like to do?",
                "Have you seen good movies?",
                "How is the weather?",
                "Do you have plans tonight?",
                "Nice talking with you!"
            ]
        }
    elif level == "intermediate":
        responses = {
            "機場對話 (Airport Conversation)": [
                "Thank you. What is the purpose of your visit?",
                "How long will you be staying in the country?",
                "Do you have anything to declare at customs?",
                "Please proceed to gate 12. Have a nice flight!",
                "Could I see your boarding pass, please?",
                "Would you prefer a window or aisle seat?"
            ],
            "餐廳點餐 (Restaurant Ordering)": [
                "What would you like to drink with your meal?",
                "Are you ready to place your order?",
                "Would you care for any appetizers to start?",
                "How would you like your steak prepared?",
                "Would you be interested in dessert?",
                "Here's your check. Thank you for dining with us!"
            ],
            "求職面試 (Job Interview)": [
                "Could you tell me about your background?",
                "What would you say are your key strengths?",
                "Why are you interested in working here?",
                "Do you have any questions about the position?",
                "When would you be available to start?",
                "Thank you for your time. We'll be in touch."
            ],
            "日常社交 (Daily Social Conversation)": [
                "How did you spend your weekend?",
                "What do you enjoy doing in your free time?",
                "Have you watched any good movies recently?",
                "What do you think of today's weather?",
                "Do you have any interesting plans this evening?",
                "It's been really nice chatting with you!"
            ]
        }
    else:
        responses = {
            "機場對話 (Airport Conversation)": [
                "Thank you for your passport. Could you please tell me the purpose of your visit?",
                "How long are you planning to stay, and do you have your return ticket?",
                "Do you have anything to declare, including gifts or items for commercial use?",
                "Please make your way to gate 12. Your flight should begin boarding in about 30 minutes.",
                "I'll need to see your boarding pass and ID for verification, please.",
                "I can offer you either a window seat with a view or an aisle seat for easier access."
            ],
            "餐廳點餐 (Restaurant Ordering)": [
                "Good evening! What can I get you to drink while you're looking over the menu?",
                "Have you had a chance to decide, or would you like me to recommend today's specials?",
                "Would you be interested in starting with any of our appetizers or sharing plates?",
                "For the steak, how would you prefer it cooked - rare, medium, or well-done?",
                "We have some excellent desserts tonight. Would you like to hear about them?",
                "Here's your bill. I hope you've enjoyed your dining experience with us tonight!"
            ],
            "求職面試 (Job Interview)": [
                "I'd like to start by having you walk me through your professional background and experience.",
                "What would you consider to be your most significant professional strengths and accomplishments?",
                "What attracts you to our company, and how do you see yourself contributing to our team?",
                "I'd be happy to answer any questions you might have about the role or our company culture.",
                "Assuming we move forward, what would be your ideal timeline for transitioning into this position?",
                "Thank you for taking the time to meet with us today. We'll follow up within the next few days."
            ],
            "日常社交 (Daily Social Conversation)": [
                "I'm curious to hear how you spent your weekend - did you get up to anything interesting?",
                "What kinds of activities do you find most enjoyable during your downtime?",
                "Have you come across any particularly good films or shows that you'd recommend lately?",
                "This weather has been quite something, hasn't it? How are you finding it?",
                "Do you have anything exciting planned for later this evening or the rest of the week?",
                "I've really enjoyed our conversation - it's been such a pleasure talking with you!"
            ]
        }
        
    return responses.get(scenario, responses["日常社交 (Daily Social Conversation)"])

class AudioProcessor:
    """音頻處理類"""
    
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
    
    def analyze_pronunciation(self, audio_path, transcribed_text, scenario, conversation_history="", difficulty="中級 (TOEIC 605-780分)"):
        """發音分析 - 包含難度級別"""
        try:
            analysis_result = self._analyze_with_audio_llm(
                audio_path, transcribed_text, scenario, conversation_history, difficulty
            )
            
            if analysis_result:
                return analysis_result
            else:
                return self._analyze_with_simple_method(transcribed_text, scenario, difficulty)
                
        except Exception as e:
            print(f"發音分析錯誤: {e}")
            return self._analyze_with_simple_method(transcribed_text, scenario, difficulty)
    
    def _analyze_with_audio_llm(self, audio_path, transcribed_text, scenario, conversation_history, difficulty):
        """使用Audio-LLM進行詳細分析 - 包含難度調整"""
        try:
            system_prompt = get_scenario_prompt(scenario, difficulty)
            
            difficulty_config = DIFFICULTY_CONFIGS.get(difficulty, DIFFICULTY_CONFIGS["中級 (TOEIC 605-780分)"])
            
            full_prompt = f"""<|im_start|>system
{system_prompt}

Student's proficiency level: {difficulty_config['level']} (TOEIC {difficulty_config['toeic_range']})
Expected competency: {difficulty_config['description']}

Previous conversation:
{conversation_history}

Student said: "{transcribed_text}"

Please provide feedback appropriate for their {difficulty_config['level']} level with {difficulty_config['encouragement_level']} tone.
<|im_end|>
<|im_start|>user
<|AUDIO|>
Please analyze the pronunciation and respond for the {scenario} scenario, considering the student's {difficulty_config['level']} proficiency level.<|im_end|>
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
    
    def _analyze_with_simple_method(self, transcribed_text, scenario, difficulty):        
        difficulty_config = DIFFICULTY_CONFIGS.get(difficulty, DIFFICULTY_CONFIGS["中級 (TOEIC 605-780分)"])
        
        pronunciation_score = self._calculate_pronunciation_score(transcribed_text)
        fluency_score = self._calculate_fluency_score(transcribed_text)
        
        pronunciation_score += difficulty_config["score_adjustment"]
        pronunciation_score = max(40, min(100, pronunciation_score))
        
        pronunciation_analysis = self._generate_difficulty_based_analysis(transcribed_text, pronunciation_score, difficulty_config)
        
        responses = get_scenario_responses(scenario, difficulty)
        response_text = random.choice(responses)
        
        return {
            "pronunciation_analysis": pronunciation_analysis,
            "response_text": response_text,
            "pronunciation_score": pronunciation_score,
            "fluency_score": fluency_score
        }
    
    def _calculate_pronunciation_score(self, text):
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
    
    def _generate_difficulty_based_analysis(self, text, score, difficulty_config):
        level = difficulty_config["level"]
        encouragement = difficulty_config["encouragement_level"]
        
        analysis = f"📊 發音評分: {score}/100 ({level} 級別標準)\n\n"
        
        if level == "beginner":
            if score >= 80:
                analysis += "🌟 對於初學者來說表現很棒！發音清晰，繼續保持這樣的練習。"
            elif score >= 70:
                analysis += "👍 很好的開始！建議多練習基本發音，特別注意清楚地說出每個單詞。"
            else:
                analysis += "💪 不要擔心，每個人都是從這裡開始的！建議先練習簡單的單詞發音。"
        
        elif level == "elementary":
            if score >= 85:
                analysis += "🎉 初級水平表現優秀！可以開始嘗試更複雜的句子結構。"
            elif score >= 75:
                analysis += "✅ 發音基礎很好，建議加強語調的自然度和句子的連貫性。"
            else:
                analysis += "📚 基礎不錯，建議多聽多模仿標準發音來提升準確度。"
        
        elif level == "intermediate":
            if score >= 90:
                analysis += "🏆 中級水平的優秀表現！語調自然，表達流暢。"
            elif score >= 80:
                analysis += "👌 表現良好，建議在語調變化和自然表達方面繼續改進。"
            else:
                analysis += "🔄 基本正確，建議加強發音準確性和表達的自然度。"
        
        elif level == "upper_intermediate":
            if score >= 92:
                analysis += "🌟 中高級水平表現卓越！接近母語者的自然度。"
            elif score >= 85:
                analysis += "💎 很好的中高級表現，建議關注更細緻的語音語調變化。"
            else:
                analysis += "⚡ 有進步空間，建議加強高級表達方式和語音細節的掌握。"
        
        else:  # advanced
            if score >= 95:
                analysis += "🎯 高級水平的完美表現！語言運用近乎母語水平。"
            elif score >= 90:
                analysis += "🚀 優秀的高級表現，建議在更複雜的語境中練習專業表達。"
            else:
                analysis += "📈 以高級標準來看還有提升空間，建議加強專業詞彙和複雜表達的掌握。"
        
        word_count = len(text.split())
        if word_count < 5 and level not in ["beginner"]:
            analysis += f"\n\n💡 建議：嘗試使用更完整和豐富的句子來表達想法，這樣更符合{level}水平的要求。"
        elif word_count >= 10 and level == "beginner":
            analysis += f"\n\n🎉 很好！您使用了比較長的句子，這對初學者來說很不錯。"
        
        return analysis
    
    def _parse_llm_response(self, response, transcribed_text, scenario, difficulty):
        lines = response.split('\n')
        
        difficulty_config = DIFFICULTY_CONFIGS.get(difficulty, DIFFICULTY_CONFIGS["中級 (TOEIC 605-780分)"])
        
        base_score = 85 + difficulty_config["score_adjustment"]
        pronunciation_score = max(40, min(100, base_score))
        fluency_score = 80
        pronunciation_analysis = ""
        response_text = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "pronunciation" in line.lower() and any(char.isdigit() for char in line):
                try:
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        score = int(numbers[0])
                        if 0 <= score <= 100:
                            pronunciation_score = score
                except:
                    pass
            
            if any(keyword in line.lower() for keyword in ["pronunciation", "accent", "clarity", "analysis"]):
                pronunciation_analysis += line + "\n"
            
            if any(keyword in line for keyword in ["?", "Please", "Would", "Can", "How", "What"]):
                response_text += line + " "
        
        if not response_text.strip():
            responses = get_scenario_responses(scenario, difficulty)
            response_text = random.choice(responses)
        
        if not pronunciation_analysis.strip():
            pronunciation_analysis = f"發音評分: {pronunciation_score}/100 ({difficulty_config['level']} 級別)\n整體表現良好，繼續保持！"
        
        return {
            "pronunciation_analysis": pronunciation_analysis.strip(),
            "response_text": response_text.strip(),
            "pronunciation_score": pronunciation_score,
            "fluency_score": fluency_score
        }

class ConversationManager:    
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.conversation_history = []
    
    def process_user_input(self, audio_path, scenario, conversation_context="", difficulty="中級 (TOEIC 605-780分)"):
        """處理用戶輸入的完整流程 - 包含難度參數"""
        result = {
            "recognized_text": "",
            "pronunciation_analysis": "",
            "response_text": "",
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
                audio_path, recognized_text, scenario, conversation_context, difficulty
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
    
    for difficulty in DIFFICULTY_CONFIGS:
        print(f"難度: {difficulty}")
        prompt = get_scenario_prompt("機場對話 (Airport Conversation)", difficulty)
        print(f"Prompt長度: {len(prompt)} 字符")
        responses = get_scenario_responses("機場對話 (Airport Conversation)", difficulty)
        print(f"回應數量: {len(responses)}")
        print("---")