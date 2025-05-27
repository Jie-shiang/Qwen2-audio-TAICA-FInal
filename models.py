# -*- coding: utf-8 -*-
"""
models.py - 模型管理中心
負責所有AI模型的載入、配置和管理
"""

import torch
import whisper
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import gc
import warnings
from memory_monitor import start_memory_monitoring, get_memory_monitor
warnings.filterwarnings("ignore")

class ModelManager:    
    def __init__(self, gpu_memory_limit=20):
        self.gpu_memory_limit = gpu_memory_limit
        self.device = None
        self.use_gpu = False
        self.whisper_model = None
        self.audio_llm_model = None
        self.audio_llm_processor = None
        self.use_audio_llm = False
        self.memory_monitor = None
        
        # 初始化
        self._setup_gpu()
        self._start_memory_monitoring()
        self._load_models()
    
    def _start_memory_monitoring(self):
        print(f"🔍 啟動記憶體監控 (限制: {self.gpu_memory_limit}GB)")
        self.memory_monitor = start_memory_monitoring(
            gpu_limit_gb=self.gpu_memory_limit,
            cpu_limit_gb=32,
            check_interval=3
        )
    
    def _setup_gpu(self):
        print("=== GPU設定檢查 ===")
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"可用GPU數量: {gpu_count}")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(0)
            self.use_gpu = True
            print(f"使用設備: {self.device}")
            
            torch.cuda.empty_cache()
            gc.collect()
        else:
            print("CUDA不可用，將使用CPU")
            self.device = torch.device("cpu")
            self.use_gpu = False
    
    def _memory_check_and_cleanup(self, operation_name=""):
        if not self.use_gpu:
            return True
            
        try:
            current_memory = torch.cuda.memory_reserved(0) / 1024**3
            if current_memory > self.gpu_memory_limit * 0.9:  # 90%警告
                print(f"⚠️  {operation_name} - 記憶體使用接近限制: {current_memory:.2f}GB")
                self.clear_gpu_memory()
                
                # 再次檢查
                current_memory = torch.cuda.memory_reserved(0) / 1024**3
                if current_memory > self.gpu_memory_limit:
                    print(f"🚨 記憶體仍超過限制: {current_memory:.2f}GB > {self.gpu_memory_limit}GB")
                    return False
            
            return True
        except Exception as e:
            print(f"記憶體檢查失敗: {e}")
            return True
    
    def _load_whisper_model(self):
        print("正在載入Whisper模型...")
        
        if not self._memory_check_and_cleanup("Whisper載入前"):
            print("記憶體不足，無法載入Whisper模型")
            return False
            
        try:
            if self.use_gpu:
                self.whisper_model = whisper.load_model("medium").to(self.device)
                print("Whisper模型已載入至GPU")
                
                if not self._memory_check_and_cleanup("Whisper載入後"):
                    print("Whisper載入後記憶體超限，降級使用base模型")
                    del self.whisper_model
                    self.clear_gpu_memory()
                    self.whisper_model = whisper.load_model("base").to(self.device)
                    
            else:
                self.whisper_model = whisper.load_model("base")
                print("Whisper模型已載入至CPU")
            return True
        except Exception as e:
            print(f"Whisper載入失敗: {e}")
            try:
                self.whisper_model = whisper.load_model("base")
                print("已載入基礎版Whisper模型")
                return True
            except Exception as e2:
                print(f"基礎版Whisper也載入失敗: {e2}")
                return False
    
    def _load_qwen_audio_model(self):
        print("正在載入Qwen2-Audio模型...")
        
        if not self._memory_check_and_cleanup("Qwen2-Audio載入前"):
            print("記憶體不足，跳過Qwen2-Audio模型載入")
            self.use_audio_llm = False
            return False
            
        try:
            if self.use_gpu:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                current_memory = torch.cuda.memory_reserved(0) / 1024**3
                available_memory = gpu_memory - current_memory
                
                print(f"GPU記憶體: 總計{gpu_memory:.1f}GB, 已用{current_memory:.1f}GB, 可用{available_memory:.1f}GB")
                
                if available_memory < 6:
                    print("可用記憶體不足6GB，使用CPU模式")
                    torch_dtype = torch.float32
                    device_map = "cpu"
                elif available_memory < 10:
                    print("可用記憶體有限，使用float16和量化")
                    torch_dtype = torch.float16
                    device_map = {"": 0}
                else:
                    torch_dtype = torch.float16
                    device_map = "auto"
            else:
                torch_dtype = torch.float32
                device_map = "cpu"

            self.audio_llm_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-Audio-7B-Instruct",
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            self.audio_llm_model.tie_weights()
            
            if not self._memory_check_and_cleanup("Qwen2-Audio載入後"):
                print("⚠️  Qwen2-Audio載入後記憶體超限")
                del self.audio_llm_model
                self.clear_gpu_memory()
                self.use_audio_llm = False
                return False

            self.audio_llm_processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2-Audio-7B-Instruct",
                trust_remote_code=True
            )

            print("Qwen2-Audio模型載入完成！")
            self.use_audio_llm = True
            return True

        except torch.cuda.OutOfMemoryError:
            print("GPU記憶體不足，嘗試使用CPU...")
            torch.cuda.empty_cache()
            gc.collect()
            try:
                self.audio_llm_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2-Audio-7B-Instruct",
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                self.audio_llm_processor = AutoProcessor.from_pretrained(
                    "Qwen/Qwen2-Audio-7B-Instruct",
                    trust_remote_code=True
                )
                print("Qwen2-Audio模型已載入至CPU")
                self.use_audio_llm = True
                return True
            except Exception as e:
                print(f"CPU載入也失敗: {e}")
                self.use_audio_llm = False
                return False

        except Exception as e:
            print(f"Qwen2-Audio模型載入失敗: {str(e)}")
            print("將使用簡化版本的發音分析功能")
            self.use_audio_llm = False
            return False
    
    def _load_models(self):
        """載入所有模型"""
        print("=== 開始載入模型 ===")
        
        whisper_success = self._load_whisper_model()
        if not whisper_success:
            raise Exception("Whisper模型載入失敗，無法繼續")
        
        qwen_success = self._load_qwen_audio_model()
        
        self._memory_check_and_cleanup("所有模型載入後")
        
        print("=== 模型載入完成 ===")
        print(f"Whisper: {'✓' if whisper_success else '✗'}")
        print(f"Qwen2-Audio: {'✓' if qwen_success else '✗'}")
        print(f"記憶體監控: {'✓' if self.memory_monitor else '✗'}")
    
    def get_device_info(self):
        """獲取設備信息"""
        info = {
            "device": str(self.device),
            "use_gpu": self.use_gpu,
            "use_audio_llm": self.use_audio_llm,
            "whisper_available": self.whisper_model is not None,
            "qwen_available": self.audio_llm_model is not None,
            "memory_limit_gb": self.gpu_memory_limit
        }
        
        if self.use_gpu:
            info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["current_memory_usage"] = torch.cuda.memory_reserved(0) / 1024**3
        
        return info
    
    def clear_gpu_memory(self):
        """清理GPU記憶體"""
        if self.use_gpu:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"🧹 GPU記憶體已清理，當前使用: {torch.cuda.memory_reserved(0) / 1024**3:.2f}GB")
    
    def transcribe_audio(self, audio_path, language="en"):
        """使用Whisper進行語音識別"""
        if self.whisper_model is None:
            raise Exception("Whisper模型未載入")
        
        if not self._memory_check_and_cleanup("語音識別前"):
            raise Exception("記憶體不足，無法進行語音識別")
        
        try:
            if self.use_gpu:
                result = self.whisper_model.transcribe(
                    audio_path, 
                    language=language, 
                    temperature=0.0, 
                    verbose=False,
                    fp16=True
                )
            else:
                result = self.whisper_model.transcribe(
                    audio_path, 
                    language=language, 
                    temperature=0.0, 
                    verbose=False
                )
            
            self._memory_check_and_cleanup("語音識別後")
            
            return result["text"].strip()
        except Exception as e:
            print(f"語音識別錯誤: {e}")
            return None
    
    def generate_audio_response(self, audio_path, prompt, max_tokens=256):
        """使用Qwen2-Audio生成回應"""
        if not self.use_audio_llm:
            return None
        
        if not self._memory_check_and_cleanup("Audio-LLM生成前"):
            print("記憶體不足，跳過Audio-LLM生成")
            return None
        
        try:
            import librosa
            audio_data, sr = librosa.load(audio_path, sr=16000)
            
            max_length = 30 * sr
            if len(audio_data) > max_length:
                audio_data = audio_data[:max_length]
            
            # 處理輸入
            with torch.no_grad():
                inputs = self.audio_llm_processor(
                    text=prompt,
                    audio=audio_data,
                    sampling_rate=sr,
                    return_tensors="pt",
                    padding=True
                )

                if self.use_gpu:
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in inputs.items()}

                if not self._memory_check_and_cleanup("Audio-LLM生成中"):
                    return None

                generate_ids = self.audio_llm_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.audio_llm_processor.tokenizer.eos_token_id
                )

                generated_ids = generate_ids[:, inputs['input_ids'].size(1):]
                response = self.audio_llm_processor.decode(generated_ids[0], skip_special_tokens=True)
                
                del inputs, generate_ids, generated_ids
                self.clear_gpu_memory()
                
                return response

        except torch.cuda.OutOfMemoryError:
            print("🚨 GPU記憶體不足，Audio-LLM生成失敗")
            self.clear_gpu_memory()
            return None
        except Exception as e:
            print(f"Audio-LLM生成錯誤: {e}")
            self.clear_gpu_memory()
            return None
    
    def get_memory_status(self):
        if self.memory_monitor:
            return self.memory_monitor.get_current_status()
        return None
    
    def __del__(self):
        try:
            if hasattr(self, 'memory_monitor') and self.memory_monitor:
                from memory_monitor import stop_memory_monitoring
                stop_memory_monitoring()
        except:
            pass

model_manager = None

def get_model_manager(gpu_memory_limit=20):
    global model_manager
    if model_manager is None:
        model_manager = ModelManager(gpu_memory_limit)
    return model_manager

def initialize_models(gpu_memory_limit=20):
    return get_model_manager(gpu_memory_limit)

if __name__ == "__main__":
    print("測試模型管理器...")
    manager = initialize_models(gpu_memory_limit=20)
    info = manager.get_device_info()
    print("設備信息:", info)
    
    memory_status = manager.get_memory_status()
    if memory_status:
        print("記憶體狀態:", memory_status)