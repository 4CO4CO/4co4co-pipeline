import torch
import numpy as np
import time
from audiocraft.models import MusicGen


class MusicGenerator:
    """audiocraft MusicGen 래퍼 클래스"""
    
    def __init__(self, model_name='facebook/musicgen-small', duration=12.0):
        self.model_name = model_name
        self.duration = duration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🎵 MusicGen 모델 로딩 중... (디바이스: {self.device})")
        self.model = MusicGen.get_pretrained(model_name)
        
        # generation_params 설정 (temperature=1.0 고정)
        self.model.set_generation_params(
            use_sampling=True,
            temperature=1.0,  # 고정
            duration=duration,
            top_k=250,
            top_p=0.0,  # top_k만 사용
            cfg_coef=3.0
        )
        print("✅ MusicGen 모델 로딩 완료!")
        
    def generate_single(self, prompt):
        """프롬프트로 음악 1개 생성"""
        try:
            start_time = time.time()
            
            # 음악 생성
            wav = self.model.generate([prompt], progress=False)
            
            # numpy 배열로 변환 (품질 검사용)
            audio_data = wav[0].cpu().numpy().squeeze()
            sample_rate = self.model.sample_rate
            duration = len(audio_data) / sample_rate
            generation_time = time.time() - start_time
            
            return {
                'success': True,
                'audio_data': audio_data,    # numpy array for analysis
                'sample_rate': sample_rate,
                'duration': duration,
                'wav_tensor': wav[0].cpu(),  # torch tensor for saving
                'prompt': prompt,
                'generation_time': generation_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'prompt': prompt
            }
        
    def generate_batch(self, prompt, count=5):
        """같은 프롬프트로 여러 개 생성"""
        results = []
        
        print(f"🎼 '{prompt}' - {count}개 생성 시작...")
        
        for i in range(count):
            print(f"  생성 {i+1}/{count}...", end=" ")
            
            result = self.generate_single(prompt)
            
            if result['success']:
                print(f"완료 ({result['generation_time']:.1f}초)")
            else:
                print(f"실패: {result['error']}")
            
            results.append(result)
            
        return results