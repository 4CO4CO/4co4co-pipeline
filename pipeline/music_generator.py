class MusicGenerator:
    """audiocraft MusicGen 래퍼 클래스"""
    
    def __init__(self, model_name='facebook/musicgen-small', duration=12.0):
        self.model_name = model_name
        self.duration = duration
        # TODO: MusicGen 모델 로딩
        # TODO: generation_params 설정 (temperature=1.0 고정)
        
    def generate_single(self, prompt):
        """프롬프트로 음악 1개 생성"""
        # TODO: 구현
        pass
        
    def generate_batch(self, prompt, count=5):
        """같은 프롬프트로 여러 개 생성"""
        # TODO: 구현
        pass