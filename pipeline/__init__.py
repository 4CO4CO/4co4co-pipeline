from .music_generator import MusicGenerator
from .quality_pipeline import MusicQualityPipeline  
from .adaptive_pipeline import AdaptiveMusicQualityPipeline

__all__ = [
    'MusicGenerator',
    'MusicQualityPipeline',        # 기존 배치 처리
    'AdaptiveMusicQualityPipeline' # 새로운 적응형 처리
]

# 버전 정보
__version__ = "2.0.0"
__author__ = "4co4co Team"
__description__ = "Adaptive Music Quality Pipeline for MusicGen"