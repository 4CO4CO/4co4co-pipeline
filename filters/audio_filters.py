import numpy as np
import librosa
from scipy import signal


class AudioQualityFilters:
    """음악 품질 검사 필터들 - 세분화된 버전"""
    
    @staticmethod
    def check_duration(audio_data, sample_rate, expected_duration=12.0, tolerance=1.0):
        """생성된 음악이 예상 길이보다 너무 짧은지 검사"""
        try:
            actual_duration = len(audio_data) / sample_rate
            min_duration = expected_duration - tolerance
            
            if actual_duration < min_duration:
                return {
                    'passed': False,
                    'score': actual_duration / expected_duration,
                    'reason': f'Too short: {actual_duration:.1f}s (expected: {expected_duration:.1f}s)'
                }
            
            return {'passed': True, 'score': 1.0, 'reason': f'Duration OK: {actual_duration:.1f}s'}
            
        except Exception as e:
            return {'passed': False, 'score': 0.0, 'reason': f'Duration check error: {e}'}
    
    @staticmethod
    def check_volume_cutoff(audio_data, sample_rate, cutoff_threshold=2.0):
        """볼륨이 끝부분에서 급격히 떨어지는지 검사"""
        try:
            # RMS 볼륨 계산 (0.1초 윈도우)
            window_size = int(0.1 * sample_rate)
            if len(audio_data) < window_size:
                return {'passed': True, 'score': 1.0, 'reason': 'Audio too short to check'}
            
            rms_values = []
            for i in range(0, len(audio_data) - window_size, window_size // 2):
                window = audio_data[i:i + window_size]
                rms = np.sqrt(np.mean(window**2))
                rms_values.append(rms)
            
            rms_values = np.array(rms_values)
            
            # 끝부분 cutoff_threshold초 검사
            cutoff_samples = int(cutoff_threshold / 0.05)  # 0.05초마다 RMS 계산
            if len(rms_values) < cutoff_samples:
                return {'passed': True, 'score': 1.0, 'reason': 'Too short to check cutoff'}
            
            end_rms = rms_values[-cutoff_samples:]
            beginning_rms = rms_values[:len(rms_values)//4]  # 앞 1/4 구간
            
            if len(beginning_rms) == 0:
                return {'passed': True, 'score': 1.0, 'reason': 'Too short'}
            
            # 볼륨 비율 계산
            avg_end = np.mean(end_rms)
            avg_beginning = np.mean(beginning_rms)
            
            if avg_beginning == 0:
                volume_ratio = 1.0
            else:
                volume_ratio = avg_end / avg_beginning
            
            # 50% 이상 볼륨 드롭시 불량
            if volume_ratio < 0.5:
                return {
                    'passed': False,
                    'score': volume_ratio,
                    'reason': f'Volume drops to {volume_ratio:.1%} at the end'
                }
            
            return {'passed': True, 'score': volume_ratio, 'reason': 'Volume stable'}
            
        except Exception as e:
            return {'passed': False, 'score': 0.0, 'reason': f'Volume check error: {e}'}
    
    @staticmethod  
    def check_high_frequency_noise(audio_data, sample_rate, freq_threshold=8000, duration_threshold=3.0):
        """고주파 노이즈가 너무 오래 지속되는지 검사"""
        try:
            # STFT 계산
            f, t, Zxx = signal.stft(audio_data, sample_rate, nperseg=1024)
            
            # 고주파 영역 인덱스
            high_freq_mask = f >= freq_threshold
            
            if not np.any(high_freq_mask):
                return {'passed': True, 'score': 1.0, 'reason': 'No high frequency content'}
            
            # 각 시간 프레임별 고주파 에너지 비율
            high_freq_energy = np.mean(np.abs(Zxx[high_freq_mask, :]), axis=0)
            total_energy = np.mean(np.abs(Zxx), axis=0)
            
            # 0으로 나누기 방지
            energy_ratio = high_freq_energy / (total_energy + 1e-8)
            
            # 고주파 우세한 프레임들 (50% 이상)
            dominant_frames = energy_ratio > 0.5
            
            # 연속된 고주파 우세 구간 찾기
            frame_duration = len(audio_data) / sample_rate / len(t)
            max_continuous_duration = 0
            current_duration = 0
            
            for is_dominant in dominant_frames:
                if is_dominant:
                    current_duration += frame_duration
                    max_continuous_duration = max(max_continuous_duration, current_duration)
                else:
                    current_duration = 0
            
            if max_continuous_duration > duration_threshold:
                return {
                    'passed': False,
                    'score': 1 - (max_continuous_duration / duration_threshold),
                    'reason': f'High frequency noise for {max_continuous_duration:.1f}s'
                }
            
            return {'passed': True, 'score': 1.0, 'reason': 'Frequency spectrum normal'}
            
        except Exception as e:
            return {'passed': False, 'score': 0.0, 'reason': f'Frequency check error: {e}'}
    
    @staticmethod
    def check_frequency_drop(audio_data, sample_rate, drop_threshold=0.3, duration_threshold=2.0):
        """고주파가 갑자기 급격히 떨어지는지 검사"""
        try:
            # Spectral Centroid 계산 (주파수 무게중심)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            
            # 중앙값으로 정규화
            centroid_median = np.median(spectral_centroids)
            normalized_centroids = spectral_centroids / centroid_median
            
            # 급격한 드롭 찾기 (30% 이상 감소)
            drop_mask = normalized_centroids < (1 - drop_threshold)
            
            # 연속된 드롭 구간 찾기
            frame_duration = len(audio_data) / sample_rate / len(spectral_centroids)
            max_drop_duration = 0
            current_drop_duration = 0
            
            for has_drop in drop_mask:
                if has_drop:
                    current_drop_duration += frame_duration
                    max_drop_duration = max(max_drop_duration, current_drop_duration)
                else:
                    current_drop_duration = 0
            
            if max_drop_duration > duration_threshold:
                return {
                    'passed': False,
                    'score': 1 - (max_drop_duration / duration_threshold),
                    'reason': f'Frequency drop for {max_drop_duration:.1f}s'
                }
            
            return {'passed': True, 'score': 1.0, 'reason': 'No significant frequency drops'}
            
        except Exception as e:
            return {'passed': False, 'score': 0.0, 'reason': f'Frequency drop check error: {e}'}
    
    @staticmethod
    def check_extreme_frequencies(audio_data, sample_rate, 
                                low_freq_threshold=80, high_freq_threshold=15000, 
                                duration_threshold=3.0):
        """너무 높은 혹은 너무 낮은 주파수가 오래 지속되는지 검사"""
        try:
            # STFT 계산
            f, t, Zxx = signal.stft(audio_data, sample_rate, nperseg=1024)
            
            # 극단적 주파수 영역 마스크
            too_low_mask = f <= low_freq_threshold
            too_high_mask = f >= high_freq_threshold
            
            # 각 시간 프레임별 극단 주파수 에너지 비율
            total_energy = np.mean(np.abs(Zxx), axis=0)
            
            # 너무 낮은 주파수 체크
            if np.any(too_low_mask):
                low_freq_energy = np.mean(np.abs(Zxx[too_low_mask, :]), axis=0)
                low_freq_ratio = low_freq_energy / (total_energy + 1e-8)
                low_dominant_frames = low_freq_ratio > 0.6  # 60% 이상
            else:
                low_dominant_frames = np.zeros(len(t), dtype=bool)
            
            # 너무 높은 주파수 체크
            if np.any(too_high_mask):
                high_freq_energy = np.mean(np.abs(Zxx[too_high_mask, :]), axis=0)
                high_freq_ratio = high_freq_energy / (total_energy + 1e-8)
                high_dominant_frames = high_freq_ratio > 0.4  # 40% 이상
            else:
                high_dominant_frames = np.zeros(len(t), dtype=bool)
            
            # 연속 구간 찾기
            frame_duration = len(audio_data) / sample_rate / len(t)
            
            # 저주파 연속 구간
            max_low_duration = 0
            current_low_duration = 0
            for is_low in low_dominant_frames:
                if is_low:
                    current_low_duration += frame_duration
                    max_low_duration = max(max_low_duration, current_low_duration)
                else:
                    current_low_duration = 0
            
            # 고주파 연속 구간
            max_high_duration = 0
            current_high_duration = 0
            for is_high in high_dominant_frames:
                if is_high:
                    current_high_duration += frame_duration
                    max_high_duration = max(max_high_duration, current_high_duration)
                else:
                    current_high_duration = 0
            
            # 실패 조건 체크
            if max_low_duration > duration_threshold:
                return {
                    'passed': False,
                    'score': 1 - (max_low_duration / duration_threshold),
                    'reason': f'Too much low frequency (<{low_freq_threshold}Hz) for {max_low_duration:.1f}s'
                }
            
            if max_high_duration > duration_threshold:
                return {
                    'passed': False,
                    'score': 1 - (max_high_duration / duration_threshold), 
                    'reason': f'Too much high frequency (>{high_freq_threshold}Hz) for {max_high_duration:.1f}s'
                }
            
            return {'passed': True, 'score': 1.0, 'reason': 'Frequency range normal'}
            
        except Exception as e:
            return {'passed': False, 'score': 0.0, 'reason': f'Extreme frequency check error: {e}'}
    
    @staticmethod
    def check_monotony(audio_data, sample_rate, variance_threshold=0.1):
        """주파수 변화가 너무 없어서 지루한지 검사"""
        try:
            # Spectral Centroid 계산 (주파수 무게중심)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            
            # 변화량 계산
            centroid_variance = np.var(spectral_centroids)
            centroid_mean = np.mean(spectral_centroids)
            
            # 정규화된 분산 (평균 대비)
            if centroid_mean > 0:
                normalized_variance = centroid_variance / (centroid_mean ** 2)
            else:
                normalized_variance = 0
            
            if normalized_variance < variance_threshold:
                return {
                    'passed': False,
                    'score': normalized_variance / variance_threshold,
                    'reason': f'Too monotonous (variance: {normalized_variance:.3f})'
                }
            
            return {'passed': True, 'score': 1.0, 'reason': 'Good musical variety'}
            
        except Exception as e:
            return {'passed': False, 'score': 0.0, 'reason': f'Monotony check error: {e}'}
        
    @classmethod
    def run_all_checks(cls, audio_data, sample_rate, expected_duration=12.0):
        """3가지 핵심 검사만 실행"""
        duration_result = cls.check_duration(audio_data, sample_rate, expected_duration)
        high_freq_result = cls.check_high_frequency_noise(audio_data, sample_rate)
        extreme_freq_result = cls.check_extreme_frequencies(audio_data, sample_rate)
        
        # 전체 통과 여부
        overall_passed = all([
            duration_result['passed'],
            high_freq_result['passed'],
            extreme_freq_result['passed']
        ])
        
        return {
            'duration': duration_result,
            'high_frequency': high_freq_result,
            'extreme_frequencies': extreme_freq_result,
            'overall_passed': overall_passed
        }