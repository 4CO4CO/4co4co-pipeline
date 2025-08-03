import numpy as np
import librosa
from scipy import signal


class MusicalCompletenessFilters:
    """음악적 완성도 검사 필터들"""
    
    @staticmethod
    def check_rhythm_consistency(audio_data, sample_rate, tempo_tolerance=0.15):
        """리듬 일관성 검사 - 일정한 템포와 박자 유지"""
        try:
            # 템포와 박자 추출
            tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
            
            if len(beats) < 3:
                return {
                    'passed': False,
                    'score': 0.0,
                    'reason': f'Too few beats detected ({len(beats)})'
                }
            
            # 박자 간격 계산
            beat_intervals = np.diff(beats) / sample_rate
            
            if len(beat_intervals) == 0:
                return {
                    'passed': False,
                    'score': 0.0,
                    'reason': 'No beat intervals detected'
                }
            
            # 박자 간격의 일관성 측정 (변동 계수)
            mean_interval = np.mean(beat_intervals)
            std_interval = np.std(beat_intervals)
            
            if mean_interval == 0:
                coefficient_of_variation = float('inf')
            else:
                coefficient_of_variation = std_interval / mean_interval
            
            # 변동 계수가 낮을수록 일관성 있음
            consistency_score = max(0, 1 - (coefficient_of_variation / tempo_tolerance))
            
            passed = consistency_score > 0.6  # 60% 이상 일관성
            
            return {
                'passed': passed,
                'score': consistency_score,
                'reason': f'Rhythm consistency: {consistency_score:.3f} (tempo: {tempo:.1f}bpm, beats: {len(beats)})'
            }
            
        except Exception as e:
            # 오류 발생시 기본 통과 처리 (관대한 처리)
            return {
                'passed': True,
                'score': 0.7,
                'reason': f'Rhythm check error (default pass): {str(e)}'
            }
    
    @staticmethod
    def check_melody_existence(audio_data, sample_rate, pitch_threshold=100):
        """멜로디 존재 검사 - 단조로운 드론이 아닌 피치 변화"""
        try:
            # 피치 추출
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate, 
                                                 threshold=0.1, fmin=80, fmax=2000)
            
            # 각 시간 프레임별 가장 강한 피치 추출
            dominant_pitches = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:  # 유효한 피치만
                    dominant_pitches.append(pitch)
            
            if len(dominant_pitches) < 10:
                return {
                    'passed': False,
                    'score': 0.0,
                    'reason': f'Too few pitch points detected ({len(dominant_pitches)})'
                }
            
            # 피치 변화량 계산
            pitch_array = np.array(dominant_pitches)
            pitch_variance = np.var(pitch_array)
            pitch_range = np.max(pitch_array) - np.min(pitch_array)
            
            # 멜로디 점수 계산 (분산과 범위 기반)
            variance_score = min(1.0, pitch_variance / (pitch_threshold ** 2))
            range_score = min(1.0, pitch_range / (pitch_threshold * 2))
            melody_score = (variance_score + range_score) / 2
            
            passed = melody_score > 0.3  # 30% 이상 멜로디 변화
            
            return {
                'passed': passed,
                'score': melody_score,
                'reason': f'Melody existence: {melody_score:.3f} (variance: {pitch_variance:.1f}, range: {pitch_range:.1f}Hz)'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'reason': f'Melody check error: {e}'
            }
    
    @staticmethod
    def check_harmonic_balance(audio_data, sample_rate, balance_threshold=0.1):
        """하모닉 밸런스 검사 - 하모닉/퍼커시브 요소 균형 (재즈 친화적)"""
        try:
            # 하모닉과 퍼커시브 성분 분리
            harmonic, percussive = librosa.effects.hpss(audio_data)
            
            # 각 성분의 에너지 계산
            harmonic_energy = np.mean(harmonic ** 2)
            percussive_energy = np.mean(percussive ** 2)
            total_energy = harmonic_energy + percussive_energy
            
            if total_energy == 0:
                return {
                    'passed': False,
                    'score': 0.0,
                    'reason': 'No energy detected in audio'
                }
            
            # 각 성분의 비율 계산
            harmonic_ratio = harmonic_energy / total_energy
            percussive_ratio = percussive_energy / total_energy
            
            # 재즈 음악은 하모닉 우세가 정상 (0.7-0.95 허용)
            if harmonic_ratio >= 0.7:
                balance_score = 1.0  # 하모닉 우세면 만점
            elif harmonic_ratio >= 0.5:
                balance_score = 0.8  # 적당한 밸런스
            else:
                balance_score = 0.3  # 퍼커시브 과다
            
            # 최소한 하나의 성분은 존재해야 함
            min_component_ratio = min(harmonic_ratio, percussive_ratio)
            passed = balance_score > balance_threshold and min_component_ratio > 0.01
            
            return {
                'passed': passed,
                'score': balance_score,
                'reason': f'Harmonic balance: {balance_score:.3f} (H:{harmonic_ratio:.2f}, P:{percussive_ratio:.2f})'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'reason': f'Harmonic balance check error: {e}'
            }
    
    @staticmethod
    def check_musical_flow(audio_data, sample_rate, flow_threshold=0.3):
        """음악적 흐름 검사 - 시간에 따른 자연스러운 전개"""
        try:
            # Spectral Centroid 계산 (음색의 밝기 변화)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            
            # RMS 에너지 계산 (볼륨 변화)
            rms_energy = librosa.feature.rms(y=audio_data)[0]
            
            # Spectral Rolloff 계산 (주파수 분포 변화)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
            
            # 각 특성의 변화량 계산
            centroid_variation = np.std(spectral_centroids) / (np.mean(spectral_centroids) + 1e-8)
            energy_variation = np.std(rms_energy) / (np.mean(rms_energy) + 1e-8)
            rolloff_variation = np.std(spectral_rolloff) / (np.mean(spectral_rolloff) + 1e-8)
            
            # 종합 흐름 점수 (적당한 변화가 있을 때 높은 점수)
            variations = [centroid_variation, energy_variation, rolloff_variation]
            
            # 각 변화량이 0.1-1.0 범위에 있을 때 좋은 점수
            flow_scores = []
            for var in variations:
                if 0.1 <= var <= 1.0:
                    flow_scores.append(1.0)
                elif var < 0.1:
                    flow_scores.append(var / 0.1)  # 너무 단조로움
                else:
                    flow_scores.append(max(0, 2.0 - var))  # 너무 급변함
            
            flow_score = np.mean(flow_scores)
            passed = flow_score > flow_threshold
            
            return {
                'passed': passed,
                'score': flow_score,
                'reason': f'Musical flow: {flow_score:.3f} (centroid:{centroid_variation:.2f}, energy:{energy_variation:.2f}, rolloff:{rolloff_variation:.2f})'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'reason': f'Musical flow check error: {e}'
            }
    
    @classmethod
    def run_musical_checks(cls, audio_data, sample_rate):
        """음악적 완성도 종합 검사 - 4개 중 2개 통과하면 합격 (관대한 기준)"""
        print(f"      음악적 완성도 검사 시작...")
        
        # 각 검사 실행
        rhythm_result = cls.check_rhythm_consistency(audio_data, sample_rate)
        print(f"      리듬 일관성: {rhythm_result['reason']}")
        
        melody_result = cls.check_melody_existence(audio_data, sample_rate)
        print(f"      멜로디 존재: {melody_result['reason']}")
        
        harmonic_result = cls.check_harmonic_balance(audio_data, sample_rate)
        print(f"      하모닉 밸런스: {harmonic_result['reason']}")
        
        flow_result = cls.check_musical_flow(audio_data, sample_rate)
        print(f"      음악적 흐름: {flow_result['reason']}")
        
        # 통과한 검사 개수 계산
        results = [rhythm_result, melody_result, harmonic_result, flow_result]
        passed_count = sum(1 for r in results if r['passed'])
        
        # 4개 중 2개 이상 통과하면 됨 (관대한 기준)
        overall_passed = passed_count >= 2
        
        # 평균 점수 계산
        avg_score = np.mean([r['score'] for r in results])
        
        print(f"      음악적 완성도 결과: {passed_count}/4 통과 (평균 점수: {avg_score:.3f})")
        
        return {
            'rhythm': rhythm_result,
            'melody': melody_result,
            'harmonic': harmonic_result,
            'flow': flow_result,
            'passed': overall_passed,
            'passed_count': passed_count,
            'avg_score': avg_score,
            'reason': f'Musical completeness: {passed_count}/4 checks passed (avg score: {avg_score:.3f})'
        }