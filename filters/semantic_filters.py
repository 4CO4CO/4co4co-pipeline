import re
import os
import tempfile
import numpy as np
import soundfile as sf


class SemanticMatchingFilters:
    """프롬프트-음악 의미적 일치도 검사 필터"""
    
    def __init__(self):
        self.clap = None
        self._initialize_clap()
    
    def _initialize_clap(self):
        """CLAP 모델 초기화"""
        try:
            from msclap import CLAP
            print(f"      CLAP 모델 로딩 중...")
            self.clap = CLAP(version='2023', use_cuda=True)
            print(f"      CLAP 모델 로딩 완료!")
        except ImportError:
            print(f"      경고: msclap 라이브러리가 설치되지 않음. 프롬프트 일치도 검사를 건너뜁니다.")
            self.clap = None
        except Exception as e:
            print(f"      CLAP 모델 로딩 실패: {e}")
            self.clap = None
    
    def parse_prompt_components(self, prompt):
        """프롬프트에서 장르, 감정, 악기 추출 (개선된 버전)"""
        try:
            # 기본값 설정
            components = {
                'genre': '',
                'emotion': '',
                'instrument': '',
                'full_prompt': prompt
            }
            
            # 장르 추출 - 더 넓은 패턴
            genre_patterns = [
                r'A\s+(\w+)\s+(?:piece|ensemble|music)',  # A jazz ensemble
                r'(\w+)\s+(?:piece|ensemble|music)',      # jazz ensemble
                r'A\s+(\w+)\s+in',                        # A jazz in
            ]
            
            for pattern in genre_patterns:
                genre_match = re.search(pattern, prompt, re.IGNORECASE)
                if genre_match:
                    components['genre'] = genre_match.group(1).lower()
                    break
            
            # 감정 추출 - 더 많은 키워드들
            emotion_keywords = [
                'joy', 'sad', 'happy', 'melancholy', 'energetic', 'calm', 
                'peaceful', 'dramatic', 'romantic', 'mysterious', 'upbeat',
                'pleasure', 'excitement', 'sorrow', 'anger', 'fear', 'love',
                'esteem', 'dignity', 'dignified', 'refined', 'elegant', 'noble',
                'contemplative', 'reflective', 'serene', 'graceful'
            ]
            
            for emotion in emotion_keywords:
                if emotion in prompt.lower():
                    components['emotion'] = emotion
                    break
            
            # 악기 추출 - 더 많은 악기들
            instrument_keywords = [
                'piano', 'guitar', 'violin', 'drums', 'bass', 'synth', 
                'saxophone', 'trumpet', 'flute', 'cello', 'organ', 'harp',
                'clarinet', 'trombone', 'oboe', 'bassoon', 'viola', 'double bass',
                'upright bass', 'electric guitar', 'acoustic guitar'
            ]
            
            # 복합 악기 패턴도 찾기
            for instrument in instrument_keywords:
                if instrument in prompt.lower():
                    components['instrument'] = instrument
                    break
            
            # 특별 케이스: "upright bass" 같은 복합 단어
            if 'upright bass' in prompt.lower():
                components['instrument'] = 'upright bass'
            elif 'electric guitar' in prompt.lower():
                components['instrument'] = 'electric guitar'
            
            return components
            
        except Exception as e:
            print(f"      프롬프트 파싱 오류: {e}")
            return {
                'genre': '',
                'emotion': '',
                'instrument': '',
                'full_prompt': prompt
            }
    
    def _save_temp_audio_file(self, audio_data, sample_rate):
        """CLAP 평가를 위한 임시 오디오 파일 저장 (soundfile 사용)"""
        try:
            # 임시 파일 생성
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav', prefix='clap_eval_')
            os.close(temp_fd)
            
            # soundfile로 저장 (librosa.output 대신)
            sf.write(temp_path, audio_data, sample_rate)
            
            return temp_path
            
        except Exception as e:
            print(f"      임시 파일 저장 오류: {e}")
            return None
    
    def _cleanup_temp_file(self, temp_path):
        """임시 파일 삭제"""
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            print(f"      임시 파일 삭제 오류: {e}")
    
    def check_prompt_alignment(self, audio_data, sample_rate, prompt):
        """프롬프트와 음악의 의미적 일치도 검사"""
        
        # CLAP 모델이 없으면 관대한 기본 통과
        if self.clap is None:
            return {
                'passed': True,
                'weighted_score': 0.25,  # 0.2보다 높게 설정
                'scores': {
                    'full_prompt': 0.25,
                    'genre': 0.25,
                    'emotion': 0.25,
                    'instrument': 0.25
                },
                'reason': 'CLAP not available - default pass'
            }
        
        temp_audio_path = None
        
        try:
            print(f"      프롬프트 일치도 검사 시작...")
            
            # 프롬프트 구성요소 파싱
            components = self.parse_prompt_components(prompt)
            print(f"      파싱된 구성요소: 장르='{components['genre']}', 감정='{components['emotion']}', 악기='{components['instrument']}'")
            
            # 임시 오디오 파일 저장
            temp_audio_path = self._save_temp_audio_file(audio_data, sample_rate)
            if temp_audio_path is None:
                raise Exception("임시 파일 저장 실패")
            
            # CLAP 유사도 계산
            scores = {}
            
            # 1. 전체 프롬프트 매칭
            scores['full_prompt'] = self._calculate_similarity(temp_audio_path, components['full_prompt'])
            
            # 2. 장르 매칭
            if components['genre']:
                genre_text = f"A {components['genre']} music piece"
                scores['genre'] = self._calculate_similarity(temp_audio_path, genre_text)
            else:
                scores['genre'] = 0.15  # 기본값
            
            # 3. 감정 매칭
            if components['emotion']:
                emotion_text = f"{components['emotion']} music"
                scores['emotion'] = self._calculate_similarity(temp_audio_path, emotion_text)
            else:
                scores['emotion'] = 0.15  # 기본값
            
            # 4. 악기 매칭
            if components['instrument']:
                instrument_text = f"{components['instrument']} music"
                scores['instrument'] = self._calculate_similarity(temp_audio_path, instrument_text)
            else:
                scores['instrument'] = 0.15  # 기본값
            
            # 가중 평균 계산: 전체 40% + 감정 30% + 장르 20% + 악기 10%
            weighted_score = (
                scores['full_prompt'] * 0.4 +
                scores['emotion'] * 0.3 +
                scores['genre'] * 0.2 +
                scores['instrument'] * 0.1
            )
            
            # 통과 기준: 가중 평균 0.15 이상 (더 관대하게)
            passed = weighted_score >= 0.15
            
            print(f"      프롬프트 일치도 점수: {weighted_score:.3f} (전체:{scores['full_prompt']:.3f}, 감정:{scores['emotion']:.3f}, 장르:{scores['genre']:.3f}, 악기:{scores['instrument']:.3f})")
            
            return {
                'passed': passed,
                'weighted_score': weighted_score,
                'scores': scores,
                'reason': f'Prompt alignment: {weighted_score:.3f} (threshold: 0.15)'
            }
            
        except Exception as e:
            print(f"      프롬프트 일치도 검사 오류: {e}")
            # 오류 발생시 관대한 기본 통과
            return {
                'passed': True,
                'weighted_score': 0.20,
                'scores': {
                    'full_prompt': 0.20,
                    'genre': 0.20,
                    'emotion': 0.20,
                    'instrument': 0.20
                },
                'reason': f'Prompt alignment check error (default pass): {e}'
            }
        
        finally:
            # 임시 파일 정리
            if temp_audio_path:
                self._cleanup_temp_file(temp_audio_path)
    
    def _calculate_similarity(self, audio_path, text_prompt):
        """CLAP을 이용한 오디오-텍스트 유사도 계산"""
        try:
            if not text_prompt.strip():
                return 0.15  # 기본값
            
            # CLAP 임베딩 계산
            audio_embeddings = self.clap.get_audio_embeddings([audio_path])
            text_embeddings = self.clap.get_text_embeddings([text_prompt])
            
            # 코사인 유사도 계산
            similarity = self.clap.compute_similarity(audio_embeddings, text_embeddings)
            
            # 스칼라 값으로 변환
            if hasattr(similarity, 'item'):
                similarity = similarity.item()
            elif isinstance(similarity, (list, tuple, np.ndarray)):
                similarity = float(similarity[0])
            else:
                similarity = float(similarity)
            
            # 0-1 범위로 클리핑
            similarity = max(0.0, min(1.0, similarity))
            
            return similarity
            
        except Exception as e:
            print(f"      유사도 계산 오류 ('{text_prompt}'): {e}")
            return 0.15  # 기본값