#!/usr/bin/env python3
"""
MusicGen 적응형 품질 필터링 파이프라인

사용법: 
    python run_pipeline.py

하나의 프롬프트 → 1개의 곡 생성 & 파이프라인 무한 검증
품질 기준을 통과할 때까지 계속 재생성합니다.
"""

import sys
import os
import time
from audiocraft.data.audio import audio_write
from pipeline.music_generator import MusicGenerator
from filters.audio_filters import AudioQualityFilters
from utils.audio_utils import print_separator, ensure_output_directory


class AdaptiveMusicPipeline:
    """적응형 음악 파이프라인"""
    
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.generator = MusicGenerator()
        self.filters = AudioQualityFilters()
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
    def process_prompt_adaptive(self, prompt):
        """품질 통과까지 재생성하는 적응형 처리"""
        start_time = time.time()
        
        print(f"프롬프트: '{prompt}'")
        print(f"목표: 품질 검사 통과한 음악 1개 생성")
        print("=" * 60)
        
        attempts = []
        attempt_count = 0
        
        while True:
            attempt_count += 1
            print(f"\n시도 {attempt_count}")
            
            # 1. 음악 생성
            print(f"  음악 생성 중...", end=" ")
            gen_result = self.generator.generate_single(prompt)
            
            if not gen_result['success']:
                print(f"실패: {gen_result['error']}")
                attempts.append({
                    'attempt': attempt_count,
                    'success': False,
                    'error': gen_result['error'],
                    'generation_time': 0
                })
                print(f"  다시 시도합니다...")
                continue
            
            print(f"완료 ({gen_result['generation_time']:.1f}초)")
            
            # 디버깅용: 생성된 데이터 정보 출력
            print(f"  생성된 데이터 정보:")
            print(f"    샘플링 레이트: {gen_result['sample_rate']}")
            print(f"    오디오 길이: {gen_result['duration']:.1f}초")
            print(f"    텐서 크기: {gen_result['wav_tensor'].shape}")
            print(f"    출력 디렉토리: {self.output_dir}")
            
            # 2. 품질 검사
            print(f"  품질 검사 중...")
            quality_result = self._run_quality_checks_with_log(
                gen_result['audio_data'], 
                gen_result['sample_rate']
            )
            
            # 3. 파일 저장
            filename = self._save_attempt_file(
                gen_result, quality_result, attempt_count
            )
            
            if filename is None:
                print(f"  파일 저장 실패!")
                attempts.append({
                    'attempt': attempt_count,
                    'success': False,
                    'error': "파일 저장 실패",
                    'generation_time': gen_result['generation_time']
                })
                print(f"  다시 시도합니다...")
                continue
            
            # 4. 시도 기록
            attempt_record = {
                'attempt': attempt_count,
                'success': True,
                'passed': quality_result['overall_passed'],
                'filename': filename,
                'quality': quality_result,
                'generation_time': gen_result['generation_time']
            }
            attempts.append(attempt_record)
            
            # 5. 결과 출력 및 종료 조건 확인
            if quality_result['overall_passed']:
                print(f"  품질 검사 통과!")
                print(f"  저장됨: {filename}")
                break
            else:
                print(f"  품질 검사 실패")
                self._print_failure_reasons(quality_result)
                print(f"  저장됨: {filename}")
                print(f"  다시 시도합니다...")
        
        # 6. 최종 결과 처리
        total_time = time.time() - start_time
        self._print_adaptive_report(attempts, total_time, prompt)
        
        return {
            'prompt': prompt,
            'attempts': attempts,
            'success_attempt_number': attempt_count,
            'total_time': total_time
        }
    
    def _run_quality_checks_with_log(self, audio_data, sample_rate):
        """품질 검사를 로그와 함께 실행"""
        # 길이 검사
        print(f"    길이 검사...", end=" ")
        duration_result = self.filters.check_duration(audio_data, sample_rate)
        if duration_result['passed']:
            print(f"통과: {duration_result['reason']}")
        else:
            print(f"실패: {duration_result['reason']}")
        
        # 고주파 노이즈 검사
        print(f"    고주파 노이즈 검사...", end=" ")
        high_freq_result = self.filters.check_high_frequency_noise(audio_data, sample_rate)
        if high_freq_result['passed']:
            print(f"통과: {high_freq_result['reason']}")
        else:
            print(f"실패: {high_freq_result['reason']}")
        
        # 극단 주파수 검사
        print(f"    극단 주파수 검사...", end=" ")
        extreme_freq_result = self.filters.check_extreme_frequencies(audio_data, sample_rate)
        if extreme_freq_result['passed']:
            print(f"통과: {extreme_freq_result['reason']}")
        else:
            print(f"실패: {extreme_freq_result['reason']}")
        
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
    
    def _save_attempt_file(self, audio_result, quality_result, attempt_count):
        """시도별 파일 저장"""
        try:
            # 출력 디렉토리 다시 확인
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir, exist_ok=True)
                print(f"    출력 디렉토리 생성: {self.output_dir}")
            
            if quality_result['overall_passed']:
                filename = f"attempt{attempt_count}_pass_FINAL"
            else:
                filename = f"attempt{attempt_count}_fail"
            
            file_path = os.path.join(self.output_dir, filename)
            print(f"    저장 경로: {file_path}")
            
            # audio_write 함수를 올바르게 사용
            audio_write(
                file_path, 
                audio_result['wav_tensor'], 
                audio_result['sample_rate'], 
                strategy="loudness"
            )
            
            # 파일이 실제로 생성되었는지 확인
            expected_file = file_path + ".wav"
            if os.path.exists(expected_file):
                print(f"    저장 성공: {expected_file}")
                return filename + ".wav"
            else:
                print(f"    저장된 파일을 찾을 수 없음: {expected_file}")
                raise FileNotFoundError(f"파일이 생성되지 않음: {expected_file}")
            
        except Exception as e:
            print(f"    파일 저장 오류: {e}")
            # 대체 저장 방법 시도
            try:
                import torch
                import torchaudio
                
                if quality_result['overall_passed']:
                    filename = f"attempt{attempt_count}_pass_FINAL.wav"
                else:
                    filename = f"attempt{attempt_count}_fail.wav"
                
                file_path = os.path.join(self.output_dir, filename)
                torchaudio.save(file_path, audio_result['wav_tensor'], audio_result['sample_rate'])
                
                if os.path.exists(file_path):
                    print(f"    대체 방법으로 저장 성공: {filename}")
                    return filename
                else:
                    print(f"    대체 저장도 파일 생성 실패")
                    return None
                
            except Exception as e2:
                print(f"    대체 저장도 실패: {e2}")
                return None
    
    def _print_failure_reasons(self, quality_result):
        """실패 이유 출력"""
        failed_reasons = []
        for check_name, check_result in quality_result.items():
            if check_name != 'overall_passed' and not check_result['passed']:
                failed_reasons.append(f"    - {check_result['reason']}")
        
        for reason in failed_reasons:
            print(reason)
    
    def _print_adaptive_report(self, attempts, total_time, prompt):
        """적응형 파이프라인 리포트 출력"""
        print(f"\n{'='*60}")
        print(f"최종 결과 리포트")
        print(f"{'='*60}")
        print(f"성공: {len(attempts)}번째 시도에서 품질 기준 통과")
        print(f"총 소요 시간: {total_time:.1f}초")
        
        final_filename = attempts[-1]['filename'] if attempts else None
        if final_filename:
            print(f"최종 파일: {final_filename}")
        
        # 시도별 상세 결과
        print(f"\n시도별 상세 결과:")
        for attempt in attempts:
            if not attempt['success']:
                print(f"  시도 {attempt['attempt']}: 생성 실패 ({attempt['error']})")
            elif attempt['passed']:
                print(f"  시도 {attempt['attempt']}: 성공 ({attempt['generation_time']:.1f}초)")
            else:
                print(f"  시도 {attempt['attempt']}: 실패 ({attempt['generation_time']:.1f}초)")
        
        # 실패 원인 통계
        failure_reasons = {}
        for attempt in attempts:
            if not attempt['passed'] and attempt['success'] and attempt.get('quality'):
                for check_name, check_result in attempt['quality'].items():
                    if check_name != 'overall_passed' and not check_result['passed']:
                        reason = check_result['reason']
                        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        if failure_reasons:
            print(f"\n실패 원인 통계:")
            for reason, count in failure_reasons.items():
                print(f"  - {reason}: {count}회")
        
        print(f"{'='*60}")


def print_welcome():
    """환영 메시지 출력"""
    print_separator("MusicGen 적응형 품질 필터링 파이프라인", width=70, char="=")
    print()
    print("프롬프트를 입력하면 품질 기준을 통과할 때까지 음악을 생성합니다.")
    print("모든 시도: attempt{번호}_fail.wav 또는 attempt{번호}_pass_FINAL.wav")
    print()
    print("품질 검사 항목:")
    print("  - 길이 검사: 11초 이상")
    print("  - 고주파 노이즈: 8kHz 이상 3초 이상 지속 금지")
    print("  - 극단 주파수: 40Hz 이하 5초 이상 지속 금지")
    print()
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print_separator(width=70, char="=")


def get_user_input():
    """사용자 입력 받기"""
    try:
        prompt = input("\n프롬프트를 입력하세요: ").strip()
        return prompt
    except KeyboardInterrupt:
        print("\n\n프로그램을 종료합니다.")
        return None
    except EOFError:
        return None


def validate_prompt(prompt):
    """프롬프트 유효성 검사"""
    if not prompt:
        print("빈 프롬프트는 입력할 수 없습니다.")
        return False
    
    if len(prompt) > 200:
        print("프롬프트가 너무 깁니다. (최대 200자)")
        return False
    
    return True


def main():
    """메인 함수"""
    print_welcome()
    
    # 출력 디렉토리 확인
    output_dir = "output"
    ensure_output_directory(output_dir)
    
    # 파이프라인 초기화
    try:
        print("\n파이프라인 초기화 중...")
        pipeline = AdaptiveMusicPipeline(output_dir=output_dir)
        print("초기화 완료!")
    except Exception as e:
        print(f"파이프라인 초기화 실패: {e}")
        print("audiocraft 라이브러리와 모델이 올바르게 설치되었는지 확인하세요.")
        return 1
    
    # 메인 루프
    session_count = 0
    
    while True:
        prompt = get_user_input()
        
        # 종료 조건
        if prompt is None:
            break
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("프로그램을 종료합니다.")
            break
        
        # 프롬프트 유효성 검사
        if not validate_prompt(prompt):
            continue
        
        # 적응형 파이프라인 실행
        try:
            session_count += 1
            print(f"\n세션 {session_count} 시작...")
            
            result = pipeline.process_prompt_adaptive(prompt)
            
            # 추가 정보 출력
            if result['success_attempt_number'] == 1:
                print(f"\n첫 번째 시도에서 바로 성공했습니다!")
            else:
                print(f"\n{result['success_attempt_number']}번의 시도 끝에 좋은 음악이 생성되었습니다!")
                
        except KeyboardInterrupt:
            print("\n\n생성을 중단했습니다.")
            continue
        except Exception as e:
            print(f"\n오류가 발생했습니다: {e}")
            print("다시 시도해보세요.")
            continue
    
    # 종료 메시지
    if session_count > 0:
        print(f"\n총 {session_count}개의 세션을 처리했습니다.")
        print(f"생성된 파일들은 '{output_dir}' 폴더에서 확인할 수 있습니다.")
    
    print("\n감사합니다!")
    return 0


if __name__ == "__main__":
    sys.exit(main())