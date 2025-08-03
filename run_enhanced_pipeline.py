#!/usr/bin/env python3
"""
MusicGen 음악 중심 적응형 품질 필터링 파이프라인

사용법: 
    python run_enhanced_pipeline.py

하나의 프롬프트 → 1개의 곡 생성 & 2단계 음악 평가
EXCELLENT/GOOD 통과할 때까지 계속 재생성합니다.
"""

import sys
import os
import time
from audiocraft.data.audio import audio_write
from pipeline.music_generator import MusicGenerator
from pipeline.enhanced_pipeline import EnhancedQualityPipeline
from utils.audio_utils import print_separator, ensure_output_directory


class EnhancedAdaptiveMusicPipeline:
    """음악 중심 적응형 파이프라인"""
    
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.generator = MusicGenerator()
        self.enhanced_pipeline = EnhancedQualityPipeline()
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
    def process_prompt_adaptive(self, prompt):
        """품질 통과까지 재생성하는 적응형 처리"""
        start_time = time.time()
        
        print(f"프롬프트: '{prompt}'")
        print(f"목표: 2단계 음악 평가 통과한 음악 1개 생성")
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
            
            # 2. 2단계 음악 평가
            print(f"  2단계 음악 평가 시작...")
            enhanced_result = self.enhanced_pipeline.evaluate_single_music(
                gen_result['audio_data'], 
                gen_result['sample_rate'],
                prompt
            )
            
            # 3. 파일 저장
            filename = self._save_attempt_file(
                gen_result, enhanced_result, attempt_count
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
                'status': enhanced_result['status'],
                'total_score': enhanced_result['total_score'],
                'filename': filename,
                'enhanced_result': enhanced_result,
                'generation_time': gen_result['generation_time']
            }
            attempts.append(attempt_record)
            
            # 5. 결과 출력 및 종료 조건 확인
            if enhanced_result['status'] in ['EXCELLENT', 'GOOD']:
                print(f"  ✅ 음악 평가 통과! (상태: {enhanced_result['status']}, 점수: {enhanced_result['total_score']:.3f})")
                print(f"  저장됨: {filename}")
                break
            else:
                print(f"  ❌ 음악 평가 실패 (상태: {enhanced_result['status']}, 점수: {enhanced_result['total_score']:.3f})")
                self._print_failure_reasons(enhanced_result)
                print(f"  저장됨: {filename}")
                print(f"  다시 시도합니다...")
        
        # 6. 최종 결과 처리
        total_time = time.time() - start_time
        self._print_enhanced_report(attempts, total_time, prompt)
        
        return {
            'prompt': prompt,
            'attempts': attempts,
            'success_attempt_number': attempt_count,
            'total_time': total_time,
            'final_status': enhanced_result['status'],
            'final_score': enhanced_result['total_score']
        }
    
    def _save_attempt_file(self, audio_result, enhanced_result, attempt_count):
        """시도별 파일 저장 (음악 평가 결과 기반)"""
        try:
            # 출력 디렉토리 다시 확인
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir, exist_ok=True)
                print(f"    출력 디렉토리 생성: {self.output_dir}")
            
            # 상태에 따른 파일명 결정
            status = enhanced_result['status']
            score = enhanced_result['total_score']
            
            if status in ['EXCELLENT', 'GOOD']:
                filename = f"attempt{attempt_count}_{status.lower()}_score{score:.2f}_FINAL"
            else:
                filename = f"attempt{attempt_count}_fail_score{score:.2f}"
            
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
                
                status = enhanced_result['status']
                score = enhanced_result['total_score']
                
                if status in ['EXCELLENT', 'GOOD']:
                    filename = f"attempt{attempt_count}_{status.lower()}_score{score:.2f}_FINAL.wav"
                else:
                    filename = f"attempt{attempt_count}_fail_score{score:.2f}.wav"
                
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
    
    def _print_failure_reasons(self, enhanced_result):
        """실패 이유 상세 출력 (2단계 버전)"""
        print(f"    📊 단계별 실패 분석:")
        
        # 1단계 기본 품질 결과
        if enhanced_result['basic_result'] and not enhanced_result['basic_result']['overall_passed']:
            print(f"      [1단계] 기본 품질 실패:")
            basic = enhanced_result['basic_result']
            if not basic['duration']['passed']:
                print(f"        - {basic['duration']['reason']}")
            if not basic['high_frequency']['passed']:
                print(f"        - {basic['high_frequency']['reason']}")
            if not basic['extreme_frequencies']['passed']:
                print(f"        - {basic['extreme_frequencies']['reason']}")
        
        # 2단계 음악적 완성도 결과
        if enhanced_result['musical_result'] and not enhanced_result['musical_result']['passed']:
            print(f"      [2단계] 음악적 완성도 부족 ({enhanced_result['musical_result']['passed_count']}/4 통과):")
            musical = enhanced_result['musical_result']
            if not musical['rhythm']['passed']:
                print(f"        - 리듬: {musical['rhythm']['reason']}")
            if not musical['melody']['passed']:
                print(f"        - 멜로디: {musical['melody']['reason']}")
            if not musical['harmonic']['passed']:
                print(f"        - 하모닉: {musical['harmonic']['reason']}")
            if not musical['flow']['passed']:
                print(f"        - 흐름: {musical['flow']['reason']}")
        
        # 개선 제안
        recommendations = self.enhanced_pipeline.get_retry_recommendations(enhanced_result)
        if recommendations:
            print(f"    💡 개선 제안:")
            for rec in recommendations:
                print(f"        - {rec}")
    
    def _print_enhanced_report(self, attempts, total_time, prompt):
        """음악 중심 파이프라인 리포트 출력"""
        print(f"\n{'='*70}")
        print(f"2단계 음악 중심 평가 최종 결과 리포트")
        print(f"{'='*70}")
        print(f"성공: {len(attempts)}번째 시도에서 품질 기준 통과")
        print(f"총 소요 시간: {total_time:.1f}초")
        
        final_attempt = attempts[-1] if attempts else None
        if final_attempt:
            print(f"최종 상태: {final_attempt['status']}")
            print(f"최종 점수: {final_attempt['total_score']:.3f}")
            print(f"최종 파일: {final_attempt['filename']}")
        
        # 시도별 상세 결과
        print(f"\n📊 시도별 상세 결과:")
        for attempt in attempts:
            if not attempt['success']:
                print(f"  시도 {attempt['attempt']}: 생성 실패 ({attempt['error']})")
            else:
                status_emoji = "🏆" if attempt['status'] == 'EXCELLENT' else "✅" if attempt['status'] == 'GOOD' else "❌"
                print(f"  시도 {attempt['attempt']}: {status_emoji} {attempt['status']} (점수: {attempt['total_score']:.3f}, {attempt['generation_time']:.1f}초)")
        
        # 상태별 통계
        status_counts = {}
        score_sum = 0
        valid_attempts = [a for a in attempts if a['success']]
        
        for attempt in valid_attempts:
            status = attempt['status']
            status_counts[status] = status_counts.get(status, 0) + 1
            score_sum += attempt['total_score']
        
        if valid_attempts:
            print(f"\n📈 평가 통계:")
            for status, count in status_counts.items():
                print(f"  - {status}: {count}회")
            print(f"  - 평균 점수: {score_sum / len(valid_attempts):.3f}")
            
            # 성공률 계산
            success_count = status_counts.get('EXCELLENT', 0) + status_counts.get('GOOD', 0)
            success_rate = success_count / len(valid_attempts) * 100
            print(f"  - 성공률: {success_rate:.1f}%")
        
        print(f"{'='*70}")
        
        # 상세 리포트 생성 (선택적)
        if final_attempt and final_attempt.get('enhanced_result'):
            print(f"\n🔍 상세 평가 리포트:")
            detailed_report = self.enhanced_pipeline.generate_detailed_report(final_attempt['enhanced_result'])
            print(detailed_report)


def print_welcome():
    """환영 메시지 출력"""
    print_separator("MusicGen 음악 중심 적응형 품질 필터링 파이프라인", width=70, char="=")
    print()
    print("🎵 2단계 음악 중심 평가 시스템:")
    print("  1단계: 기본 품질 필터 (길이, 노이즈, 극단주파수) - 30%")
    print("  2단계: 음악적 완성도 (리듬, 멜로디, 하모닉, 흐름) - 70%")
    print()
    print("🎯 평가 기준:")
    print("  - EXCELLENT (0.8+): 재생성 불필요, 우수한 품질")
    print("  - GOOD (0.65+): 재생성 불필요, 양호한 품질")
    print("  - RETRY (0.65-): 재생성 권장")
    print()
    print("💾 파일 저장: attempt{번호}_{상태}_score{점수}_FINAL.wav")
    print()
    print("🎼 음악적 완성도에 집중한 순수 음악 평가!")
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
        print("\n🔧 음악 중심 파이프라인 초기화 중...")
        pipeline = EnhancedAdaptiveMusicPipeline(output_dir=output_dir)
        print("✅ 초기화 완료!")
    except Exception as e:
        print(f"❌ 파이프라인 초기화 실패: {e}")
        print("필요한 라이브러리가 설치되었는지 확인하세요:")
        print("  - pip install librosa soundfile")
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
        
        # 음악 중심 적응형 파이프라인 실행
        try:
            session_count += 1
            print(f"\n🚀 세션 {session_count} 시작...")
            
            result = pipeline.process_prompt_adaptive(prompt)
            
            # 추가 정보 출력
            if result['success_attempt_number'] == 1:
                print(f"\n🎉 첫 번째 시도에서 바로 {result['final_status']} 달성!")
            else:
                print(f"\n🎊 {result['success_attempt_number']}번의 시도 끝에 {result['final_status']} 품질 음악 생성!")
                
        except KeyboardInterrupt:
            print("\n\n⏹️ 생성을 중단했습니다.")
            continue
        except Exception as e:
            print(f"\n❌ 오류가 발생했습니다: {e}")
            print("다시 시도해보세요.")
            continue
    
    # 종료 메시지
    if session_count > 0:
        print(f"\n🏁 총 {session_count}개의 세션을 처리했습니다.")
        print(f"📁 생성된 파일들은 '{output_dir}' 폴더에서 확인할 수 있습니다.")
    
    print("\n🙏 감사합니다!")
    return 0


if __name__ == "__main__":
    sys.exit(main())