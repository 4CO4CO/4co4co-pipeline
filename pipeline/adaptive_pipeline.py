import os
import time
from audiocraft.data.audio import audio_write
from .music_generator import MusicGenerator
from filters.audio_filters import AudioQualityFilters


class AdaptiveMusicQualityPipeline:
    """적응형 음악 품질 파이프라인 - 품질 통과까지 재생성"""
    
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
            
            # 4. 시도 기록
            attempt_record = {
                'attempt': attempt_count,
                'success': True,
                'passed': quality_result['overall_passed'],
                'filename': filename,
                'quality': quality_result,
                'generation_time': gen_result['generation_time'],
                'audio_data': gen_result['audio_data'],
                'sample_rate': gen_result['sample_rate'],
                'wav_tensor': gen_result['wav_tensor']
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
        final_result = attempts[-1]  # 마지막(성공한) 시도
        
        # 7. 종합 리포트 생성
        summary = self._generate_adaptive_summary(attempts, total_time, prompt)
        
        pipeline_result = {
            'prompt': prompt,
            'attempts': attempts,
            'final_result': final_result,
            'summary': summary
        }
        
        # 8. 최종 리포트 출력
        self._print_adaptive_report(pipeline_result)
        
        return pipeline_result
    
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
        if quality_result['overall_passed']:
            filename = f"attempt{attempt_count}_pass_FINAL"
        else:
            filename = f"attempt{attempt_count}_fail"
        
        file_path = os.path.join(self.output_dir, filename)
        audio_write(
            file_path, 
            audio_result['wav_tensor'], 
            audio_result['sample_rate'], 
            strategy="loudness"
        )
        
        return filename + ".wav"
    
    def _print_failure_reasons(self, quality_result):
        """실패 이유 출력"""
        failed_reasons = []
        for check_name, check_result in quality_result.items():
            if check_name != 'overall_passed' and not check_result['passed']:
                failed_reasons.append(f"    - {check_result['reason']}")
        
        for reason in failed_reasons:
            print(reason)
    
    def _generate_adaptive_summary(self, attempts, total_time, prompt):
        """적응형 파이프라인 요약 생성"""
        successful_attempts = [a for a in attempts if a['success']]
        passed_attempts = [a for a in successful_attempts if a.get('passed', False)]
        
        # 실패 원인 통계
        failure_reasons = {}
        for attempt in successful_attempts:
            if not attempt.get('passed', False) and attempt.get('quality'):
                for check_name, check_result in attempt['quality'].items():
                    if check_name != 'overall_passed' and not check_result['passed']:
                        reason = check_result['reason']
                        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        return {
            'total_attempts': len(attempts),
            'successful_generations': len(successful_attempts),
            'success_attempt_number': len(attempts),  # 마지막 시도가 성공
            'total_time': total_time,
            'avg_generation_time': sum(a.get('generation_time', 0) for a in successful_attempts) / len(successful_attempts) if successful_attempts else 0,
            'failure_reasons': failure_reasons,
            'final_filename': attempts[-1]['filename'] if attempts else None
        }
    
    def _print_adaptive_report(self, pipeline_result):
        """적응형 파이프라인 리포트 출력"""
        summary = pipeline_result['summary']
        
        print(f"\n{'='*60}")
        print(f"최종 결과 리포트")
        print(f"{'='*60}")
        print(f"성공: {summary['success_attempt_number']}번째 시도에서 품질 기준 통과")
        print(f"총 소요 시간: {summary['total_time']:.1f}초")
        print(f"최종 파일: {summary['final_filename']}")
        
        # 시도별 상세 결과
        print(f"\n시도별 상세 결과:")
        for attempt in pipeline_result['attempts']:
            if not attempt['success']:
                print(f"  시도 {attempt['attempt']}: 생성 실패 ({attempt['error']})")
            elif attempt.get('passed', False):
                print(f"  시도 {attempt['attempt']}: 성공 ({attempt['generation_time']:.1f}초)")
            else:
                print(f"  시도 {attempt['attempt']}: 실패 ({attempt['generation_time']:.1f}초)")
        
        # 실패 원인 통계
        if summary['failure_reasons']:
            print(f"\n실패 원인 통계:")
            for reason, count in summary['failure_reasons'].items():
                print(f"  - {reason}: {count}회")
        
        print(f"{'='*60}")