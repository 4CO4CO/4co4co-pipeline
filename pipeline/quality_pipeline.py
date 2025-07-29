import os
import time
from audiocraft.data.audio import audio_write
from .music_generator import MusicGenerator
from filters.audio_filters import AudioQualityFilters


class MusicQualityPipeline:
    """메인 파이프라인 클래스"""
    
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.generator = MusicGenerator()
        self.filters = AudioQualityFilters()
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
    def process_prompt(self, prompt, batch_size=5):
        """프롬프트 처리 메인 함수"""
        start_time = time.time()
        
        # 1. batch_size만큼 음악 생성
        generation_results = self.generator.generate_batch(prompt, batch_size)
        
        # 2. 각각 품질 검사 및 저장
        results = []
        
        print(f"\n🔍 품질 검사 및 파일 저장...")
        
        for i, gen_result in enumerate(generation_results, 1):
            if not gen_result['success']:
                # 생성 실패한 경우
                result = {
                    'index': i,
                    'passed': False,
                    'filename': None,
                    'quality': None,
                    'generation_time': 0,
                    'error': gen_result['error']
                }
                print(f"  샘플 {i}: ❌ 생성 실패 ({gen_result['error']})")
            else:
                # 3. 품질 검사
                quality_result = self.filters.run_all_checks(
                    gen_result['audio_data'], 
                    gen_result['sample_rate']
                )
                
                # 4. 파일 저장
                filename = self._save_audio_with_result(
                    gen_result, quality_result, i, prompt
                )
                
                result = {
                    'index': i,
                    'passed': quality_result['overall_passed'],
                    'filename': filename,
                    'quality': quality_result,
                    'generation_time': gen_result['generation_time']
                }
                
                # 결과 출력
                if quality_result['overall_passed']:
                    print(f"  샘플 {i}: ✅ 통과 → {filename}")
                else:
                    failed_reasons = []
                    for check_name, check_result in quality_result.items():
                        if check_name != 'overall_passed' and not check_result['passed']:
                            failed_reasons.append(check_result['reason'])
                    print(f"  샘플 {i}: ❌ 실패 → {filename}")
                    for reason in failed_reasons:
                        print(f"    - {reason}")
            
            results.append(result)
        
        # 5. 결과 리포트 생성
        total_time = time.time() - start_time
        summary = self._generate_summary(results, total_time)
        
        pipeline_result = {
            'prompt': prompt,
            'batch_size': batch_size,
            'results': results,
            'summary': summary
        }
        
        # 6. 결과 리포트 출력
        self._print_report(pipeline_result)
        
        return pipeline_result
        
    def _save_audio_with_result(self, audio_result, quality_result, index, prompt):
        """품질 검사 결과에 따라 파일 저장"""
        # 파일명 결정
        if quality_result['overall_passed']:
            filename = f"sample{index}_pass"
        else:
            filename = f"sample{index}_fail"
        
        # 파일 저장
        file_path = os.path.join(self.output_dir, filename)
        audio_write(
            file_path, 
            audio_result['wav_tensor'], 
            audio_result['sample_rate'], 
            strategy="loudness"
        )
        
        return filename + ".wav"
        
    def _generate_summary(self, results, total_time):
        """결과 요약 생성"""
        success_count = sum(1 for r in results if r['passed'])
        fail_count = len(results) - success_count
        success_rate = success_count / len(results) if results else 0
        
        passed_files = [r['filename'] for r in results if r['passed'] and r['filename']]
        failed_files = [r['filename'] for r in results if not r['passed'] and r['filename']]
        
        # 실패 원인 통계
        failure_reasons = {}
        for result in results:
            if not result['passed'] and result['quality']:
                for check_name, check_result in result['quality'].items():
                    if check_name != 'overall_passed' and not check_result['passed']:
                        reason = check_result['reason']
                        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        return {
            'success_count': success_count,
            'fail_count': fail_count,
            'success_rate': success_rate,
            'total_time': total_time,
            'passed_files': passed_files,
            'failed_files': failed_files,
            'failure_reasons': failure_reasons
        }
        
    def _print_report(self, pipeline_result):
        """결과 리포트 출력"""
        summary = pipeline_result['summary']
        
        print(f"\n{'='*60}")
        print(f"📊 결과 리포트")
        print(f"{'='*60}")
        print(f"프롬프트: '{pipeline_result['prompt']}'")
        print(f"전체 시간: {summary['total_time']:.1f}초")
        print(f"성공률: {summary['success_count']}/{pipeline_result['batch_size']} ({summary['success_rate']:.1%})")
        
        if summary['passed_files']:
            print(f"\n✅ 통과한 파일 ({len(summary['passed_files'])}개):")
            for filename in summary['passed_files']:
                print(f"  - {filename}")
        
        if summary['failed_files']:
            print(f"\n❌ 실패한 파일 ({len(summary['failed_files'])}개):")
            for filename in summary['failed_files']:
                print(f"  - {filename}")
        
        if summary['failure_reasons']:
            print(f"\n🔍 실패 원인 통계:")
            for reason, count in summary['failure_reasons'].items():
                print(f"  - {reason}: {count}회")
        
        print(f"{'='*60}")