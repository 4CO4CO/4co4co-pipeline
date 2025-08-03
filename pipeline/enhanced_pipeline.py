import time
import numpy as np
from filters.audio_filters import AudioQualityFilters
from filters.musical_filters import MusicalCompletenessFilters  
from filters.semantic_filters import SemanticMatchingFilters


class EnhancedQualityPipeline:
    """3단계 종합 평가 파이프라인"""
    
    def __init__(self):
        print(f"🔧 강화된 평가 파이프라인 초기화 중...")
        
        # 각 단계별 필터 초기화
        self.basic_filters = AudioQualityFilters()
        self.musical_filters = MusicalCompletenessFilters()
        self.semantic_filters = SemanticMatchingFilters()
        
        print(f"✅ 강화된 평가 파이프라인 초기화 완료!")
    
    def evaluate_single_music(self, audio_data, sample_rate, prompt):
        """개별 음악에 대한 3단계 종합 평가"""
        start_time = time.time()
        
        print(f"    🔍 3단계 종합 평가 시작...")
        
        try:
            # 1단계: 기본 품질 필터 (3초 목표)
            print(f"    [1단계] 기본 품질 필터 검사...")
            stage1_start = time.time()
            
            basic_result = self.basic_filters.run_all_checks(audio_data, sample_rate)
            
            stage1_time = time.time() - stage1_start
            print(f"    [1단계] 완료 ({stage1_time:.1f}초) - 통과: {basic_result['overall_passed']}")
            
            # 1단계 실패시 조기 종료
            if not basic_result['overall_passed']:
                total_time = time.time() - start_time
                return {
                    'status': 'RETRY',
                    'total_score': 0.0,
                    'stage_completed': 1,
                    'basic_result': basic_result,
                    'musical_result': None,
                    'semantic_result': None,
                    'evaluation_time': total_time,
                    'reason': 'Failed basic quality checks'
                }
            
            # 2단계: 음악적 완성도 (4초 목표)
            print(f"    [2단계] 음악적 완성도 검사...")
            stage2_start = time.time()
            
            musical_result = self.musical_filters.run_musical_checks(audio_data, sample_rate)
            
            stage2_time = time.time() - stage2_start
            print(f"    [2단계] 완료 ({stage2_time:.1f}초) - 통과: {musical_result['passed']} ({musical_result['passed_count']}/4)")
            
            # 3단계: 프롬프트 일치도 (5초 목표)
            print(f"    [3단계] 프롬프트 일치도 검사...")
            stage3_start = time.time()
            
            semantic_result = self.semantic_filters.check_prompt_alignment(audio_data, sample_rate, prompt)
            
            stage3_time = time.time() - stage3_start
            print(f"    [3단계] 완료 ({stage3_time:.1f}초) - 통과: {semantic_result['passed']} (점수: {semantic_result['weighted_score']:.3f})")
            
            # 종합 점수 계산 및 최종 판정
            total_score = self._calculate_total_score(basic_result, musical_result, semantic_result)
            status = self._determine_status(total_score, basic_result, musical_result, semantic_result)
            
            total_time = time.time() - start_time
            
            print(f"    🎯 종합 평가 완료 ({total_time:.1f}초) - 상태: {status}, 점수: {total_score:.3f}")
            
            return {
                'status': status,
                'total_score': total_score,
                'stage_completed': 3,
                'basic_result': basic_result,
                'musical_result': musical_result,
                'semantic_result': semantic_result,
                'evaluation_time': total_time,
                'stage_times': {
                    'basic': stage1_time,
                    'musical': stage2_time,
                    'semantic': stage3_time
                },
                'reason': f'Enhanced evaluation: {status} (score: {total_score:.3f})'
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            print(f"    ❌ 평가 중 오류 발생: {e}")
            
            return {
                'status': 'RETRY',
                'total_score': 0.0,
                'stage_completed': 0,
                'basic_result': None,
                'musical_result': None,
                'semantic_result': None,
                'evaluation_time': total_time,
                'reason': f'Evaluation error: {e}'
            }
    
    def _calculate_total_score(self, basic_result, musical_result, semantic_result):
        """종합 점수 계산"""
        try:
            # 1단계: 기본 품질 점수 (20%)
            basic_score = 1.0 if basic_result['overall_passed'] else 0.0
            
            # 2단계: 음악적 완성도 점수 (30%)
            musical_score = musical_result['avg_score'] if musical_result['passed'] else 0.0
            
            # 3단계: 프롬프트 일치도 점수 (40%)
            semantic_score = semantic_result['weighted_score']
            
            # 보너스 점수 (10%) - 모든 단계 통과시 추가 점수
            bonus_score = 0.1 if (basic_result['overall_passed'] and 
                                musical_result['passed'] and 
                                semantic_result['passed']) else 0.0
            
            # 가중 평균 계산
            total_score = (
                basic_score * 0.2 +
                musical_score * 0.3 +
                semantic_score * 0.4 +
                bonus_score
            )
            
            # 0-1 범위로 클리핑
            total_score = max(0.0, min(1.0, total_score))
            
            return total_score
            
        except Exception as e:
            print(f"    점수 계산 오류: {e}")
            return 0.0
    
    def _determine_status(self, total_score, basic_result, musical_result, semantic_result):
        """총점을 기반으로 최종 상태 결정"""
        try:
            # 기본 품질을 통과하지 못하면 무조건 RETRY
            if not basic_result['overall_passed']:
                return 'RETRY'
            
            # 점수 기반 판정
            if total_score >= 0.7:
                return 'EXCELLENT'
            elif total_score >= 0.5:
                return 'GOOD'
            else:
                return 'RETRY'
                
        except Exception as e:
            print(f"    상태 결정 오류: {e}")
            return 'RETRY'
    
    def generate_detailed_report(self, evaluation_result):
        """상세 평가 리포트 생성"""
        try:
            report = []
            report.append(f"=== 3단계 종합 평가 리포트 ===")
            report.append(f"최종 상태: {evaluation_result['status']}")
            report.append(f"종합 점수: {evaluation_result['total_score']:.3f}")
            report.append(f"평가 시간: {evaluation_result['evaluation_time']:.1f}초")
            report.append(f"완료 단계: {evaluation_result['stage_completed']}/3")
            report.append("")
            
            # 1단계 상세 결과
            if evaluation_result['basic_result']:
                basic = evaluation_result['basic_result']
                report.append(f"[1단계] 기본 품질 필터:")
                report.append(f"  - 전체 통과: {basic['overall_passed']}")
                report.append(f"  - 길이 검사: {basic['duration']['reason']}")
                report.append(f"  - 고주파 노이즈: {basic['high_frequency']['reason']}")
                report.append(f"  - 극단 주파수: {basic['extreme_frequencies']['reason']}")
                report.append("")
            
            # 2단계 상세 결과
            if evaluation_result['musical_result']:
                musical = evaluation_result['musical_result']
                report.append(f"[2단계] 음악적 완성도:")
                report.append(f"  - 전체 통과: {musical['passed']} ({musical['passed_count']}/4)")
                report.append(f"  - 평균 점수: {musical['avg_score']:.3f}")
                report.append(f"  - 리듬 일관성: {'✅' if musical['rhythm']['passed'] else '❌'} ({musical['rhythm']['score']:.3f})")
                report.append(f"  - 멜로디 존재: {'✅' if musical['melody']['passed'] else '❌'} ({musical['melody']['score']:.3f})")
                report.append(f"  - 하모닉 밸런스: {'✅' if musical['harmonic']['passed'] else '❌'} ({musical['harmonic']['score']:.3f})")
                report.append(f"  - 음악적 흐름: {'✅' if musical['flow']['passed'] else '❌'} ({musical['flow']['score']:.3f})")
                report.append("")
            
            # 3단계 상세 결과
            if evaluation_result['semantic_result']:
                semantic = evaluation_result['semantic_result']
                report.append(f"[3단계] 프롬프트 일치도:")
                report.append(f"  - 전체 통과: {semantic['passed']}")
                report.append(f"  - 가중 점수: {semantic['weighted_score']:.3f}")
                if 'scores' in semantic:
                    scores = semantic['scores']
                    report.append(f"  - 전체 프롬프트: {scores['full_prompt']:.3f}")
                    report.append(f"  - 감정 매칭: {scores['emotion']:.3f}")
                    report.append(f"  - 장르 매칭: {scores['genre']:.3f}")
                    report.append(f"  - 악기 매칭: {scores['instrument']:.3f}")
                report.append("")
            
            # 단계별 실행 시간
            if 'stage_times' in evaluation_result:
                times = evaluation_result['stage_times']
                report.append(f"단계별 실행 시간:")
                report.append(f"  - 1단계 (기본): {times['basic']:.1f}초")
                report.append(f"  - 2단계 (음악): {times['musical']:.1f}초")
                report.append(f"  - 3단계 (의미): {times['semantic']:.1f}초")
                report.append("")
            
            # 개선 제안
            report.append(f"개선 제안:")
            if evaluation_result['status'] == 'RETRY':
                if evaluation_result['basic_result'] and not evaluation_result['basic_result']['overall_passed']:
                    report.append(f"  - 기본 품질 문제로 재생성 필요")
                elif evaluation_result['musical_result'] and not evaluation_result['musical_result']['passed']:
                    report.append(f"  - 음악적 완성도 부족 (리듬, 멜로디, 하모니, 흐름 개선 필요)")
                elif evaluation_result['semantic_result'] and not evaluation_result['semantic_result']['passed']:
                    report.append(f"  - 프롬프트 일치도 부족 (더 명확한 프롬프트 필요)")
                else:
                    report.append(f"  - 종합 점수 부족 (전반적 품질 개선 필요)")
            elif evaluation_result['status'] == 'GOOD':
                report.append(f"  - 양호한 품질, 추가 개선 여지 있음")
            else:
                report.append(f"  - 우수한 품질, 개선 불필요")
            
            return "\n".join(report)
            
        except Exception as e:
            return f"리포트 생성 오류: {e}"
    
    def evaluate_batch(self, music_data_list, prompts):
        """여러 음악에 대한 배치 평가 (3곡 처리용)"""
        results = []
        
        print(f"🎵 배치 평가 시작 ({len(music_data_list)}곡)...")
        
        for i, (audio_data, sample_rate) in enumerate(music_data_list):
            print(f"\n[음악 {i+1}/{len(music_data_list)}] 평가 중...")
            
            prompt = prompts[i] if i < len(prompts) else prompts[0]
            result = self.evaluate_single_music(audio_data, sample_rate, prompt)
            results.append(result)
            
            print(f"[음악 {i+1}] 결과: {result['status']} (점수: {result['total_score']:.3f})")
        
        # 배치 결과 요약
        excellent_count = sum(1 for r in results if r['status'] == 'EXCELLENT')
        good_count = sum(1 for r in results if r['status'] == 'GOOD')
        retry_count = sum(1 for r in results if r['status'] == 'RETRY')
        
        print(f"\n🎯 배치 평가 완료:")
        print(f"  - EXCELLENT: {excellent_count}곡")
        print(f"  - GOOD: {good_count}곡")
        print(f"  - RETRY: {retry_count}곡")
        print(f"  - 통과율: {((excellent_count + good_count) / len(results) * 100):.1f}%")
        
        return {
            'results': results,
            'summary': {
                'excellent_count': excellent_count,
                'good_count': good_count,
                'retry_count': retry_count,
                'pass_rate': (excellent_count + good_count) / len(results),
                'avg_score': sum(r['total_score'] for r in results) / len(results)
            }
        }
    
    def get_retry_recommendations(self, evaluation_result):
        """재생성 권장 사항 반환"""
        recommendations = []
        
        try:
            if evaluation_result['status'] != 'RETRY':
                return recommendations
            
            # 1단계 실패 분석
            if evaluation_result['basic_result'] and not evaluation_result['basic_result']['overall_passed']:
                basic = evaluation_result['basic_result']
                if not basic['duration']['passed']:
                    recommendations.append("생성 길이 늘리기 (12초 이상)")
                if not basic['high_frequency']['passed']:
                    recommendations.append("고주파 노이즈 줄이기")
                if not basic['extreme_frequencies']['passed']:
                    recommendations.append("극단 주파수 문제 해결 (드론/럼블 제거)")
            
            # 2단계 실패 분석
            if evaluation_result['musical_result'] and not evaluation_result['musical_result']['passed']:
                musical = evaluation_result['musical_result']
                if not musical['rhythm']['passed']:
                    recommendations.append("리듬 일관성 개선")
                if not musical['melody']['passed']:
                    recommendations.append("멜로디 라인 강화")
                if not musical['harmonic']['passed']:
                    recommendations.append("하모닉-퍼커시브 밸런스 조정")
                if not musical['flow']['passed']:
                    recommendations.append("음악적 흐름 개선")
            
            # 3단계 실패 분석
            if evaluation_result['semantic_result'] and not evaluation_result['semantic_result']['passed']:
                recommendations.append("프롬프트 명확성 개선")
                recommendations.append("장르, 감정, 악기 키워드 재검토")
            
            return recommendations
            
        except Exception as e:
            print(f"권장사항 생성 오류: {e}")
            return ["재생성 권장"]