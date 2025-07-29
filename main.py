#!/usr/bin/env python3
"""
🎵 MusicGen 품질 필터링 파이프라인

사용법: 
    python main.py

프롬프트를 입력하면 5개의 음악을 생성하고 품질 검사를 수행합니다.
통과한 음악은 sample{번호}_pass.wav로, 
실패한 음악은 sample{번호}_fail.wav로 저장됩니다.
"""

import sys
import os
from pipeline.quality_pipeline import MusicQualityPipeline
from utils.audio_utils import print_separator, ensure_output_directory


def print_welcome():
    """환영 메시지 출력"""
    print_separator("MusicGen 품질 필터링 파이프라인", width=70, char="🎵")
    print()
    print("📝 프롬프트를 입력하면 5개의 음악을 생성하고 품질을 검사합니다.")
    print("✅ 통과한 음악: sample{번호}_pass.wav")
    print("❌ 실패한 음악: sample{번호}_fail.wav")
    print()
    print("💡 품질 검사 항목:")
    print("  - 길이 검사: 예상보다 너무 짧은 음악인지 확인")
    print("  - 고주파 노이즈: 8kHz 이상 고주파가 3초 이상 지속되는지 확인")
    print("  - 드론/럼블 검사: 40Hz 이하 단조로운 소리가 5초 이상 지속되는지 확인")
    print()
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print_separator(width=70, char="🎵")


def get_user_input():
    """사용자 입력 받기"""
    try:
        prompt = input("\n🎼 프롬프트를 입력하세요: ").strip()
        return prompt
    except KeyboardInterrupt:
        print("\n\n👋 프로그램을 종료합니다.")
        return None
    except EOFError:
        return None


def validate_prompt(prompt):
    """프롬프트 유효성 검사"""
    if not prompt:
        print("❌ 빈 프롬프트는 입력할 수 없습니다.")
        return False
    
    if len(prompt) > 200:
        print("❌ 프롬프트가 너무 깁니다. (최대 200자)")
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
        print("\n🚀 파이프라인 초기화 중...")
        pipeline = MusicQualityPipeline(output_dir=output_dir)
        print("✅ 초기화 완료!")
    except Exception as e:
        print(f"❌ 파이프라인 초기화 실패: {e}")
        print("💡 audiocraft 라이브러리와 모델이 올바르게 설치되었는지 확인하세요.")
        return 1
    
    # 메인 루프
    session_count = 0
    
    while True:
        prompt = get_user_input()
        
        # 종료 조건
        if prompt is None:
            break
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("👋 프로그램을 종료합니다.")
            break
        
        # 프롬프트 유효성 검사
        if not validate_prompt(prompt):
            continue
        
        # 파이프라인 실행
        try:
            session_count += 1
            print(f"\n🎵 세션 {session_count} 시작...")
            
            result = pipeline.process_prompt(prompt, batch_size=5)
            
            # 추가 정보 출력
            if result['summary']['success_count'] > 0:
                print(f"\n🎉 {result['summary']['success_count']}개의 좋은 음악이 생성되었습니다!")
            else:
                print(f"\n😅 이번에는 품질 기준을 통과한 음악이 없네요. 다시 시도해보세요!")
                
        except KeyboardInterrupt:
            print("\n\n⏹️  생성을 중단했습니다.")
            continue
        except Exception as e:
            print(f"\n❌ 오류가 발생했습니다: {e}")
            print("💡 다시 시도해보세요.")
            continue
    
    # 종료 메시지
    if session_count > 0:
        print(f"\n📈 총 {session_count}개의 세션을 처리했습니다.")
        print(f"📁 생성된 파일들은 '{output_dir}' 폴더에서 확인할 수 있습니다.")
    
    print("\n🎵 감사합니다!")
    return 0


if __name__ == "__main__":
    sys.exit(main())