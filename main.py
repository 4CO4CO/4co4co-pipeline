def main():
    print("🎵 MusicGen 품질 필터링 파이프라인")
    
    # 파이프라인 초기화
    pipeline = MusicQualityPipeline(output_dir="output")
    
    while True:
        # 프롬프트 입력 받기
        prompt = input("\n프롬프트를 입력하세요 (종료: 'quit'): ")
        
        if prompt.lower() == 'quit':
            break
            
        print(f"\n🎼 '{prompt}' 처리 시작...")
        
        # 파이프라인 실행
        results = pipeline.process_prompt(prompt, batch_size=5)
        
        # 결과 출력
        print_results(results)

def print_results(results):
    """결과를 깔끔하게 출력"""
    # 성공률, 파일 목록, 실패 원인 등 표시

if __name__ == "__main__":
    main()