import re
import os


def create_safe_filename(prompt, max_length=50):
    """프롬프트를 안전한 파일명으로 변환"""
    # 특수문자 제거 및 공백을 언더스코어로 변경
    safe_name = re.sub(r'[^\w\s-]', '', prompt)
    safe_name = re.sub(r'[-\s]+', '_', safe_name)
    
    # 길이 제한
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length]
    
    # 앞뒤 언더스코어 제거
    safe_name = safe_name.strip('_')
    
    # 빈 문자열인 경우 기본값
    if not safe_name:
        safe_name = "untitled"
    
    return safe_name.lower()


def format_duration(seconds):
    """초를 mm:ss 형식으로 변환"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def calculate_success_rate(results):
    """통과율 계산"""
    if not results:
        return 0.0
    
    passed_count = sum(1 for r in results if r.get('passed', False))
    return passed_count / len(results)


def ensure_output_directory(output_dir):
    """출력 디렉토리가 존재하는지 확인하고 생성"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 출력 디렉토리 생성: {output_dir}")
    return output_dir


def get_file_size_mb(filepath):
    """파일 크기를 MB 단위로 반환"""
    if os.path.exists(filepath):
        size_bytes = os.path.getsize(filepath)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    return 0.0


def print_separator(title="", width=60, char="="):
    """구분선 출력"""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"{char * padding} {title} {char * padding}")
    else:
        print(char * width)


def format_quality_score(score, passed):
    """품질 점수를 색상과 함께 포맷"""
    if passed:
        return f"✅ {score:.2f}"
    else:
        return f"❌ {score:.2f}"


def get_audio_info_summary(audio_data, sample_rate):
    """오디오 정보 요약 반환"""
    duration = len(audio_data) / sample_rate
    max_amplitude = max(abs(audio_data.max()), abs(audio_data.min()))
    rms = (audio_data ** 2).mean() ** 0.5
    
    return {
        'duration': duration,
        'sample_rate': sample_rate,
        'max_amplitude': max_amplitude,
        'rms': rms,
        'length_samples': len(audio_data)
    }