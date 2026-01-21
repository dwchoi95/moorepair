"""
Defects4DS 데이터를 JSON 형태로 변환하는 스크립트

알고리즘:
1. labeled_information.jsonl 파일을 읽어서 각 problem_id별로 그룹화
2. 각 problem_id에 대해:
   a. assignment_description에서 문제 설명 파싱
   b. test_cases XML 파일에서 테스트 케이스 추출
   c. student_submissions에서 각 user_id의 코드 읽기
   d. labeled_information에서 각 제출의 상태 정보 추출
3. 모든 정보를 통합하여 JSON 형식으로 저장
"""

import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any

def parse_test_cases(file_path: str) -> List[Dict[str, str]]:
    """
    XML 테스트 케이스 파일을 파싱하여 리스트로 변환
    
    Returns:
        [{"id": int, "input": str, "output": str}]
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    test_cases = []
    count = int(root.get('count', 0))
    
    for i in range(1, count + 1):
        test_data = root.find(f'testData{i}')
        if test_data is not None:
            input_elem = test_data.find('input')
            output_elem = test_data.find('output')
            
            test_cases.append({
                "id": i,
                "input": input_elem.text if input_elem is not None and input_elem.text else "",
                "output": output_elem.text if output_elem is not None and output_elem.text else ""
            })
    
    return test_cases


def read_code_file(file_path: str) -> str:
    """코드 파일 읽기"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # UTF-8로 읽기 실패시 다른 인코딩 시도
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()


def get_file_extension(file_path: str) -> str:
    """파일 확장자 추출"""
    return os.path.splitext(file_path)[1][1:]  # .c -> c


def convert_defects4ds_to_json(base_path: str, output_dir: str):
    """
    Defects4DS 데이터를 problemID별 JSON 파일로 변환
    
    Args:
        base_path: Defects4DS 폴더 경로
        output_dir: 출력 JSON 파일들을 저장할 디렉토리
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # labeled_information.jsonl 읽기
    labeled_info_path = os.path.join(base_path, 'labeled_information.jsonl')
    labeled_data = set()
    
    with open(labeled_info_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            problem_id = data['problem_id']
            labeled_data.add(problem_id)
    
    # 각 problem_id에 대해 JSON 생성
    for problem_id in sorted(labeled_data):
        print(f"Processing {problem_id}...")
        
        # Assignment description 파싱
        assignment_info = {}
        assignment_info['id'] = problem_id
        desc_file = os.path.join(base_path, 'assignment_description', f'{problem_id}.txt')
        with open(desc_file, 'r', encoding='utf-8') as f:
            assignment_info['description'] = f.read()
        
        # Test cases 파싱
        test_cases_file = os.path.join(base_path, 'test_cases', f'{problem_id}.xml')
        test_cases = parse_test_cases(test_cases_file)
        
        # Submissions 수집
        submissions = []
        submissions_dir = os.path.join(base_path, 'student_submissions', problem_id)
        
        if os.path.exists(submissions_dir):
            for user_id in sorted(os.listdir(submissions_dir)):
                if not user_id.startswith('userID_'):
                    continue
                
                user_dir = os.path.join(submissions_dir, user_id)
                if not os.path.isdir(user_dir):
                    continue
                
                file = None
                for filename in os.listdir(user_dir):
                    if filename.startswith('buggy') or filename.startswith('correct'):
                        file = os.path.join(user_dir, filename)
                        
                        code = read_code_file(file)
                        ext = get_file_extension(file)
                        status = "buggy" if filename.startswith('buggy') else "correct"
                        
                        sub_id = user_id.split('_')[1]
                        submission = {
                            "id": sub_id,
                            "code": code,
                            "ext": ext,
                            "status": status
                        }
                        
                        submissions.append(submission)
        
        # 최종 JSON 구조 생성
        output_data = {
            "assignment": assignment_info,
            "submissions": submissions,
            "test_cases": test_cases
        }
        
        # problem_id별 디렉토리 생성 및 JSON 파일로 저장
        problem_dir = os.path.join(output_dir, problem_id)
        os.makedirs(problem_dir, exist_ok=True)
        output_file = os.path.join(problem_dir, 'dataset.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"  - Created {output_file}")
        print(f"  - {len(submissions)} submissions")
        print(f"  - {len(test_cases)} test cases")
        print()


if __name__ == '__main__':
    base_path = './Defects4DS'
    output_dir = './data'
    convert_defects4ds_to_json(base_path, output_dir)