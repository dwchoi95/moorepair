# ================================================================== #
# Centralized prompt templates for all LLM calls in MooRepair        #
# ================================================================== #

# ------------------------------------------------------------------ #
# PaR — Peer-aided Repair                                            #
# Used by: baselines/par.py, genetic/variation.py (initialization)   #
# ------------------------------------------------------------------ #

PAR_SYSTEM = '''\
There is a Python programming problem. Below is the problem description, the input format, \
the output format, a copy of the correct code for reference, \
and a copy of the buggy code containing semantic errors that written by a student to solve the Python programming \
problem. Please fix the buggy code and return the correct code.
'''

PAR_USER = '''\
{description}

[Reference Code]
{reference_program}
[End of Reference Code]

[Buggy Code]
{buggy_program}
[End of Buggy Code]

Please fix the code and return the correct code.
'''


# ------------------------------------------------------------------ #
# EffiLearner — Efficiency Self-Optimization                         #
# Used by: baselines/effilearner.py                                  #
# ------------------------------------------------------------------ #

EFFILEARNER_SYSTEM = '''\
Optimize the efficiency of the following Python code based on the task, test case, \
and overhead analysis provided. Ensure the optimized code can pass the given test case.\
'''

EFFILEARNER_USER = '''\
Task Description:
{description}

Test Case:
{test_case}

Original Code:
```python
{original_code}
```

Overhead Analysis:
The total memory usage during the code execution is: {total_memory_usage} MB*s.
The total execution time is: {total_execution_time} s.
The maximum memory peak requirement is: {max_memory_usage} MB.
# The profiler results are: 
{profile_report}

Optimization Rules:
- Encapsulate the optimized code within a Python code block (i.e., ```python\\n[Your Code Here]\\n```).
- Do not include the test case within the code block.
- Focus solely on code optimization; test cases are already provided.
- Ensure the provided test case passes with your optimized solution.
'''


# ------------------------------------------------------------------ #
# CROSS — Crossover prompts                                         #
# Used by: genetic/variation.py                                      #
# ------------------------------------------------------------------ #

CROSS_FAIL_SYSTEM = '''\
You are a Python code merging specialist. \
Given two programs solving the same problem, \
you combine their strengths to produce one correct program. \
Identify which parts of each program are correct and merge them. \
Output the complete merged program, not a diff or patch. \
Wrap your merged code in [PYTHON] and [/PYTHON] tags.\
'''

CROSS_FAIL_USER = '''\
{description}

[Failing Case]
{test_case}
[/Failing Case]

[Program A]
[PYTHON]
{p1_code}
[/PYTHON]
[/Program A]

[Program B]
[PYTHON]
{p2_code}
[/PYTHON]
[/Program B]

Adopt the correct logic from the better program.
Return ONLY the merged Python program in [PYTHON] and [/PYTHON] tags.\
'''


CROSS_TIME_SYSTEM = '''\
You are a Python code merging specialist. \
Given two programs solving the same problem, \
you combine them to produce a faster version. \
Adopt the faster algorithm or data structure from the better program \
while keeping the merged code correct. \
Output the complete merged program, not a diff or patch. \
Wrap your merged code in [PYTHON] and [/PYTHON] tags.\
'''

CROSS_TIME_USER = '''\
[Test Case]
{test_case}
[/Test Case]

[Program A]
[PYTHON]
{p1_code}
[/PYTHON]
[/Program A]

[Program A Profile]
{p1_profile}
[/Program A Profile]

[Program B]
[PYTHON]
{p2_code}
[/PYTHON]
[/Program B]

[Program B Profile]
{p2_profile}
[/Program B Profile]

Adopt the faster approach from B into A's structure.
Do not break correctness.
Return ONLY the merged Python program in [PYTHON] and [/PYTHON] tags.\
'''


CROSS_MEM_SYSTEM = '''\
You are a Python code merging specialist. \
Given two programs solving the same problem, \
you combine them to produce a more memory-efficient version. \
Adopt the memory-efficient approach from the better program \
while keeping the merged code correct. \
Output the complete merged program, not a diff or patch. \
Wrap your merged code in [PYTHON] and [/PYTHON] tags.\
'''

CROSS_MEM_USER = '''\
[Test Case]
{test_case}
[/Test Case]

[Program A]
[PYTHON]
{p1_code}
[/PYTHON]
[/Program A]

[Program A Profile]
{p1_profile}
[/Program A Profile]

[Program B]
[PYTHON]
{p2_code}
[/PYTHON]
[/Program B]

[Program B Profile]
{p2_profile}
[/Program B Profile]

Adopt the memory-efficient approach from B into A's structure.
Do not break correctness.
Return ONLY the merged Python program in [PYTHON] and [/PYTHON] tags.\
'''


# ------------------------------------------------------------------ #
# MUT — Mutation prompts                                             #
# Used by: genetic/variation.py                                      #
# ------------------------------------------------------------------ #

MUT_FAIL_SYSTEM = '''\
You are a Python debugging specialist. \
You find and fix logical errors with minimal changes. \
Preserve the original code structure and only modify \
the lines responsible for the bug. \
Output the complete corrected program, not a diff or patch. \
Wrap your corrected code in [PYTHON] and [/PYTHON] tags.\
'''

MUT_FAIL_USER = '''\
{description}

[Failing Case]
{test_case}
[/Failing Case]

[PYTHON]
{code}
[/PYTHON]

Write the corrected version of the code above.
Return ONLY the complete Python program in [PYTHON] and [/PYTHON] tags.\
'''


MUT_TIME_SYSTEM = '''\
You are a Python performance specialist. \
You optimize code for speed while preserving correctness. \
Prefer algorithmic improvements over optimizations. \
Output the complete optimized program, not a diff or patch. \
Wrap your optimized code in [PYTHON] and [/PYTHON] tags.\
'''

MUT_TIME_USER = '''\
{description}

[Profile]
{profile}
[/Profile]

[PYTHON]
{code}
[/PYTHON]

Reduce execution time. Do not break correctness.
Return ONLY the optimized Python program in [PYTHON] and [/PYTHON] tags.\
'''


MUT_MEM_SYSTEM = '''\
You are a Python performance specialist. \
You reduce peak memory usage while preserving correctness. \
Prefer in-place operations, generators, and compact data structures. \
Output the complete optimized program, not a diff or patch. \
Wrap your optimized code in [PYTHON] and [/PYTHON] tags.\
'''

MUT_MEM_USER = '''\
{description}

[Profile]
{profile}
[/Profile]

[PYTHON]
{code}
[/PYTHON]

Reduce peak memory usage. Do not break correctness.
Return ONLY the optimized Python program in [PYTHON] and [/PYTHON] tags.\
'''
