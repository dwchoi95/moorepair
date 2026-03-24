import re
import asyncio
from tqdm.asyncio import tqdm as tqdm_async

from ..execution.program import Program
from ..execution.testcases import TestCase
from ..utils import Randoms


# ------------------------------------------------------------------ #
# Mutation prompts                                                    #
# ------------------------------------------------------------------ #

_MUT_FAIL_SYSTEM = (
    "You are an expert competitive programmer. "
    "Fix the bug in the given {language} code so that it produces the correct output."
)
_MUT_FAIL_USER = """\
## Problem
{description}

## Failing Case
Input:    {tc_input}
Expected: {tc_expected}
Got:      {tc_actual}

## Code
{code}

Fix the logical error. Return ONLY the fixed {language} code."""


_MUT_TIME_SYSTEM = (
    "You are an expert competitive programmer. "
    "Optimize the given {language} code for speed without breaking correctness."
)
_MUT_TIME_USER = """\
## Problem
{description}

## Representative Case
Input:     {tc_input}
exec_time: {exec_time_ms:.2f} ms

## Profile (on this case)
{profile}

Reduce execution time. Do not break correctness.
Return ONLY the optimized {language} code."""


_MUT_MEM_SYSTEM = (
    "You are an expert competitive programmer. "
    "Optimize the given {language} code for memory usage without breaking correctness."
)
_MUT_MEM_USER = """\
## Problem
{description}

## Representative Case
Input:    {tc_input}
peak_mem: {peak_mem_mb:.4f} MB

## Profile (on this case)
{profile}

Reduce peak memory usage. Do not break correctness.
Return ONLY the optimized {language} code."""


# ------------------------------------------------------------------ #
# Crossover prompts                                                   #
# ------------------------------------------------------------------ #

_CROSS_FAIL_SYSTEM = (
    "You are an expert competitive programmer. "
    "Combine two {language} programs to produce a correct merged version."
)
_CROSS_FAIL_USER = """\
## Problem
{description}

## Failing Case
Input:    {tc_input}
Expected: {tc_expected}
Got:      {tc_actual}

## Program A
{code_a}

## Program B
{code_b}

Adopt the correct logic from the better program.
Return ONLY the merged {language} code."""


_CROSS_TIME_SYSTEM = (
    "You are an expert competitive programmer. "
    "Combine two {language} programs to improve speed."
)
_CROSS_TIME_USER = """\
## Representative Case  (A is slow, B is fast)
Input:         {tc_input}
A exec_time:   {exec_time_a:.2f} ms
B exec_time:   {exec_time_b:.2f} ms

## Program A — Profile (on this case)
{profile_a}

## Program B — Profile (on this case)
{profile_b}

Adopt the faster approach from B into A's structure.
Do not break correctness.
Return ONLY the merged {language} code."""


_CROSS_MEM_SYSTEM = (
    "You are an expert competitive programmer. "
    "Combine two {language} programs to reduce memory usage."
)
_CROSS_MEM_USER = """\
## Representative Case  (A is heavy, B is light)
Input:        {tc_input}
A peak_mem:   {peak_mem_a:.4f} MB
B peak_mem:   {peak_mem_b:.4f} MB

## Program A — Profile (on this case)
{profile_a}

## Program B — Profile (on this case)
{profile_b}

Adopt the memory-efficient approach from B into A's structure.
Do not break correctness.
Return ONLY the merged {language} code."""


# ------------------------------------------------------------------ #
# Initial-population prompt                                           #
# ------------------------------------------------------------------ #

_INIT_SYSTEM = """\
# Role
You are an expert programmer.

# Task
Please fix bugs and optimize the efficiency of the following Python code based on the problem description and reference code.
Ensure the fixed code can pass the given test case.

# Output Format
Please provide the fixed code as string.
Do not include any explanations or comments"""

_INIT_USER = """\
# Inputs
## Problem Description
{description}

## Buggy Code
{buggy_code}

## Reference Code
{refer_code}


# Output
## Fixed Code"""


# ------------------------------------------------------------------ #
# Variation class                                                     #
# ------------------------------------------------------------------ #

class Variation:
    def __init__(self, description: str = ""):
        self.description = description

    # ---- profile helpers ----------------------------------------- #

    def _get_profile(self, program: Program, tc: TestCase) -> dict:
        """Return the profile dict for *tc* from program.results, or {}."""
        tr = self._get_tc_result(program, tc)
        if tr and tr.result and isinstance(tr.result.profile, dict):
            return tr.result.profile
        return {}

    def _format_time_profile(self, program: Program, tc: TestCase) -> str:
        """Format line-level runtime profile from profile data."""
        profile = self._get_profile(program, tc)
        if not profile:
            return "(no profile available)"
        lines = []
        for lineno, data in sorted(profile.items(), key=lambda x: int(x[0])):
            hits = data.get("hits", 0)
            runtime_ms = data.get("runtime", 0.0) * 1000.0
            lines.append(f"L{lineno}: {runtime_ms:.3f} ms ({hits} hits)")
        return "\n".join(lines) if lines else "(no profile available)"

    def _format_memory_profile(self, program: Program, tc: TestCase) -> str:
        """Format line-level memory profile from profile data."""
        profile = self._get_profile(program, tc)
        if not profile:
            return "(no profile available)"
        lines = []
        for lineno, data in sorted(profile.items(), key=lambda x: int(x[0])):
            hits = data.get("hits", 0)
            memory_mb = data.get("memory", 0.0)
            lines.append(f"L{lineno}: {memory_mb:.4f} MB ({hits} hits)")
        return "\n".join(lines) if lines else "(no profile available)"

    # ---- prompt builders ----------------------------------------- #

    def _init_prompt(self, buggy: Program, reference: Program) -> tuple[str, str]:
        system = _INIT_SYSTEM
        user = _INIT_USER.format(
            description=self.description,
            buggy_code=f"```{buggy.ext}\n{buggy.code}\n```",
            refer_code=f"```{reference.ext}\n{reference.code}\n```",
        )
        return system, user

    def _get_tc_result(self, program: Program, tc: TestCase):
        """Return the TestcaseResult for *tc* from program.results, or None."""
        if program.results is None:
            return None
        for tr in program.results:
            if tr.testcase.id == tc.id:
                return tr
        return None

    def _mutation_prompt(
        self, p1: Program, tc: TestCase, strategy: str
    ) -> tuple[str, str]:
        lang = p1.ext.capitalize()
        if strategy == "f_fail":
            tr = self._get_tc_result(p1, tc)
            actual = tr.result.stdout.strip() if (tr and tr.result) else ""
            system = _MUT_FAIL_SYSTEM.format(language=lang)
            user = _MUT_FAIL_USER.format(
                description=self.description,
                tc_input=tc.input.strip(),
                tc_expected=tc.output.strip(),
                tc_actual=actual,
                code=p1.code,
                language=lang,
            )
        elif strategy == "f_time":
            tr = self._get_tc_result(p1, tc)
            exec_ms = (tr.result.runtime * 1000.0) if (tr and tr.result) else 0.0
            profile = self._format_time_profile(p1, tc)
            system = _MUT_TIME_SYSTEM.format(language=lang)
            user = _MUT_TIME_USER.format(
                description=self.description,
                tc_input=tc.input.strip(),
                exec_time_ms=exec_ms,
                profile=profile or "(no profile available)",
                language=lang,
            )
        else:  # f_mem
            tr = self._get_tc_result(p1, tc)
            peak_mb = tr.result.memory if (tr and tr.result) else 0.0
            profile = self._format_memory_profile(p1, tc)
            system = _MUT_MEM_SYSTEM.format(language=lang)
            user = _MUT_MEM_USER.format(
                description=self.description,
                tc_input=tc.input.strip(),
                peak_mem_mb=peak_mb,
                profile=profile or "(no profile available)",
                language=lang,
            )
        return system, user

    def _crossover_prompt(
        self, p1: Program, p2: Program, tc: TestCase, strategy: str
    ) -> tuple[str, str]:
        lang = p1.ext.capitalize()
        if strategy == "f_fail":
            tr1 = self._get_tc_result(p1, tc)
            actual = tr1.result.stdout.strip() if (tr1 and tr1.result) else ""
            system = _CROSS_FAIL_SYSTEM.format(language=lang)
            user = _CROSS_FAIL_USER.format(
                description=self.description,
                tc_input=tc.input.strip(),
                tc_expected=tc.output.strip(),
                tc_actual=actual,
                code_a=p1.code,
                code_b=p2.code,
                language=lang,
            )
        elif strategy == "f_time":
            tr1 = self._get_tc_result(p1, tc)
            tr2 = self._get_tc_result(p2, tc)
            exec_a = (tr1.result.runtime * 1000.0) if (tr1 and tr1.result) else 0.0
            exec_b = (tr2.result.runtime * 1000.0) if (tr2 and tr2.result) else 0.0
            prof_a = self._format_time_profile(p1, tc)
            prof_b = self._format_time_profile(p2, tc)
            system = _CROSS_TIME_SYSTEM.format(language=lang)
            user = _CROSS_TIME_USER.format(
                tc_input=tc.input.strip(),
                exec_time_a=exec_a,
                exec_time_b=exec_b,
                profile_a=prof_a or "(no profile available)",
                profile_b=prof_b or "(no profile available)",
                language=lang,
            )
        else:  # f_mem
            tr1 = self._get_tc_result(p1, tc)
            tr2 = self._get_tc_result(p2, tc)
            mem_a = tr1.result.memory if (tr1 and tr1.result) else 0.0
            mem_b = tr2.result.memory if (tr2 and tr2.result) else 0.0
            prof_a = self._format_memory_profile(p1, tc)
            prof_b = self._format_memory_profile(p2, tc)
            system = _CROSS_MEM_SYSTEM.format(language=lang)
            user = _CROSS_MEM_USER.format(
                tc_input=tc.input.strip(),
                peak_mem_a=mem_a,
                peak_mem_b=mem_b,
                profile_a=prof_a or "(no profile available)",
                profile_b=prof_b or "(no profile available)",
                language=lang,
            )
        return system, user

    # ---- LLM async task ------------------------------------------ #

    async def _task(self, system: str, user: str) -> str | None:
        from ..llms import Models
        return await Models.run(system, user)



    # ---- async orchestration ------------------------------------- #

    async def _run_init_async(self, buggy: Program, references: list[Program], count: int) -> list[Program]:
        """Generate *count* initial candidates (syntax-only validation done in GA)."""
        if len(references) >= count:
            selected = Randoms.sample(list(references), count)
        else:
            selected = [Randoms.choice(list(references)) for _ in range(count)]
        tasks = [asyncio.create_task(self._task(*self._init_prompt(buggy, reference))) for reference in selected]

        programs = []
        pbar = tqdm_async(total=len(tasks), desc="Init", leave=False, position=2)
        for coro in asyncio.as_completed(tasks):
            patch: str | None = await coro
            pbar.update(1)
            if patch is None or not patch.strip():
                continue
            programs.append(
                Program(
                    id=f"init_{len(programs) + 1}",
                    code=patch,
                    ext=buggy.ext,
                )
            )
        pbar.close()
        return programs

    async def _run_variation_async(
        self, buggy: Program, pairs: list[tuple]
    ) -> list[Program]:
        """Generate one crossover + one mutation offspring per pair."""
        tasks = []
        meta = []  # (is_crossover, ext) for each task

        for p1, p2, t_star in pairs:
            if t_star is None:
                continue
            strategy = p1.strategy or "f_fail"
            # Crossover task
            sys_c, usr_c = self._crossover_prompt(p1, p2, t_star, strategy)
            tasks.append(asyncio.create_task(self._task(sys_c, usr_c)))
            meta.append(buggy.ext)
            # Mutation task
            sys_m, usr_m = self._mutation_prompt(p1, t_star, strategy)
            tasks.append(asyncio.create_task(self._task(sys_m, usr_m)))
            meta.append(buggy.ext)

        programs = []
        pbar = tqdm_async(total=len(tasks), desc="Variation", leave=False, position=2)
        for coro, ext in zip(asyncio.as_completed(tasks), meta):
            patch: str | None = await coro
            pbar.update(1)
            if patch is None or not patch.strip():
                continue
            programs.append(
                Program(
                    id=f"child_{len(programs) + 1}",
                    code=patch,
                    ext=ext,
                )
            )
        pbar.close()
        return programs

    # ---- public API ---------------------------------------------- #

    def _asyncio_loop(self):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop

    def run_init(self, buggy: Program, references: list[Program], count: int) -> list[Program]:
        """Generate *count* candidate programs for initial population."""
        loop = self._asyncio_loop()
        return loop.run_until_complete(self._run_init_async(buggy, references, count))

    def run(self, buggy: Program, pairs: list[tuple]) -> list[Program]:
        """Generate offspring from (p1, p2, t*) pairs."""
        loop = self._asyncio_loop()
        return loop.run_until_complete(self._run_variation_async(buggy, pairs))
