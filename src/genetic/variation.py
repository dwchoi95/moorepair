import asyncio
from tqdm.asyncio import tqdm as tqdm_async

from ..execution import Program, TestCase, Tester
from ..utils import Randoms
from ..llms import prompts


class Variation:
    def __init__(self, description: str = ""):
        self.description = description
    
    # ---- prompt builders ----------------------------------------- #

    def _correct_prompt(self, buggy: Program, reference: Program) -> tuple[str, str]:
        system = prompts.PAR_SYSTEM
        user = prompts.PAR_USER.format(
            description=self.description,
            buggy_program=buggy.code,
            reference_program=reference.code,
        )
        return system, user
    
    def _efficient_prompt(self, correct: Program) -> tuple[str, str]:
        results = Tester.run(correct, profiling=True)
        system = prompts.EFFILEARNER_SYSTEM
        user = prompts.EFFILEARNER_USER.format(
            description=self.description,
            test_case=str(Tester.testcases),
            original_code=correct.code,
            total_memory_usage=results.mem_usage(),
            total_execution_time=results.exec_time(),
            max_memory_usage=results.mem_usage_max(),
            mem_usage_report=results.report_mem(),
            exec_time_report=results.report_time(),
        )
        return system, user
    
    def _crossover_prompt(
        self, p1: Program, p2: Program, tc: TestCase, strategy: str) -> tuple[str, str]:
        if strategy == "f_fail":
            system = prompts.CROSS_FAIL_SYSTEM
            user = prompts.CROSS_FAIL_USER.format(
                description=self.description,
                test_case=str(p1.results.print_tc_result(tc)),
                p1_code=p1.code,
                p2_code=p2.code,
            )
        elif strategy == "f_time":
            system = prompts.CROSS_TIME_SYSTEM
            user = prompts.CROSS_TIME_USER.format(
                test_case=str(p1.results.print_tc_result(tc)),
                p1_code=p1.code,
                p1_profile=p1.results.report_time(tc),
                p2_code=p2.code,
                p2_profile=p2.results.report_time(tc)
            )
        else:  # f_mem
            system = prompts.CROSS_MEM_SYSTEM
            user = prompts.CROSS_MEM_USER.format(
                test_case=str(p1.results.print_tc_result(tc)),
                p1_code=p1.code,
                p1_profile=p1.results.report_mem(tc),
                p2_code=p2.code,
                p2_profile=p2.results.report_mem(tc)
            )
        return system, user
    
    def _mutation_prompt(
        self, p1: Program, tc: TestCase, strategy: str
    ) -> tuple[str, str]:
        if strategy == "f_fail":
            system = prompts.MUT_FAIL_SYSTEM
            user = prompts.MUT_FAIL_USER.format(
                description=self.description,
                test_case=str(p1.results.print_tc_result(tc)),
                code=p1.code
            )
        elif strategy == "f_time":
            system = prompts.MUT_TIME_SYSTEM
            user = prompts.MUT_TIME_USER.format(
                description=self.description,
                profile=p1.results.report_time(tc),
                code=p1.code
            )
        else:  # f_mem
            system = prompts.MUT_MEM_SYSTEM
            user = prompts.MUT_MEM_USER.format(
                description=self.description,
                profile=p1.results.report_mem(tc),
                code=p1.code
            )
        return system, user

    # ---- LLM async task ------------------------------------------ #

    async def _task(self, system: str, user: str) -> str | None:
        from ..llms import Models
        return await Models.run(system, user)

    # ---- async orchestration ------------------------------------- #

    async def _run_correct_async(self, buggy: Program, references: list[Program], count: int) -> list[Program]:
        """Generate *count* initial candidates (syntax-only validation done in GA)."""
        if len(references) >= count:
            selected = Randoms.sample(list(references), count)
        else:
            selected = [Randoms.choice(list(references)) for _ in range(count)]
        tasks = [asyncio.create_task(self._task(
                *self._correct_prompt(buggy, reference)
            )) for reference in selected]

        programs = []
        pbar = tqdm_async(total=len(tasks), desc="Init", leave=False, position=2)
        for coro in asyncio.as_completed(tasks):
            patch: str | None = await coro
            pbar.update(1)
            if patch is None or not patch.strip():
                continue
            programs.append(
                Program(
                    id=f"correct_{len(programs) + 1}",
                    code=patch,
                    ext=buggy.ext,
                )
            )
        pbar.close()
        return programs

    async def _run_efficient_async(self, corrects: list[Program]) -> list[Program]:
        """Generate candidates for EffiLearner (validation passed in PaR)."""
        tasks = [asyncio.create_task(self._task(
                *self._efficient_prompt(correct)
            )) for correct in corrects]

        programs = []
        pbar = tqdm_async(total=len(tasks), desc="Efficient", leave=False, position=2)
        for coro in asyncio.as_completed(tasks):
            patch: str | None = await coro
            pbar.update(1)
            if patch is None or not patch.strip():
                continue
            programs.append(
                Program(
                    id=f"efficient_{len(programs) + 1}",
                    code=patch,
                    ext="py",
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

    def correct(self, buggy: Program, references: list[Program], count: int) -> list[Program]:
        """Generate *count* candidate programs for initial population."""
        loop = self._asyncio_loop()
        return loop.run_until_complete(self._run_correct_async(buggy, references, count))
    
    def efficient(self, corrects: list[Program]) -> list[Program]:
        """Generate *count* candidate programs for EffiLearner."""
        loop = self._asyncio_loop()
        return loop.run_until_complete(self._run_efficient_async(corrects))

    def run(self, buggy: Program, pairs: list[tuple]) -> list[Program]:
        """Generate offspring from (p1, p2, t*) pairs."""
        loop = self._asyncio_loop()
        return loop.run_until_complete(self._run_variation_async(buggy, pairs))
