import asyncio
from tqdm.asyncio import tqdm as tqdm_async

from ..execution import Program, TestCase, Tester
from ..llms import prompts


class Variation:
    def __init__(self, assignment: dict = {}):
        self.description = assignment.get("description", "")
        self.input_format = assignment.get("input_format", "")
        self.output_format = assignment.get("output_format", "")
    
    # ---- prompt builders ----------------------------------------- #

    async def _correct_prompt(self, buggy: Program, reference: Program) -> tuple[str, str]:
        from ..llms import Models
        system = prompts.PAR_SYSTEM
        user = prompts.PAR_USER.format(
            description=self.description,
            input_format=self.input_format,
            output_format=self.output_format,
            buggy_program=buggy.code,
            reference_program=reference.code,
        )
        return await Models.run(system, user)
    
    async def _efficient_prompt(self, correct: Program) -> tuple[str, str]:
        from ..llms import Models
        results = Tester.run(correct, profiling=True)
        system = prompts.EFFILEARNER_SYSTEM
        user = prompts.EFFILEARNER_USER.format(
            description=self.description,
            input_format=self.input_format,
            output_format=self.output_format,
            test_case=str(Tester.testcases),
            original_code=correct.code,
            total_memory_usage=results.mem_usage(),
            total_execution_time=results.exec_time(),
            max_memory_usage=results.mem_usage_max(),
            line_profiler_results=results.report_time(),
            memory_report=results.report_mem()
        )
        return await Models.run(system, user)

    async def _crossover_prompt(
        self, p1: Program, p2: Program, tc: TestCase
    ) -> tuple[str, str]:
        from ..llms import Models
        strategy = p1.strategy or "f_fail"
        if strategy == "f_fail":
            system = prompts.CROSS_FAIL_SYSTEM
            user = prompts.CROSS_FAIL_USER.format(
                description=self.description,
                input_format=self.input_format,
                output_format=self.output_format,
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
        return await Models.run(system, user), p1.fitness, p1.ext
    
    async def _mutation_prompt(
        self, p1: Program, tc: TestCase, 
    ) -> tuple[str, str]:
        from ..llms import Models
        strategy = p1.strategy or "f_fail"
        if strategy == "f_fail":
            system = prompts.MUT_FAIL_SYSTEM
            user = prompts.MUT_FAIL_USER.format(
                description=self.description,
                input_format=self.input_format,
                output_format=self.output_format,
                test_case=str(p1.results.print_tc_result(tc)),
                code=p1.code
            )
        elif strategy == "f_time":
            system = prompts.MUT_TIME_SYSTEM
            user = prompts.MUT_TIME_USER.format(
                description=self.description,
                input_format=self.input_format,
                output_format=self.output_format,
                profile=p1.results.report_time(tc),
                code=p1.code
            )
        else:  # f_mem
            system = prompts.MUT_MEM_SYSTEM
            user = prompts.MUT_MEM_USER.format(
                description=self.description,
                input_format=self.input_format,
                output_format=self.output_format,
                profile=p1.results.report_mem(tc),
                code=p1.code
            )
        return await Models.run(system, user), p1.fitness, p1.ext

    # ---- async orchestration ------------------------------------- #

    async def _run_correct_async(self, buggy: Program, references: list[Program]) -> list[Program]:
        """Generate *count* initial candidates (syntax-only validation done in GA)."""
        tasks = [asyncio.create_task(self._correct_prompt(buggy, reference)) 
            for reference in references]

        programs = []
        pbar = tqdm_async(total=len(tasks), desc="Correct", leave=False, position=2)
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
        tasks = [asyncio.create_task(self._efficient_prompt(correct)) 
            for correct in corrects]

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
    
    async def _run_variation_async(self, pairs: list[tuple]) -> list[Program]:
        """Generate one crossover + one mutation offspring per pair."""
        tasks = []

        for p1, p2, t_star in pairs:
            if t_star is None:
                continue
            # Crossover task
            tasks.append(asyncio.create_task(
                self._crossover_prompt(p1, p2, t_star)))
            # Mutation task
            tasks.append(asyncio.create_task(
                self._mutation_prompt(p1, t_star)))

        programs = []
        pbar = tqdm_async(total=len(tasks), desc="Variation", leave=False, position=2)
        for coro in asyncio.as_completed(tasks):
            patch, fitness, ext = await coro
            pbar.update(1)
            if patch is None or not patch.strip():
                continue
            child = Program(
                id=f"child_{len(programs) + 1}",
                code=patch,
                ext=ext,
            )
            child.prev_fitness = fitness
            programs.append(child)
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

    def correct(self, buggy: Program, references: list[Program]) -> list[Program]:
        """Generate *count* candidate programs for initial population."""
        loop = self._asyncio_loop()
        return loop.run_until_complete(self._run_correct_async(buggy, references))

    def efficient(self, corrects: list[Program]) -> list[Program]:
        """Generate *count* candidate programs for EffiLearner."""
        loop = self._asyncio_loop()
        return loop.run_until_complete(self._run_efficient_async(corrects))

    def run(self, pairs: list[tuple]) -> list[Program]:
        """Generate offspring from (p1, p2, t*) pairs."""
        loop = self._asyncio_loop()
        return loop.run_until_complete(self._run_variation_async(pairs))
