"""Run every offline test in fieldvision_agents (fakes and made-up data only: no model calls, no network).

    fieldvision test                                # or: python -m fieldvision_agents.run_tests
                                                    # exit code 1 if anything fails
"""

import importlib
import sys
import traceback

TEST_MODULES = [
    "fieldvision_agents.react.test_supervisor",
    "fieldvision_agents.reporter.test_reporter",
    "fieldvision_agents.analyst_screen.test_views",
    "fieldvision_agents.evaluation.test_evaluate",
    "fieldvision_agents.vision.test_vision",
    "fieldvision_agents.planner.test_planner",
    "fieldvision_agents.benchmark.test_benchmark",
    "fieldvision_agents.fieldvision_bridge.tests.test_prevalence",
    "fieldvision_agents.fieldvision_bridge.tests.test_toolbox",
    "fieldvision_agents.fieldvision_bridge.tests.test_field_qa",
    "fieldvision_agents.fieldvision_bridge.tests.test_rag",
    "fieldvision_agents.tempdiff.test_tempdiff",
    "fieldvision_agents.market.test_price_impact",
    "fieldvision_agents.app.test_app",
    "fieldvision_agents.gap.test_gap",
    "fieldvision_agents.gap.test_gap_tool",
]


def run_all(modules=TEST_MODULES):
    passed, failed = 0, []
    for name in modules:
        module = importlib.import_module(name)
        tests = [(n, fn) for n, fn in vars(module).items()
                 if n.startswith("test_") and callable(fn) and fn.__module__ == module.__name__]
        ok = 0
        for test_name, fn in tests:
            try:
                fn()
                ok += 1
            except Exception:
                failed.append(f"{name}.{test_name}")
                traceback.print_exc()
        passed += ok
        print(f"{'PASS' if ok == len(tests) else 'FAIL'}  {name}: {ok}/{len(tests)}")
    print(f"\n{passed} passed, {len(failed)} failed" + (f": {failed}" if failed else ""))
    return not failed


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
