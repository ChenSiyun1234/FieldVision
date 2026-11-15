"""Run every bridge test: python -m fieldvision_agents.fieldvision_bridge.tests.run"""

from . import test_field_qa, test_prevalence, test_rag, test_toolbox


def run_all():
    passed = 0
    for module in (test_prevalence, test_toolbox, test_field_qa, test_rag):
        for name, fn in vars(module).items():
            if name.startswith("test_") and callable(fn):
                fn()
                print(f"PASS  {module.__name__.split('.')[-1]}.{name}")
                passed += 1
    print(f"\n{passed} tests passed")
    return passed


if __name__ == "__main__":
    run_all()
