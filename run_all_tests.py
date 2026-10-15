import unittest
import sys


def run_tests():

    loader = unittest.TestLoader()

    suite = unittest.TestSuite()


    test_modules = [

        "tests.test_intent",
        "tests.test_graph_routing",
        "tests.test_memory",
        "tests.test_profile_extractor",
        "tests.test_memory_pipeline",
        "tests.test_checkpoint_memory",

    ]


    for module in test_modules:

        tests = loader.loadTestsFromName(
            module
        )

        suite.addTests(tests)


    runner = unittest.TextTestRunner(
        verbosity=2
    )


    result = runner.run(suite)


    if not result.wasSuccessful():

        sys.exit(1)



if __name__ == "__main__":

    run_tests()