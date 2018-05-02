import numpy as np
from algorithms import helpers


def test_QR(Ntests):
    passed = 0
    critical = 0
    for _ in range(Ntests):
        try:
            n = np.random.randint(2, 11)
            X = np.random.uniform(low=0.0,
                                  high=100.0,
                                  size=(n, n))
            Q, R = helpers.qr_factorize(X)
            assert all(np.isclose(Q.dot(R), X).flatten())
            passed += 1
        except AssertionError:
            print("AssertionError with:")
            print(X)
            continue
        except Exception:
            print("Other Error with:")
            print(X)
            critical += 1

    print("Test Results:")
    print("Passed {} of {} Tests.".format(passed, Ntests))
    print("Failed {} tests.".format(Ntests-passed-critical))
    print("{} tests failed critically".format(critical))
    if passed == Ntests:
        return True
    else:
        return False


assert test_QR(1000)

A = np.array([[2, 4, 2],
              [-1, 0, -4],
              [2, 2, -1]])
Q, R = helpers.qr_factorize(A)
Q
np.round(R)
np.round(Q.T.dot(Q))
A1 = np.dot(Q.T, A).dot(Q)
Q1, R1 = helpers.qr_factorize(A1)
A2 = np.dot(Q1.T, A1).dot(Q1)
A2sym = A2 + A2.T
np.tril(A2sym) == np.triu(A2sym).T
