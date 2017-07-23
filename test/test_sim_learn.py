import unittest
import sim_learn


class TestSimLearn(unittest.TestCase):
    def test_id(self):
        sim_learn.main(["sim_learn", "8", "2", "id", "xor", "20", "1", "2", "1234", "1234"])

    def test_atf(self):
        sim_learn.main(["sim_learn", "8", "2", "atf", "xor", "20", "1", "2", "1234", "1234"])

    def test_mm(self):
        sim_learn.main(["sim_learn", "8", "2", "mm", "xor", "20", "1", "2", "1234", "1234"])

    def test_lightweight_secure(self):
        sim_learn.main(["sim_learn", "8", "2", "lightweight_secure", "xor", "20", "1", "2", "1234", "1234"])

    def test_1_n_bent(self):
        sim_learn.main(["sim_learn", "8", "2", "1_n_bent", "xor", "20", "1", "2", "1234", "1234"])

    def test_1_1_bent(self):
        sim_learn.main(["sim_learn", "8", "2", "1_1_bent", "xor", "20", "1", "2", "1234", "1234"])

    def test_ip_mod2_id(self):
        sim_learn.main(["sim_learn", "8", "2", "id", "ip_mod2", "20", "1", "2", "1234", "1234"])

    def test_ip_mod2_atf(self):
        sim_learn.main(["sim_learn", "8", "2", "atf", "ip_mod2", "20", "1", "2", "1234", "1234"])

    def test_ip_mod2_mm(self):
        sim_learn.main(["sim_learn", "8", "2", "mm", "ip_mod2", "20", "1", "2", "1234", "1234"])

    def test_ip_mod2_lightweight_secure(self):
        sim_learn.main(["sim_learn", "8", "2", "lightweight_secure", "ip_mod2", "20", "1", "2", "1234", "1234"])

    def test_ip_mod2_1_n_bent(self):
        sim_learn.main(["sim_learn", "8", "2", "1_n_bent", "ip_mod2", "20", "1", "2", "1234", "1234"])

    def test_ip_mod2_1_1_bent(self):
        sim_learn.main(["sim_learn", "8", "2", "1_1_bent", "ip_mod2", "20", "1", "2", "1234", "1234"])


