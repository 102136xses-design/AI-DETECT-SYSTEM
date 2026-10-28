import unittest
import numpy as np
from analytics import angle,extract_pose_metrics,spacing_proxy,summarize_frames

class TestAnalytics(unittest.TestCase):
    def test_right_angle(self):self.assertAlmostEqual(angle((1,0),(0,0),(0,1)),90)
    def test_straight(self):self.assertAlmostEqual(angle((-1,0),(0,0),(1,0)),180)
    def test_missing_keypoints(self):
        k=np.zeros((17,3));self.assertIsNone(extract_pose_metrics(k)['left_knee_deg'])
    def test_spacing(self):self.assertEqual(spacing_proxy([(0,0),(50,20)],100),.5)
    def test_summary(self):self.assertEqual(summarize_frames([])['frames'],0)
if __name__=='__main__':unittest.main()
