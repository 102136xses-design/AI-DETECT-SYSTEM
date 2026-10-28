"""Explainable, non-medical video analytics. No fabricated injury probabilities."""
from __future__ import annotations
import math
import numpy as np

# COCO-17 pose indices: shoulders 5/6, elbows 7/8, wrists 9/10,
# hips 11/12, knees 13/14, ankles 15/16.
SKELETON = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

def angle(a,b,c):
    """Joint angle in degrees, b is vertex; None for degenerate vectors."""
    a,b,c = [np.asarray(p,dtype=float) for p in (a,b,c)]
    u,v=a-b,c-b
    d=np.linalg.norm(u)*np.linalg.norm(v)
    if d<1e-8: return None
    return float(np.degrees(np.arccos(np.clip(np.dot(u,v)/d,-1,1))))

def extract_pose_metrics(keypoints, confidence=0.35):
    """keypoints: (17,3) x,y,confidence, in image pixels."""
    k=np.asarray(keypoints,dtype=float)
    if k.shape != (17,3): raise ValueError('Expected 17 COCO keypoints with x,y,confidence')
    def joint(a,b,c):
        if min(k[a,2],k[b,2],k[c,2])<confidence: return None
        return angle(k[a,:2],k[b,:2],k[c,:2])
    return {'left_knee_deg':joint(11,13,15),'right_knee_deg':joint(12,14,16),
            'left_elbow_deg':joint(5,7,9),'right_elbow_deg':joint(6,8,10),
            'left_hip_deg':joint(5,11,13),'right_hip_deg':joint(6,12,14)}

def movement_observations(metrics):
    """Non-diagnostic coaching prompts; thresholds are demo heuristics."""
    out=[]
    lk,rk=metrics.get('left_knee_deg'),metrics.get('right_knee_deg')
    if lk is not None and rk is not None and abs(lk-rk)>25:
        out.append('Large left/right knee-angle difference in this frame; review camera perspective and technique.')
    le,re=metrics.get('left_elbow_deg'),metrics.get('right_elbow_deg')
    if le is not None and re is not None and abs(le-re)>35:
        out.append('Large elbow-angle difference in this frame; compare with the intended drill and camera angle.')
    if not out: out.append('No demo threshold exceeded in this sampled frame; this does not establish safe mechanics.')
    return out

def spacing_proxy(centers, width):
    """Screen-space horizontal spread, NOT basketball court spacing in meters."""
    if len(centers)<2 or width<=0: return None
    xs=[float(p[0]) for p in centers]
    return round((max(xs)-min(xs))/float(width),3)

def summarize_frames(rows):
    """Aggregate detected people and available joint-angle measurements."""
    if not rows:return {'frames':0,'mean_people':0,'metrics':{}}
    names=['left_knee_deg','right_knee_deg','left_elbow_deg','right_elbow_deg','left_hip_deg','right_hip_deg']
    result={}
    for name in names:
        vals=[r[name] for r in rows if r.get(name) is not None]
        result[name]=round(float(np.median(vals)),1) if vals else None
    return {'frames':len(rows),'mean_people':round(float(np.mean([r['people_count'] for r in rows])),2),'metrics':result}
